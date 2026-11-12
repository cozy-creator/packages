"""Krea 2 (Raw and Turbo), component-staged over Diffusers' modules.

The DiT and VAE are Diffusers' (`Krea2Transformer2DModel`, `AutoencoderKLQwenImage`, 0.42.1); the text
encoder is Transformers' Qwen3-VL-4B, its vision tower read only with pictures. Prompt conditioning, the
schedule and the sampler are Diffusers' `Krea2Pipeline`, Krea's reference code (`krea-ai/krea-2` db3984f) in
its form. Runtime constructs and fills the modules and owns their residency.

Krea 2 takes no pictures upstream; its community adapters (`adapters.py`) add them as clean VAE tokens in the
DiT's sequence (`Reference`), as Qwen3-VL vision tokens in its text (`encode_pictures`), or as a control
latent beside each image token (`denoise(control=)`). An adapter's LoRA reaches the DiT through permanent
hooks (`cozy_runtime.lora.Hook`), which Runtime bakes into the resident weights while it is armed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import torch
from cozy_runtime import rng, tiles
from cozy_runtime.accel import fusion, hold_dtype, lora
from cozy_runtime.author import (
    Config,
    Loader,
    Model,
    ModelAssets,
    OutputError,
    Serving,
    memoize,
    preview,
    uses_components,
)
from cozy_runtime.accel.kernels import Plan
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
    QwenImageCausalConv3d,
    QwenImageDecoder3d,
    QwenImageResample,
    QwenImageResidualBlock,
    QwenImageRMS_norm,
    QwenImageUpsample,
)
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel, Krea2TransformerBlock
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast, Qwen3VLConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLVisionModel

if TYPE_CHECKING:
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageUpBlock

#: Pixels per DiT token: the VAE's 8x, then 2x2 patches.
PATCH = 16
#: The DiT's activations; its RMSNorm scales stay F32 (Diffusers' `_keep_in_fp32_modules`).
DTYPE = torch.bfloat16
#: The text encoder's hidden states the DiT reads (`Krea2Pipeline`'s `text_encoder_select_layers`).
TAPS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
#: Diffusers' fixed text length: at most 512 tokens after the dropped system prefix.
TEXT_TOKENS = 512
#: `Krea2Pipeline`'s template (Qwen-Image's): the prompt between a system prefix of 34 tokens, which the
#: DiT never reads, and a 5-token assistant suffix.
PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS, SUFFIX_TOKENS = 34, 5
#: Turbo's fixed timestep shift (Diffusers' `is_distilled`); Raw's follows the token count.
DISTILLED_MU = 1.15
_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
#: Qwen3-VL-4B-Instruct's image processor (`preprocessor_config.json`): 16 px patches merged 2x2, each patch
#: twice in time, mean and std 0.5, 64Ki to 16Mi pixels on the 32 px grid.
VISION_PATCH, VISION_MERGE, VISION_FRAMES = 16, 2, 2
VISION_PIXELS = (65536, 16777216)
IMAGE_PAD = "<|image_pad|>"
VISION = "<|vision_start|>" + IMAGE_PAD + "<|vision_end|>"
#: Adapters armed at once: an edit's and a control's (`Krea2.denoise`'s `overlays`).
SLOTS = 2


class TextEncoder(nn.Module):
    """Qwen3-VL-4B: its language model (`language_model.*`) up to the deepest tapped layer, and its vision
    tower (`visual.*`), which only a prompt with pictures runs. The later layers and the final norm are never
    built, so their bytes are never read; with the norm an identity, hidden state `depth` stays the pre-norm
    output Diffusers taps."""

    def __init__(self, config: Qwen3VLConfig, depth: int) -> None:
        super().__init__()
        self.language_model: Qwen3VLTextModel = Qwen3VLTextModel._from_config(  # type: ignore[no-untyped-call]  # Transformers' _from_config is unannotated
            config.text_config, dtype=DTYPE
        )
        self.language_model.layers = nn.ModuleList(list(self.language_model.layers)[:depth])
        self.language_model.norm = nn.Identity()  # type: ignore[assignment, unused-ignore]  # the final norm, removed
        self.visual: Qwen3VLVisionModel = Qwen3VLVisionModel._from_config(  # type: ignore[no-untyped-call]  # Transformers' _from_config is unannotated
            config.vision_config, dtype=DTYPE
        )
        self.image_token_id = int(config.image_token_id)


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Transformers' Qwen2-VL `smart_resize` at Qwen3-VL's factor (32) and pixel range."""
    factor = VISION_PATCH * VISION_MERGE
    low, high = VISION_PIXELS
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > high:
        beta = math.sqrt(height * width / high)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < low:
        beta = math.sqrt(low / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def vision_patches(picture: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Transformers' Qwen2-VL image processor on one picture: bicubic to `smart_resize`, normalized by mean
    and std 0.5, cut into 16 px patches in merge order, each repeated over 2 frames: [patches, 1536] F32 and
    its (1, rows, columns) grid in patches."""
    height, width = smart_resize(picture.height, picture.width)
    pixels = np.asarray(picture.convert("RGB").resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
    x = (torch.from_numpy(pixels).permute(2, 0, 1) / 255.0 - 0.5) / 0.5
    rows, columns, p, m = height // VISION_PATCH, width // VISION_PATCH, VISION_PATCH, VISION_MERGE
    x = x.reshape(3, rows // m, m, p, columns // m, m, p).permute(1, 4, 2, 5, 0, 3, 6)
    x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, VISION_FRAMES, -1, -1)
    return x.reshape(rows * columns, 3 * VISION_FRAMES * p * p), (1, rows, columns)


def mrope(ids: Sequence[int], pad: int, grids: Sequence[tuple[int, int, int]]) -> torch.Tensor:
    """Qwen3-VL's `get_rope_index` for one sequence: text counts up on all three axes; a picture's merged
    tokens sit on its (frame, row, column) grid from where the text reached, which then moves on by its
    longer side. [3, 1, len(ids)]."""
    parts: list[torch.Tensor] = []
    at, i, pictures = 0, 0, iter(grids)
    while i < len(ids):
        if ids[i] == pad:
            frames, rows, columns = next(pictures)
            rows, columns = rows // VISION_MERGE, columns // VISION_MERGE
            t, h, w = torch.meshgrid(torch.arange(frames), torch.arange(rows), torch.arange(columns), indexing="ij")
            parts.append(torch.stack([t, h, w]).reshape(3, -1) + at)
            at += max(rows, columns)
            i += frames * rows * columns
            continue
        run = i
        while run < len(ids) and ids[run] != pad:
            run += 1
        parts.append(torch.arange(at, at + run - i).view(1, -1).expand(3, -1))
        at += run - i
        i = run
    return torch.cat(parts, dim=1)[:, None]


class Block(Krea2TransformerBlock):
    """Diffusers' block with its modulation in the activations' dtype: the checkpoint stores the
    block's `scale_shift_table` in F32, which Diffusers' bf16 load would cast. With `split`, `temb` holds
    two rows: the tokens before `split` read the first, the rest the second (an adapter's references at
    time 0)."""

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor,
                image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None = None, split: int | None = None) -> torch.Tensor:
        rows = temb.unflatten(-1, (6, -1))
        modulation = (rows + self.scale_shift_table).to(hidden_states.dtype)
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        def modulated(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            if split is None:
                return (1.0 + scale) * x + shift
            return torch.cat(((1.0 + scale[0]) * x[:, :split] + shift[0], (1.0 + scale[1]) * x[:, split:] + shift[1]), 1)

        def gated(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            return gate * x if split is None else torch.cat((gate[0] * x[:, :split], gate[1] * x[:, split:]), 1)

        attn_out = self.attn(modulated(self.norm1(hidden_states), prescale, preshift),
                             attention_mask=attention_mask, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gated(attn_out, pregate)
        ff_out = self.ff(modulated(self.norm2(hidden_states), postscale, postshift))
        out: torch.Tensor = hidden_states + gated(ff_out, postgate)
        return out


def _modulation(block: nn.Module, arguments: Mapping[str, object]) -> fusion.Modulation:
    """`Block.forward`'s six adaLN rows in the activations' dtype; a batch of one reads row 0, and with
    `split` the tokens from it on read row 1."""
    # Runtime fuses the `Block`s and binds their forward's tensors.
    hidden, temb = cast(torch.Tensor, arguments["hidden_states"]), cast(torch.Tensor, arguments["temb"])
    rows6 = temb.unflatten(-1, (6, -1))
    table = (rows6 + cast(Block, block).scale_shift_table).to(hidden.dtype)
    rows = [row.reshape(-1, row.shape[-1]) for row in table.unbind(-2)]
    batch, tokens = hidden.shape[:2]
    split = cast("int | None", arguments.get("split"))
    if split is not None:
        index = (torch.arange(tokens, device=hidden.device) >= split).long()
    else:
        index = torch.arange(batch, device=hidden.device).repeat_interleave(tokens)
        index = index[:1] if batch == 1 else index
    prescale, preshift, pregate, postscale, postshift, postgate = rows
    return fusion.Modulation(index, prescale, pregate, postscale, postgate, preshift, postshift)


#: `Block` as Runtime fuses it: zero-centered fp32 RMSNorms (rounding once), modulated by `1 + scale`
#: and `shift`, identity gates, per-head q/k norms with Flux's interleaved RoPE, a SwiGLU MLP. The GEMMs
#: stay the model's own (int8 leaves quantize each normed row once for every projection reading it).
BLOCK = fusion.DiTBlock(
    Block,
    width=6144,
    head_dim=128,
    modulation=_modulation,
    norms=("norm1", "norm2"),
    norm="rms",
    attention="attn",
    arguments=("attention_mask", "image_rotary_emb"),
    mlp=fusion.GatedMLP("ff", gate="gate", up="up", down="down"),
    rope=fusion.Rope("image_rotary_emb", interleaved=True),
    shift=True,
    weights="fp32",
    centered=True,
    rms_rounding="once",
)
#: The VAE's dtype on CUDA. Its F32 lane holds exactly Qwen-Image's bf16 values, which fp16 holds exactly
#: too: against an F32 decode, fp16 measures 62.9 dB and bf16 53.4 (dtype audit M1).
VAE_DTYPE = torch.float16
#: The VAE's RMS norms in one kernel each, with the SiLU each residual block and the decoder's head apply
#: after them; Diffusers computes them in F32 copies of every activation.
VAE_NORM = fusion.ChannelNorm(
    QwenImageRMS_norm,
    silu=((QwenImageResidualBlock, ("norm1", "norm2")), (QwenImageDecoder3d, ("norm_out",))),
    dtype="fp16",
)
#: A lone frame's 3x3x3 causal convolutions on their last temporal tap, the only one it meets (Diffusers
#: pads two zero frames and convolves all 27 taps), and its nearest upsampling in its own dtype, not F32.
VAE_FRAME = fusion.LoneFrame(QwenImageCausalConv3d, upsample=QwenImageUpsample)


def as_stored(config: Config, component: str, root: nn.Module) -> None:
    """Each parameter in the dtype the checkpoint stores it in, which the Runtime fills without a
    cast. Krea's Diffusers checkpoint keeps the norms, the modulation tables, the embedders and the
    final layer in F32 (the reference checkpoint has those in bf16, its norms in F32)."""
    for path, module in root.named_modules():
        for name, param in list(module.named_parameters(recurse=False)):
            stored = config.tensor_dtype(f"{component}.{path}.{name}" if path else f"{component}.{name}", default="bf16")
            dtype = _DTYPES.get(stored)
            if dtype is not None and param.dtype != dtype:
                module.register_parameter(name, nn.Parameter(param.to(dtype), requires_grad=False))


@dataclass(frozen=True)
class Reference:
    """A clean picture in the DiT's sequence, as an adapter places it: its VAE latents [1, 16, h/8, w/8]
    (normalized, on the CPU), its RoPE frame and the (row, column) its grid starts at, in tokens."""

    latents: torch.Tensor
    frame: int
    origin: tuple[float, float] = (0.0, 0.0)
    #: The target's attention to it, multiplied (Identity Edit's `ref_boost`: a logit bias of its log).
    boost: float = 1.0


@dataclass(frozen=True)
class Branch:
    """One prompt of a request (positive or negative) and the pictures its sequence carries: `before` the
    target at the step's time (Identity Edit), `after` it at time 0 (ai-toolkit's edits), the latter
    attending only among themselves when `isolated` (a LoRA trained with ai-toolkit's `kv_cache`)."""

    text: torch.Tensor
    before: tuple[Reference, ...] = ()
    after: tuple[Reference, ...] = ()
    isolated: bool = False


@dataclass(frozen=True)
class Stream:
    """A branch as each step reads it, built once: its fused text, its pictures through `img_in`, its
    rotary tables, its attention bias, and where its target tokens sit."""

    text: torch.Tensor
    before: torch.Tensor | None
    after: torch.Tensor | None
    rotary: tuple[torch.Tensor, torch.Tensor]
    mask: torch.Tensor | None
    target: int


def packed(latents: torch.Tensor) -> torch.Tensor:
    """VAE latents [1, C, 2r, 2c] as DiT tokens [1, r*c, 4C], `unpack`'s inverse."""
    _, channels, height, width = latents.shape
    rows, columns = height // 2, width // 2
    x = latents.view(1, channels, rows, 2, columns, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(1, rows * columns, channels * 4)


def grid(frame: float, rows: int, columns: int, origin: tuple[float, float], device: torch.device) -> torch.Tensor:
    """A (frame, row, column) RoPE coordinate per token of a rows x columns grid starting at `origin`."""
    ids = torch.zeros(rows, columns, 3, device=device)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(rows, device=device)[:, None] + origin[0]
    ids[..., 2] = torch.arange(columns, device=device)[None, :] + origin[1]
    return ids.reshape(rows * columns, 3)


class Transformer(Krea2Transformer2DModel):
    """`Krea2Transformer2DModel` whose forward is one sampling step for every prompt of a request."""

    #: Each slot's armed adapter, its factors by linear (`adapters.Adapter.applied`), else None.
    _cozy_overlays: list[Mapping[str, lora.Factors] | None]
    #: A control adapter's expanded input projection while it is armed (`adapters.Adapter.applied`): its
    #: weight [6144, 128] over each image token and its control token side by side, and its bias.
    _cozy_projection: tuple[torch.Tensor, torch.Tensor] | None = None

    def embed(self, latents: torch.Tensor, control: torch.Tensor | None) -> torch.Tensor:
        """`img_in`, in its stored dtype, or with `control` tokens the armed control adapter's projection over
        both (Patil's `ControlInputLayer`)."""
        if control is None:
            return cast(torch.Tensor, self.img_in(latents.to(_dtype(self.img_in)))).to(latents.dtype)
        if self._cozy_projection is None:
            raise ValueError("a control map reaches Krea 2 only through an armed control adapter")
        weight, bias = self._cozy_projection
        x = torch.cat((latents, control.to(latents.dtype)), dim=-1).to(weight.dtype)
        return F.linear(x, weight, bias.to(weight.dtype)).to(latents.dtype)

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor, streams: list[Stream],
                control: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Each prompt's velocity at `timestep` [1] (0-1): `Krea2Transformer2DModel.forward` op for op past
        what no step changes (`Krea2.stream`). A branch's pictures before the target share the step's
        modulation; those after it take time 0's. The prompts pass through each block in turn, so a
        streamed block comes in once per step."""
        # The embedders and the final layer run in their stored F32; the blocks in bf16.
        temb = self.time_embed(timestep, dtype=_dtype(self.time_embed))
        modulation = self.time_mod_proj(F.gelu(temb, approximate="tanh").to(_dtype(self.time_mod_proj)))
        both = modulation
        if any(stream.after is not None for stream in streams):
            rest = self.time_embed(torch.zeros_like(timestep), dtype=_dtype(self.time_embed))
            rest = self.time_mod_proj(F.gelu(rest, approximate="tanh").to(_dtype(self.time_mod_proj)))
            both = torch.cat((modulation, rest))
        image = self.embed(latents, control)
        hidden, splits = [], []
        for stream in streams:
            parts = [stream.text, *([stream.before] if stream.before is not None else []), image]
            splits.append(None if stream.after is None else sum(int(p.shape[1]) for p in parts))
            hidden.append(torch.cat([*parts, *([stream.after] if stream.after is not None else [])], dim=1))
        for block in self.transformer_blocks:
            hidden = [block(h, modulation if split is None else both, stream.rotary, stream.mask, split)
                      for h, stream, split in zip(hidden, streams, splits, strict=True)]
        size = int(image.shape[1])
        return [self.final_layer(h[:, stream.target:stream.target + size], temb.to(_dtype(self.final_layer)))
                for h, stream in zip(hidden, streams, strict=True)]


def _armed(transformer: nn.Module, slot: int, path: str) -> lora.Factors | None:
    """An armed adapter's factors at `path` in `slot` (`Krea2.denoise`'s `overlays`), else None."""
    armed: list[Mapping[str, lora.Factors] | None] = getattr(transformer, "_cozy_overlays", [None] * SLOTS)
    sites = armed[slot]
    return None if sites is None else sites.get(path)


class Krea2Graph:
    def __init__(self, config: Config) -> None:
        with no_init_weights():
            transformer = construct(Krea2Transformer2DModel, config.section("transformer").mapping())
            encoder = TextEncoder(Qwen3VLConfig(**config.section("text_encoder").mapping()), max(TAPS))
            vae = construct(AutoencoderKLQwenImage, config.section("vae").mapping())
        transformer.__class__ = Transformer
        for block in transformer.transformer_blocks:
            block.__class__ = Block
        # Every linear of the DiT serves an armed adapter's factors (`adapters.Adapter`), one hook per slot.
        dit = cast(Transformer, transformer)  # its class, set above
        dit._cozy_overlays = [None] * SLOTS  # a plain attribute: nn.Module keeps a list as such
        for path, module in dit.named_modules():
            if isinstance(module, nn.Linear):
                for slot in range(SLOTS):
                    module.register_forward_hook(lora.Hook(partial(_armed, dit, slot, path)))
        as_stored(config, "transformer", transformer)
        as_stored(config, "vae", vae)
        self.components: Components = {
            "transformer": cast(Transformer, transformer).eval(),  # its class, set above
            "text_encoder": encoder.eval(),
            "vae": vae.eval(),
        }
        self.scheduler_config: Mapping[str, object] = config.section("scheduler").mapping()


class Components(TypedDict):
    transformer: Transformer
    text_encoder: TextEncoder
    vae: AutoencoderKLQwenImage


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def build_tokenizer(assets: ModelAssets) -> PreTrainedTokenizerFast:
    """The checkpoint's Qwen2 tokenizer (`tokenizer/`), parsed in memory."""
    config = json.loads(assets.read("tokenizer/tokenizer_config.json"))
    for key in ("tokenizer_class", "backend", "is_local", "name_or_path", "tokenizer_file",
                "added_tokens_decoder", "chat_template", "extra_special_tokens", "errors"):
        config.pop(key, None)
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # Transformers' tokenizer __init__ is unannotated
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer/tokenizer.json").decode()), **config
    )


def unpack(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Diffusers' `_unpack_latents`: DiT tokens [1, (H/16)(W/16), 64] as one VAE frame [1, 16, 1, H/8, W/8]."""
    rows, columns = height // PATCH, width // PATCH
    channels = latents.shape[-1] // 4
    latents = latents.view(1, rows, columns, channels, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(1, channels, 1, rows * 2, columns * 2)


def _dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def fused_text(transformer: Krea2Transformer2DModel, text: torch.Tensor) -> torch.Tensor:
    """`Krea2TextFusion` then `txt_in` over one prompt's tapped states, each module in its stored dtype:
    the step-invariant text stream of the DiT's forward."""
    fusion = transformer.text_fusion
    batch, length, layers, width = text.shape
    hidden = text.reshape(batch * length, layers, width)
    for block in fusion.layerwise_blocks:
        hidden = block(hidden.contiguous())
    hidden = hidden.reshape(batch, length, layers, width).permute(0, 1, 3, 2)
    hidden = fusion.projector(hidden.to(_dtype(fusion.projector))).squeeze(-1).to(text.dtype)
    for block in fusion.refiner_blocks:
        hidden = block(hidden)
    out: torch.Tensor = transformer.txt_in(hidden.to(_dtype(transformer.txt_in)))
    return out.to(text.dtype)


class Krea2(Model[Krea2Graph]):
    """One Krea 2 checkpoint, Raw or Turbo. Their configs are identical and the pipeline's `is_distilled` is
    not in the model's config, so the caller names the regime (`denoise(distilled=)`)."""

    #: Caller LoRAs on the DiT, at any strength: the Runtime bakes them into the weights.
    serving = Serving(loras=("transformer",))

    graph: Krea2Graph
    tokenizer: PreTrainedTokenizerFast

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Krea2Graph)
        self.tokenizer = build_tokenizer(loader.assets)

    def execution_plan(self, plan: Plan) -> str:
        vae = self.graph.components["vae"]
        encoder = vae.encoder
        # Not the encoder's temporal downsamplers: their `time_conv` only ever runs on a next frame's cache.
        frame = (
            *(block for block in encoder.down_blocks if not isinstance(block, QwenImageResample)),
            encoder.conv_in, encoder.mid_block, encoder.conv_out, vae.quant_conv,
            vae.decoder, vae.post_quant_conv,
        )
        installed = (
            fusion.install(plan, BLOCK, self.graph.components["transformer"]),
            fusion.install_norms(plan, VAE_NORM, vae),
            fusion.install_lone_frame(plan, VAE_FRAME, *frame),
        )
        return "+".join(identity for identity in installed if identity)

    def hold_vae(self) -> None:
        """Encode and decode in `VAE_DTYPE` on CUDA; the CPU keeps the stored F32. The Runtime casts the VAE's
        stored F32 when it next fills it. Called before any VAE scope opens."""
        if torch.device(str(self.device)).type == "cuda":
            hold_dtype(self.graph.components["vae"], VAE_DTYPE)

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str) -> torch.Tensor:
        """The DiT's text conditioning [1, S, 12, 2560] (bf16, on the CPU): Diffusers' Qwen-Image chat
        template, at most 512 tokens after its 34-token system prefix, the 12 tapped hidden states.

        Diffusers pads the prompt to 512 tokens and masks the padding, with positions counting real
        tokens only; the DiT reads text keys at rotary position 0 and masks padded ones. Unpadded, the
        same tokens get the same positions and every real token the same values, without the padding's
        compute or a mask (which leaves only SDPA among attention kernels)."""
        encoder = self.graph.components["text_encoder"].language_model
        device = torch.device(str(self.device))
        tokens = self.tokenizer(
            PROMPT_PREFIX + prompt, truncation=True,
            max_length=TEXT_TOKENS + PREFIX_TOKENS - SUFFIX_TOKENS,
        ).input_ids + self.tokenizer(PROMPT_SUFFIX).input_ids
        ids = torch.tensor([tokens], device=device)
        positions = torch.arange(ids.shape[1], device=device).view(1, 1, -1).expand(3, 1, -1)
        with torch.inference_mode():
            hidden = encoder(input_ids=ids, position_ids=positions, output_hidden_states=True,
                             use_cache=False).hidden_states
            stacked = torch.stack([hidden[i] for i in TAPS], dim=2)[:, PREFIX_TOKENS:]
        return stacked.to(DTYPE).cpu()

    @memoize
    @uses_components("text_encoder")
    def encode_pictures(self, prompt: str, pictures: tuple[Image.Image, ...], labels: bool) -> torch.Tensor:
        """`encode` with `pictures` read by Qwen3-VL's vision tower in the user's turn, before the prompt, each
        `Picture N: ` first with `labels` (ai-toolkit's edits) or bare (Identity Edit), as Transformers' Qwen3-VL
        reads them (DeepStack included). The pictures come at the size the adapter reads them; their tokens
        stay in the conditioning, after the 34-token prefix. [1, S, 12, 2560] bf16, on the CPU."""
        encoder = self.graph.components["text_encoder"]
        device = torch.device(str(self.device))
        heads = "".join(f"Picture {n}: {VISION}" if labels else VISION for n in range(1, len(pictures) + 1))
        words = self.tokenizer(prompt).input_ids[:TEXT_TOKENS - SUFFIX_TOKENS]
        pad = encoder.image_token_id
        patches, grids = zip(*(vision_patches(picture) for picture in pictures), strict=True)
        ids: list[int] = []
        sizes = iter(rows * columns // VISION_MERGE**2 for _, rows, columns in grids)
        for token in self.tokenizer(PROMPT_PREFIX + heads).input_ids + words + self.tokenizer(PROMPT_SUFFIX).input_ids:
            ids.extend([pad] * next(sizes) if token == pad else [token])
        positions = mrope(ids, pad, grids).to(device)
        tokens = torch.tensor([ids], device=device)
        visual = tokens == pad
        with torch.inference_mode():
            seen = encoder.visual(torch.cat(patches).to(device, DTYPE), grid_thw=torch.tensor(grids, device=device))
            language = encoder.language_model
            embeds = language.embed_tokens(tokens)
            embeds = embeds.masked_scatter(visual[..., None], seen.pooler_output.to(embeds.dtype))
            hidden = language(inputs_embeds=embeds, position_ids=positions, visual_pos_masks=visual,
                              deepstack_visual_embeds=list(seen.deepstack_features), output_hidden_states=True,
                              use_cache=False).hidden_states
            stacked = torch.stack([hidden[i] for i in TAPS], dim=2)[:, PREFIX_TOKENS:]
        return stacked.to(DTYPE).cpu()

    @memoize
    @uses_components("vae")
    def encode_image(self, picture: Image.Image) -> torch.Tensor:
        """`decode` inverted: a picture already at its size (16 px grid) through the VAE's encoder, its mode,
        normalized: [1, 16, H/8, W/8] F32 on the CPU. Diffusers' `encode` of one frame without its feature
        cache (each convolution's input kept for a next frame an image never has)."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        z_dim = int(vae.config["z_dim"])
        with torch.inference_mode():
            pixels = torch.from_numpy(np.asarray(picture.convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1
            moments = vae.quant_conv(vae.encoder(pixels[None, :, None].to(device, vae.dtype)))
            z = moments[:, :z_dim, 0].float()
            mean = z.new_tensor(vae.config["latents_mean"]).view(1, -1, 1, 1)
            std = z.new_tensor(vae.config["latents_std"]).view(1, -1, 1, 1)
            return ((z - mean) / std).cpu()

    def stream(self, branch: Branch, rows: int, columns: int, device: torch.device) -> Stream:
        """What no step changes of one branch: its fused text, its pictures through `img_in` (clean, so
        once), its rotary tables over [text | before | target | after], and its attention bias: the target's
        logits on a reference raised by log `boost`, and an isolated branch's `after` blind to the rest."""
        transformer = self.graph.components["transformer"]
        text = fused_text(transformer, branch.text.to(device=device, dtype=DTYPE))
        ids = [torch.zeros(text.shape[1], 3, device=device)]
        boosts: list[tuple[int, int, float]] = []
        at = int(text.shape[1])

        def place(references: tuple[Reference, ...]) -> torch.Tensor | None:
            nonlocal at
            tokens = []
            for reference in references:
                latents = packed(reference.latents.to(device=device, dtype=DTYPE))
                height, width = reference.latents.shape[-2] // 2, reference.latents.shape[-1] // 2
                tokens.append(transformer.img_in(latents.to(_dtype(transformer.img_in))).to(DTYPE))
                ids.append(grid(reference.frame, height, width, reference.origin, device))
                boosts.append((at, at + height * width, reference.boost))
                at += height * width
            return torch.cat(tokens, dim=1) if tokens else None

        before = place(branch.before)
        target, size = at, rows * columns
        ids.append(grid(0, rows, columns, (0.0, 0.0), device))
        at += size
        after = place(branch.after)
        isolated = branch.isolated and after is not None
        mask = None
        if isolated or any(boost != 1.0 for _, _, boost in boosts):
            mask = torch.zeros(1, 1, at, at, device=device, dtype=DTYPE)
            for start, end, boost in boosts:
                if boost != 1.0:
                    mask[..., target:target + size, start:end] = math.log(max(boost, 1e-4))
            if isolated:
                mask[..., target + size:, :target + size] = float("-inf")
        return Stream(text, before, after, transformer.rotary_emb(torch.cat(ids)), mask, target)

    @uses_components("transformer")
    def denoise(
        self,
        positive: Branch,
        negative: Branch | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        distilled: bool,
        progress: Callable[[int, int], None],
        start: torch.Tensor | None = None,
        strength: float = 1.0,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`Krea2Pipeline.__call__`'s loop: bf16 noise from `seed` drawn on the CPU,
        `FlowMatchEulerDiscreteScheduler` on `linspace(1, 1/steps)` time-shifted by mu, and Krea's guidance
        `cond + g(cond - uncond)` when `negative` is given. Returns the packed latents [1, (H/16)(W/16), 64]
        on the CPU.

        With `start` (`encode_image` at the output's size), image-to-image: the steps whose sigma is above
        `strength` are skipped and the state begins as `start` noised to the first kept sigma. With `control`
        (`encode_image` at the output's size), each image token reads its control token beside it through the
        armed control adapter's projection (`Transformer.embed`).

        Each step is one call of the transformer (`Transformer.forward`), with what no step changes built
        once before the loop (`stream`)."""
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        channels: int = transformer.config["in_channels"] // 4
        scheduler = construct(FlowMatchEulerDiscreteScheduler, dict(self.graph.scheduler_config))
        config = scheduler.config
        mu = DISTILLED_MU if distilled else calculate_shift(
            rows * columns, config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 6400),
            config.get("base_shift", 0.5), config.get("max_shift", 1.15),
        )
        # Diffusers types sigmas as a list; the scheduler takes the array as it is.
        retrieve_timesteps(scheduler, steps, device, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)  # type: ignore[arg-type]
        first = 0 if start is None else min(int((scheduler.sigmas[:-1] > strength).sum()), steps - 1)
        if start is not None:
            scheduler.set_begin_index(first)
        with torch.no_grad():
            noise = rng.noise((1, channels, rows * 2, columns * 2), seed, dtype=DTYPE)
            latents = noise.view(1, channels, rows, 2, columns, 2).permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(1, rows * columns, channels * 4).to(device)
            if start is not None:
                sigma = float(scheduler.sigmas[first])
                latents = ((1 - sigma) * packed(start.to(device)) + sigma * latents.float()).to(DTYPE)
            tokens = None if control is None else packed(control.to(device)).to(DTYPE)
            streams = [self.stream(branch, rows, columns, device)
                       for branch in (positive, *([] if negative is None else [negative]))]
            for i, t in enumerate(scheduler.timesteps):
                if i < first:
                    continue
                velocity: list[torch.Tensor] = transformer(
                    latents, (t / config["num_train_timesteps"]).expand(1).to(DTYPE), streams, tokens
                )
                prediction = velocity[0]
                if len(velocity) > 1:
                    prediction = prediction + guidance * (prediction - velocity[1])
                # The state stays bf16, as the pipeline's with a bf16 final layer.
                preview(lambda: latents - scheduler.sigmas[i] * prediction, "qwen-image", grid=(rows, columns))
                latents = scheduler.step(prediction, t, latents, return_dict=False)[0].to(DTYPE)
                progress(i, steps)
        return latents.cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor, *, width: int, height: int) -> Image.Image:
        """Diffusers' decode of the one frame (`_decoded`), its pixels quantized on the device as
        `VaeImageProcessor` quantizes them: a 4 MP canvas sends 12 MB of bytes to the host, not 50 MB of
        floats for host passes."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        with torch.inference_mode():
            z = unpack(latents, height, width).to(device=device, dtype=vae.dtype)
            mean = z.new_tensor(vae.config["latents_mean"]).view(1, -1, 1, 1, 1)
            inverse = 1.0 / z.new_tensor(vae.config["latents_std"]).view(1, -1, 1, 1, 1)
            image = _decoded(vae, z / inverse + mean)[0, :, 0].float()
            if not torch.isfinite(image).all():
                raise OutputError(
                    "Krea 2's decoder produced non-finite pixels, so there is no picture: try another seed, "
                    "and if it repeats another lane with `--lane bf16`",
                    code="output_integrity",
                )
            pixels = ((image.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
            return Image.fromarray(pixels.permute(1, 2, 0).contiguous().cpu().numpy())


def _decoded(vae: AutoencoderKLQwenImage, z: torch.Tensor) -> torch.Tensor:
    """The pipeline's decode of one frame, the whole canvas: Diffusers' `decode` without its feature cache
    (each convolution's input kept for a next frame an image never has), every residual and resample block
    over the canvas in strips of rows (`tiles.rows`: exact, a strip's working set)."""
    decoder = vae.decoder
    x = decoder.conv_in(vae.post_quant_conv(z))
    mid = decoder.mid_block
    x = tiles.rows(mid.resnets[0], x, halo=2)
    for attention, resnet in zip(mid.attentions, mid.resnets[1:], strict=True):
        x = tiles.rows(resnet, attention(x), halo=2)
    for block in cast("list[QwenImageUpBlock]", decoder.up_blocks):  # a ModuleList of these
        for resnet in block.resnets:
            x = tiles.rows(resnet, x, halo=2)
        if block.upsamplers is not None:
            x = tiles.rows(block.upsamplers[0], x, halo=1, scale=2)
    return tiles.rows(lambda s: decoder.conv_out(decoder.nonlinearity(decoder.norm_out(s))), x, halo=1)
