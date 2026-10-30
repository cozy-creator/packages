"""Krea 2 (Raw and Turbo), component-staged over Diffusers' modules.

The DiT and VAE are Diffusers' (`Krea2Transformer2DModel`, `AutoencoderKLQwenImage`, 0.42.1); the text
encoder is Transformers' Qwen3-VL-4B text model. Prompt conditioning, the schedule and the sampler are
Diffusers' `Krea2Pipeline`, Krea's reference code (`krea-ai/krea-2` db3984f) in its form. Runtime
constructs and fills the modules and owns their residency.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from cozy_runtime import fusion, rng, tiles
from cozy_runtime.author import (
    AdapterCompatibility,
    Config,
    Loader,
    MemoDistribution,
    Model,
    ModelAssets,
    denoise_loop,
    hold_dtype,
    uses_components,
)
from cozy_runtime.author.kernels import Plan
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    QwenImageCausalConv3d,
    QwenImageDecoder3d,
    QwenImageResidualBlock,
    QwenImageRMS_norm,
    QwenImageUpsample,
)
from diffusers.models.transformers.transformer_krea2 import Krea2TransformerBlock
from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline, calculate_shift, retrieve_timesteps
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast, Qwen3VLConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

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


class TextEncoder(nn.Module):
    """Qwen3-VL-4B's language model (`language_model.*`) up to the deepest tapped layer. The vision
    tower, the later layers and the final norm are never built, so their bytes are never read; with the
    norm an identity, hidden state `depth` stays the pre-norm output Diffusers taps."""

    def __init__(self, config: Qwen3VLConfig, depth: int) -> None:
        super().__init__()
        self.language_model = Qwen3VLTextModel._from_config(config.text_config, dtype=DTYPE)
        self.language_model.layers = nn.ModuleList(list(self.language_model.layers)[:depth])
        self.language_model.norm = nn.Identity()


class Block(Krea2TransformerBlock):
    """Diffusers' block with its modulation in the activations' dtype: the checkpoint stores the
    block's `scale_shift_table` in F32, which Diffusers' bf16 load would cast."""

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, image_rotary_emb: Any,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        modulation = (temb.unflatten(-1, (6, -1)) + self.scale_shift_table).to(hidden_states.dtype)
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)
        attn_out = self.attn((1.0 + prescale) * self.norm1(hidden_states) + preshift,
                             attention_mask=attention_mask, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        return hidden_states + postgate * ff_out


def _modulation(block: Any, arguments: Mapping[str, Any]) -> fusion.Modulation:
    """`Block.forward`'s six adaLN rows in the activations' dtype; a batch of one reads row 0."""
    hidden = arguments["hidden_states"]
    table = (arguments["temb"].unflatten(-1, (6, -1)) + block.scale_shift_table).to(hidden.dtype)
    rows = [row.reshape(-1, row.shape[-1]) for row in table.unbind(-2)]
    batch, tokens = hidden.shape[:2]
    index = torch.arange(batch, device=hidden.device).repeat_interleave(tokens)
    prescale, preshift, pregate, postscale, postshift, postgate = rows
    return fusion.Modulation(index[:1] if batch == 1 else index, prescale, pregate, postscale,
                             postgate, preshift, postshift)


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


class Krea2Graph:
    def __init__(self, config: Config) -> None:
        with no_init_weights():
            transformer = Krea2Transformer2DModel.from_config(config.section("transformer").mapping())
            encoder = TextEncoder(Qwen3VLConfig(**config.section("text_encoder").mapping()), max(TAPS))
            vae = AutoencoderKLQwenImage.from_config(config.section("vae").mapping())
        for block in transformer.transformer_blocks:
            block.__class__ = Block
        as_stored(config, "transformer", transformer)
        as_stored(config, "vae", vae)
        self.components: dict[str, nn.Module] = {
            "transformer": transformer.eval(),
            "text_encoder": encoder.eval(),
            "vae": vae.eval(),
        }
        self.scheduler_config: Mapping[str, Any] = config.section("scheduler").mapping()


def build_tokenizer(assets: ModelAssets) -> PreTrainedTokenizerFast:
    """The checkpoint's Qwen2 tokenizer (`tokenizer/`), parsed in memory."""
    config = json.loads(assets.read("tokenizer/tokenizer_config.json"))
    for key in ("tokenizer_class", "backend", "is_local", "name_or_path", "tokenizer_file",
                "added_tokens_decoder", "chat_template", "extra_special_tokens", "errors"):
        config.pop(key, None)
    return PreTrainedTokenizerFast(
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


def fused_text(transformer: Any, text: torch.Tensor) -> torch.Tensor:
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
    return transformer.txt_in(hidden.to(_dtype(transformer.txt_in))).to(text.dtype)  # type: ignore[no-any-return]


#: Code the memoized encode reads beyond its own body.
_CODE = (build_tokenizer, TextEncoder, MemoDistribution("transformers"), MemoDistribution("tokenizers"))


class Krea2(Model[Krea2Graph], encoded_leaves="accept"):
    """One Krea 2 checkpoint, Raw or Turbo. Their configs are identical and the pipeline's `is_distilled` is
    not in the model's config, so the caller names the regime (`denoise(distilled=)`)."""

    #: Caller LoRAs on the DiT, at any strength: the Runtime bakes them into the weights.
    __adapter_compatibility__ = (AdapterCompatibility("lora", "", ("transformer",), -math.inf, math.inf),)

    graph: Krea2Graph
    tokenizer: PreTrainedTokenizerFast

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Krea2Graph, factory=Krea2Graph)
        self.tokenizer = build_tokenizer(loader.assets)

    def execution_plan(self, plan: Plan) -> str:
        vae = self.graph.components["vae"]
        installed = (
            fusion.install(plan, BLOCK, self.graph.components["transformer"]),
            fusion.install_norms(plan, VAE_NORM, vae),
            fusion.install_lone_frame(plan, VAE_FRAME, vae.decoder, vae.post_quant_conv),
        )
        return "+".join(identity for identity in installed if identity)

    def hold_vae(self) -> None:
        """Decode in `VAE_DTYPE` on CUDA; the CPU keeps the stored F32. The Runtime casts the VAE's
        stored F32 when it next fills it. Called before any VAE scope opens."""
        if torch.device(str(self.device)).type == "cuda":
            hold_dtype(self.graph.components["vae"], VAE_DTYPE)

    @uses_components("text_encoder", memoize=True, memo_dependencies=_CODE)
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

    @uses_components("transformer")
    def denoise(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        distilled: bool,
        on_step: Callable[[int], None],
    ) -> torch.Tensor:
        """`Krea2Pipeline.__call__`'s loop as one `denoise_loop`: bf16 noise from `seed` drawn on the CPU,
        Euler on `linspace(1, 1/steps)` time-shifted by mu, and Krea's guidance `cond + g(cond - uncond)`
        when `negative` is given. Returns the packed latents [1, (H/16)(W/16), 64] on the CPU.

        The step is `Krea2Transformer2DModel.forward` op for op, with what no step changes built once
        before the loop: each prompt's text-fusion and projection, and its rotary tables. With CFG both
        branches pass through each block in turn, so a streamed block comes in once per step."""
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        channels = transformer.config.in_channels // 4
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(self.graph.scheduler_config))
        config = scheduler.config
        mu = DISTILLED_MU if distilled else calculate_shift(
            rows * columns, config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 6400),
            config.get("base_shift", 0.5), config.get("max_shift", 1.15),
        )
        retrieve_timesteps(scheduler, steps, device, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
        with torch.inference_mode(False), torch.no_grad():
            noise = rng.noise((1, channels, rows * 2, columns * 2), seed, dtype=DTYPE)
            latents = noise.view(1, channels, rows, 2, columns, 2).permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(1, rows * columns, channels * 4).to(device)
            branches: list[torch.Tensor] = []
            for text in (positive, *([] if negative is None else [negative])):
                text = text.to(device=device, dtype=DTYPE)
                ids = Krea2Pipeline.prepare_position_ids(text.shape[1], rows, columns, device)
                branches += [fused_text(transformer, text), *transformer.rotary_emb(ids)]
            # The pipeline divides the f32 timestep by 1000 and then casts it; Euler adds the bf16
            # product of the f32 sigma delta and the bf16 velocity to the f32 latents.
            times = (scheduler.timesteps / config.num_train_timesteps).to(DTYPE)
            deltas = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).to(device)
        blocks = transformer.transformer_blocks

        def step(i: torch.Tensor, latents: torch.Tensor, times: torch.Tensor, deltas: torch.Tensor,
                 *branches: torch.Tensor) -> None:
            # The embedders and the final layer run in their stored F32; the blocks in bf16.
            temb = transformer.time_embed(times.index_select(0, i), dtype=_dtype(transformer.time_embed))
            modulation = transformer.time_mod_proj(F.gelu(temb, approximate="tanh").to(_dtype(transformer.time_mod_proj)))
            image = transformer.img_in(latents.to(_dtype(transformer.img_in))).to(latents.dtype)
            texts = branches[0::3]
            hidden = [torch.cat([text, image], dim=1) for text in texts]
            rotary = list(zip(branches[1::3], branches[2::3], strict=True))
            for block in blocks:
                hidden = [block(h, modulation, rope) for h, rope in zip(hidden, rotary, strict=True)]
            velocity = [transformer.final_layer(h[:, text.shape[1]:], temb.to(_dtype(transformer.final_layer)))
                        for h, text in zip(hidden, texts, strict=True)]
            prediction = velocity[0]
            if len(velocity) > 1:
                prediction = prediction + guidance * (prediction - velocity[1])
            delta = deltas.index_select(0, i).reshape(())
            latents.copy_((latents.float() + delta * prediction).to(latents.dtype))

        denoise_loop(step, range(steps), state=(latents,), inputs=(times, deltas, *branches), on_step=on_step)
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
            mean = z.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            inverse = 1.0 / z.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            image = _decoded(vae, z / inverse + mean)[0, :, 0].float()
            if not torch.isfinite(image).all():
                raise ValueError("Krea 2's decoder produced non-finite pixels")
            pixels = ((image.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
            return Image.fromarray(pixels.permute(1, 2, 0).contiguous().cpu().numpy())


def _decoded(vae: Any, z: torch.Tensor) -> torch.Tensor:
    """The pipeline's decode of one frame, the whole canvas: Diffusers' `decode` without its feature cache
    (each convolution's input kept for a next frame an image never has), every residual and resample block
    over the canvas in strips of rows (`tiles.rows`: exact, a strip's working set)."""
    decoder = vae.decoder
    x = decoder.conv_in(vae.post_quant_conv(z))
    mid = decoder.mid_block
    x = tiles.rows(mid.resnets[0], x, halo=2)
    for attention, resnet in zip(mid.attentions, mid.resnets[1:], strict=True):
        x = tiles.rows(resnet, attention(x), halo=2)
    for block in decoder.up_blocks:
        for resnet in block.resnets:
            x = tiles.rows(resnet, x, halo=2)
        if block.upsamplers is not None:
            x = tiles.rows(block.upsamplers[0], x, halo=1, scale=2)
    return tiles.rows(lambda s: decoder.conv_out(decoder.nonlinearity(decoder.norm_out(s))), x, halo=1)
