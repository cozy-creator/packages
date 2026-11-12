"""MOSS-Image: a 5B single-stream DiT over Qwen3.5-9B's last hidden state, decoded by FLUX.2's VAE.

The DiT, its timestep grid and the sampling loop follow upstream's `pipeline_moss_image.py`
(OpenMOSS-Team/MOSS-Image-SFT `f972bd4b`, Apache-2.0) module for module, with its state-dict names;
Diffusers has no MOSS-Image. The text encoder is Transformers' Qwen3.5 (its vision tower reads the pictures
AI-Toolkit's edit LoRAs show it) and the VAE Diffusers' `AutoencoderKLFlux2`. Runtime constructs and fills the
modules and owns their residency.

On a GPU a DiT block runs as ComfyUI's MOSS-Image does, with the Runtime's prebuilt kernels: each
modulated RMSNorm and SiTU comes out of the int8 leaves' quantization pass (`int8_rows`, the norm
weight folded into `1 + scale`), the q/k norms and split-half RoPE are one kernel
(`rms_rope_split_half`) and each gated residual one in-place `addcmul`. The DiT blocks attend at 8
bits, as Comfy-Org's int8 file declares (`comfy_attention.config`); `--attention-kernel exact` runs
them exactly. Elsewhere (the CPU, the bf16 lane's norms and SiTU) a block runs upstream's eager
composition.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Literal, TypedDict, cast

import numpy as np
import torch
from cozy_runtime import rng
from cozy_runtime.attention import Attention
from cozy_runtime.accel import hold_dtype
from cozy_runtime.author import (
    Config,
    Loader,
    Model,
    ModelAssets,
    OutputError,
    Serving,
    Telemetry,
    UnsupportedInput,
    memoize,
    preview,
    uses_components,
)
from cozy_runtime.accel.kernels import PreQuantized, int8_rows, native, quantize_once, rms_rope_split_half
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from PIL import Image
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, Qwen3_5Config
from transformers.initialization import no_init_weights
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

#: The DiT's activations and the dtype upstream casts each linear's input to (its weights' in bf16).
DTYPE = torch.bfloat16
#: The VAE's dtype on a GPU, as ComfyUI runs FLUX.2's VAE: its activations fit fp16 (dtype audit M1).
VAE_DTYPE = torch.float16
EPS = 1e-6
MAX_TOKENS = 4096
#: Pixels per DiT token: the VAE's 8x, then 2x2 patches.
PATCH = 16
#: Upstream's default guidance: the negative branch skips this block (zero-based).
UNCOND_SKIP = frozenset({22})
#: A block's keys and values of the references, each [1, N, heads, head_dim].
type Reference = tuple[torch.Tensor, torch.Tensor]
#: The RoPE layout an edit LoRA trained with (ComfyUI's `TextEncodeMOSSImageEdit` edit formats): Moss-Image-Edit's
#: text from f = 0 and the image after it; AI-Toolkit's the image at f = 0 beside its references, the text before it.
type EditFormat = Literal["moss-image-edit", "ai-toolkit"]
#: Qwen3.5's image processor (Qwen3.5-9B's, which MOSS's checkpoint carries since 0.1.1).
PROCESSOR = "text_encoder/processor/preprocessor_config.json"
#: The most pixels the text encoder sees a picture at before its processor's grid (ComfyUI's `TextEncodeMOSSImageEdit`).
GLIMPSE_PIXELS = 384 * 384
#: A picture in the user turn, ahead of the text, as AI-Toolkit's edit LoRAs read it: `{}` its vision tokens.
PICTURE = "Picture {}: <|vision_start|>{}<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE of `x` [B, N, heads, D] by the `[1, N, D]` sin|cos table."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(
    site: Attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope: torch.Tensor
) -> torch.Tensor:
    shape = (*q.shape[:2], site.heads, site.head_dim)
    out: torch.Tensor = site(rotate(q.reshape(shape), rope), rotate(k.reshape(shape), rope), v.reshape(shape))
    return out.flatten(2)


def path(rows: torch.Tensor | PreQuantized | None) -> str:
    return "int8 prologue" if isinstance(rows, PreQuantized) else "eager"


def modulated(x: torch.Tensor, norm: RMSNorm, scale: torch.Tensor, shift: torch.Tensor,
              consumers: tuple[torch.nn.Module, ...]) -> torch.Tensor | PreQuantized:
    """`norm(x) * (1 + scale) + shift` for `consumers`: quantized for them in the norm's one pass when
    they are int8 leaves (the norm weight folded into the scale, as ComfyUI's `rms_adaln`), else as
    upstream computes it."""
    folded = (norm.weight.float() * (1 + scale.float()) - 1).to(x.dtype)
    rows = int8_rows(x, consumers, norm="rms", eps=norm.eps, scale=folded, shift=shift)
    if rows is None:
        rows = (norm(x) * (1 + scale[:, None]) + shift[:, None]).to(DTYPE)
    return rows


class TextBlock(torch.nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads, self.attn = heads, Attention(heads, dim // heads, DTYPE)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.proj = torch.nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = torch.nn.Linear(dim, ffn_dim), torch.nn.Linear(dim, ffn_dim)
        self.ffn_down = torch.nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(self.attn, q, k, v, rope))
        y = self.norm2(x)
        out: torch.Tensor = x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))
        return out


class TextProjection(torch.nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(text_dim, dim)
        self.blocks = torch.nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        x = self.proj(x.to(DTYPE))
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(torch.nn.Module):
    """One DiT block over one sample's stream `x` [1, N, dim] (bf16), updated in place. `rope` is the
    `[1, N, 1, D/2, 2, 2]` rotation matrices on the fused path, the `[1, N, D]` sin|cos table eager."""

    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads, self.attn = heads, Attention(heads, dim // heads, DTYPE)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = torch.nn.ModuleDict(
            {name: torch.nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = torch.nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = torch.nn.Linear(ffn_dim, dim, bias=False)
        #: How its last call ran the modulated norms and SiTU, for the run record.
        self.served = {"norms": "", "situ": ""}

    def forward(self, x: torch.Tensor, rope: torch.Tensor, modulation: torch.Tensor,
                references: Reference | None = None, keep: list[Reference] | None = None) -> torch.Tensor:
        """`references`: the references' keys and values here (`MossTransformer.references`), which every query
        also attends to; `keep` collects this block's own keys and values (the references' pass)."""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        proj = self.proj
        qkv = proj["q"], proj["k"], proj["v"]
        rows = modulated(x, self.norm1, scale_a, shift_a, qkv)
        with quantize_once(rows):  # type: ignore[arg-type, unused-ignore]  # quantized rows share as they are
            q, k, v = (p(rows).unflatten(-1, (self.heads, -1)) for p in qkv)
        if rope.dim() == 6:
            q, k = rms_rope_split_half(q, k, rope, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps)
        else:
            q, k = rotate(self.q_norm(q), rope), rotate(self.k_norm(k), rope)
        if keep is not None:
            keep.append((k, v))
        if references is not None:
            k, v = torch.cat((k, references[0]), 1), torch.cat((v, references[1]), 1)
        x.addcmul_(proj["o"](self.attn(q, k, v).flatten(2)), gate_a[:, None])
        hidden = proj["ffn_up_gate"](modulated(x, self.norm2, scale_f, shift_f, (proj["ffn_up_gate"],)))
        activated: torch.Tensor | PreQuantized | None = int8_rows(hidden, (proj["ffn_down"],), activation="situ")
        self.served = {"norms": path(rows), "situ": path(activated)}
        if activated is None:
            activated = situ(*hidden.chunk(2, -1))
        return x.addcmul_(proj["ffn_down"](activated), gate_f[:, None])


class MossTransformer(torch.nn.Module):
    """Upstream's `MOSSImageTransformer` for one sample: image tokens then text tokens in one stream,
    one shared modulation for every block. Its RoPE bands are derived constants, computed in f32 on the
    CPU as training built them; the checkpoint's `rope.bands` is not read. The text stream
    and RoPE are built once a request (`text`, `table`/`rotations`); `forward` is one step for its prompts."""

    def __init__(self, dim: int = 3072, ffn_dim: int = 8192, num_heads: int = 24, num_layers: int = 42,
                 text_dim: int = 4096, freq_dim: int = 256, latent_channels: int = 128,
                 rope_base: float = 10000.0) -> None:
        super().__init__()
        self.heads, self.freq_dim, self.latent_channels = num_heads, freq_dim, latent_channels
        self.rope_base, self.num_bands = rope_base, (dim // num_heads) // 8
        self.video_embedder = torch.nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(freq_dim, dim), torch.nn.SiLU(), torch.nn.Linear(dim, dim)
        )
        self.shared_block_modulation = torch.nn.Linear(dim, 6 * dim)
        self.layers = torch.nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = torch.nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = torch.nn.Linear(dim, 2 * dim)

    def phase(self, positions: torch.Tensor) -> torch.Tensor:
        """RoPE angles [N, D/2] of (r, f, h, w) `positions` [N, 4]: axis-major, 16 bands an axis."""
        bands = 1 / (self.rope_base ** (torch.arange(self.num_bands, dtype=torch.float32, device="cpu") / self.num_bands))
        return (positions.unsqueeze(-1) * bands.to(positions.device)).flatten(-2)

    def positions(self, height: int, width: int, text_length: int, device: torch.device,
                  offset: int = 0) -> torch.Tensor:
        """(r, f, h, w) per token: the image centred on its grid with f = the text length, then the text
        counting up on f, all shifted by `offset` on f (`-text_length` puts the image at f = 0, ComfyUI's
        `f_offset`)."""
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        f = torch.full_like(h, text_length + offset)
        image = torch.stack((torch.zeros_like(h), f, h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        return torch.cat((image, self.text_positions(text_length, device, offset)), 0)

    @staticmethod
    def reference_positions(shapes: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
        """(r, f, h, w) of each reference's tokens: reference i on r = i + 1 and f = 0, its grid centred (ComfyUI's
        `MOSSImage.rope`)."""
        out = []
        for index, (height, width) in enumerate(shapes):
            h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
            grid = torch.stack((torch.full_like(h, index + 1), torch.zeros_like(h), h, w), -1).flatten(0, 1).float()
            grid[:, 2] -= (height - 1) / 2
            grid[:, 3] -= (width - 1) / 2
            out.append(grid)
        return torch.cat(out, 0)

    def references(self, latents: list[torch.Tensor], fused: bool) -> list[Reference]:
        """Each block's keys and values of the references `latents` [1, 128, h, w] (an edit LoRA's): free of noise
        at t = 0 and attending only to each other, so they depend on themselves alone and are computed once a
        request (ComfyUI's `MOSSImage.preprocess_refs`)."""
        device = latents[0].device
        x = torch.cat([self.video_embedder(latent.flatten(2).transpose(1, 2).to(DTYPE)) for latent in latents], 1)
        phase = self.phase(self.reference_positions([(latent.shape[2], latent.shape[3]) for latent in latents], device))
        rope = self.rotations(phase) if fused else self.table(phase)
        modulation, _ = self.modulation(torch.zeros(1, device=device))
        kept: list[Reference] = []
        x = x.contiguous()
        for block in self.layers:
            x = block(x, rope, modulation, keep=kept)
        return kept

    @staticmethod
    def text_positions(text_length: int, device: torch.device, offset: int = 0) -> torch.Tensor:
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device) + offset
        return text

    @staticmethod
    def table(phase: torch.Tensor) -> torch.Tensor:
        """The eager sin|cos table [1, N, D] in bf16, as upstream builds it."""
        return torch.cat((phase.sin(), phase.cos()), -1).unsqueeze(0).to(DTYPE)

    @staticmethod
    def rotations(phase: torch.Tensor) -> torch.Tensor:
        """The fused kernel's f32 rotation matrices [1, N, 1, D/2, 2, 2]."""
        cos, sin = phase.cos(), phase.sin()
        out: torch.Tensor = torch.stack((cos, -sin, sin, cos), -1).unflatten(-1, (2, 2))
        return out.unsqueeze(0).unsqueeze(2)

    def text(self, context: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The step-invariant text stream [1, S, dim] of `context` [1, S, text_dim], its positions shifted by
        `offset` on f as the stream's (`positions`)."""
        phase = self.phase(self.text_positions(context.shape[1], context.device, offset))
        return self.text_embedder(context, self.table(phase))  # type: ignore[no-any-return]

    def modulation(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The blocks' shared modulation [1, 6, dim] and the head's [1, 2, dim] at `timestep` [1] (0-1000).
        Training computes the sinusoid in f64 and feeds f32 into its MLP."""
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=timestep.device, dtype=torch.float64) / half)
        phase = timestep.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(DTYPE))
        return self.shared_block_modulation(time).view(1, 6, -1), self.video_head_modulation(time).view(1, 2, -1)

    def velocity(self, x: torch.Tensor, head: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """The f32 velocity [1, C, H, W] from the stream's image tokens."""
        x = self.final_norm_video(x[:, : height * width].to(DTYPE))
        x = self.final_linear_video((x * (1 + head[:, 1, None]) + head[:, 0, None]).to(DTYPE))
        out: torch.Tensor = x.transpose(1, 2).reshape(1, -1, height, width)
        return out.float()

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor, texts: list[torch.Tensor],
                ropes: list[torch.Tensor], references: list[Reference] | None = None) -> list[torch.Tensor]:
        """Each prompt's f32 velocity [1, C, H, W] of `latents` at `timestep` [1]: its text stream and
        RoPE, the conditional prompt first. The prompts pass through each block in turn, so a streamed
        block comes in once per step; a later prompt skips `UNCOND_SKIP`, upstream's unconditional branch.
        Both attend to `references` (`references`), as the edit LoRAs' sampling conditions both."""
        modulation, head = self.modulation(timestep)
        video = self.video_embedder(latents.flatten(2).transpose(1, 2).to(DTYPE).contiguous())
        streams = [torch.cat((video, text), 1) for text in texts]
        for index, block in enumerate(self.layers):
            held = None if references is None else references[index]
            streams = [x if branch and index in UNCOND_SKIP else block(x, rope, modulation, held)
                       for branch, (x, rope) in enumerate(zip(streams, ropes, strict=True))]
        return [self.velocity(x, head, latents.shape[2], latents.shape[3]) for x in streams]


class MossScheduler(FlowMatchEulerDiscreteScheduler):
    """Upstream's `MOSSImageScheduler`: Euler over sigmas shifted by `shift`, then snapped to the
    nearest of the 1000 training sigmas."""

    @register_to_config  # type: ignore[untyped-decorator]
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, training_shift: float = 1.0) -> None:  # type: ignore[misc]  # untyped after Diffusers' register_to_config
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)  # type: ignore[no-untyped-call]  # Diffusers' register_to_config is unannotated

    def set_timesteps(self, num_inference_steps: int = 50, device: str | torch.device | None = None) -> None:  # type: ignore[override]
        total: int = self.config["num_train_timesteps"]

        def grid(count: int, shift: float) -> torch.Tensor:
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count, device="cpu")
            return shift * values / (1 + (shift - 1) * values)

        sigmas = grid(num_inference_steps, self.config["shift"])
        training = grid(total, self.config["training_shift"]).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
        sigmas = torch.unique_consecutive(training[nearest])
        # Already shifted and snapped: no second shift.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())  # type: ignore[arg-type]  # Diffusers' own default is None


class MossGraph:
    def __init__(self, config: Config) -> None:
        with no_init_weights():
            self.components: Components = {
                "transformer": MossTransformer(**config.section("transformer").mapping()).to(DTYPE).eval(),
                "text_encoder": Qwen3_5Model._from_config(  # type: ignore[no-untyped-call]  # Transformers' _from_config is unannotated
                    Qwen3_5Config(**config.section("text_encoder").mapping()), dtype=DTYPE
                ).eval(),
                "vae": _to(construct(AutoencoderKLFlux2, config.section("vae").mapping()), torch.float32).eval(),
            }
        self.scheduler_config: dict[str, object] = config.section("scheduler").mapping()


class Components(TypedDict):
    transformer: MossTransformer
    #: Qwen3.5-9B: its text model (`language_model.*`) and its vision tower (`visual.*`), which only a prompt with
    #: pictures runs.
    text_encoder: Qwen3_5Model
    vae: AutoencoderKLFlux2


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def _to[M: torch.nn.Module](module: M, dtype: torch.dtype) -> M:
    """`module.to(dtype)`, typed: Diffusers' `to` is `functools.wraps`'d, which the checkers bind without `self`."""
    return module.to(dtype)


def build_tokenizer(assets: ModelAssets) -> PreTrainedTokenizerFast:
    """Qwen3.5's tokenizer and chat template from the checkpoint's `tokenizer/` files, parsed in memory."""
    config = json.loads(assets.read("tokenizer/tokenizer_config.json"))
    for key in ("tokenizer_file", "name_or_path", "added_tokens_decoder", "tokenizer_class", "processor_class",
                "backend", "is_local", "local_files_only", "chat_template", "model_specific_special_tokens"):
        config.pop(key, None)
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # Transformers' tokenizer __init__ is unannotated
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer/tokenizer.json").decode()),
        chat_template=assets.read("tokenizer/chat_template.jinja").decode(),
        **config,
    )


def build_processor(assets: ModelAssets) -> Qwen2VLImageProcessorPil | None:
    """Qwen3.5's image processor from the checkpoint's `PROCESSOR`; None for a checkpoint without one (MOSS-Image
    0.1.0), which reads no pictures."""
    if PROCESSOR not in assets.names():
        return None
    config = json.loads(assets.read(PROCESSOR))
    for key in ("processor_class", "image_processor_type"):
        config.pop(key, None)
    return Qwen2VLImageProcessorPil(**config)


def glimpse(picture: Image.Image) -> Image.Image:
    """A picture as the text encoder takes it: at most `GLIMPSE_PIXELS`, its ratio kept (area resampling, ComfyUI's
    `TextEncodeMOSSImageEdit`). Its processor then snaps it to its grid."""
    picture = picture.convert("RGB")
    scale = min(1.0, math.sqrt(GLIMPSE_PIXELS / (picture.width * picture.height)))
    size = (max(1, round(picture.width * scale)), max(1, round(picture.height * scale)))
    return picture if size == picture.size else picture.resize(size, Image.Resampling.BOX)


class MossImage(Model[MossGraph]):
    """One MOSS-Image checkpoint (Base or SFT)."""

    #: Caller LoRAs on the DiT, at any strength: the Runtime bakes them into the weights.
    serving = Serving(attention_bits={"transformer.layers.": 8}, loras=("transformer",))

    graph: MossGraph
    tokenizer: PreTrainedTokenizerFast
    processor: Qwen2VLImageProcessorPil | None

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(MossGraph)
        self.tokenizer = build_tokenizer(loader.assets)
        self.processor = build_processor(loader.assets)

    def hold_vae(self, tel: Telemetry) -> None:
        """Serve the VAE at `VAE_DTYPE` on a GPU from now on, before any VAE scope opens: the Runtime
        casts the stored float32 bytes when it next fills them. A CPU keeps float32."""
        vae = self.graph.components["vae"]
        if torch.device(str(self.device)).type == "cuda" and next(vae.parameters()).dtype != VAE_DTYPE:
            hold_dtype(vae, VAE_DTYPE)
            tel.log("vae representation", dtype=str(VAE_DTYPE).removeprefix("torch."))

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str, pictures: tuple[Image.Image, ...] = ()) -> torch.Tensor:
        """The prompt's last hidden state [S, 4096] (bf16, on the CPU): the user turn of the chat
        template with thinking open, at most 4096 tokens, one unpadded forward. One prompt a call, so
        a request's fixed negative is a memo hit. `pictures` come before the text, each `Picture N: ` and its
        vision tokens, one a merged patch of its `glimpse` (AI-Toolkit's edit LoRAs; ComfyUI's
        `TextEncodeMOSSImageEdit`), with Qwen3.5's multimodal positions."""
        encoder = self.graph.components["text_encoder"]
        device = torch.device(str(self.device))
        seen: dict[str, torch.Tensor] = {}
        if pictures:
            if self.processor is None:
                raise UnsupportedInput(
                    f"this MOSS-Image checkpoint has no {PROCESSOR}, which reading a picture needs: run "
                    "openmoss-team/moss-image@0.1.1 or later", code="model_asset_missing",
                )
            seen = dict(self.processor.preprocess([glimpse(p) for p in pictures], return_tensors="pt"))
            merged = (seen["image_grid_thw"].prod(-1) // self.processor.merge_size**2).tolist()
            prompt = "".join(PICTURE.format(n, IMAGE_PAD * int(tokens)) for n, tokens in enumerate(merged, 1)) + prompt
        with torch.inference_mode():
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            ids = inputs.input_ids.to(device)
            visual = {key: value.to(device) for key, value in seen.items()}
            if seen:
                visual["mm_token_type_ids"] = (ids == encoder.config.image_token_id).int()
            out = encoder(input_ids=ids, attention_mask=inputs.attention_mask.to(device), use_cache=False, **visual)
            return out.last_hidden_state[0].cpu()  # type: ignore[no-any-return]

    @uses_components("vae")
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """`decode` inverted: the picture's VAE mode in 2x2 patches, BatchNorm-normalized, [1, 128, H/16, W/16]
        in f32 on the CPU."""
        vae = self.graph.components["vae"]
        with torch.inference_mode():
            pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1.0
            z = vae.encode(pixels[None].to(str(self.device), vae.dtype)).latent_dist.mode().float()
            b, c, h, w = z.shape
            z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
            bn, eps = vae.bn, vae.config["batch_norm_eps"]
            std = (bn.running_var.float() + eps).sqrt().view(1, -1, 1, 1).to(z.device)  # type: ignore[union-attr]  # tracked
            normed: torch.Tensor = (z - bn.running_mean.float().to(z.device).view(1, -1, 1, 1)) / std  # type: ignore[union-attr]  # tracked
            return normed.cpu()

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
        progress: Callable[[int, int], None],
        tel: Telemetry,
        start: torch.Tensor | None = None,
        strength: float = 1.0,
        references: list[torch.Tensor] | None = None,
        edit_format: EditFormat = "moss-image-edit",
    ) -> torch.Tensor:
        """Upstream's sampler: f32 noise from `seed` on the CPU, CFG against `negative` with block 22
        skipped in its branch, Euler (`MossScheduler`) in f32. Returns the latents [1, 128, H/16, W/16]
        on the CPU. `references` (`encode_image`'s) condition both branches, laid out on RoPE as the edit
        LoRA trained (`edit_format`).

        Each prompt's text stream and RoPE are built once, before the loop; with CFG both branches
        pass through each block in turn inside one call of the transformer, so a streamed block comes in
        once per step. Their text lengths differ, so the branches are two streams, not one batch."""
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        fused = device.type == "cuda" and not native()
        scheduler = construct(MossScheduler, self.graph.scheduler_config)
        scheduler.set_timesteps(steps, device=device)
        # With `start` (`encode_image`), img2img: the steps whose sigma is above `strength` are skipped and the
        # state begins as `start` noised to the first kept sigma.
        first = 0 if start is None else min(int((scheduler.sigmas[:-1] > strength).sum()), len(scheduler.timesteps) - 1)
        if start is not None:
            scheduler.set_begin_index(first)
        with torch.no_grad():
            latents = rng.noise((1, transformer.latent_channels, rows, columns), seed, device=device)
            if start is not None:
                sigma = float(scheduler.sigmas[first])
                latents = (1 - sigma) * start.to(device=device, dtype=latents.dtype) + sigma * latents
            held = None if not references else transformer.references([r.to(device) for r in references], fused)
            texts, ropes = [], []
            for text in (positive, *([] if negative is None else [negative])):
                context = text[None].to(device)
                offset = -context.shape[1] if held is not None and edit_format == "ai-toolkit" else 0
                phase = transformer.phase(transformer.positions(rows, columns, context.shape[1], device, offset))
                texts.append(transformer.text(context, offset))
                ropes.append(transformer.rotations(phase) if fused else transformer.table(phase))
            timesteps = scheduler.timesteps
            for i, t in enumerate(timesteps):
                if i < first:
                    continue
                velocity = transformer(latents, t[None], texts, ropes, held)
                prediction = velocity[0]
                if len(velocity) > 1:
                    prediction = velocity[1] + guidance * (velocity[0] - velocity[1])
                preview(lambda: latents - scheduler.sigmas[i] * prediction, "flux2")
                latents = scheduler.step(prediction, t, latents, return_dict=False)[0]
                progress(i, len(timesteps))
        tel.log("dit block", rope="fused" if fused else "eager", **cast(Block, transformer.layers[0]).served)  # its blocks
        return latents.cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> Image.Image:
        """The latents' BatchNorm statistics restored in f32, 2x2 patches unfolded, then the small
        decoder in the VAE's dtype."""
        vae = self.graph.components["vae"]
        with torch.inference_mode():
            z = latents.to(str(self.device), torch.float32)
            bn, eps = vae.bn, vae.config["batch_norm_eps"]
            std = (bn.running_var.float() + eps).sqrt().view(1, -1, 1, 1)  # type: ignore[union-attr]  # tracked: it keeps its statistics
            z = z * std.to(z.device) + bn.running_mean.float().to(z.device).view(1, -1, 1, 1)  # type: ignore[union-attr]  # tracked
            b, c, h, w = z.shape
            z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
            image = vae.decode(z.to(vae.dtype), return_dict=False)[0].clamp(-1, 1)
            if not torch.isfinite(image).all():
                raise OutputError(
                    "MOSS-Image's decoder produced non-finite pixels, so there is no picture: try another seed, "
                    "and if it repeats another lane with `--lane bf16`",
                    code="output_integrity",
                )
            (picture,) = VaeImageProcessor(vae_scale_factor=PATCH).postprocess(image.float(), output_type="pil")
        return picture  # type: ignore[no-any-return]
