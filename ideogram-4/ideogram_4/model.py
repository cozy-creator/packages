"""Ideogram 4, component-staged over Diffusers' `Ideogram4Pipeline` (0.41.0).

Two DiTs run every step: the conditional one over the caption's text tokens and the image tokens,
and the unconditional one over the image tokens alone (Ideogram's asymmetric CFG). The text encoder
is the stock Qwen3-VL-8B-Instruct Ideogram froze, a slot of its own (`text_encoder.py`). Runtime
constructs and fills the modules and owns their residency.

On a GPU a DiT block runs as ComfyUI's Ideogram 4 does, on the Runtime's prebuilt kernels: each
modulated pre-norm comes out of the int8 leaves' quantization pass (`int8_rows`, the norm weight
folded into `1 + scale`), as does the SwiGLU product the down projection reads; the q/k norms and
split-half RoPE are one kernel (`rms_rope_split_half`); each post-norm is torch's fused RMSNorm and
its gated residual one in-place `addcmul`. Linears that are not int8 take the norms as Diffusers
computes them, and on the CPU a block is Diffusers' composition.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypedDict, cast

import torch
import torch.nn.functional as F
from cozy_runtime import rng
from cozy_runtime.accel import hold_dtype
from cozy_runtime.accel.kernels import PreQuantized, int8_rows, native, quantize_once, rms_rope_split_half
from cozy_runtime.author import Config, Loader, Model, OutputError, Telemetry, preview, uses_components
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_ideogram4 import (
    Ideogram4Attention,
    Ideogram4Transformer2DModel,
    Ideogram4TransformerBlock,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
    Ideogram4Pipeline,
    _logit_normal_sigmas,
    _resolution_aware_mu,
)
from PIL import Image

from . import latent_norm

if TYPE_CHECKING:
    from diffusers.models._modeling_parallel import ParallelConfig

#: Pixels per latent token: the VAE's 8x, then 2x2 patches.
PATCH = 16
COMPONENTS = ("transformer", "unconditional_transformer", "vae")
#: The DiTs' activations.
DTYPE = torch.bfloat16
#: The VAE's dtype on a GPU, as ComfyUI runs FLUX.2's VAE: its activations fit fp16 (dtype audit M1).
VAE_DTYPE = torch.float16


#: A block's RoPE: the f32 rotation matrices `[1, tokens, 1, head_dim / 2, 2, 2]` the fused q/k kernel reads
#: (`rms_rope_split_half`), else Diffusers' `(cos, sin)` tables in the activations' dtype.
type Rope = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _weight(norm: RMSNorm) -> torch.Tensor:
    """An Ideogram 4 RMSNorm's weight: every one is affine."""
    return cast(torch.Tensor, norm.weight)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _modulated(x: torch.Tensor, norm: RMSNorm, scale: torch.Tensor,
               consumers: tuple[torch.nn.Module, ...]) -> torch.Tensor | PreQuantized:
    """`norm(x) * (1 + scale)` for `consumers`: quantized for them in the norm's one pass when they are
    int8 leaves (the norm weight folded into the scale, as ComfyUI's `int8_linear_rms_modulated`), else
    as Diffusers computes it."""
    folded = (_weight(norm).float() * (1 + scale.float()) - 1).to(x.dtype)
    rows = int8_rows(x, consumers, norm="rms", eps=norm.eps, scale=folded)
    if rows is None:
        eager: torch.Tensor = norm(x) * (1.0 + scale)
        return eager
    return rows


def _gated(x: torch.Tensor, norm: RMSNorm, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """`x + tanh(gate) * norm(branch)`: on a GPU torch's fused RMSNorm, then one in-place `addcmul`."""
    if not x.is_cuda:
        eager: torch.Tensor = x + torch.tanh(gate) * norm(branch)
        return eager
    normed = F.rms_norm(branch, (branch.shape[-1],), _weight(norm), norm.eps)
    return x.addcmul_(normed, torch.tanh(gate))


class Processor:
    """`Ideogram4AttnProcessor` for rows its block may hand quantized (`int8_rows`), the q/k norms
    and split-half RoPE one kernel when the RoPE is rotation matrices. A Runtime attention site: the
    Runtime sets each one's kernel."""

    _attention_backend: AttentionBackendName | None = None
    _parallel_config: ParallelConfig | None = None

    def __call__(
        self, attn: Ideogram4Attention, hidden_states: torch.Tensor | PreQuantized,
        attention_mask: torch.Tensor | None, image_rotary_emb: Rope,
    ) -> torch.Tensor:
        heads = (attn.num_heads, attn.head_dim)
        with quantize_once(hidden_states if isinstance(hidden_states, torch.Tensor) else None):
            q, k, v = (p(hidden_states).unflatten(-1, heads) for p in (attn.to_q, attn.to_k, attn.to_v))
        if isinstance(image_rotary_emb, torch.Tensor):
            q, k = rms_rope_split_half(
                q, k, image_rotary_emb, _weight(attn.norm_q), _weight(attn.norm_k), attn.norm_q.eps
            )
        else:
            cos, sin = (t.unsqueeze(2) for t in image_rotary_emb)
            q, k = (norm(t) for norm, t in ((attn.norm_q, q), (attn.norm_k, k)))
            q, k = (t * cos + _rotate_half(t) * sin for t in (q, k))
        attended = dispatch_attention_fn(
            q, k, v, attn_mask=attention_mask, backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        out: torch.Tensor = attn.to_out[0](attended.flatten(2, 3))
        return out


class Block(Ideogram4TransformerBlock):
    """`Ideogram4TransformerBlock`, its norms, modulation, RoPE and SwiGLU fused (module docstring)."""

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None, image_rotary_emb: Rope,
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln_modulation(adaln_input).chunk(4, dim=-1)
        attn, ff = self.attention, self.feed_forward
        rows = _modulated(hidden_states, self.attention_norm1, scale_msa, (attn.to_q, attn.to_k, attn.to_v))
        x = _gated(hidden_states, self.attention_norm2, attn(rows, attention_mask, image_rotary_emb), gate_msa)
        rows = _modulated(x, self.ffn_norm1, scale_mlp, (ff.w1, ff.w3))
        with quantize_once(rows if isinstance(rows, torch.Tensor) else None):
            gate, up = ff.w1(rows), ff.w3(rows)
        product = int8_rows(gate, (ff.w2,), activation="swiglu", up=up)
        return _gated(x, self.ffn_norm2, ff.w2(F.silu(gate) * up if product is None else product), gate_mlp)


class Transformer(Ideogram4Transformer2DModel):
    """`Ideogram4Transformer2DModel` whose forward is one velocity past the text rows no step changes."""

    def forward(self, image: torch.Tensor, prefix: torch.Tensor | None, t: torch.Tensor, rope: Rope) -> torch.Tensor:
        """`Ideogram4Transformer2DModel.forward` past its text rows: the image tokens embedded beside
        `prefix` (`_text_rows`; None: image tokens alone), the blocks, and the final layer over the
        image rows. One unpadded caption is one segment, so attention is full and takes no mask:
        an all-true mask only kept SDPA off its flash kernel."""
        hidden = self.input_proj(image) + self.embed_image_indicator(torch.ones(1, dtype=torch.long, device=image.device))
        if prefix is not None:
            hidden = torch.cat([prefix, hidden], dim=1)
        adaln = F.silu(self.adaln_proj(self.t_embedding(t).unsqueeze(1)))
        for block in self.layers:
            hidden = block(hidden, None, rope, adaln)
        return self.final_layer(hidden[:, hidden.shape[1] - image.shape[1] :], conditioning=adaln)  # type: ignore[no-any-return]


class IdeogramGraph:
    def __init__(self, config: Config) -> None:
        dits = {
            name: _to(construct(Ideogram4Transformer2DModel, config.section(name).mapping()), torch.bfloat16)
            for name in ("transformer", "unconditional_transformer")
        }
        for model in dits.values():
            model.__class__ = Transformer
            for block in model.layers:
                block.__class__ = Block
                cast(Block, block).attention.set_processor(Processor())  # type: ignore[arg-type]  # the package's own
        self.components: Components = {
            "transformer": cast(Transformer, dits["transformer"]),  # its class, set above
            "unconditional_transformer": cast(Transformer, dits["unconditional_transformer"]),
            "vae": _to(construct(AutoencoderKLFlux2, config.section("vae").mapping()), torch.bfloat16),
        }
        self.scheduler_config: dict[str, object] = config.section("scheduler").mapping()


class Components(TypedDict):
    transformer: Transformer
    unconditional_transformer: Transformer
    vae: AutoencoderKLFlux2


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def _to[M: torch.nn.Module](module: M, dtype: torch.dtype) -> M:
    """`module.to(dtype=dtype)`, typed: Diffusers' `to` is `functools.wraps`'d, which the checkers bind without `self`."""
    return module.to(dtype=dtype)


def guidance_schedule(steps: int, guidance: float) -> list[float]:
    """Ideogram's presets, first step first: `guidance` (7) on the main steps, then 3 on the last
    `ceil(steps / 16)` "polish" steps (48 -> 3, 20 -> 2, 12 -> 1), never above `guidance`."""
    polish = min(-(-steps // 16), steps - 1)
    return [guidance] * (steps - polish) + [min(3.0, guidance)] * polish


def schedule_shape(steps: int) -> tuple[float, float]:
    """The logit-normal schedule's (mu, std) of the nearest preset: V4_QUALITY_48, V4_DEFAULT_20
    or V4_TURBO_12 (`ideogram4.sampler_configs`)."""
    if steps >= 34:
        return 0.0, 1.5
    if steps >= 16:
        return 0.0, 1.75
    return 0.5, 1.75


def _text_rows(tf: Transformer, features: torch.Tensor) -> torch.Tensor:
    """`Ideogram4Transformer2DModel.forward`'s text rows, which no step changes: the caption's
    features normed and projected, plus the text indicator's embedding. [1, texts, hidden]."""
    text = tf.llm_cond_proj(tf.llm_cond_norm(features[None]))
    out: torch.Tensor = text + tf.embed_image_indicator(torch.zeros(1, dtype=torch.long, device=features.device))
    return out


def _rotary(tf: Transformer, position_ids: torch.Tensor) -> Rope:
    """The forward's MRoPE for `position_ids`: on a GPU whose Runtime carries its kernels, the f32 rotation
    matrices of each split-half pair, as ComfyUI's `_split_half_rope_matrix` (Ideogram's tables repeat one
    half); else Diffusers' tables in the activations' dtype."""
    cos, sin = tf.rotary_emb(position_ids)
    if position_ids.is_cuda and not native():
        half = cos.shape[-1] // 2
        c, s = cos[..., :half], sin[..., :half]
        matrices: torch.Tensor = torch.stack((c, -s, s, c), -1).unflatten(-1, (2, 2))
        return matrices.unsqueeze(2)
    return cos.to(DTYPE), sin.to(DTYPE)


class Ideogram4(Model[IdeogramGraph]):
    """Ideogram 4's two DiTs and its VAE."""

    graph: IdeogramGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(IdeogramGraph)

    def hold_vae(self, tel: Telemetry) -> None:
        """Serve the VAE at `VAE_DTYPE` on a GPU from now on, before any VAE scope opens: the Runtime
        casts the stored bfloat16 bytes when it next fills them. A CPU keeps bfloat16."""
        vae = self.graph.components["vae"]
        if torch.device(str(self.device)).type == "cuda" and next(vae.parameters()).dtype != VAE_DTYPE:
            hold_dtype(vae, VAE_DTYPE)
            tel.log("vae representation", dtype=str(VAE_DTYPE).removeprefix("torch."))

    @uses_components("transformer", "unconditional_transformer")
    def denoise(
        self,
        features: torch.Tensor,
        *,
        width: int,
        height: int,
        guidance: Sequence[float],
        seed: int,
        progress: Callable[[int, int], None],
    ) -> torch.Tensor:
        """`Ideogram4Pipeline.__call__`'s loop for one caption whose `[tokens, 53248]` text features
        are `features`: `FlowMatchEulerDiscreteScheduler` over the resolution-shifted logit-normal sigmas,
        each step the conditional and unconditional velocities blended by its guidance weight,
        latents in float32, noise drawn on the CPU from `seed`.

        The text rows' conditioning is step-invariant, so it is projected once before the loop
        (`_text_rows`); the image rows never carry the zero text features Diffusers pads them with."""
        tf = self.graph.components["transformer"]
        uncond = self.graph.components["unconditional_transformer"]
        device = torch.device(str(self.device))
        steps = len(guidance)
        rows, columns = height // PATCH, width // PATCH
        tokens, texts = rows * columns, features.shape[0]
        mu, std = schedule_shape(steps)
        scheduler = construct(FlowMatchEulerDiscreteScheduler, self.graph.scheduler_config)
        if scheduler.config["stochastic_sampling"]:
            raise ValueError("Ideogram 4 samples with deterministic Euler")
        sigmas = _logit_normal_sigmas(steps, _resolution_aware_mu(height, width, mu), std=std, device=device)
        scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        with torch.no_grad():
            position_ids, _, _ = Ideogram4Pipeline._prepare_ids([texts], rows, columns, texts, device)
            latents = rng.noise((1, tokens, tf.config["in_channels"]), seed, device=device)
            prefix = _text_rows(tf, features.to(device, DTYPE))
            positive_rope = _rotary(tf, position_ids)
            negative_rope = _rotary(uncond, position_ids[:, texts:])
            num_train_timesteps = scheduler.config["num_train_timesteps"]
            for i, t in enumerate(scheduler.timesteps):
                t_model = (1.0 - t.float() / num_train_timesteps).expand(1).to(DTYPE)
                image = latents.to(DTYPE)
                positive = tf(image, prefix, t_model, positive_rope).to(torch.float32)
                negative = uncond(image, None, t_model, negative_rope).to(torch.float32)
                velocity = guidance[i] * positive + (1.0 - guidance[i]) * negative
                preview(lambda: latents + scheduler.sigmas[i] * velocity, "flux2", grid=(rows, columns))
                latents = scheduler.step(-velocity, t, latents, return_dict=False)[0]
                progress(i, steps)
        return latents.cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor, *, width: int, height: int) -> Image.Image:
        """Ideogram's decode (`Ideogram4Pipeline._decode`): its own latent normalization undone on the packed
        latents (`latent_norm`; Diffusers undoes the VAE's BatchNorm statistics instead), 2x2 patches unpacked,
        then the decoder."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        with torch.inference_mode():
            z = latents.to(device)
            scale = torch.tensor(latent_norm.SCALE, dtype=z.dtype, device=device).view(1, 1, -1)
            z = z * scale + torch.tensor(latent_norm.SHIFT, dtype=z.dtype, device=device).view(1, 1, -1)
            channels = z.shape[-1] // 4
            z = z.view(1, rows, columns, 2, 2, channels).permute(0, 5, 1, 3, 2, 4).reshape(1, channels, rows * 2, columns * 2)
            pixels = vae.decode(z.to(vae.dtype), return_dict=False)[0]
            if not torch.isfinite(pixels).all():
                raise OutputError(
                    "Ideogram 4's decoder produced non-finite pixels, so there is no picture: try another seed, "
                    "and if it repeats another lane with `--lane bf16`",
                    code="output_integrity",
                )
            (image,) = VaeImageProcessor(vae_scale_factor=PATCH).postprocess(pixels.float(), output_type="pil")
        return image  # type: ignore[no-any-return]  # a PIL picture, for "pil"
