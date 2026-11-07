"""Ideogram 4, component-staged over Diffusers' `Ideogram4Pipeline` (0.41.0).

Two DiTs run every step: the conditional one over the caption's text tokens and the image tokens,
and the unconditional one over the image tokens alone (Ideogram's asymmetric CFG). The text encoder
is the stock Qwen3-VL-8B-Instruct Ideogram froze, a slot of its own (`text_encoder.py`). Runtime
constructs and fills the modules and owns their residency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from cozy_runtime import rng
from cozy_runtime.author import OutputError, Config, Loader, Model, Telemetry, denoise_loop, hold_dtype, uses_components
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Ideogram4Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
    Ideogram4Pipeline,
    _logit_normal_sigmas,
    _resolution_aware_mu,
)
from PIL import Image

#: Pixels per latent token: the VAE's 8x, then 2x2 patches.
PATCH = 16
COMPONENTS = ("transformer", "unconditional_transformer", "vae")
#: The DiTs' activations.
DTYPE = torch.bfloat16
#: The VAE's dtype on a GPU, as ComfyUI runs FLUX.2's VAE: its activations fit fp16 (dtype audit M1).
VAE_DTYPE = torch.float16


class IdeogramGraph:
    def __init__(self, config: Config) -> None:
        # Ideogram's official fp8 release carries no transformer configs: Diffusers' defaults are its geometry.
        self.components: dict[str, torch.nn.Module] = {
            name: Ideogram4Transformer2DModel.from_config(
                config.section(name).mapping() if name in config.keys() else {}
            ).to(dtype=torch.bfloat16)
            for name in ("transformer", "unconditional_transformer")
        }
        self.components["vae"] = AutoencoderKLFlux2.from_config(config.section("vae").mapping()).to(dtype=torch.bfloat16)
        self.scheduler_config = config.section("scheduler").mapping()


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


def _text_rows(tf: Any, features: torch.Tensor) -> torch.Tensor:
    """`Ideogram4Transformer2DModel.forward`'s text rows, which no step changes: the caption's
    features normed and projected, plus the text indicator's embedding. [1, texts, hidden]."""
    text = tf.llm_cond_proj(tf.llm_cond_norm(features[None]))
    return text + tf.embed_image_indicator(torch.zeros(1, dtype=torch.long, device=features.device))


def _rotary(tf: Any, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The forward's MRoPE tables for `position_ids`, in its activations' dtype."""
    cos, sin = tf.rotary_emb(position_ids)
    return cos.to(DTYPE), sin.to(DTYPE)


def _velocity(tf: Any, image: torch.Tensor, prefix: torch.Tensor | None, t: torch.Tensor, cos: torch.Tensor,
              sin: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """`Ideogram4Transformer2DModel.forward` past its text rows: the image tokens embedded beside
    `prefix` (None: image tokens alone), the blocks, and the final layer over the image rows."""
    hidden = tf.input_proj(image) + tf.embed_image_indicator(torch.ones(1, dtype=torch.long, device=image.device))
    if prefix is not None:
        hidden = torch.cat([prefix, hidden], dim=1)
    adaln = F.silu(tf.adaln_proj(tf.t_embedding(t).unsqueeze(1)))
    mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
    for block in tf.layers:
        hidden = block(hidden, mask, (cos, sin), adaln)
    return tf.final_layer(hidden[:, hidden.shape[1] - image.shape[1] :], conditioning=adaln)  # type: ignore[no-any-return]


class Ideogram4(Model[IdeogramGraph], encoded_leaves="accept"):
    """Ideogram 4's two DiTs and its VAE."""

    graph: IdeogramGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(IdeogramGraph, factory=IdeogramGraph)

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
        on_step: Callable[[int], None],
    ) -> torch.Tensor:
        """`Ideogram4Pipeline.__call__`'s loop for one caption whose `[tokens, 53248]` text features
        are `features`, as one `denoise_loop`: Euler over the resolution-shifted logit-normal sigmas,
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
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config)
        if scheduler.config.stochastic_sampling:
            raise ValueError("Ideogram 4 samples with deterministic Euler")
        sigmas = _logit_normal_sigmas(steps, _resolution_aware_mu(height, width, mu), std=std, device=device)
        scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        with torch.inference_mode(False), torch.no_grad():
            position_ids, segment_ids, _ = Ideogram4Pipeline._prepare_ids([texts], rows, columns, texts, device)
            latents = rng.noise((1, tokens, tf.config.in_channels), seed, device=device)
            prefix = _text_rows(tf, features.to(device, DTYPE))
            times = (1.0 - scheduler.timesteps.float() / scheduler.config.num_train_timesteps).to(DTYPE)
            deltas = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).to(device)
            weights = torch.tensor([[w for w in guidance], [1.0 - w for w in guidance]], dtype=torch.float64)
            weights = weights.float().to(device)
            branches = (
                *_rotary(tf, position_ids), segment_ids,
                *_rotary(uncond, position_ids[:, texts:]), segment_ids[:, texts:],
            )

        def step(i: torch.Tensor, latents: torch.Tensor, times: torch.Tensor, deltas: torch.Tensor,
                 weights: torch.Tensor, prefix: torch.Tensor, *branches: torch.Tensor) -> None:
            t = times.index_select(0, i)
            image = latents.to(DTYPE)
            positive = _velocity(tf, image, prefix, t, *branches[:3]).to(torch.float32)
            negative = _velocity(uncond, image, None, t, *branches[3:]).to(torch.float32)
            weight, rest = weights.index_select(1, i).unbind(0)
            velocity = weight.reshape(()) * positive + rest.reshape(()) * negative
            latents.copy_(latents + deltas.index_select(0, i).reshape(()) * -velocity)

        denoise_loop(step, range(steps), state=(latents,), inputs=(times, deltas, weights, prefix, *branches),
                     key=("ideogram-4.denoise/1",), on_step=on_step)
        return latents.cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor, *, width: int, height: int) -> Image.Image:
        """The pipeline's decode: the VAE's BatchNorm statistics undone on the packed latents,
        2x2 patches unpacked, then the decoder."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        with torch.inference_mode():
            z = latents.to(device)
            mean = vae.bn.running_mean.view(1, 1, -1).to(z)
            std = torch.sqrt(vae.bn.running_var + vae.config.batch_norm_eps).view(1, 1, -1).to(z)
            z = z * std + mean
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
        return image
