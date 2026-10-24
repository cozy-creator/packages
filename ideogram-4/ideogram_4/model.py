"""Ideogram 4, component-staged over Diffusers' `Ideogram4Pipeline` (0.41.0).

Two DiTs run every step: the conditional one over the caption's text tokens and the image tokens,
and the unconditional one over the image tokens alone (Ideogram's asymmetric CFG). The text encoder
is the stock Qwen3-VL-8B-Instruct Ideogram froze, a slot of its own (`text_encoder.py`). Runtime
constructs and fills the modules and owns their residency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from cozy_runtime import rng
from cozy_runtime.author import Config, Loader, Model, uses_components
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


def section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Ideogram 4 {name} configuration must be a mapping")
    return {key: item for key, item in value.items() if not key.startswith("_")}


class IdeogramGraph:
    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        self.components: dict[str, torch.nn.Module] = {
            name: Ideogram4Transformer2DModel.from_config(section(mapping, name)).to(dtype=torch.bfloat16)
            for name in ("transformer", "unconditional_transformer")
        }
        self.components["vae"] = AutoencoderKLFlux2.from_config(section(mapping, "vae")).to(dtype=torch.bfloat16)
        self.scheduler_config = section(mapping, "scheduler")


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


class Ideogram4(Model[IdeogramGraph], encoded_leaves="accept"):
    """Ideogram 4's two DiTs and its VAE."""

    graph: IdeogramGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(IdeogramGraph, factory=IdeogramGraph)

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
        cancel: Callable[[], None],
    ) -> torch.Tensor:
        """`Ideogram4Pipeline.__call__`'s loop for one caption whose `[tokens, 53248]` text features
        are `features`: Euler over the resolution-shifted logit-normal sigmas, each step the
        conditional and unconditional velocities blended by its guidance weight, latents in float32,
        noise drawn on the CPU from `seed`."""
        tf = self.graph.components["transformer"]
        uncond = self.graph.components["unconditional_transformer"]
        device = torch.device(str(self.device))
        steps = len(guidance)
        rows, columns = height // PATCH, width // PATCH
        tokens, texts = rows * columns, features.shape[0]
        position_ids, segment_ids, indicator = Ideogram4Pipeline._prepare_ids([texts], rows, columns, texts, device)
        dtype = torch.bfloat16
        llm = torch.cat([features, features.new_zeros(tokens, features.shape[1])])[None].to(device, dtype)
        neg_llm = torch.zeros(1, tokens, features.shape[1], dtype=dtype, device=device)
        mu, std = schedule_shape(steps)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config)
        sigmas = _logit_normal_sigmas(steps, _resolution_aware_mu(height, width, mu), std=std, device=device)
        scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        channels = tf.config.in_channels
        latents = rng.noise((1, tokens, channels), seed, device=device)
        padding = torch.zeros(1, texts, channels, dtype=torch.float32, device=device)
        train_steps = scheduler.config.num_train_timesteps
        with torch.inference_mode():
            for i, t in enumerate(scheduler.timesteps):
                cancel()
                t_model = (1.0 - t.float() / train_steps).expand(1).to(dtype)
                positive = tf(
                    hidden_states=torch.cat([padding, latents], dim=1).to(dtype), timestep=t_model,
                    encoder_hidden_states=llm, position_ids=position_ids, segment_ids=segment_ids,
                    indicator=indicator, return_dict=False,
                )[0][:, texts:].to(torch.float32)
                negative = uncond(
                    hidden_states=latents.to(dtype), timestep=t_model, encoder_hidden_states=neg_llm,
                    position_ids=position_ids[:, texts:], segment_ids=segment_ids[:, texts:],
                    indicator=indicator[:, texts:], return_dict=False,
                )[0].to(torch.float32)
                weight = guidance[i]
                velocity = weight * positive + (1.0 - weight) * negative
                latents = scheduler.step(-velocity, t, latents, return_dict=False)[0]
                on_step(i)
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
                raise ValueError("Ideogram 4's decoder produced non-finite pixels")
            (image,) = VaeImageProcessor(vae_scale_factor=PATCH).postprocess(pixels.float(), output_type="pil")
        return image
