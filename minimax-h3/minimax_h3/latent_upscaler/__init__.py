"""LBH-123-AI's 3D-conv H3 latent upscaler (v1), as Open-H3-Regenerate-2K runs it: the whole
latent at once, from H3-normalized latents normalized again by the network's own statistics, as
the ComfyUI node does to the already-normalized latents it receives.

Its int8 lane (`UPSCALER_INT8`, cr-327) stores the 512-channel convs as int8-convrot filters; Runtime runs
them on the int8 tensor cores, each block's GroupNorm, modulation and SiLU folded into the pass that
quantizes its activations (`BLOCKS`). Against f32, 63.65 dB where bf16 is 63.85 dB."""

from __future__ import annotations

import math
from typing import cast

import msgspec
import torch
from cozy_runtime.accel import fusion, sequence_parallel
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import (
    Config,
    InvalidRequest,
    Loader,
    Model,
    QuantizationPlan,
    Recipe,
    uses_components,
)

from .architecture import LatentResizer3D, ResBlockEmb3D, TemporalConv

ARCHITECTURE = "minimax-h3-latent-upscaler-3d-conv-v1"


class _Normalization(msgspec.Struct, frozen=True):
    enabled: bool
    mean: tuple[float, ...]
    std: tuple[float, ...]


def _normalization(config: Config) -> _Normalization:
    stats = config.section("latent_normalization").typed(_Normalization)
    if not stats.enabled:
        raise InvalidRequest(
            "latent_normalization.enabled must be true", code="config_invalid",
            fields=["latent_normalization.enabled"],
        )
    for name, values in (("mean", stats.mean), ("std", stats.std)):
        if len(values) != 24 or any(
            not math.isfinite(value) or (name == "std" and value <= 0) for value in values
        ):
            raise InvalidRequest(
                f"latent_normalization.{name} must contain 24 finite {'positive ' if name == 'std' else ''}numbers",
                code="config_invalid", fields=[f"latent_normalization.{name}"],
            )
    return stats


class UpscalerGraph:
    def __init__(self, config: Config) -> None:
        # The checkpoint owns the architecture identity and learned normalization.
        if config.mapping().get("architecture") != ARCHITECTURE:
            raise ValueError("the latent upscaler requires LBH-123-AI's 3D-conv v1 checkpoint")
        self.normalization = _normalization(config)
        self.components = {"upscaler": LatentResizer3D().eval().bfloat16()}


def build_upscaler(config: Config) -> UpscalerGraph:
    return UpscalerGraph(config)


#: The upscaler's blocks as Runtime runs them on an int8 lane: each `silu(norm(x))` before a conv
#: one pass with its quantize (the second norm modulated by the scale embedding), the residual in
#: the conv's epilogue; the temporal blocks' depthwise conv in the pass before their 1x1x1 conv.
BLOCKS = fusion.ConvBlocks(
    blocks=(
        fusion.ConvResidual(
            ResBlockEmb3D,
            steps=(("in_layers.0", "in_layers.2"), ("out_norm", "out_layers.2")),
            shortcut="skip",
            modulation=fusion.ConvModulation(step=1, rows="emb_layers"),
        ),
        fusion.TemporalMix(TemporalConv, norm="norm", depthwise="dwconv", pointwise="pwconv"),
    )
)


@sequence_parallel(degrees=(2, 4, 7, 8))
class H3LatentUpscaler(Model[UpscalerGraph]):
    """Replicated beside H3 on every CP rank; it runs whole on each."""

    graph: UpscalerGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(build_upscaler)

    def execution_plan(self, plan: Plan) -> str:
        return fusion.install_conv_blocks(plan, BLOCKS, self.graph.components["upscaler"])

    @uses_components("upscaler")
    def upscale(self, latents: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Normalized `(1, 24, T, h, w)` H3 latents to `(1, 24, T, height, width)`, float32 on the
        device; time is unchanged. The node's TARGET_DIMENSIONS scale embedding, no chunking."""
        if latents.ndim != 5 or latents.shape[:2] != (1, 24):
            raise ValueError("expected batch-one, 24-channel H3 video latents")
        in_height, in_width = (int(value) for value in latents.shape[-2:])
        if not (in_height <= height <= 4 * in_height and in_width <= width <= 4 * in_width):
            raise ValueError("learned latent upscaling supports spatial factors from 1 to 4")
        model = self.graph.components["upscaler"]
        device = next(model.parameters()).device
        stats = self.graph.normalization
        mean = torch.tensor(stats.mean, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(stats.std, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        scale = (height / in_height + width / in_width) / 2.0
        with torch.inference_mode():
            normalized = ((latents.to(device, torch.float32) - mean) / std).to(torch.bfloat16)
            # NDHWC from the first conv on: the int8 lane's blocks read it so (`BLOCKS`).
            normalized = normalized.contiguous(memory_format=torch.channels_last_3d)
            result = model(normalized, scale=scale, target_size=(latents.shape[2], height, width))
            return cast(torch.Tensor, result.float() * std + mean)


#: The int8 lane: every 512-channel conv in int8-convrot (the 48 3x3x3 convs and the 12 temporal
#: blocks' 1x1x1 convs), its filter stored channels-last. The 24-channel first and last convs, the
#: depthwise temporal convs, the norms and the scale embedding keep their bf16 bytes.
UPSCALER_INT8 = Recipe(
    lane="int8",
    model=H3LatentUpscaler,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("upscaler",),
        include=(
            "upscaler.*_blocks.*.in_layers.2.weight",
            "upscaler.*_blocks.*.out_layers.2.weight",
            "upscaler.*_blocks.*.pwconv.weight",
        ),
        output_precision="preserve",
    ),
)
