"""LBH-123-AI's 3D-conv H3 latent upscaler (v1), as Open-H3-Regenerate-2K runs it: the whole
latent at once, from H3-normalized latents normalized again by the network's own statistics, as
the ComfyUI node does to the already-normalized latents it receives."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import cast

import torch
from cozy_runtime.author import Config, Loader, Model, uses_components

from .architecture import LatentResizer3D

ARCHITECTURE = "minimax-h3-latent-upscaler-3d-conv-v1"
_STATS = json.loads(files(__name__).joinpath("config.json").read_text())
MEAN = torch.tensor(_STATS["latent_normalization"]["mean"]).view(1, -1, 1, 1, 1)
STD = torch.tensor(_STATS["latent_normalization"]["std"]).view(1, -1, 1, 1, 1)


class UpscalerGraph:
    def __init__(self, config: Config) -> None:
        if config.mapping().get("architecture") != ARCHITECTURE:
            raise ValueError("the latent upscaler requires LBH-123-AI's 3D-conv v1 checkpoint")
        self.components = {"upscaler": LatentResizer3D().eval().bfloat16()}  # type: ignore[no-untyped-call]


def build_upscaler(config: Config) -> UpscalerGraph:
    return UpscalerGraph(config)


class H3LatentUpscaler(Model[UpscalerGraph]):
    graph: UpscalerGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpscalerGraph, factory=build_upscaler)

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
        mean, std = MEAN.to(device), STD.to(device)
        scale = (height / in_height + width / in_width) / 2.0
        with torch.inference_mode():
            normalized = ((latents.to(device, torch.float32) - mean) / std).to(torch.bfloat16)
            result = model(normalized, scale=scale, target_size=(latents.shape[2], height, width),
                           enable_chunking=False)  # fmt: skip
            return cast(torch.Tensor, result.float() * std + mean)
