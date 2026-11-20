"""Runtime-owned Practical-RIFE weights and placement."""

from __future__ import annotations

from typing import cast

import torch
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from cozy_runtime.media import DecodedVideoFrame
from torch import Tensor
from torch.nn import functional as F

from .net import IFNet


class Graph:
    def __init__(self, config: Config) -> None:
        if config.as_str("architecture") != "Practical-RIFE 4.26":
            raise ValueError(
                "Anime Frame Interpolation requires the Practical-RIFE 4.26 checkpoint"
            )
        self.components = {"rife": IFNet().eval()}


class Rife(Model[Graph]):
    serving = Serving(encoded_leaves=False, min_bits=16)
    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)

    @uses_components("rife")
    def interpolate(
        self, left: DecodedVideoFrame, right: DecodedVideoFrame, position: float
    ) -> bytes:
        net = self.graph.components["rife"]
        return interpolate(net, left, right, position, torch.device(str(self.device)))


@torch.inference_mode()
def interpolate(
    net: IFNet,
    left: DecodedVideoFrame,
    right: DecodedVideoFrame,
    position: float,
    device: torch.device,
) -> bytes:
    height, width = left.height, left.width
    padding = (0, -width % 64, 0, -height % 64)

    def padded(frame: DecodedVideoFrame) -> Tensor:
        pixels = torch.frombuffer(bytearray(frame.rgb), dtype=torch.uint8).view(height, width, 3)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device, net.encode.cnn0.weight.dtype) / 255
        return F.pad(pixels, padding)

    output = cast(Tensor, net(padded(left), padded(right), position))[0, :, :height, :width]
    pixels = output.mul(255).round().clamp(0, 255).to(torch.uint8)
    return cast(bytes, pixels.permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
