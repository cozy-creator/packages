"""Upstream BiRefNet, with construction and weight residency owned by Runtime."""

from __future__ import annotations

from typing import cast

import torch
from PIL import Image
from torchvision import transforms

from cozy_runtime.author import Config, Loader, Model, uses_components

from .upstream.BiRefNet_config import BiRefNetConfig
from .upstream.birefnet import BiRefNet


class BiRefNetGraph:
    def __init__(self, config: Config) -> None:
        if config.mapping().get("bb_pretrained") is not False:
            raise ValueError("BiRefNet's checkpoint is the complete model (bb_pretrained: false)")
        self.components = {
            "model": BiRefNet(config=BiRefNetConfig(bb_pretrained=False)).eval().half(),
        }


def build_birefnet(config: Config) -> BiRefNetGraph:
    return BiRefNetGraph(config)


class BackgroundRemovalModel(Model[BiRefNetGraph]):
    graph: BiRefNetGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(BiRefNetGraph, factory=build_birefnet)

    @uses_components("model")
    def mask(self, image: Image.Image) -> Image.Image:
        model = self.graph.components["model"]
        device = torch.device(str(self.device))
        # The official model's input transform and final sigmoid, using torchvision.
        prepare = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.inference_mode():
            pixels = prepare(image).unsqueeze(0).to(device=device, dtype=torch.float16)
            probability = model(pixels)[-1].sigmoid().float().cpu()[0].squeeze(0)
        mask = cast(Image.Image, transforms.ToPILImage()(probability))
        return mask.resize(image.size, Image.Resampling.BILINEAR)
