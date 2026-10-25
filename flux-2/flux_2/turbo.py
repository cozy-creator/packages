"""fal's FLUX.2 [dev] Turbo: a rank-256 LoRA over [dev]'s linears (`fal/FLUX.2-dev-Turbo`), its own model.

The overlay is never merged into a [dev] lane. While `applied` runs, [dev]'s linear hooks
(`cozy_runtime.lora.Hook`) serve its factors: Runtime bakes them into the resident weights by default, and a
hook adds the update itself where they are not baked.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import torch
from cozy_runtime import lora
from cozy_runtime.author import Config, Loader, Model, uses_components
from diffusers import Flux2Transformer2DModel
from torch import nn


T = TypeVar("T")
#: The upload's one component: the LoRA file at the repository's root.
COMPONENT = "model"


class Factors(nn.Module):
    """One site's `lora_A` [r, in] and `lora_B` [out, r], applied as scale · B·A."""

    def __init__(self, inputs: int, outputs: int, rank: int, scale: float) -> None:
        super().__init__()
        self.lora_A = nn.Linear(inputs, rank, bias=False)
        self.lora_B = nn.Linear(rank, outputs, bias=False)
        self.scale = scale

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> None:
        lora.accumulate(x, out, self.lora_A.weight, self.lora_B.weight, self.scale)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self.lora_A.weight, self.lora_B.weight, self.scale


class TurboGraph:
    def __init__(self, config: Config, geometry: Mapping[str, Any] | None = None) -> None:
        adapter = config.section(COMPONENT).mapping()
        rank = int(adapter["r"])
        scale = float(adapter.get("lora_alpha", rank)) / rank
        with torch.device("meta"):  # the base's sites; the class defaults are [dev]'s geometry
            dev = Flux2Transformer2DModel(**(geometry or {}))
        root = nn.Module()
        self.sites: dict[str, Factors] = {}
        for target in adapter["target_modules"]:
            path = target.removeprefix("transformer.")
            linear = dev.get_submodule(path)
            factors = Factors(linear.in_features, linear.out_features, rank, scale)
            parent = root
            for name in target.split(".")[:-1]:
                if not hasattr(parent, name):
                    parent.add_module(name, nn.Module())
                parent = getattr(parent, name)
            parent.add_module(target.rsplit(".", 1)[-1], factors)
            self.sites[path] = factors
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(torch.bfloat16)}


class Flux2TurboLoRA(Model[TurboGraph], min_bits=16):
    """The Turbo LoRA's factors, kept at their stored 16 bits."""

    graph: TurboGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(TurboGraph, factory=TurboGraph)

    @uses_components(COMPONENT)
    def applied(self, transformer: nn.Module, body: Callable[[], T]) -> T:
        """`body` with this overlay armed on [dev]'s `transformer`."""
        if not hasattr(transformer, "_cozy_turbo"):
            raise ValueError("the Turbo LoRA adapts FLUX.2 [dev] only")
        transformer._cozy_turbo = self.graph.sites
        try:
            return body()
        finally:
            transformer._cozy_turbo = None
