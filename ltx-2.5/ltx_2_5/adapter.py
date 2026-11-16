"""Lightricks' Ingredients IC-LoRA for LTX-2.5 (`Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`, mirrored as
`lightricks/ingredients`): characters, props and places from a reference sheet carried into the video. It arms on the
DiT's `lora.Sites` for stage 1 only, the stage that takes its reference (ltx_pipelines/ic_lora.py
`DEFAULT_IC_LORA_STAGES`), at strength 1: "the weights ship pre-scaled, so full strength is the intended default"
(its model card). TensorFS stores it as PEFT's form under Diffusers' module names: `<module>.lora_A.weight` [r, in],
`.lora_B.weight` [out, r], scaled by α over the module's rank: its own `.alpha` (F32, where ranks differ), else the
config's `lora_alpha`, else r (TensorFS import/lora.rs)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch
from cozy_runtime.accel import lora
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from torch import nn
from transformers.initialization import no_init_weights

from .model import stored

R = TypeVar("R")
COMPONENT = "transformer"


def _add(root: nn.Module, path: str, module: nn.Module) -> None:
    parent = root
    *parents, leaf = path.split(".")
    for name in parents:
        if name not in parent._modules:
            parent.add_module(name, nn.Module())
        parent = parent.get_submodule(name)
    parent.add_module(leaf, module)


class Graph:
    def __init__(self, config: Config) -> None:
        settings = (config.section(COMPONENT) if COMPONENT in config.keys() else config).mapping()
        alpha = settings.get("lora_alpha")
        root = nn.Module()
        self.factors: dict[str, lora.Pair] = {}
        with no_init_weights():
            for path in (str(target) for target in settings["target_modules"]):
                rank, fan_in = config.tensor_shape(f"{COMPONENT}.{path}.lora_A.weight", default=(0, 0))
                fan_out, _ = config.tensor_shape(f"{COMPONENT}.{path}.lora_B.weight", default=(0, 0))
                factors = lora.Pair(fan_in, fan_out, rank, float(alpha) / rank if isinstance(alpha, (int, float)) else 1.0,
                                    dtype=torch.bfloat16)
                if config.tensor_shape(f"{COMPONENT}.{path}.alpha", default=(-1,)) != (-1,):
                    factors.register_buffer("alpha", torch.ones((), dtype=torch.float32))
                _add(root, path, factors)
                self.factors[path] = factors
        self.components = {COMPONENT: stored(root, COMPONENT, config)}


class Ingredients(Model[Graph]):
    serving = Serving(encoded_leaves=False, min_bits=16)
    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)

    @uses_components(COMPONENT)
    def applied(self, sites: lora.Sites, body: Callable[[], R]) -> R:
        """`body` with this LoRA armed on the DiT's `sites`: the Runtime bakes it into the resident weights, else the
        sites add it per call."""
        for factors in self.graph.factors.values():
            if hasattr(factors, "alpha"):
                factors.scale = float(factors.alpha) / factors.lora_A.weight.shape[0]
        with sites.applied(self.graph.factors):
            return body()
