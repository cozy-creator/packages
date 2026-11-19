"""A Qwen Image 2.1 LoRA a function arms from its own slot (ausboss's Restore LoRA, `upscale`'s default), its own model.

Its factors at each of its `target_modules` (PEFT's config, which TensorFS writes with `lora_alpha` when every module has
one rank) on the DiT's `adapters` sites (`model.QwenImage21DiTGraph`), beside Turbo's on `sites`: the Runtime bakes both
into the resident weights while they are armed. A caller's `--lora` stays the Runtime's (`serving.loras`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import torch
from cozy_runtime.accel import lora
from cozy_runtime.author import Config, Loader, Model, Serving, UnsupportedInput, uses_components
from torch import nn

from .model import SITES, QwenImage21DiT

#: An uploaded Qwen Image 2.1 LoRA's one component (TensorFS's `qwenimage21.lora`).
COMPONENT = "transformer"
_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
T = TypeVar("T")


class LoRAGraph:
    def __init__(self, config: Config) -> None:
        adapter = config.section(COMPONENT) if COMPONENT in config.keys() else config
        settings = adapter.mapping()
        if "lora_alpha" not in settings:
            raise UnsupportedInput(
                "the LoRA's modules state their own α (no single lora_alpha), which this slot doesn't read",
                code="artifact_config",
                fields=["lora_alpha"],
            )
        rank = adapter.as_int("r")
        scale = adapter.as_float("lora_alpha") / rank
        named = [str(target) for target in settings["target_modules"]]
        matched = {target: [path for path in SITES if path == target or path.endswith("." + target)] for target in named}
        #: What it adapts that the slot can't arm: the output projection, which Turbo's per-step heads replace, or no DiT
        #: linear. The run warns of them and arms the rest, as the Runtime does a request LoRA's partial fit (cl-1241).
        self.unfit = tuple(target for target, paths in matched.items() if not paths)
        targets = sorted({path for paths in matched.values() for path in paths})
        if not targets:
            raise UnsupportedInput(
                f"no module of the LoRA is a Qwen Image 2.1 DiT linear the slot arms ({', '.join(named[:3])}, …)",
                code="artifact_config",
                fields=["target_modules"],
            )
        root = nn.Module()
        self.factors: dict[str, lora.Pair] = {}
        for path in targets:
            inputs, outputs = SITES[path]
            parent = root
            *parents, leaf = path.split(".")
            for name in parents:
                if name not in parent._modules:
                    parent.add_module(name, nn.Module())
                parent = cast(nn.Module, parent._modules[name])  # added above
            self.factors[path] = lora.Pair(inputs, outputs, rank, scale)
            parent.add_module(leaf, self.factors[path])
        dtype = _DTYPES[adapter.tensor_dtype(f"{COMPONENT}.{targets[0]}.lora_A.weight", default="bf16")]
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(dtype)}


class QwenImage21LoRA(Model[LoRAGraph]):
    """A LoRA's factors, kept at their stored 16 bits."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: LoRAGraph

    @property
    def unfit(self) -> tuple[str, ...]:
        return self.graph.unfit

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(LoRAGraph)

    @uses_components(COMPONENT)
    def over(self, base: QwenImage21DiT, run: Callable[[], T]) -> T:
        """`run` with this LoRA armed on `base`'s DiT."""
        with base.graph.adapters.applied(self.graph.factors):
            return run()
