"""Qwen Image 2's Turbo: Alibaba PAI's Qwen-Image-2.1-Fun-Acc-4Step over the base model, its own model.

A Parallel Decoding Distillation export (hf://alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs@f7545234,
`qwenimage21_pdd.py`): rank-64 factors on every DiT linear but the output projection, one output
projection per step (`proj_out.heads`), 4 steps on its fixed grid, CFG 1, an F32 state. Its
replacement q/k and text norms equal the base's to bf16 rounding, so TensorFS leaves them out.

The overlay is never merged into a base lane: while `denoise` runs, the base DiT's linear hooks
(`model.SITES`) serve its factors, and Runtime bakes them into the resident weights.
"""

from __future__ import annotations

from typing import Any

import torch
from cozy_runtime.accel import lora
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from torch import nn

from .model import PROJ_OUT, SITES, QwenImage21Model

#: The upload's one component (TensorFS's `qwenimage21.lora`).
COMPONENT = "transformer"
STEPS = 4
#: `pdd_config.json` `pdd_sigmas`, shifted at distillation and never again.
SIGMAS = (1.0, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471)
#: Over the base's scheduler: the grid as given, without the dynamic shift or the terminal stretch.
SCHEDULER = {"use_dynamic_shifting": False, "shift_terminal": None}
_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}


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


class FunAccGraph:
    def __init__(self, config: Config) -> None:
        # The construction config is the component's PEFT adapter config.
        rank = config.as_int("r")
        scale = config.as_float("lora_alpha", float(rank)) / rank
        dtype = _DTYPES[config.tensor_dtype(f"{COMPONENT}.proj_out.heads", default="bf16")]
        root = nn.Module()
        self.sites: dict[str, Factors] = {}
        for path, (inputs, outputs) in SITES.items():
            parent = root
            *parents, leaf = path.split(".")
            for name in parents:
                if name not in parent._modules:
                    parent.add_module(name, nn.Module())
                parent = parent._modules[name]
            self.sites[path] = Factors(inputs, outputs, rank, scale)
            parent.add_module(leaf, self.sites[path])
        self.heads = nn.Module()
        self.heads.heads = nn.Parameter(torch.empty(STEPS, *PROJ_OUT), requires_grad=False)
        root.add_module("proj_out", self.heads)
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(dtype)}


class FunAccLoRA(Model[FunAccGraph]):
    """Fun Acc's factors and heads, kept at their stored 16 bits."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: FunAccGraph
    steps = STEPS

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(FunAccGraph)

    @uses_components(COMPONENT)
    def denoise(self, base: QwenImage21Model, *args: Any, **kwargs: Any) -> Any:
        """`base.denoise` with this overlay armed on its DiT, on Fun Acc's grid and per-step heads."""
        transformer = base.graph.components["transformer"]
        heads = self.graph.heads.heads.to(device=torch.device(str(base.device))).clone()
        transformer._cozy_overlay = self.graph.sites
        try:
            return base.denoise(*args, sigmas=SIGMAS, scheduler=SCHEDULER, heads=heads, **kwargs)
        finally:
            transformer._cozy_overlay = None
