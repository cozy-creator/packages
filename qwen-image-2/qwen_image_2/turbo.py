"""Qwen Image 2's Turbo: Alibaba PAI's Qwen-Image-2.1-Fun-Acc-4Step over the base model, its own model.

A Parallel Decoding Distillation export (hf://alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs@f7545234,
`qwenimage21_pdd.py`): rank-64 factors on every DiT linear but the output projection, one output
projection per step (`proj_out.heads`), CFG 1, an F32 state. Its steps and sigma grid are the
checkpoint's `pdd_config.json` (`read_plan`). Its replacement q/k and text norms equal the base's to bf16
rounding, so TensorFS leaves them out.

The overlay is never merged into a base lane: while `denoise` runs, the base DiT's linear hooks
(`model.SITES`) serve its factors, and Runtime bakes them into the resident weights.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
from cozy_runtime.accel import lora
from cozy_runtime.author import Config, Loader, Model, Serving, UnsupportedInput, uses_components
from PIL import Image
from torch import nn

from .model import PROJ_OUT, SITES, QwenImage21DiT

#: The upload's one component (TensorFS's `qwenimage21.lora`).
COMPONENT = "transformer"
#: Its sampling plan: upstream's `models/pdd_config.json`, the component's file.
PLAN = f"{COMPONENT}/pdd_config.json"
#: `pdd_sigmas` are shifted and stretched at distillation, never again (`qwenimage21_pdd.py`): over the
#: base's scheduler, the grid as given, without the dynamic shift or the terminal stretch.
SCHEDULER = {"use_dynamic_shifting": False, "shift_terminal": None}
_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}


@dataclass(frozen=True)
class Plan:
    """One model call per sigma, each with its own output projection."""

    steps: int
    sigmas: tuple[float, ...]


def read_plan(raw: bytes) -> Plan:
    """`pdd_config.json`: `pdd_num_steps` calls on the first `pdd_num_steps` of its `pdd_sigmas` (the last is 0)."""
    config = json.loads(raw)
    steps, grid = int(config["pdd_num_steps"]), [float(sigma) for sigma in config["pdd_sigmas"]]
    if len(grid) != steps + 1 or config.get("pdd_block_size", 1) != 1:
        raise UnsupportedInput(
            f"{PLAN}: {len(grid)} sigmas for {steps} steps in blocks of {config.get('pdd_block_size', 1)}; "
            "Qwen Image 2 runs a grid of steps + 1 sigmas, one step a block",
            code="artifact_config",
            fields=[PLAN],
        )
    return Plan(steps, tuple(grid[:-1]))


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
    def __init__(self, config: Config, steps: int) -> None:
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
                parent = cast(nn.Module, parent._modules[name])  # added above
            self.sites[path] = Factors(inputs, outputs, rank, scale)
            parent.add_module(leaf, self.sites[path])
        self.heads = nn.Module()
        self.heads.heads = nn.Parameter(torch.empty(steps, *PROJ_OUT), requires_grad=False)
        root.add_module("proj_out", self.heads)
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(dtype)}


class FunAccLoRA(Model[FunAccGraph]):
    """Fun Acc's factors and heads, kept at their stored 16 bits."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: FunAccGraph
    plan: Plan

    @property
    def steps(self) -> int:
        return self.plan.steps

    def load(self, loader: Loader) -> None:
        self.plan = read_plan(loader.assets.read(PLAN))
        self.graph = loader.construct(partial(FunAccGraph, steps=self.plan.steps))

    @uses_components(COMPONENT)
    def denoise(
        self,
        base: QwenImage21DiT,
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        width: int,
        height: int,
        steps: int,
        seed: int,
        progress: Callable[[int, int], None],
        reference_images: list[Image.Image] | None = None,
        reference_latents: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
        start: torch.Tensor | None = None,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """`base.denoise` with this overlay armed on its DiT, on Fun Acc's grid and per-step heads."""
        transformer = base.graph.components["transformer"]
        heads = cast(torch.Tensor, self.graph.heads.heads).to(device=torch.device(str(base.device))).clone()  # its Parameter
        transformer._cozy_overlay = self.graph.sites
        try:
            return base.denoise(
                embeds, mask, width=width, height=height, steps=steps, seed=seed, progress=progress,
                sigmas=self.plan.sigmas, scheduler=SCHEDULER, heads=heads, reference_images=reference_images,
                reference_latents=reference_latents, image_pad_mask=image_pad_mask, start=start, strength=strength,
            )
        finally:
            transformer._cozy_overlay = None
