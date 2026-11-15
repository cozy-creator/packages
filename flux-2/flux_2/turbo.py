"""fal's FLUX.2 [dev] Turbo: a rank-256 LoRA over [dev]'s linears (`fal/FLUX.2-dev-Turbo`), its own model.

The overlay is never merged into a [dev] lane. While `applied` runs, [dev]'s linear hooks
(`cozy_runtime.lora.Hook`) serve its factors: Runtime bakes them into the resident weights by default, and a
hook adds the update itself where they are not baked. Its sites are the checkpoint's: each target's stored
factors give their shapes (`Config.tensor_shape`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch
from cozy_runtime.accel import lora
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from torch import nn


T = TypeVar("T")
#: The upload's one component: TensorFS's `flux2.lora`, PEFT factors under [dev]'s module names.
COMPONENT = "transformer"
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


class TurboGraph:
    def __init__(self, config: Config) -> None:
        adapter = config.mapping()  # the one component's PEFT adapter config
        root = nn.Module()
        self.sites: dict[str, Factors] = {}
        dtype = torch.bfloat16
        for target in adapter["target_modules"]:
            path = target.removeprefix("transformer.")
            key = f"{COMPONENT}.{path}"
            down = config.tensor_shape(f"{key}.lora_A.weight", default=())
            up = config.tensor_shape(f"{key}.lora_B.weight", default=())
            if len(down) != 2 or len(up) != 2:
                raise ValueError(f"the Turbo LoRA's {path} has no stored factors to build it from")
            (rank, inputs), (outputs, _) = down, up
            # α: the config's `lora_alpha` (one rank, so no module states its own), else r.
            factors = Factors(inputs, outputs, rank, float(adapter.get("lora_alpha", rank)) / rank)
            dtype = _DTYPES[config.tensor_dtype(f"{key}.lora_A.weight", default="bf16")]
            parent = root
            for name in path.split(".")[:-1]:
                if not hasattr(parent, name):
                    parent.add_module(name, nn.Module())
                parent = getattr(parent, name)
            parent.add_module(path.rsplit(".", 1)[-1], factors)
            self.sites[path] = factors
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(dtype)}


class Flux2TurboLoRA(Model[TurboGraph]):
    """The Turbo LoRA's factors, kept at their stored 16 bits."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: TurboGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(TurboGraph)

    @uses_components(COMPONENT)
    def applied(self, transformer: nn.Module, body: Callable[[], T]) -> T:
        """`body` with this overlay armed on [dev]'s `transformer`."""
        if not hasattr(transformer, "turbo_overlay"):
            raise ValueError("the Turbo LoRA adapts FLUX.2 [dev] only")
        transformer.turbo_overlay = self.graph.sites  # type: ignore[assignment]  # a plain attribute: nn.Module keeps non-tensors as such
        try:
            return body()
        finally:
            transformer.turbo_overlay = None  # type: ignore[assignment]  # a plain attribute
