"""Build the Open-H3-Regenerate-2K refine overlay: its merged LoRA and the refine plan's tables.

NVlabs' release (`Efficient-Large-Model/Open-H3-Regenerate-2K`, code `NVlabs/Sana` `sol-engine`
`models/Open-H3-Regenerate-2K`) refines an upscaled H3 latent through the FL2VA DiT with one merged
LoRA (refiner + LightX2V FL2VA Turbo + LMS Sharp, rank 576, scale 1) and LMS Sharp's AdaLN curve
branches. The curve's update depends on the timestep alone, so it lands in the overlay's tables at
the plan's exact timesteps beside the base modulation; the factors stay factors (the Runtime bakes
them).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from cozy_runtime.author import (
    Context,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    Weights,
    canonical_json,
    invocable,
)
from cozy_runtime.weights_store import Derivation

from ._memo import REGENERATE
from .adaln_operations import _sections, _validate_generators
from .kernel import CurveAdapter, H3Topology, table_shapes
from .model_config import DIT_COMPONENT
from .plans import TimestepPlan, refine_plan
from .source import TIMESTEP_PLAN, Rows, open_source, plain, put, read, rows, write_config, write_tables

COMPONENT = "fl2va_regenerate"
RANK = 576
CURVE_RANK = 64
SHIFT = 6.0
#: The release's 3-step grid (`open_h3/sigmas.py` SCHEDULES[3]).
RELEASE = (0.909375, 0.8571428571428571, 0.6666666666666666)
#: The starts a request may name (`upscale.strength`): the release's, and two gentler ones.
STARTS = (0.7, 0.8, RELEASE[0])


def shifted_grid(start: float, steps: int, shift: float = SHIFT) -> tuple[float, ...]:
    """`open_h3/sigmas.py`'s shifted_grid without its terminal zero: even steps in unshifted
    sigma from `start`, then `s·x / (1 + (s - 1)·x)`."""
    raw = start / (shift - (shift - 1) * start)
    points = (raw * (1 - i / steps) for i in range(1, steps))
    return (start, *(shift * x / (1 + (shift - 1) * x) for x in points))


def refine_sigmas(start: float) -> tuple[float, ...]:
    """The release grid at its own start; elsewhere the shift-6 grid with the release's unshifted
    step size or finer: 3·u(start)/u(0.909375) steps, rounded up (0.7 gives the README's 2)."""
    if start == RELEASE[0]:
        return RELEASE

    def unshifted(value: float) -> float:
        return value / (SHIFT - (SHIFT - 1) * value)

    return shifted_grid(start, math.ceil(3 * unshifted(start) / unshifted(RELEASE[0]) - 1e-9))


def plan() -> TimestepPlan:
    return refine_plan("fl2va", [refine_sigmas(start) for start in STARTS], shift=SHIFT)


def _sites(config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Every LoRA site of the merged adapter, in the overlay's construction order, with [in, out]."""
    hidden, inner, ffn = (
        int(config["hidden_size"]),
        int(config["num_attention_heads"]) * int(config["attention_head_dim"]),
        int(config["ffn_dim"]),
    )
    families = {
        "attn.to_q": (hidden, inner),
        "attn.to_k": (hidden, inner),
        "attn.to_v": (hidden, inner),
        "attn.to_out.0": (inner, hidden),
        "ff.net.0.proj": (hidden, 2 * ffn),
        "ff.net.2": (ffn, hidden),
    }
    sites = {}
    for prefix, count in (
        ("token_refiner.refiner_blocks", int(config["num_refiner_layers"])),
        ("transformer_blocks", int(config["num_layers"])),
    ):
        for index in range(count):
            for family, io in families.items():
                sites[f"{prefix}.{index}.{family}"] = io
    return sites


def overlay_order(config: dict[str, Any], plan: TimestepPlan) -> list[str]:
    """The overlay's keys in construction order (minimax-h3 `Overlay`, no heads)."""
    keys = []
    blocks = int(config["num_layers"])
    for site in _sites(config):
        keys += [f"{site}.lora_down", f"{site}.lora_up"]
        if site.startswith("transformer_blocks.") and site.endswith(".ff.net.2"):
            keys.append(site.rsplit(".ff.", 1)[0] + ".adaln_proj.table")
    assert sum(k.endswith(".adaln_proj.table") for k in keys) == blocks
    return [*keys, "norm_out.table"]


def _source(present: Rows, config: dict[str, Any]) -> str:
    """The converted release's one component: the merged factors and the curve sites (h3.lora)."""
    if len(present) != 1:
        raise UnsupportedInput(
            f"the Open-H3 source holds one h3.lora component, not {sorted(present)}",
            code="h3_regenerate_source",
        )
    (name, found), = present.items()
    expected = {}
    for site, (inputs, outputs) in _sites(config).items():
        expected[f"{site}.lora_A.weight"] = (RANK, inputs)
        expected[f"{site}.lora_B.weight"] = (outputs, RANK)
    for index in range(int(config["num_layers"])):
        site = f"transformer_blocks.{index}.adaln_curve"
        expected[f"{site}.lora_A.weight"] = (CURVE_RANK, 8)
        expected[f"{site}.lora_B.weight"] = (18 * int(config["hidden_size"]), CURVE_RANK)
        expected[f"{site}.table"] = (1025, 8)
    tensors = {key: row for key, row in found.items() if not key.endswith(".alpha")}
    if set(tensors) != set(expected) or not all(
        plain(tensors[key], "f32", shape) for key, shape in expected.items()
    ):
        missing, unknown = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
        raise UnsupportedInput(
            f"the Open-H3 source must hold exactly the rank-{RANK} F32 merged LoRA on "
            f"{len(_sites(config))} sites and the 50 curve sites (missing {missing[:3]}, "
            f"unknown {unknown[:3]})",
            code="h3_regenerate_source",
        )
    return name


@invocable(memo_version="h3-regenerate-lora/1", memo_dependencies=REGENERATE)
async def regenerate_lora(
    ctx: Context,
    *,
    adapters: Weights[Any],
    base: Weights[Any],
    tel: Telemetry,
) -> ModelArtifact:
    """The `fl2va_regenerate` overlay from the converted Open-H3-Regenerate-2K LoRA folder.

    ``adapters`` is the release's `lora/` folder through h3.lora. ``base`` is any full-precision
    H3 checkpoint; only its FL2VA modulation weights are read. Factors are cast to bf16; each
    table row is the base modulation plus the LMS Sharp curve update at that exact timestep.
    """
    config = _sections()[DIT_COMPONENT["fl2va"]]
    topology = H3Topology.from_config(config)
    full, source = open_source(ctx, base), open_source(ctx, adapters)
    _validate_generators(rows(full), "fl2va", topology)
    name = _source(rows(source), config)
    refine = plan()
    order = [(COMPONENT, key) for key in overlay_order(config, refine)]
    document = {
        COMPONENT: {
            **config,
            "cozy_h3": {
                "task": "fl2va",
                "modulation": "adaln-pruned",
                "distillation": "regenerate",
                "table_keys": refine.table_keys,
                "lora_rank": RANK,
                "lora_alpha": float(RANK),
            },
        }
    }
    if ctx.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    computed = reused = 0
    with ctx.output("bf16", Derivation(base=None, sources=(full, source), order=order)) as out:
        write_config(out, canonical_json.encode(document))
        if out.done is None:
            out.component(COMPONENT).asset(TIMESTEP_PLAN, refine.canonical_bytes)
        for site, (inputs, outputs) in _sites(config).items():
            ctx.raise_if_cancelled()
            for suffix, key, shape in (
                ("lora_down", f"{site}.lora_A.weight", (RANK, inputs)),
                ("lora_up", f"{site}.lora_B.weight", (outputs, RANK)),
            ):
                if out.has(COMPONENT, f"{site}.{suffix}"):
                    reused += 1
                    continue
                put(out, COMPONENT, f"{site}.{suffix}", read(source, name, key, torch.float32, shape).bfloat16())
                computed += 1

        def reader(key: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
            return read(source, name, key, dtype, shape)

        _, _, kept = write_tables(
            ctx,
            plan=refine,
            topology=topology,
            source=full,
            component=DIT_COMPONENT["fl2va"],
            target=COMPONENT,
            outputs=(out,),
            progress=lambda done, total: tel.progress(done / total, stage="regenerate-tables"),
            curve=CurveAdapter(CURVE_RANK, reader),
        )
        tables = len(table_shapes(topology, refine))
        reused, computed = reused + kept, computed + tables - kept
        artifact = ctx.adopt_model(out)
    tel.metric("h3.regenerate.reused_tensors", float(reused))
    tel.metric("h3.regenerate.computed_tensors", float(computed))
    return artifact
