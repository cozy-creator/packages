"""Build a standalone PDD-8 adapter with LoRA factors, modulation tables and heads."""

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
from cozy_runtime.weights_store import Checkpoint, Derivation

from .adaln_operations import _sections, _validate_generators
from .kernel import H3Topology, LowRankAdapter, adapter_shapes, table_shapes
from .model_config import DIT_COMPONENT
from .plans import Task, TimestepPlan
from .source import (
    Rows, copy_assets, open_source, plain, put, read, rows, timestep_plan, write_config, write_tables,
)

RANK = 64
NUM_STEPS = 32
BLOCK_SIZE = 4
TASKS: tuple[Task, ...] = ("fl2va", "ref2va")


def turbo_plan(source: Checkpoint, task: Task, component: str) -> TimestepPlan:
    """The adapter's own plan: one schedule, one evaluation per distilled PDD block."""
    plan = timestep_plan(source, task, component=component)
    if plan.steps != (NUM_STEPS // BLOCK_SIZE,):
        raise UnsupportedInput(
            f"the PDD adapter's {task} plan serves {plan.steps}, not its one "
            f"{NUM_STEPS // BLOCK_SIZE}-evaluation schedule",
            code="adaln_plan",
        )
    return plan


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shift * sigma / (1 + (shift - 1) * sigma)


def pdd_head_plan(shift: float, num_steps: int, block_size: int) -> torch.Tensor:
    if num_steps < 1 or block_size < 1 or num_steps % block_size:
        raise ValueError("the PDD head grid must divide into complete blocks")
    sizes = pdd_time_grid(shift, num_steps).diff()
    plan = torch.zeros(num_steps // block_size, num_steps, dtype=torch.float64)
    for index in range(num_steps // block_size):
        span = sizes[index * block_size : (index + 1) * block_size]
        plan[index, index * block_size : (index + 1) * block_size] = span / span.sum()
    return plan


def collapse_head_bank(
    weight: torch.Tensor, bias: torch.Tensor, plan: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the upstream float32 head fusion, including its rounding order."""
    if (
        weight.ndim != 3
        or bias.shape != weight.shape[:2]
        or plan.ndim != 2
        or plan.shape[1] != weight.shape[0]
    ):
        raise ValueError("head bank, bias and plan do not describe one N-interval head")
    # Upstream converts its float64 grid coefficients to the head's float32 dtype
    # before einsum. Float64 accumulation followed by casting changes the result.
    plan = plan.to(device=weight.device, dtype=torch.float32)
    return (
        torch.einsum("pn,noi->poi", plan, weight.float()),
        torch.einsum("pn,no->po", plan, bias.float()),
    )


def overlay_shapes(
    config: dict[str, Any], plan: TimestepPlan
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Constructor order of the overlay, with all six inference LoRA families."""
    hidden, inner, ffn = (
        int(config["hidden_size"]),
        int(config["num_attention_heads"]) * int(config["attention_head_dim"]),
        int(config["ffn_dim"]),
    )
    shapes: dict[str, tuple[str, tuple[int, ...]]] = {}
    families = {
        "attn.to_q": (hidden, inner),
        "attn.to_k": (hidden, inner),
        "attn.to_v": (hidden, inner),
        "attn.to_out.0": (inner, hidden),
        "ff.net.0.proj": (hidden, 2 * ffn),
        "ff.net.2": (ffn, hidden),
    }
    for prefix, count in (
        ("token_refiner.refiner_blocks", int(config["num_refiner_layers"])),
        ("transformer_blocks", int(config["num_layers"])),
    ):
        for index in range(count):
            for family, (inputs, outputs) in families.items():
                key = f"{prefix}.{index}.{family}"
                shapes[f"{key}.lora_down"] = ("bf16", (RANK, inputs))
                shapes[f"{key}.lora_up"] = ("bf16", (outputs, RANK))
            if prefix == "transformer_blocks":
                shapes[f"{prefix}.{index}.adaln_proj.table"] = (
                    "bf16",
                    (len(plan.block_rows), 6, hidden),
                )
    shapes["norm_out.table"] = ("bf16", (len(plan.timesteps), 2, hidden))
    for prefix, outputs in (
        ("proj_out", int(config["in_channels"]) * math.prod(config["patch_size"])),
        ("audio_proj_out", int(config["audio_in_channels"])),
    ):
        shapes[f"{prefix}.weight"] = ("f32", (NUM_STEPS // BLOCK_SIZE, outputs, hidden))
        shapes[f"{prefix}.bias"] = ("f32", (NUM_STEPS // BLOCK_SIZE, outputs))
    return shapes


def _adapter_component(
    present: Rows, component: str, config: dict[str, Any], plan: TimestepPlan, topology: H3Topology
) -> None:
    expected = {
        key: ("bf16", ((NUM_STEPS, *shape[1:]) if dtype == "f32" else shape))
        for key, (dtype, shape) in overlay_shapes(config, plan).items()
        if not key.endswith(".table")
    }
    expected.update(
        {key: ("bf16", shape) for key, (_, shape) in adapter_shapes(topology, RANK).items()}
    )
    found = present[component]
    if set(found) != set(expected) or not all(
        plain(found[key], dtype, shape) for key, (dtype, shape) in expected.items()
    ):
        raise UnsupportedInput(
            f"PDD adapter component {component} must hold exactly its plain bf16 LoRA factors "
            f"and 32 heads ({len(found)} rows, expected {len(expected)})",
            code="h3_turbo_adapter",
        )


def adapter_components(present: Rows) -> dict[Task, str]:
    """Name each task's adapter in one PDD checkpoint: the one component naming the task."""
    found = {task: [name for name in present if task in name.lower()] for task in TASKS}
    if any(len(names) != 1 for names in found.values()) or len(present) != 2:
        raise UnsupportedInput(
            "the PDD adapter checkpoint must hold exactly two components, one naming fl2va and "
            f"one naming ref2va; it holds {sorted(present)}",
            code="h3_turbo_adapter",
        )
    return {task: names[0] for task, names in found.items()}


def _topologies() -> tuple[dict[Task, dict[str, Any]], dict[Task, H3Topology]]:
    sections = _sections()
    configs = {task: sections[DIT_COMPONENT[task]] for task in TASKS}
    return configs, {task: H3Topology.from_config(configs[task]) for task in TASKS}


@invocable
async def turbo_lora(
    ctx: Context,
    *,
    adapters: Weights[Any],
    base: Weights[Any],
    tel: Telemetry,
) -> ModelArtifact:
    """Build the standalone PDD-8 turbo LoRA from the upstream PDD adapter pair.

    ``adapters`` is the converted alibaba-pai/MiniMax-H3-Acc-LoRAs release: one component per
    task. ``base`` is any full-precision H3 checkpoint; only its AdaLN modulation weights are
    read. LoRA factors keep the adapter objects; adapted timestep tables and collapsed heads
    are computed per tensor, so an interrupted run resumes where it stopped.
    """
    configs, topologies = _topologies()
    full, pair = open_source(ctx, base), open_source(ctx, adapters)
    for task in TASKS:
        _validate_generators(rows(full), task, topologies[task])
    names = adapter_components(rows(pair))
    plans = {task: turbo_plan(pair, task, names[task]) for task in TASKS}
    for task in TASKS:
        _adapter_component(rows(pair), names[task], configs[task], plans[task], topologies[task])
    order = [
        (f"{task}_turbo", key) for task in TASKS for key in overlay_shapes(configs[task], plans[task])
    ]
    config = {
        f"{task}_turbo": {
            **configs[task],
            "cozy_h3": {
                "task": task,
                "modulation": "adaln-pruned",
                "distillation": "pdd",
                "table_keys": plans[task].table_keys,
                "lora_rank": RANK,
                "lora_alpha": float(RANK),
                "pdd_num_steps": NUM_STEPS,
                "pdd_block_size": BLOCK_SIZE,
            },
        }
        for task in TASKS
    }
    reused = computed = 0
    with ctx.output("pdd8", Derivation(base=None, sources=(full, pair), order=order)) as out:
        write_config(out, canonical_json.encode(config))
        for task in TASKS:
            ctx.raise_if_cancelled()
            component, adapter = f"{task}_turbo", names[task]
            copy_assets(out, pair, adapter, target=component)
            specs = overlay_shapes(configs[task], plans[task])
            for key in specs:
                if key.endswith((".lora_down", ".lora_up")) and not out.has(component, key):
                    out.copy(component, key, pair, from_component=adapter, from_key=key)

            def reader(
                key: str, dtype: torch.dtype, shape: tuple[int, ...], adapter: str = adapter
            ) -> torch.Tensor:
                return read(pair, adapter, key, dtype, shape)

            def progress(done: int, total: int, task: Task = task) -> None:
                tel.progress(done / total, stage=f"turbo-{task}")

            _, _, kept = write_tables(
                ctx,
                plan=plans[task],
                topology=topologies[task],
                source=full,
                component=DIT_COMPONENT[task],
                target=component,
                outputs=(out,),
                progress=progress,
                adapter=LowRankAdapter(RANK, 1.0, reader),
            )
            tables = len(table_shapes(topologies[task], plans[task]))
            reused, computed = reused + kept, computed + tables - kept
            for prefix, shift in zip(("proj_out", "audio_proj_out"), plans[task].shifts, strict=True):
                ctx.raise_if_cancelled()
                keys = (f"{prefix}.weight", f"{prefix}.bias")
                if all(out.has(component, key) for key in keys):
                    reused += 2
                    continue
                shape = specs[f"{prefix}.weight"][1]
                weight = reader(f"{prefix}.weight", torch.bfloat16, (NUM_STEPS, *shape[1:]))
                bias = reader(f"{prefix}.bias", torch.bfloat16, (NUM_STEPS, shape[1]))
                fused = collapse_head_bank(weight, bias, pdd_head_plan(shift, NUM_STEPS, BLOCK_SIZE))
                for key, value in zip(keys, fused, strict=True):
                    if not out.has(component, key):
                        put(out, component, key, value)
                computed += 2
        artifact = ctx.adopt_model(out)
    tel.metric("h3.turbo.reused_tensors", float(reused))
    tel.metric("h3.turbo.computed_tensors", float(computed))
    return artifact
