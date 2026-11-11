#!/usr/bin/env python
"""Generate MiniMax-H3 model assets from the official Diffusers scheduler.

    python scripts/h3_plans.py --output /path/to/model-assets 30 40 50
    python scripts/h3_plans.py --output /path/to/turbo-assets --turbo 8

The explicit output directory receives <component>/timestep-plan.json, ready to embed in
that component's model header. This script never writes package resources or restamps
conformance expectations. The oracle helpers below are CPU-only proof fixtures; serving
and producer code must read the selected model's assets instead.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minimax-h3"))

import torch  # noqa: E402
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler  # noqa: E402
from minimax_h3.official import Schedule, Task, TimestepPlan, Trunk  # noqa: E402

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
TASKS: tuple[Trunk, ...] = ("fl2va", "ref2va")
BASE_STEPS = (30, 40, 50)
TURBO_STEPS = 8


def official_schedule(steps: int) -> Schedule:
    if steps < 1:
        raise ValueError("a schedule needs at least one transformer evaluation")
    sigmas = []
    # Proof callers sometimes construct models under torch.device('meta'). The oracle
    # always computes real float32 CPU values, independent of that construction context.
    with torch.device("cpu"):
        for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(steps + 1)
            grid = cast(torch.Tensor, scheduler.sigmas)  # set by set_timesteps
            sigmas.append(tuple(float(value) for value in grid.float()))
    schedule = Schedule(steps + 1, sigmas[0], sigmas[1])
    if schedule.transformer_evaluations != steps:
        raise ValueError(f"{steps} steps collide on the float32 grid; pick another count")
    return schedule


@lru_cache(maxsize=4)
def oracle_plan(task: Task) -> TimestepPlan:
    """Derive a historical release fixture; never a production asset fallback."""
    trunk = task.removesuffix("_turbo")
    if trunk not in TASKS:
        raise ValueError(f"unknown H3 task {task!r}")
    steps = (TURBO_STEPS,) if task.endswith("_turbo") else BASE_STEPS
    return TimestepPlan(
        trunk, VIDEO_SHIFT, AUDIO_SHIFT,
        tuple(official_schedule(count) for count in steps),
    )


def oracle_plans(*, turbo: bool = False) -> dict[Trunk, TimestepPlan]:
    return {
        trunk: oracle_plan(cast(Task, f"{trunk}_turbo" if turbo else trunk))
        for trunk in TASKS
    }


def main(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("steps", type=int, nargs="+")
    args = parser.parse_args(arguments)
    steps = sorted(set(args.steps))
    if args.turbo and steps != [TURBO_STEPS]:
        parser.error("the supported PDD overlay requires its fixed eight-evaluation schedule")
    schedules = tuple(official_schedule(count) for count in steps)
    for task, base_component in zip(TASKS, ("transformer", "transformer_ref"), strict=True):
        plan = TimestepPlan(task, VIDEO_SHIFT, AUDIO_SHIFT, schedules)
        component = f"{task}_turbo" if args.turbo else base_component
        destination = args.output / component / "timestep-plan.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(plan.canonical_bytes())
        final_rows, block_rows = plan.table_layout()
        print(
            f"{destination}: sha256:{plan.digest} steps={plan.steps} "
            f"final_rows={len(final_rows)} block_rows={len(block_rows)}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
