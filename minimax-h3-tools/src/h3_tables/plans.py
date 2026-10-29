"""Read the canonical MiniMax-H3 timestep-plan handoff.

The plan, not a step-count label, is the numerical contract. Its exact bytes bind every
schedule's sigmas and forward evaluations, every row class, and the one union table order
the producer emits so a single pruned checkpoint serves every listed step count.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

from cozy_runtime.author import canonical_json

Task = Literal["fl2va", "ref2va"]
TASKS: tuple[Task, ...] = ("fl2va", "ref2va")

_TOP_LEVEL = {
    "adaln_row_index",
    "audio_shift",
    "table_keys",
    "table_order",
    "final_norm_row_index",
    "fps",
    "row_timestep_reduction",
    "scalar_encoding",
    "scheduler_semantics",
    "schedules",
    "task",
    "video_shift",
}
_SCHEDULE = {"transformer_evaluations", "sigma_grid_points", "evaluations", "terminal"}
#: A refine plan (`"kind": "refine"`) re-noises a finished latent to its schedule's first video sigma
#: and denoises video alone: every evaluation's audio sigma is zero, its rows clean conditioning.
REFINE = "refine"


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    task: Task
    digest: str
    canonical_bytes: bytes
    timesteps: tuple[float, ...]
    block_rows: tuple[tuple[int, int], ...]
    steps: tuple[int, ...]
    #: Each schedule's first video sigma; a refine plan's schedules differ by it.
    starts: tuple[float, ...] = ()
    refine: bool = False

    @property
    def table_keys(self) -> dict[str, Any]:
        return cast(dict[str, Any], canonical_json.decode(self.canonical_bytes)["table_keys"])

    @property
    def shifts(self) -> tuple[float, float]:
        """The video and audio scheduler shifts the plan's sigmas were built with."""
        document = canonical_json.decode(self.canonical_bytes)
        return float.fromhex(document["video_shift"]), float.fromhex(document["audio_shift"])


def _f32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def _hex_f32(value: object, field: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field} is not an IEEE-754 hexadecimal string")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a hexadecimal float") from exc
    if not math.isfinite(parsed) or parsed != _f32(parsed) or parsed.hex() != value:
        raise ValueError(f"{field} is not one exact canonical float32 value")
    return parsed


def _closed(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{field} has fields {actual}, expected {sorted(keys)}")
    return cast(dict[str, Any], value)


def _canonical(document: dict[str, Any]) -> bytes:
    return canonical_json.encode(document)


_EXPECTED_CLASSES = (
    ("target_video", "video", 0, "always"),
    ("text", "text", 1, "always"),
    ("target_audio", "audio", 2, "always"),
    ("condition_video", "video", 0, "if_condition_video_rows"),
    ("condition_audio", "audio", 2, "if_condition_audio_rows"),
)


def _parse_schedule(
    value: object, field: str, class_rows: list[dict[str, Any]], *, refine: bool = False
) -> tuple[int, float]:
    """Validate one schedule, append its row classes in order, return its step count and first
    video sigma."""
    schedule = _closed(value, _SCHEDULE, field)
    grid_points = schedule["sigma_grid_points"]
    forwards = schedule["transformer_evaluations"]
    evaluations = schedule["evaluations"]
    if (
        not isinstance(grid_points, int)
        or not isinstance(forwards, int)
        or forwards < 1
        or forwards + 1 > grid_points
        or not isinstance(evaluations, list)
        or len(evaluations) != forwards
    ):
        raise ValueError(f"{field} needs N indexed evaluations under at least N+1 grid points")
    terminal = _closed(
        schedule["terminal"],
        {"video_sigma", "audio_sigma", "transformer_evaluation"},
        f"{field}.terminal",
    )
    if (
        _hex_f32(terminal["video_sigma"], f"{field}.terminal.video_sigma") != 0.0
        or _hex_f32(terminal["audio_sigma"], f"{field}.terminal.audio_sigma") != 0.0
        or terminal["transformer_evaluation"] is not False
    ):
        raise ValueError("terminal sigma zero is not a transformer evaluation")
    video_sigmas: list[float] = []
    audio_sigmas: list[float] = []
    clean_video = _f32(0.999)
    for index, value in enumerate(evaluations):
        name = f"{field}.evaluations[{index}]"
        row = _closed(value, {"index", "video_sigma", "audio_sigma", "modulation_classes"}, name)
        if row["index"] != index:
            raise ValueError("evaluation indexes are not contiguous from zero")
        video_sigma = _hex_f32(row["video_sigma"], f"{name}.video_sigma")
        audio_sigma = _hex_f32(row["audio_sigma"], f"{name}.audio_sigma")
        video_sigmas.append(video_sigma)
        audio_sigmas.append(audio_sigma)
        classes = row["modulation_classes"]
        if not isinstance(classes, list) or len(classes) != len(_EXPECTED_CLASSES):
            raise ValueError(f"{name} does not declare the five H3 row classes")
        wanted_timesteps = (
            _f32(1.0 - video_sigma),
            _f32(1.0 - video_sigma),
            _f32(1.0 - audio_sigma),
            max(_f32(1.0 - video_sigma), clean_video),
            _f32(1.0),
        )
        for position, (item_value, expected, timestep) in enumerate(
            zip(classes, _EXPECTED_CLASSES, wanted_timesteps, strict=True)
        ):
            item = _closed(
                item_value,
                {"name", "modality", "modality_tag", "presence", "timestep"},
                f"{name}.modulation_classes[{position}]",
            )
            if (
                tuple(item[key] for key in ("name", "modality", "modality_tag", "presence"))
                != expected
            ):
                raise ValueError(f"{name} row class {position} changed meaning")
            if _hex_f32(item["timestep"], f"{name} timestep") != timestep:
                raise ValueError(f"{name} row class {position} has the wrong timestep")
            class_rows.append(item)
    if refine:
        if not 0.0 < video_sigmas[0] <= 1.0 or any(a <= b or b <= 0.0 for a, b in pairwise(video_sigmas)):
            raise ValueError(f"{field} video sigmas are not strictly descending within (0, 1]")
        if any(sigma != 0.0 for sigma in audio_sigmas):
            raise ValueError(f"{field} refines video alone: every audio sigma is zero")
        return forwards, video_sigmas[0]
    for modality, sigmas in (("video", video_sigmas), ("audio", audio_sigmas)):
        if sigmas[0] != 1.0 or any(a <= b or b <= 0.0 for a, b in pairwise(sigmas)):
            raise ValueError(
                f"{field} {modality} sigmas are not strictly descending from one toward zero"
            )
    return forwards, video_sigmas[0]


def parse_plan(raw: bytes, *, task: str) -> TimestepPlan:
    """Validate one complete plan and return its exact table selectors. The model that carries
    the plan is its authority: any exact schedule set passes."""
    if task not in TASKS:
        raise ValueError(f"unknown MiniMax-H3 task {task!r}")
    try:
        decoded = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError("timestep plan is not JSON") from exc
    refine = isinstance(decoded, dict) and "kind" in decoded
    document = _closed(decoded, _TOP_LEVEL | ({"kind"} if refine else set()), "TimestepPlan")
    if refine and document["kind"] != REFINE:
        raise ValueError(f"timestep plan kind {document['kind']!r} is not {REFINE!r}")
    canonical = _canonical(document)
    typed_task: Task = task
    if document["task"] != task:
        raise ValueError(f"timestep plan does not declare task {task!r}")
    fixed = {
        "scalar_encoding": "ieee754-binary32-hex",
        "scheduler_semantics": "minimax-h3-data-ward-rf-euler/1",
        "row_timestep_reduction": "unique-sorted-return-inverse",
        "adaln_row_index": "timestep_index*3+modality_tag",
        "final_norm_row_index": "timestep_index",
        "table_order": "first-distinct-evaluation-class-occurrence",
        "fps": 24,
    }
    for name, expected in fixed.items():
        if document[name] != expected:
            raise ValueError(f"timestep plan {name} is {document[name]!r}, expected {expected!r}")
    _hex_f32(document["video_shift"], "video_shift")
    _hex_f32(document["audio_shift"], "audio_shift")

    schedules = document["schedules"]
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("the plan must list at least one schedule")
    class_rows: list[dict[str, Any]] = []
    parsed = [
        _parse_schedule(value, f"schedules[{index}]", class_rows, refine=refine)
        for index, value in enumerate(schedules)
    ]
    steps, starts = tuple(n for n, _ in parsed), tuple(start for _, start in parsed)
    if refine and any(left >= right for left, right in pairwise(starts)):
        raise ValueError("refine schedules must be listed by strictly ascending start sigma")
    if not refine and any(left >= right for left, right in pairwise(steps)):
        raise ValueError("schedules must be listed by strictly ascending step count")

    keys = _closed(
        document["table_keys"],
        {"block_modulation", "final_normalization"},
        "table_keys",
    )
    block_expected: list[dict[str, Any]] = []
    final_expected: list[dict[str, Any]] = []
    seen_blocks: set[tuple[str, int]] = set()
    seen_final: set[str] = set()
    for item in class_rows:
        pair = (str(item["timestep"]), int(item["modality_tag"]))
        if pair not in seen_blocks:
            seen_blocks.add(pair)
            block_expected.append(
                {
                    "index": len(block_expected),
                    "timestep": pair[0],
                    "modality": item["modality"],
                    "modality_tag": pair[1],
                }
            )
        if pair[0] not in seen_final:
            seen_final.add(pair[0])
            final_expected.append({"index": len(final_expected), "timestep": pair[0]})
    if keys["block_modulation"] != block_expected or keys["final_normalization"] != final_expected:
        raise ValueError("table keys are not the first-distinct exact evaluation coverage")

    timesteps = tuple(
        _hex_f32(row["timestep"], "final_normalization.timestep") for row in final_expected
    )
    timestep_index = {value: index for index, value in enumerate(timesteps)}
    block_rows = tuple(
        (
            timestep_index[_hex_f32(row["timestep"], "block_modulation.timestep")],
            row["modality_tag"],
        )
        for row in block_expected
    )
    digest = canonical_json.digest(document)
    return TimestepPlan(typed_task, digest, canonical, timesteps, block_rows, steps, starts, refine)


def refine_plan(task: Task, schedules: list[tuple[float, ...]], *, shift: float) -> TimestepPlan:
    """The canonical refine plan of `schedules` (each its video sigmas, terminal zero excluded),
    as minimax-h3's `TimestepPlan(kind="refine")` writes it, parsed back."""

    def hexf(value: float) -> str:
        return _f32(value).hex()

    def modulation(name: str, timestep: float, modality: str, tag: int, presence: str) -> dict[str, Any]:
        return {"name": name, "timestep": hexf(timestep), "modality": modality,
                "modality_tag": tag, "presence": presence}  # fmt: skip

    document_schedules, blocks, finals = [], [], []
    seen_blocks: set[tuple[str, int]] = set()
    seen_final: set[str] = set()
    for sigmas in schedules:
        evaluations = []
        for index, sigma in enumerate(sigmas):
            video = _f32(1.0 - _f32(sigma))
            classes = [
                modulation("target_video", video, "video", 0, "always"),
                modulation("text", video, "text", 1, "always"),
                modulation("target_audio", 1.0, "audio", 2, "always"),
                modulation("condition_video", max(video, _f32(0.999)), "video", 0, "if_condition_video_rows"),
                modulation("condition_audio", 1.0, "audio", 2, "if_condition_audio_rows"),
            ]
            evaluations.append({"index": index, "video_sigma": hexf(sigma), "audio_sigma": hexf(0.0),
                                "modulation_classes": classes})  # fmt: skip
            for item in classes:
                pair = (item["timestep"], item["modality_tag"])
                if pair not in seen_blocks:
                    seen_blocks.add(pair)
                    blocks.append({"index": len(blocks), "timestep": pair[0],
                                   "modality": item["modality"], "modality_tag": pair[1]})  # fmt: skip
                if pair[0] not in seen_final:
                    seen_final.add(pair[0])
                    finals.append({"index": len(finals), "timestep": pair[0]})
        document_schedules.append({
            "transformer_evaluations": len(sigmas), "sigma_grid_points": len(sigmas) + 1,
            "evaluations": evaluations,
            "terminal": {"video_sigma": hexf(0.0), "audio_sigma": hexf(0.0), "transformer_evaluation": False},
        })  # fmt: skip
    document = {
        "task": task, "kind": REFINE, "scalar_encoding": "ieee754-binary32-hex",
        "scheduler_semantics": "minimax-h3-data-ward-rf-euler/1",
        "row_timestep_reduction": "unique-sorted-return-inverse",
        "adaln_row_index": "timestep_index*3+modality_tag", "final_norm_row_index": "timestep_index",
        "table_order": "first-distinct-evaluation-class-occurrence", "fps": 24,
        "video_shift": hexf(shift), "audio_shift": hexf(1.0), "schedules": document_schedules,
        "table_keys": {"block_modulation": blocks, "final_normalization": finals},
    }  # fmt: skip
    return parse_plan(_canonical(document), task=task)


def parse_declared_plan(raw: bytes) -> TimestepPlan:
    """Parse one plan whose own closed task field selects the strict task validator."""
    try:
        value = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError("timestep plan is not JSON") from exc
    if not isinstance(value, dict) or value.get("task") not in TASKS:
        raise ValueError("timestep plan does not declare fl2va or ref2va")
    return parse_plan(raw, task=str(value["task"]))
