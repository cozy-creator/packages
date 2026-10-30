"""The official Diffusers 0.40 MiniMax-H3 graph, staged by weighted component.

This module contains no media decoder and no checkpoint loader. Runtime reads the reference
handles into immutable values and fills the component roots constructed here. Diffusers owns every model
operation: presentation, conditioning, layout, schedules, solver, and decode. Artifact config
selects the official FULL modulation modules or AdaLN-pruned table replacements before fill.
"""

from __future__ import annotations

import contextvars
import hashlib
import math
import random
import struct
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import CancelledError, Future
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import msgspec
import numpy as np
import torch
from diffusers import (
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
)
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.utils.torch_utils import randn_tensor
from torch.nn.utils.weight_norm import WeightNorm

from cozy_runtime import vlm
from cozy_runtime.author import (
    AttentionLayout,
    Config,
    ConformanceError,
    DecodedAudio,
    DecodedAudioChunk,
    DecodedVideoFrame,
    DerivedCache,
    Device,
    Image,
    InvalidRequest,
    ModelAssets,
    ThreadPoolExecutor,
    attention_scope,
    canonical_json,
    expect_attention,
    spread,
)
from h3_tables.table_layout import TableLayout

from .adaln_pruned import AdaLNPrunedMiniMaxH3Transformer, step_invariants
from .conditioner import FinalHiddenState, PresentationLength, build_text_conditioner
from .continuation import (
    AVContext,
    ContextReferenceLayout,
    ContinuationPlan,
    DeliveredFrames,
    align_context_rows,
    export_context,
)
from .guide_rows import ANCHOR_LATENTS, ANCHORS, GUIDE, GUIDE_LATENTS, GuideClip, anchor_rows, guide_rows
from .reference_rows import ClipRows, motion_size, public_video
from .steps import denoise_fl2va
from .overlay import ATTENTION_KWARG, REGENERATE_BANK, TURBO_BANK, Overlay, TurboSchedule
from .vae_tiles import TileBatchedVideoVAE, encode_condition

#: A trunk is one DiT and its conditioning workflow; a task is one served function — the
#: trunk's official plan, or the trunk under the turbo bank's fixed 8-evaluation plan.
Trunk = Literal["fl2va", "ref2va"]
Task = Literal["fl2va", "ref2va", "fl2va_turbo", "ref2va_turbo"]
_TRUNKS: tuple[Trunk, ...] = ("fl2va", "ref2va")
_TRUNK: dict[Task, Trunk] = {
    "fl2va": "fl2va",
    "ref2va": "ref2va",
    "fl2va_turbo": "fl2va",
    "ref2va_turbo": "ref2va",
}
_BANK: dict[Task, str | None] = {
    "fl2va": None,
    "ref2va": None,
    "fl2va_turbo": TURBO_BANK,
    "ref2va_turbo": TURBO_BANK,
}
#: Six workflows over four tasks. `t2va` is the keyframe-free payload shape of `fl2va`, and
#: `t2va_turbo` is the same shape of `fl2va_turbo`; each turbo workflow runs the official
#: block graph of its base name with the trunk DiT armed under the turbo bank.
_WORKFLOW_TASKS: dict[str, Task] = {
    "t2va": "fl2va",
    "fl2va": "fl2va",
    "ref2va": "ref2va",
    "t2va_turbo": "fl2va_turbo",
    "fl2va_turbo": "fl2va_turbo",
    "ref2va_turbo": "ref2va_turbo",
}

FPS = 24
#: The video VAE's temporal chunking: `clip_length` pixel frames per chunk keeping
#: `tokens_chunk_size` latents, so only `17n + 5` frame counts decode. Refused at
#: construction if the artifact config drifts (`_validate_model_contract`).
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
#: The clip envelope the upstream pipeline declares, in seconds. A DEPENDENCY fact we read
#: and pin, not a serving policy we choose: `_validate_model_contract` refuses if the
#: installed library moves either bound. What this release serves is the frame envelope
#: below, because frames are the real constraint.
MIN_DURATION = 5.0
MAX_DURATION = 15.0
#: Packed-sequence geometry: the VAE's spatial compression, the DiT's patch, and the
#: stereo audio latents' channel-major packing. All three are contract-checked.
VAE_SPATIAL_RATIO = 16
PATCH_SIZE = (1, 2, 2)
AUDIO_CHANNELS = 2
MAX_IMAGE_REFERENCES = 9
MAX_VIDEO_REFERENCES = 3
MAX_AUDIO_REFERENCES = 3
MAX_REFERENCES = 12
MAX_CONDITIONER_VISION_TOKENS = 32768
REFERENCE_IMAGE_SHORT_EDGE = 2048
_WEIGHTED_CONFIG_SECTIONS = {"audio_vae", "text_encoder", "transformer", "transformer_ref", "vae"}
#: The independent turbo LoRA checkpoint always carries both overlay components.
_OVERLAY_CONFIG_SECTIONS = {"fl2va_turbo", "ref2va_turbo"}
_DIT_COMPONENT: dict[Trunk, str] = {"fl2va": "transformer", "ref2va": "transformer_ref"}
_OVERLAY_COMPONENT: dict[Trunk, str] = {"fl2va": "fl2va_turbo", "ref2va": "ref2va_turbo"}
#: The Regenerate refine overlay: one component, over the FL2VA DiT (NVlabs' refine trunk).
REGENERATE_COMPONENT = "fl2va_regenerate"
_REGENERATE_RANK = 576
_DISTILLATION = "pdd"
#: The timesteps and sigmas a component's modulation tables hold rows for, in its folder.
PLAN_ASSET = "timestep-plan.json"
#: The first release of each model whose components carry their plans.
PLANS_SINCE = {False: "minimaxai/minimax-h3@1.0.0-rc.4", True: "alibaba-pai/minimax-h3-turbo-lora@1.0.1"}
_ASSETS = Path(__file__).resolve().parent
# NVIDIA's H3 sparse policy protects conditioning and the initial sensitive work.
# This supplies semantics only; Runtime chooses the kernels.
_DENSE_ATTENTION_PATHS = ("token_refiner", "transformer_blocks.0", "transformer_blocks.1")
#: Layers whose dense rows take Runtime's exact kernel (Kitchen INT8 where the route has it, else
#: BF16) when the dense kernel quantizes coarser, beside the ones Runtime finds by their key RMSNorm
#: gain (blocks 45, 48 and 49): the text refiner, and block 1, whose key outliers are in its
#: activations (FA4 FP8 on B200 0.10-0.12 rel L2; Sage3 FP4 on an RTX 5090 0.23 at step 0, INT8 0.015).
_EXACT_ATTENTION_PATHS = ("token_refiner", "transformer_blocks.1")


def _validate_sol_dense_steps(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("sol_dense_steps must be a nonnegative integer")


def _attention_layout(state: Any, step: int, *, sol_dense_steps: int = 10) -> AttentionLayout:
    fields = state.denoiser_input_fields
    # NVIDIA's H3 policy protects every row before target video, including target
    # audio. Diffusers packs text, reference media, target audio, then target video.
    # Shape/count metadata is host-resident; do not read GPU values.
    return AttentionLayout(
        live_tokens=fields["token_tags"].numel(),
        protected_prefix=(
            fields["text_indices"].numel()
            + state.num_condition_video_rows
            + fields["audio_indices"].numel()
        ),
        step=step,
        dense_until_step=sol_dense_steps,
        dense_paths=_DENSE_ATTENTION_PATHS,
        exact_paths=_EXACT_ATTENTION_PATHS,
    )


def frames_for(duration_s: int) -> int:
    """Whole seconds onto the video VAE's own `17n + 5` grid, by the official snap.

    The snap only ever goes UP, so a clip is never shorter than the seconds asked for; the
    exact delivered length is `frames_for(seconds) / FPS`.
    """
    return int(align_num_frames(duration_s * FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK))


#: The envelope this release serves, on the video VAE's own `17n + 5` grid. It is stated in
#: FRAMES because frames are what constrains: the grid decides what decodes, and the clock
#: only reads it. Each bound is a declared second carried onto the grid by the same upward
#: snap a request gets, so the ceiling is the top grid point of a 15-second model rather than
#: a clock reading that a legal grid point overshoots. The grid has no point at
#: `15.0 * 24 = 360`, so a ceiling held in seconds makes a 15-second model structurally
#: incapable of a 15-second clip — the defect se-053 removes.
MIN_FRAMES = frames_for(math.ceil(MIN_DURATION))
MAX_FRAMES = frames_for(math.floor(MAX_DURATION))
#: The frame envelope restated in the units the upstream blocks compare in. Both official
#: layout blocks compute `aligned_frames / fps` and refuse it outside
#: `[min_duration, max_duration]`, so the ceiling has to reach the top grid point in seconds
#: or 362 frames refuses upstream. Upstream's own floor already tolerates exactly this snap
#: (124 frames = 5.167 s passes a 5.0 s floor); only its ceiling does not, and that asymmetry
#: is an oversight rather than a capability bound — v1's ie#658 shipped, billed and
#: pixel-checked the 362-frame cell, and nothing mechanical stands behind the bare
#: `return 15.0`: RoPE is per request and both VAEs are chunked convolutions. Handed to the
#: blocks through `_ScopedPipeline`, so the override is per request, visible, and revertible
#: the day the library states its ceiling in frames.
_CEILING_S = MAX_FRAMES / FPS


def supported_durations() -> tuple[int, ...]:
    """The whole seconds this release serves: those whose snapped frame count lands inside
    the frame envelope. 4 s snaps to 107 frames (below `MIN_FRAMES`) and 16 s to 396 (above
    `MAX_FRAMES`), so the set is 5..15 s = 124..362 frames."""
    return tuple(
        seconds
        for seconds in range(1, 1 + math.floor(MAX_DURATION))
        if MIN_FRAMES <= frames_for(seconds) <= MAX_FRAMES
    )


def assert_duration_envelope(served: tuple[int, int]) -> None:
    """Refuse unless the official geometry serves exactly the whole seconds on the wire.

    The wire states its own bounds, because `describe` reads this package's source and
    never runs it (#713). This is what holds those two numbers to the 17n+5 snap and the
    official envelope, so the pair cannot drift from the geometry it advertises.
    """
    durations = supported_durations()
    if (durations[0], durations[-1]) != served or durations != tuple(
        range(served[0], served[1] + 1)
    ):
        raise ConformanceError(
            f"the official MiniMax-H3 geometry serves {durations} whole seconds, not the "
            f"contiguous {served[0]}..{served[1]} this release admits",
            code="artifact_config",
        )


def denoise_rows(frames: int, height: int, width: int) -> int:
    """Rows of the one packed sequence the DiT attends over — what attention is quadratic
    in, and therefore what a length choice actually buys."""
    latent_frames = video_latent_num_frames(frames, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK)
    patch_t, patch_h, patch_w = PATCH_SIZE
    video = (
        latent_frames
        // patch_t
        * (height // VAE_SPATIAL_RATIO // patch_h)
        * (width // VAE_SPATIAL_RATIO // patch_w)
    )
    return int(video + audio_latent_num_frames(frames) * AUDIO_CHANNELS)


@dataclass(frozen=True, slots=True)
class Schedule:
    """One official sigma grid per modality: `MiniMaxH3Scheduler.set_timesteps(points)`.

    `linspace(1, 0, sigma_grid_points)` through each modality's exponential shift, float32
    collisions collapsed; the terminal zero is a grid point with no transformer evaluation.
    A refine schedule (`refine`) starts its video at its first sigma, below one, and holds the
    audio clean: every audio sigma is zero.
    """

    sigma_grid_points: int
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]
    refine: bool = False

    def __post_init__(self) -> None:
        points = len(self.video_sigmas)
        if points < 2 or len(self.audio_sigmas) != points or points > self.sigma_grid_points:
            raise ValueError(
                "a MiniMax-H3 schedule holds equally many video and audio sigmas, "
                "at most one per grid point"
            )
        descending = (self.video_sigmas,) if self.refine else (self.video_sigmas, self.audio_sigmas)
        for sigmas in descending:
            if (
                not 0.0 < sigmas[0] <= 1.0
                or (sigmas[0] != 1.0 and not self.refine)
                or sigmas[-1] != 0.0
                or any(not math.isfinite(value) or value != _as_float32(value) for value in sigmas)
                or any(left <= right for left, right in pairwise(sigmas))
            ):
                raise ValueError(
                    "MiniMax-H3 sigmas must be exact float32 values descending to zero, "
                    "from one unless the schedule refines"
                )
        if self.refine and any(value != 0.0 for value in self.audio_sigmas):
            raise ValueError("a refine schedule holds its audio clean: every audio sigma is zero")

    @property
    def start(self) -> float:
        """The video's first sigma: one, or where a refine re-noises to."""
        return self.video_sigmas[0]

    @property
    def transformer_evaluations(self) -> int:
        return len(self.video_sigmas) - 1

    @property
    def video_timesteps(self) -> tuple[float, ...]:
        return tuple(_as_float32(1.0 - sigma) for sigma in self.video_sigmas[:-1])

    @property
    def audio_timesteps(self) -> tuple[float, ...]:
        return tuple(_as_float32(1.0 - sigma) for sigma in self.audio_sigmas[:-1])


@dataclass(frozen=True, slots=True)
class ScheduleFacts:
    """The receipt of the one schedule an attempt executed, under its plan identity."""

    timestep_plan_digest: str
    transformer_evaluations: int
    sigma_grid_points: int
    video_sigma_digest: str
    audio_sigma_digest: str
    video_timestep_digest: str
    audio_timestep_digest: str

    @classmethod
    def of(cls, plan: TimestepPlan, schedule: Schedule) -> ScheduleFacts:
        return cls(
            timestep_plan_digest=plan.digest,
            transformer_evaluations=schedule.transformer_evaluations,
            sigma_grid_points=schedule.sigma_grid_points,
            video_sigma_digest=_float32_digest(schedule.video_sigmas),
            audio_sigma_digest=_float32_digest(schedule.audio_sigmas),
            video_timestep_digest=_float32_digest(schedule.video_timesteps),
            audio_timestep_digest=_float32_digest(schedule.audio_timesteps),
        )


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    """The exact official two-modality schedules one task serves; no weights required.

    An AdaLN-pruned checkpoint carries modulation rows for the union of these schedules,
    so the plan identity names both the step choices a request may make and the table
    layout the checkpoint must contain.
    """

    task: Trunk
    video_shift: float
    audio_shift: float
    schedules: tuple[Schedule, ...]
    #: A refine plan (the Regenerate overlay's): its schedules re-noise a finished latent to
    #: their start and denoise video alone, listed by strictly ascending start.
    refine: bool = False

    def __post_init__(self) -> None:
        if self.task not in _TRUNKS:
            raise ValueError(f"unknown MiniMax-H3 task {self.task!r}")
        if any(
            not math.isfinite(shift) or shift <= 0 or shift != _as_float32(shift)
            for shift in (self.video_shift, self.audio_shift)
        ):
            raise ValueError("MiniMax-H3 shifts must be finite positive float32 values")
        if not self.schedules or any(s.refine != self.refine for s in self.schedules):
            raise ValueError("a MiniMax-H3 plan's schedules are all of its kind")
        if self.refine and any(
            left.start >= right.start for left, right in pairwise(self.schedules)
        ):
            raise ValueError("a refine plan lists its schedules by strictly ascending start")
        if not self.refine and any(
            left.transformer_evaluations >= right.transformer_evaluations
            for left, right in pairwise(self.schedules)
        ):
            raise ValueError(
                "a MiniMax-H3 plan lists its schedules by strictly ascending evaluation count"
            )

    @property
    def strengths(self) -> tuple[float, ...]:
        """A refine plan's starts as a request names them, to three decimals."""
        return tuple(round(schedule.start, 3) for schedule in self.schedules)

    def refined(self, strength: float) -> Schedule:
        """The refine schedule starting at `strength` (to three decimals)."""
        for schedule in self.schedules:
            if round(schedule.start, 3) == round(strength, 3):
                return schedule
        raise InvalidRequest(
            f"regenerate serves strength {', '.join(map(str, self.strengths))}, not {strength}",
            code="strength",
            fields=["upscale", "strength"],
        )

    @property
    def steps(self) -> tuple[int, ...]:
        """The denoise step counts (transformer evaluations) this plan serves."""
        return tuple(schedule.transformer_evaluations for schedule in self.schedules)

    def schedule(self, steps: int | None = None) -> Schedule:
        """The schedule of `steps` evaluations; absent, the plan's fewest."""
        for schedule in self.schedules:
            if steps is None or schedule.transformer_evaluations == steps:
                return schedule
        raise InvalidRequest(
            f"this release serves {', '.join(map(str, self.steps))} denoise steps, not {steps}",
            code="steps",
            fields=["steps"],
        )

    def executed(
        self, video_timesteps: Sequence[float], audio_timesteps: Sequence[float]
    ) -> Schedule:
        """The one schedule whose exact timesteps the official pipeline just built."""
        for schedule in self.schedules:
            if (
                tuple(video_timesteps) == schedule.video_timesteps
                and tuple(audio_timesteps) == schedule.audio_timesteps
            ):
                return schedule
        raise ConformanceError(
            f"the official {self.task} schedule differs from every plan schedule",
            code="artifact_config",
        )

    @property
    def digest(self) -> str:
        return timestep_plan_digest(self.canonical_bytes())

    def canonical_bytes(self) -> bytes:
        """Canonical producer handoff; float32 values are exact hex strings."""
        return canonical_json.encode(self._document())

    def table_layout(self) -> tuple[tuple[float, ...], tuple[tuple[int, int], ...]]:
        """The table order named by these exact canonical plan bytes."""
        keys = self._document()["table_keys"]
        final_rows = keys["final_normalization"]
        timesteps = tuple(float.fromhex(row["timestep"]) for row in final_rows)
        timestep_rows = {value: index for index, value in enumerate(timesteps)}
        block_rows = tuple(
            (timestep_rows[float.fromhex(row["timestep"])], int(row["modality_tag"]))
            for row in keys["block_modulation"]
        )
        return timesteps, block_rows

    def _document(self) -> dict[str, Any]:
        clean_video = _as_float32(0.999)
        condition_audio = _as_float32(1.0)
        schedules: list[dict[str, Any]] = []
        block_keys: list[dict[str, Any]] = []
        final_keys: list[dict[str, Any]] = []
        seen_blocks: set[tuple[str, int]] = set()
        seen_final: set[str] = set()
        for schedule in self.schedules:
            evaluations: list[dict[str, Any]] = []
            for index, (video_timestep, audio_timestep) in enumerate(
                zip(schedule.video_timesteps, schedule.audio_timesteps, strict=True)
            ):
                classes = [
                    _modulation_class("target_video", video_timestep, "video", 0, "always"),
                    _modulation_class("text", video_timestep, "text", 1, "always"),
                    _modulation_class("target_audio", audio_timestep, "audio", 2, "always"),
                    _modulation_class(
                        "condition_video",
                        max(video_timestep, clean_video),
                        "video",
                        0,
                        "if_condition_video_rows",
                    ),
                    _modulation_class(
                        "condition_audio",
                        condition_audio,
                        "audio",
                        2,
                        "if_condition_audio_rows",
                    ),
                ]
                evaluations.append(
                    {
                        "index": index,
                        "video_sigma": _float_hex(schedule.video_sigmas[index]),
                        "audio_sigma": _float_hex(schedule.audio_sigmas[index]),
                        "modulation_classes": classes,
                    }
                )
                for modulation in classes:
                    timestep = modulation["timestep"]
                    block_key = (timestep, modulation["modality_tag"])
                    if block_key not in seen_blocks:
                        seen_blocks.add(block_key)
                        block_keys.append(
                            {
                                "index": len(block_keys),
                                "timestep": timestep,
                                "modality": modulation["modality"],
                                "modality_tag": modulation["modality_tag"],
                            }
                        )
                    if timestep not in seen_final:
                        seen_final.add(timestep)
                        final_keys.append({"index": len(final_keys), "timestep": timestep})
            schedules.append(
                {
                    "transformer_evaluations": schedule.transformer_evaluations,
                    "sigma_grid_points": schedule.sigma_grid_points,
                    "evaluations": evaluations,
                    "terminal": {
                        "video_sigma": _float_hex(schedule.video_sigmas[-1]),
                        "audio_sigma": _float_hex(schedule.audio_sigmas[-1]),
                        "transformer_evaluation": False,
                    },
                }
            )
        return {
            "task": self.task,
            **({"kind": "refine"} if self.refine else {}),
            "scalar_encoding": "ieee754-binary32-hex",
            "scheduler_semantics": "minimax-h3-data-ward-rf-euler/1",
            "row_timestep_reduction": "unique-sorted-return-inverse",
            "adaln_row_index": "timestep_index*3+modality_tag",
            "final_norm_row_index": "timestep_index",
            "table_order": "first-distinct-evaluation-class-occurrence",
            # A plan holds one row per (timestep, modality) and no row depends on the frame
            # count, so it carries no frame stamp: se-047 kept one and it made the release's
            # longest clip a digest input, which is why raising the ceiling to 362 costs a
            # retable at all. Removed here so a length can never bind a checkpoint again.
            "fps": FPS,
            "video_shift": _float_hex(self.video_shift),
            "audio_shift": _float_hex(self.audio_shift),
            "schedules": schedules,
            "table_keys": {
                "block_modulation": block_keys,
                "final_normalization": final_keys,
            },
        }


def timestep_plan_digest(raw: bytes) -> str:
    """Formatting-invariant identity for one parsed timestep-plan document."""
    document = canonical_json.decode(raw)
    if not isinstance(document, dict):
        raise ValueError("a timestep plan must be a JSON object")
    return canonical_json.digest(document).removeprefix("sha256:")


class ReferencePolicyFacts(msgspec.Struct, frozen=True):
    """Preflight facts: the one reference-cardinality record both modules share."""

    images: int
    videos: int
    audios: int
    total: int


def validate_reference_policy(kinds: Sequence[str]) -> ReferencePolicyFacts:
    """Validate the official ordered-reference cardinality policy before hydration."""
    unknown = [kind for kind in kinds if kind not in {"image", "video", "audio"}]
    if unknown:
        raise ValueError(f"unknown MiniMax-H3 reference kind {unknown[0]!r}")
    facts = ReferencePolicyFacts(
        images=kinds.count("image"),
        videos=kinds.count("video"),
        audios=kinds.count("audio"),
        total=len(kinds),
    )
    for name, value, limit in (
        ("image", facts.images, MAX_IMAGE_REFERENCES),
        ("video", facts.videos, MAX_VIDEO_REFERENCES),
        ("audio", facts.audios, MAX_AUDIO_REFERENCES),
    ):
        if value > limit:
            raise ValueError(f"MiniMax-H3 accepts at most {limit} {name} references, got {value}")
    if not 1 <= facts.total <= MAX_REFERENCES:
        raise ValueError(f"MiniMax-H3 needs 1..{MAX_REFERENCES} references, got {facts.total}")
    if facts.audios == facts.total:
        raise ValueError(
            "an audio reference must be paired with at least one image or video reference"
        )
    return facts


def timestep_plan(raw: bytes, trunk: Trunk) -> TimestepPlan:
    """Read and self-verify one plan asset without a tensor device. A turbo plan is stamped with
    its TRUNK, because the producer tables one trunk's modulation rows at it."""
    try:
        document = canonical_json.decode(raw)
        if not isinstance(document, dict):
            raise TypeError("timestep plan is not an object")
        refine = document.get("kind") == "refine"
        schedules = []
        for row in document["schedules"]:
            evaluations = row["evaluations"]
            schedules.append(
                Schedule(
                    refine=refine,
                    sigma_grid_points=int(row["sigma_grid_points"]),
                    video_sigmas=tuple(
                        [float.fromhex(entry["video_sigma"]) for entry in evaluations]
                        + [float.fromhex(row["terminal"]["video_sigma"])]
                    ),
                    audio_sigmas=tuple(
                        [float.fromhex(entry["audio_sigma"]) for entry in evaluations]
                        + [float.fromhex(row["terminal"]["audio_sigma"])]
                    ),
                )
            )
        plan = TimestepPlan(
            task=trunk,
            video_shift=float.fromhex(document["video_shift"]),
            audio_shift=float.fromhex(document["audio_shift"]),
            schedules=tuple(schedules),
            refine=refine,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ConformanceError(
            f"the model's {trunk} timestep plan is malformed", code="artifact_config"
        ) from exc
    if plan.canonical_bytes() != canonical_json.encode(document):
        raise ConformanceError(
            f"the model's {trunk} timestep plan has unexpected semantics", code="artifact_config"
        )
    return plan


def model_plans(assets: ModelAssets, *, turbo: bool = False) -> dict[Trunk, TimestepPlan]:
    """Each trunk's plan: the asset of the component (DiT or turbo overlay) whose tables it labels."""
    names = {trunk: f"{name}/{PLAN_ASSET}" for trunk, name in
             (_OVERLAY_COMPONENT if turbo else _DIT_COMPONENT).items()}
    missing = sorted(set(names.values()) - set(assets.names()))
    if missing:
        raise InvalidRequest(
            f"this model release carries no {', '.join(missing)}: H3 reads its timestep plans "
            f"from the model, first carried by {PLANS_SINCE[turbo]}",
            code="model_asset_missing",
            fields=missing,
        )
    return {trunk: timestep_plan(assets.read(name), trunk) for trunk, name in names.items()}


def regenerate_plan(assets: ModelAssets) -> TimestepPlan:
    """The Regenerate overlay's refine plan, from its component's asset."""
    name = f"{REGENERATE_COMPONENT}/{PLAN_ASSET}"
    if name not in assets.names():
        raise InvalidRequest(
            f"this model release carries no {name}", code="model_asset_missing", fields=[name]
        )
    plan = timestep_plan(assets.read(name), "fl2va")
    if not plan.refine:
        raise ConformanceError("the Regenerate overlay's plan is not a refine plan", code="artifact_config")
    return plan


def _validate_plan(plan: TimestepPlan, trunk: Trunk, layout: TableLayout | None) -> None:
    """Refuse a mismatched selected asset before constructing weighted components.

    Header rows are authoritative: their order may differ and they may cover additional
    schedules. Every schedule this plan offers must be covered by those actual rows.
    """
    if plan.task != trunk:
        raise ConformanceError(
            f"the model's {trunk} timestep plan names {plan.task}", code="artifact_config"
        )
    if layout is not None:
        for schedule in plan.schedules:
            layout.require(schedule.video_timesteps, schedule.audio_timesteps)


def _schedulers(plan: TimestepPlan) -> dict[str, MiniMaxH3Scheduler]:
    """Fresh official schedulers at the plan's shifts; `executed` proves the grid they build."""
    return {
        "scheduler": MiniMaxH3Scheduler(shift=plan.video_shift),
        "audio_scheduler": MiniMaxH3Scheduler(shift=plan.audio_shift),
    }


def reference_image_size(
    width: int, height: int, short_edge: int = REFERENCE_IMAGE_SHORT_EDGE
) -> tuple[int, int]:
    """The upstream 32-pixel-grid image geometry, in width/height order."""
    scale = short_edge / min(width, height)
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def reference_image_vision_tokens(
    width: int, height: int, short_edge: int = REFERENCE_IMAGE_SHORT_EDGE
) -> int:
    """Official short-edge, 16-patch, 2x2-merge image demand."""
    target_width, target_height = reference_image_size(width, height, short_edge)
    return target_height * target_width // (16 * 16 * 2 * 2)


#: The canvas rule a video reference shares with the target: multiple, short edge, area.
CANVAS = (32, 768, 768 * 1344)


def reference_video_vision_tokens(width: int, height: int, duration: Fraction) -> int:
    """Exact 2-fps, pair-merged vision demand at the clip's `motion_size` of its canvas."""
    canvas_height, canvas_width = clip_size(width, height, guide=False)
    frames_at_24fps = _round_fraction(duration * FPS)
    sampled_frames = (frames_at_24fps + 11) // 12
    temporal_blocks = (sampled_frames + 1) // 2
    spatial_tokens = canvas_height * canvas_width // (16 * 16 * 2 * 2)
    return temporal_blocks * spatial_tokens


@dataclass(frozen=True, slots=True)
class ReferenceMemo:
    """Where one generation keeps its reference latents (`loader.cache`), and the identity of
    the construction that encoded them (its checkpoint's manifest)."""

    cache: DerivedCache
    identity: str


def _unreachable() -> Any:
    raise AssertionError("a memo hit was just checked")


def _returning(value: Any) -> Callable[[], Any]:
    return lambda: value


def _encode_identity(view: Any, memo: ReferenceMemo | None) -> tuple[Any, ...]:
    """What, beside the pixels, decides a condition encode's latents."""
    return (
        memo.identity if memo is not None else "",
        str(view._execution_device.type),
        str(next(view.vae.parameters()).dtype),
        tuple(view.pixel_mean),
        tuple(view.pixel_std),
        int(view.keyframe_encode_seed),
    )


def _fitted(image: Any, height: int, width: int) -> np.ndarray:
    """An image center-cropped to the canvas's aspect and resized to it, as uint8 `(H, W, 3)`."""
    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(width, round(image.width * scale)), max(height, round(image.height * scale))),
                           resample=3)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8).copy()


def _snapped(frames: int) -> int:
    """A clip's frames snapped down to `17n + 5`, as the official encode step does."""
    chunks = max(1, (frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK)
    return chunks * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK


def _clip_latent_frames(frames: int) -> int:
    return video_latent_num_frames(_snapped(frames), FRAMES_PER_CHUNK, LATENTS_PER_CHUNK)


def _soundtrack_rows(audio_vae: Any, waveform: torch.Tensor) -> torch.Tensor:
    """A stereo soundtrack's normalized latents, `(channels, latents, width)`: the channels are
    batch items of the mono audio VAE, at the posterior mean, never sampled."""
    posterior = audio_vae.encode(waveform[:, None], return_dict=False)[0]
    rows = posterior.mode().float().cpu().transpose(1, 2)
    mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
    return (rows - mean) / std


def reference_digest(pixels: Any) -> tuple[str, tuple[int, ...], str]:
    """The exact encoder input: dtype, shape and the SHA-256 of the pixel bytes."""
    contiguous = np.ascontiguousarray(pixels)
    return (
        str(contiguous.dtype),
        tuple(int(d) for d in contiguous.shape),
        hashlib.sha256(contiguous).hexdigest(),
    )


class _ScopedPipeline:
    """A request-local view whose execution device follows the admitted component.

    Diffusers normally finds a device by scanning every registered module. Runtime stages
    one weighted root at a time, so that scan can see an inactive sibling first. The view
    changes no placement; it merely reports the device of the root whose Runtime scope is
    already active.
    """

    def __init__(
        self,
        pipe: Any,
        component: Any = None,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self._pipe = pipe
        self._component = component
        self._overrides = {} if overrides is None else dict(overrides)

    @property
    def _execution_device(self) -> Any:
        return self._pipe._execution_device if self._component is None else self._component.device

    @property
    def device(self) -> Any:
        return self._pipe.device if self._component is None else self._component.device

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._pipe, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Blocks communicate through PipelineState. A write landing on this per-call view
        # would evaporate silently, and forwarding it would mutate the shared pipeline
        # across requests — so a non-private write is a typed refusal, never a silent one.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        raise ConformanceError(
            f"a wrapped Diffusers block set {name!r} on the request-local pipeline view",
            code="artifact_config",
        )


class _ImageEdges:
    """Request-local configuration view for the official ``ref2va`` setup step.

    The step reads ``reference_image_short_edge`` once per image reference, in packed order,
    so this view hands every image its own resolved edge and the official resize does the
    per-image work — no copied geometry, and the shared pipeline config is never touched.
    A read count that differs from the image count is upstream drift and refuses typed.
    """

    def __init__(self, config: Mapping[str, Any], edges: Sequence[int]) -> None:
        self._config = config
        self._edges = iter(edges)
        self._count = len(edges)

    def __getattr__(self, name: str) -> Any:
        if name != "reference_image_short_edge":
            return getattr(self._config, name)
        edge = next(self._edges, None)
        if edge is None:
            raise ConformanceError(
                f"official ref2va setup read more than {self._count} image short edges",
                code="artifact_config",
                fields=["pipeline", "reference_image_short_edge"],
            )
        return edge

    def settle(self) -> None:
        if next(self._edges, None) is not None:
            raise ConformanceError(
                f"official ref2va setup read fewer than {self._count} image short edges",
                code="artifact_config",
                fields=["pipeline", "reference_image_short_edge"],
            )


#: The request state's handle on its intake length statement (`OfficialH3Pipeline._expect`).
_INTAKE = "intake_length_statement"
#: The request state's timestep plan, the one `_start` scheduled and `denoise` proves.
_PLAN = "timestep_plan"


class _IntakeLengths:
    """One pipeline's early attention-length statements (h3a-087), run off the request thread
    by one non-daemon worker, so no statement outlives its owner. A request settles its own
    before its first DiT step (`_settle_intake`), so what it records lands in that request; a
    newer statement cancels an older one still queued; `close` or the interpreter's exit waits
    out the one running rather than finalizing torch under it. The worker ends with the pool."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minimax-h3-intake")
        self._lock = threading.Lock()
        self._queued: Future[None] | None = None
        self._closed = False

    def submit(self, statement: Callable[[], None]) -> Future[None] | None:
        """None once closed: the statement is advisory, and the first dense step states it."""
        # The copied context carries the request's observation scope to the kernel rows.
        runner = contextvars.copy_context().run
        with self._lock:
            if self._closed:
                return None
            if self._queued is not None:
                self._queued.cancel()
            self._queued = future = self._pool.submit(runner, statement)
        return future

    def close(self) -> None:
        """Cancel what is queued and wait for the statement running, if any."""
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)


def _settle_intake(state: Any) -> None:
    """Wait for this request's statement, if it made one, to finish or be cancelled. Not
    `concurrent.futures.wait`: a future `shutdown` cancels from the queue never notifies it."""
    statement = state.get(_INTAKE)
    if statement is not None:
        state.set(_INTAKE, None)
        with suppress(CancelledError):
            statement.result()


@contextmanager
def _audio_weight_norm_scope(audio_vae: Any) -> Iterator[None]:
    """Release only legacy WeightNorm's recomputable, unregistered attributes."""
    try:
        yield
    finally:
        for module in audio_vae.modules():
            for hook in module._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    module.__dict__.pop(hook.name, None)


class OfficialH3Pipeline:
    """Both official task workflows over one shared, dual-DiT construction."""

    def __init__(self, config: Config, plans: Mapping[Trunk, TimestepPlan]) -> None:
        dit_specs = _dit_specs(_artifact_sections(config))
        for trunk, (_upstream, _structure, layout) in dit_specs.items():
            _validate_plan(plans[trunk], trunk, layout)
        blocks = {
            name: MiniMaxH3Blocks().get_workflow(_diffusers_workflow(name))
            for name in _WORKFLOW_TASKS
        }
        pipes = {name: MiniMaxH3ModularPipeline(blocks=block) for name, block in blocks.items()}
        dits = {
            trunk: _build_dit(upstream, structure, layout)
            for trunk, (upstream, structure, layout) in dit_specs.items()
        }
        _validate_dual_topology(dits, _DIT_COMPONENT)
        text_encoder = build_text_conditioner(config.section("text_encoder").mapping())
        video_vae = _apply_video_vae_dtype(
            TileBatchedVideoVAE.from_config(config.section("vae").mapping()), config
        )
        audio = config.section("audio_vae").mapping()
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_config(audio).eval()
        for trunk in _TRUNKS:
            _validate_model_contract(pipes[trunk], dits[trunk], video_vae, audio_vae)
        processor = vlm.processor(_ASSETS / "processor")
        tokenizer = processor.tokenizer

        for workflow, task in _WORKFLOW_TASKS.items():
            pipes[workflow].register_components(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                processor=processor,
                vae=video_vae,
                audio_vae=audio_vae,
                **_schedulers(plans[_TRUNK[task]]),
                **{_DIT_COMPONENT[_TRUNK[task]]: dits[_TRUNK[task]]},
            )

        # Runtime reads this mapping and nothing under ``pipe`` when deriving/filling
        # checkpoint destinations, under Diffusers' component names. Config-only processors
        # and schedulers are deliberately absent. LoRA roots belong to their own separately
        # bound construction.
        self.components: dict[str, Any] = {
            "transformer": dits["fl2va"],
            "transformer_ref": dits["ref2va"],
            "text_encoder": text_encoder,
            "vae": video_vae,
            "audio_vae": audio_vae,
        }
        # The one release audio clock, read off the contract-checked artifact config
        # rather than respelled by callers.
        self.sample_rate = int(audio_vae.config.sampling_rate)
        self._blocks = blocks
        self._pipes = pipes
        self.plans: dict[Trunk, TimestepPlan] = {trunk: plans[trunk] for trunk in _TRUNKS}
        self._dit_specs = dit_specs
        self._intake = _IntakeLengths()

    def close(self) -> None:
        """End this pipeline's background work: queued intake statements are cancelled and the
        running one is waited for. Requests after this state nothing early."""
        self._intake.close()

    def generator(self, source: object) -> Any:
        """Adapt Runtime's public request generator to Diffusers' torch generator."""
        if isinstance(source, torch.Generator):
            return source
        if not isinstance(source, random.Random):
            raise ConformanceError(
                f"request generator has unsupported type {type(source).__name__}",
                code="artifact_config",
            )
        return torch.Generator().manual_seed(source.getrandbits(63))

    def image_reference(self, image: Image) -> Any:
        return MiniMaxH3ImageReference(image=image if image.mode == "RGB" else image.convert("RGB"))

    def audio_reference(self, audio: DecodedAudio) -> Any:
        waveform = _resample(_audio_tensor(audio), audio.sample_rate, self.sample_rate)
        return MiniMaxH3AudioReference(audio=waveform, sample_rate=self.sample_rate)

    def video_reference(self, clip: ReferenceClip) -> Any:
        """A clip already at the size it is used at (`read_reference_video`). The official
        normalization sees a free canvas-sized stand-in it leaves as it is, and `start_ref2va`
        puts the real frames in its place, so nothing is resized twice."""
        soundtrack = clip.soundtrack
        if soundtrack is not None:
            soundtrack = _resample(soundtrack, clip.sample_rate, self.sample_rate)
        height, width = (int(v) for v in resolve_canvas_size(clip.width, clip.height, *CANVAS))
        stand_in = np.broadcast_to(
            np.zeros((1, 1, 1, 3), dtype=np.uint8), (len(clip.frames), height, width, 3)
        )
        return MotionVideo(
            frames=stand_in,
            fps=float(FPS),
            audio=soundtrack,
            sample_rate=None if soundtrack is None else self.sample_rate,
            motion=clip.frames,
        )

    def guide_clip(self, clip: ReferenceClip) -> GuideClip:
        """A guide video, read at the render's canvas, with its soundtrack in stereo at the
        audio VAE's rate, as the official normalization puts a reference's."""
        audio = clip.soundtrack
        if audio is not None:
            audio = _resample(audio, clip.sample_rate, self.sample_rate).to(torch.float32)
            audio = audio.expand(2, -1).contiguous() if audio.shape[0] == 1 else audio
        return GuideClip(clip.frames, audio)

    def start_fl2va(
        self,
        *,
        prompt: str,
        first_frame: Any | None,
        last_frame: Any | None,
        generator: Any,
        steps: int,
        frames: int,
        task: Task = "fl2va",
        plan: TimestepPlan | None = None,
        delivery: ContinuationPlan | None = None,
    ) -> Any:
        """`delivery` is the segment's clock when its successor carries a picture or sound
        (`export_context`); an FL2VA segment carries nothing in itself."""
        if delivery is not None and (delivery.carried or frames != delivery.sample_frames):
            raise InvalidRequest("an FL2VA segment carries nothing in and samples its own window",
                                 code="h3_continuation")
        # With no anchor, state the official default canvas. With an anchor, leaving the
        # dimensions absent makes the first supplied keyframe the geometry authority.
        height, width = (768, 1344) if first_frame is None and last_frame is None else (None, None)
        state = self._start(
            _trunk_task(task, "fl2va"),
            prompt=prompt,
            generator=generator,
            steps=steps,
            frames=frames,
            image=first_frame,
            last_image=last_frame,
            height=height,
            width=width,
            plan=plan,
        )
        state.set("continuation_delivery", delivery)
        self._expect(task, state)
        return state

    def start_ref2va(
        self,
        *,
        prompt: str,
        references: Sequence[Any],
        generator: Any,
        steps: int,
        frames: int,
        reference_image_short_edges: Sequence[int],
        task: Task = "ref2va",
        plan: TimestepPlan | None = None,
        context: AVContext | None = None,
        delivery: ContinuationPlan | None = None,
        expected_context_provenance: str | None = None,
        guide: GuideClip | None = None,
        anchors: Sequence[tuple[Any, Literal["first", "last"]]] = (),
    ) -> Any:
        """``reference_image_short_edges`` is one resolved edge per image reference, in
        packed order; the official setup step resizes each image to its own. A `guide` sets
        the render's canvas (`guide_rows`), else the carried picture does, else the first
        keyframe `anchor`, as FL2VA's first keyframe does; each anchor is fitted to it
        (`anchor_rows`)."""
        if context is not None:
            context.validate()
            if delivery is None or delivery.carried != context.carry:
                raise InvalidRequest("context and delivery plan disagree", code="h3_continuation")
            if expected_context_provenance != context.provenance:
                raise InvalidRequest(
                    "continuation renderer provenance differs", code="h3_continuation"
                )
        elif delivery is not None and delivery.carried:
            raise InvalidRequest(
                "a carried picture or sound requires a completed context", code="h3_continuation"
            )
        if delivery is not None and frames != delivery.sample_frames:
            raise InvalidRequest("sample frames differ from delivery plan", code="h3_continuation")
        if guide is not None:
            canvas = dict(zip(("height", "width"), map(int, guide.frames.shape[1:3]), strict=True))
        elif context is not None and context.canvas is not None:
            canvas = dict(zip(("height", "width"), context.canvas, strict=True))
        elif anchors:
            width, height = anchors[0][0].size
            canvas = dict(zip(("height", "width"),
                              (int(v) for v in resolve_canvas_size(width, height, *CANVAS)), strict=True))
        else:
            canvas = {}
        state = self._start(
            _trunk_task(task, "ref2va"),
            prompt=prompt,
            generator=generator,
            steps=steps,
            frames=frames,
            references=list(references),
            image_short_edges=tuple(reference_image_short_edges),
            plan=plan,
            **canvas,
        )
        state.set("normalized_references", [
            replace(normalized, frames=source.motion[:frames])
            if isinstance(source, MotionVideo) else normalized
            for source, normalized in zip(references, state.normalized_references, strict=True)
        ])
        state.set(GUIDE, guide)
        state.set(ANCHORS, [(_fitted(image, state.height, state.width), at) for image, at in anchors])
        if context is not None and context.canvas not in (None, (state.height, state.width)):
            raise InvalidRequest(
                "continuation canvas differs from this render", code="h3_continuation"
            )
        state.set("continuation_context", context)
        state.set("continuation_delivery", delivery)
        self._expect(task, state)
        return state

    def _start(
        self,
        task: Task,
        *,
        prompt: str,
        generator: Any,
        steps: int,
        frames: int,
        image_short_edges: Sequence[int] = (),
        plan: TimestepPlan | None = None,
        **values: Any,
    ) -> Any:
        # The plan, not the request, spells the official grid: `steps` transformer
        # evaluations are the schedule's grid points less the terminal zero. A turbo task's
        # plan is its turbo LoRA's, which the caller passes.
        if plan is None:
            if _BANK[task] is not None:
                raise ConformanceError(
                    "a turbo request requires its selected overlay's timestep plan",
                    code="artifact_config",
                )
            plan = self.plans[_TRUNK[task]]
        if plan.task != _TRUNK[task]:
            raise ConformanceError("timestep plan belongs to a different trunk", code="artifact_config")
        schedule = plan.schedule(steps)
        state = PipelineState()
        fixed = {
            _PLAN: plan,
            "prompt": prompt,
            "generator": generator,
            "num_frames": frames,
            "num_inference_steps": schedule.sigma_grid_points,
            "output_type": "pt",
            **values,
        }
        bank = _BANK[task]
        if bank is not None:
            # Every DiT forward of this request names the bank it is served under; the
            # official loop hands `attention_kwargs` to the transformer call unchanged.
            fixed["attention_kwargs"] = {ATTENTION_KWARG: bank}
        for name, value in fixed.items():
            state.set(name, value)
        workflow = self._workflow(task, state)
        if "before_encode" in self._blocks[workflow].sub_blocks:
            pipe = self._pipes[workflow]
            edges = None
            if _TRUNK[task] == "ref2va":
                # Only setup uses this per-request geometry. A view leaves the shared
                # pipeline unchanged, even if preprocessing raises or calls overlap.
                # `ref2va` setup is the one before-encode block that also gates on the
                # seconds ceiling, so it is scoped here as well as at denoise.
                edges = _ImageEdges(pipe.config, image_short_edges)
                pipe = _ScopedPipeline(
                    pipe, overrides={"config": edges, "max_duration": _CEILING_S}
                )
            self._run_with(task, pipe, "before_encode", state)
            if edges is not None:
                edges.settle()
        return state

    def expected_live_tokens(self, task: Task, state: Any) -> int | None:
        """The packed length this request's sparse steps will attend over, read at intake
        (h3a-087): a clip's latent rows have left the sequence by then (`ClipRows.release`).

        The workflow's own text block tokenizes the presentation beside a conditioner that
        only counts (`PresentationLength`), and its own layout block packs it, all on `meta`.
        The one input not yet known is each reference's latent block, sized here by the video
        VAE's grid exactly as `condition_media` will encode it; an audio-bearing public
        reference, whose rows are the audio VAE's own count, answers None.
        """
        workflow = self._workflow(task, state)
        pipe, blocks = self._pipes[workflow], self._blocks[workflow].sub_blocks
        scratch = PipelineState(
            dict(state.values), {k: list(v) for k, v in state.kwargs_mapping.items()}
        )
        shell = SimpleNamespace(device=torch.device("meta"))
        counted = PresentationLength(self.components["text_encoder"], torch.device("meta"))
        blocks["text_encoder"](
            _ScopedPipeline(pipe, shell, overrides={"text_encoder": counted}), scratch
        )
        if _TRUNK[task] == "ref2va":
            references = list(scratch.get("normalized_references"))
            if any(reference.has_audio for reference in references):
                return None
            video: list[Any] = []
            audio: list[Any] = []
            released = 0
            for reference in references:
                if reference.kind == "image":
                    width, height = reference.image.size
                    frames = 1
                else:
                    count, height, width = reference.frames.shape[:3]
                    frames = _clip_latent_frames(count)
                    if public_video(reference):
                        released += (
                            frames
                            * (height // VAE_SPATIAL_RATIO // PATCH_SIZE[1])
                            * (width // VAE_SPATIAL_RATIO // PATCH_SIZE[2])
                        )
                video.append(
                    torch.empty(
                        (1, 1, frames, height // VAE_SPATIAL_RATIO, width // VAE_SPATIAL_RATIO),
                        device="meta",
                    )
                )
            context = scratch.get("continuation_context")
            if context is not None:
                # `condition_media` appends the carried latents after the public references.
                video.extend(t for t in (context.video,) if t is not None)
                audio.extend(t for t in (context.audio,) if t is not None)
                references.append(ContextReferenceLayout.of(context))
            scratch.set("normalized_references", references)
            scratch.set("condition_latents", video)
            scratch.set("audio_condition_latents", audio)
        elif "denoise.no_keyframe_anchors" in blocks:
            blocks["denoise.no_keyframe_anchors"](pipe, scratch)
        # Under `denoise`'s ceiling: upstream's own refuses a full-length (362-frame) clip.
        blocks["denoise.prepare_layout"](
            _ScopedPipeline(pipe, shell, overrides={"max_duration": _CEILING_S}), scratch
        )
        tokens = int(scratch.get("token_tags").numel())
        for pixels, _ in scratch.get(ANCHORS) or ():
            tokens += (pixels.shape[0] // VAE_SPATIAL_RATIO // PATCH_SIZE[1]) * (
                pixels.shape[1] // VAE_SPATIAL_RATIO // PATCH_SIZE[2]
            )
        guide = scratch.get(GUIDE)
        if guide is not None:
            if guide.audio is not None:
                return None
            count, height, width = guide.frames.shape[:3]
            tokens += (
                _clip_latent_frames(count)
                * (height // VAE_SPATIAL_RATIO // PATCH_SIZE[1])
                * (width // VAE_SPATIAL_RATIO // PATCH_SIZE[2])
            )
        return tokens - released if _TRUNK[task] == "ref2va" else tokens

    def _expect(self, task: Task, state: Any) -> None:
        """Tell the runtime this request's attention length while conditioning runs, so a
        length-specialized kernel starts preparing now. Off the request's thread (tokenizing
        the references' vision blocks is ~0.4 s at their 2048-pixel edge) and advisory: a
        failure is reported, never raised, and a wrong answer only costs a background compile.
        `denoise` settles the statement before the first DiT step."""
        snapshot = PipelineState(
            dict(state.values), {k: list(v) for k, v in state.kwargs_mapping.items()}
        )
        dit = self.components[_DIT_COMPONENT[_TRUNK[task]]]

        def state_length() -> None:
            # Reported where it fails, whether or not its request reaches the DiT to settle it.
            try:
                tokens = self.expected_live_tokens(task, snapshot)
                if tokens is not None:
                    expect_attention(tokens, dit)
            except Exception as exc:
                print(f"[minimax-h3] no intake attention length: {exc!r}"[:300], file=sys.stderr)

        state.set(_INTAKE, self._intake.submit(state_length))

    def text_state(self, state: Any) -> Any:
        """A copy of `state` for the text block alone: it reads the prompt and the references
        as they stand now, whatever the media block appends meanwhile (`adopt_text`)."""
        return PipelineState(
            dict(state.values), {k: list(v) for k, v in state.kwargs_mapping.items()}
        )

    def adopt_text(self, task: Task, state: Any, text: Any) -> None:
        """The text block's outputs, from its copy into `state`, as the block writes them."""
        block = self._blocks[self._workflow(task, state)].sub_blocks["text_encoder"]
        for output in block.intermediate_outputs:
            state.set(output.name, text.get(output.name), output.kwargs_type)

    def condition_text(self, task: Task, state: Any, device: Device) -> None:
        conditioner = FinalHiddenState(self.components["text_encoder"], torch.device(str(device)))
        view = _ScopedPipeline(
            self._pipes[self._workflow(task, state)],
            conditioner,
            overrides={"text_encoder": conditioner},
        )
        self._run_with(task, view, "text_encoder", state)

    def condition_media(
        self,
        task: Task,
        state: Any,
        *,
        memo: ReferenceMemo | None = None,
    ) -> tuple[int, int]:
        """Encode the request's keyframes or references. For `ref2va`, the (hits, encoded)
        count of reference latents (`_encode_references`)."""
        # Encode public references first; the carried latents never reach Qwen or a VAE.
        # They are the predecessor's own denoised latents, in the native normalization.
        counts = (0, 0)
        if _TRUNK[task] == "ref2va":
            with _audio_weight_norm_scope(self.components["audio_vae"]):
                counts = self._encode_references(task, state, memo)
        else:
            self._run(task, "vae_encoder", state, component="vae")
        if state.get(GUIDE) is not None:
            with _audio_weight_norm_scope(self.components["audio_vae"]):
                self._encode_guide(task, state, memo)
        if state.get(ANCHORS):
            self._encode_anchors(task, state)
        context = state.get("continuation_context")
        if context is not None:
            if context.video is not None:
                state.set("condition_latents", [*state.condition_latents, context.video])
            if context.audio is not None:
                audio = [*state.audio_condition_latents, context.audio]
                state.set("audio_condition_latents", audio)
            state.set(
                "normalized_references",
                [*state.normalized_references, ContextReferenceLayout.of(context)],
            )
        return counts

    def _encode_references(
        self, task: Task, state: Any, memo: ReferenceMemo | None
    ) -> tuple[int, int]:
        """Diffusers' `MiniMaxH3Ref2VAReferenceEncoderStep`, one reference at a time.

        The same arithmetic: each image or video reference through `encode_vae_condition`
        (sampled under its own fixed seed, rounded to float16, normalized on the CPU), each
        soundtrack through the audio VAE's posterior mode. Two things change where it runs:

        - A video-VAE encode is a pure function of the reference's pixels, so its latents are
          memoized per generation by an exact key (`ReferenceMemo`): a long-form piece's
          later shots reuse their references' latents. The audio encode is not: H3's audio
          is not bitwise reproducible run to run (h3a-088), so it runs every time.
        - Image encodes that miss run across the group's idle ranks (`author.spread`,
          sparing: the DiT the denoise needs next stays where it is), each on one rank from
          its own pixels, so each result is what rank 0 alone computes on an identical card.
        """
        with torch.no_grad():
            workflow = self._workflow(task, state)
            view = _ScopedPipeline(self._pipes[workflow], self.components["vae"])
            device = view._execution_device
            vae, audio_vae = view.vae, view.audio_vae
            pixel_mean, pixel_std = tuple(view.pixel_mean), tuple(view.pixel_std)
            seed = int(view.keyframe_encode_seed)
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            identity = _encode_identity(view, memo)
            references = state.normalized_references
            visual: dict[int, tuple[Any, tuple[Any, ...]]] = {}
            for index, reference in enumerate(references):
                if reference.kind == "image":
                    pixels = np.array(reference.image)
                elif reference.kind == "video":
                    pixels = reference.frames[: _snapped(reference.frames.shape[0])].copy()
                else:
                    continue
                key = () if memo is None else (*identity, *reference_digest(pixels))
                visual[index] = (pixels, key)
            latents: dict[int, Any] = {}
            if memo is not None:
                for index, (_, key) in visual.items():
                    if key in memo.cache:
                        latents[index] = memo.cache.get(key, build=_unreachable)
            missing = [index for index in visual if index not in latents]
            images = [index for index in missing if visual[index][0].ndim == 3]
            arguments = (
                (torch.from_numpy(visual[i][0]), pixel_mean, pixel_std, seed) for i in images
            )
            # Only H3's own VAE class carries the method a group rank can run.
            encoded = (
                spread(vae, "encode_condition", arguments, spare=True)
                if isinstance(vae, TileBatchedVideoVAE)
                else (encode_condition(vae, *args) for args in arguments)
            )
            for index, raw in zip(images, encoded, strict=True):
                latents[index] = (raw.cpu() - latents_mean) / latents_std
            for index in missing:
                if index not in latents:
                    raw = encode_condition(
                        vae, torch.from_numpy(visual[index][0]), pixel_mean, pixel_std, seed
                    )
                    latents[index] = (raw.cpu() - latents_mean) / latents_std
            if memo is not None:
                for index in missing:
                    memo.cache.get(visual[index][1], build=_returning(latents[index]))
            # The memo keeps its own tensors: the request gets copies it cannot share.
            condition_latents = [
                latents[index].clone() if memo is not None else latents[index] for index in visual
            ]

            audio_condition_latents = [
                _soundtrack_rows(audio_vae, reference.audio.to(device)).reshape(
                    -1, view.audio_latent_channels
                )
                for reference in references
                if reference.has_audio
            ]
            state.set("condition_latents", condition_latents)
            state.set("audio_condition_latents", audio_condition_latents)
            return len(visual) - len(missing), len(missing)

    def _encode_guide(self, task: Task, state: Any, memo: ReferenceMemo | None) -> None:
        """The guide's latents and soundtrack rows, encoded as a reference video's are."""
        guide: GuideClip = state.get(GUIDE)
        with torch.no_grad():
            view = _ScopedPipeline(self._pipes[self._workflow(task, state)], self.components["vae"])
            vae = view.vae
            pixels = guide.frames[: _snapped(guide.frames.shape[0])].copy()
            key = None
            if memo is not None:
                key = (*_encode_identity(view, memo), *reference_digest(pixels))
            if key is not None and key in memo.cache:
                latents = memo.cache.get(key, build=_unreachable).clone()
            else:
                raw = encode_condition(
                    vae,
                    torch.from_numpy(pixels),
                    tuple(view.pixel_mean),
                    tuple(view.pixel_std),
                    int(view.keyframe_encode_seed),
                )
                mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
                std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
                latents = (raw.cpu() - mean) / std
                if key is not None:
                    latents = memo.cache.get(key, build=_returning(latents)).clone()
            audio = None
            if guide.audio is not None:
                audio = _soundtrack_rows(view.audio_vae, guide.audio.to(view._execution_device))
        state.set(GUIDE_LATENTS, (latents, audio))

    def _encode_anchors(self, task: Task, state: Any) -> None:
        """Each keyframe anchor as one latent frame, encoded as FL2VA's keyframes are."""
        with torch.no_grad():
            view = _ScopedPipeline(self._pipes[self._workflow(task, state)], self.components["vae"])
            vae = view.vae
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            encoded = []
            for pixels, at in state.get(ANCHORS):
                raw = encode_condition(
                    vae, torch.from_numpy(pixels), tuple(view.pixel_mean), tuple(view.pixel_std),
                    int(view.keyframe_encode_seed),
                )
                encoded.append(((raw.cpu() - mean) / std, at))
        state.set(ANCHOR_LATENTS, encoded)

    def denoise(
        self,
        task: Task,
        state: Any,
        *,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
        sol_dense_steps: int = 10,
    ) -> ScheduleFacts:
        _validate_sol_dense_steps(sol_dense_steps)
        _settle_intake(state)
        blocks = self._blocks[self._workflow(task, state)].sub_blocks
        scoped = _ScopedPipeline(
            self._pipes[self._workflow(task, state)],
            self.components[_DIT_COMPONENT[_TRUNK[task]]],
            overrides={
                **_schedulers(state.get(_PLAN)),
                "max_duration": _CEILING_S,
            },
        )
        # The selected upstream workflow owns the ordered preparation steps.
        # Text-only requests have no keyframe tensors to encode or concatenate.
        guided = 0
        for name in blocks:
            if name == "denoise.denoise":
                break
            if not name.startswith("denoise."):
                continue
            if name == "denoise.set_timesteps" and state.get(GUIDE_LATENTS) is not None:
                guided = self._pack_guide(scoped, state)
            if name == "denoise.set_timesteps" and state.get(ANCHOR_LATENTS):
                guided += self._pack_anchors(scoped, state)
            self._run_with(task, scoped, name, state)

        context = state.get("continuation_context")
        if context is not None:
            align_context_rows(state, context, tuple(scoped.patch_size))

        # The receipt names the schedule the official blocks actually built, proven
        # against the plan by exact float32 equality of both modality vectors.
        plan = state.get(_PLAN)
        schedule = plan.executed(_float_tuple(state.timesteps), _float_tuple(state.audio_timesteps))
        if _BANK[task] is None:
            layout = self._dit_specs[_TRUNK[task]][2]
            if layout is not None:
                layout.require(schedule.video_timesteps, schedule.audio_timesteps)
        facts = ScheduleFacts.of(plan, schedule)
        loop = blocks["denoise.denoise"]
        block_state = loop.get_block_state(state)
        clips = None
        if _TRUNK[task] == "ref2va":
            references, latents = state.normalized_references, state.condition_latents
            clips = ClipRows(block_state, references, latents, scoped.patch_size, start=guided)
            clips.thin(block_state)
        if (
            task == "fl2va"
            and clips is None
            and context is None
            and not block_state.attention_kwargs
            and getattr(scoped.transformer, "_parallel_config", None) is None
        ):
            denoise_fl2va(
                scoped.transformer,
                scoped,
                block_state,
                lambda step: _attention_layout(block_state, step, sol_dense_steps=sol_dense_steps),
                sol_dense_steps,
                on_step,
            )
        else:
            with step_invariants():
                for index, timestep in enumerate(block_state.timesteps):
                    cancel()
                    if clips is not None and index == sol_dense_steps:
                        clips.release(block_state)
                    with attention_scope(
                        _attention_layout(block_state, index, sol_dense_steps=sol_dense_steps)
                    ):
                        _, block_state = loop.loop_step(scoped, block_state, i=index, t=timestep)
                    on_step(index)
        loop.set_block_state(state, block_state)
        self._run_with(task, scoped, "denoise.after_denoise", state)
        return facts

    @staticmethod
    def _pack_guide(scoped: Any, state: Any) -> int:
        """The guide as its rows (`guide_rows`), noised as FL2VA's keyframes are, after every
        other draw of the request's generator; its video row count."""
        latents, audio = state.get(GUIDE_LATENTS)
        device = state.latents.device
        noise = randn_tensor(
            latents.shape, generator=state.generator, device=device, dtype=torch.float32
        )
        noised = scoped.scheduler.scale_noise(latents.to(device), scoped.keyframe_noise_aug, noise)
        video = patchify_video_latents(noised, tuple(scoped.patch_size))
        if audio is not None:
            audio = audio[:, : state.num_audio_latents]
        guide_rows(state, video, audio)
        return int(video.shape[0])

    @staticmethod
    def _pack_anchors(scoped: Any, state: Any) -> int:
        """Each keyframe anchor as its rows (`anchor_rows`), noised as FL2VA's keyframes are,
        after the guide's draws; their video row count."""
        count = 0
        device = state.latents.device
        for latents, at in state.get(ANCHOR_LATENTS):
            noise = randn_tensor(
                latents.shape, generator=state.generator, device=device, dtype=torch.float32
            )
            noised = scoped.scheduler.scale_noise(latents.to(device), scoped.keyframe_noise_aug, noise)
            video = patchify_video_latents(noised, tuple(scoped.patch_size))
            anchor_rows(state, video, last=at == "last")
            count += int(video.shape[0])
        return count

    def decode_audio(self, task: Task, state: Any) -> Any:
        # Repeated BigVGAN decodes can differ on the same input even
        # with benchmarking disabled. Scope only algorithm determinism to this
        # serial component call; preserve TF32, dtype and every other backend setting.
        deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        try:
            self._run(task, "decode.audio", state, component="audio_vae")
        finally:
            torch.backends.cudnn.deterministic = deterministic
        delivery = state.get("continuation_delivery")
        if delivery is None:
            return state.audio
        context = state.get("continuation_context")
        entry = context.audio_offset if context is not None else Fraction(0)
        start, end = delivery.audio_samples(entry, int(state.sampling_rate))
        delivered = state.audio[..., start:end]
        # Only a picture end the grid rounds the sound short of: at most a third of a tick.
        return torch.nn.functional.pad(delivered, (0, end - start - delivered.shape[-1]))

    def decode_video(self, task: Task, state: Any) -> Any:
        if state.get("continuation_delivery") is not None:
            raise InvalidRequest(
                "continuation requires streaming video decode", code="h3_continuation"
            )
        self._run(task, "decode.video", state, component="vae")
        return state.videos

    def decode_video_chunks(self, task: Task, state: Any, on_chunk: Callable[[Any], None]) -> int:
        """`decode.video` with the VAE's temporal chunks handed out as they finish (h3a-017).

        Upstream's `MiniMaxH3VideoDecodeStep` denormalizes the latents, decodes under
        float16 autocast over the float32 VAE, and reverts the ImageNet normalization on
        the whole clip. This is that arithmetic per chunk, under the same request-local
        device scope. `on_chunk` receives each `(t, 3, H, W)` float32 piece before clipping,
        so integrity checks can reject infinities before RGB8 conversion clamps them.
        Valid pixels match the official block after clipping. Returns the frame count.
        """
        workflow = self._workflow(task, state)
        components = _ScopedPipeline(self._pipes[workflow], self.components["vae"])
        device = components._execution_device
        vae = components.vae
        latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
        frames = 0
        delivery = state.get("continuation_delivery")
        tail = DeliveredFrames(delivery) if delivery is not None else None
        with torch.no_grad():
            chunks = vae.decode_chunks(state.latents * latents_std + latents_mean)
            while True:
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    chunk = next(chunks, None)
                if chunk is None:
                    return frames if tail is None else tail.finish()
                video = chunk.float() * pixel_std + pixel_mean
                frames += int(video.shape[2])
                selected = video[0].permute(1, 0, 2, 3)
                if tail is not None:
                    selected = tail.push(selected)
                if selected.shape[0]:
                    on_chunk(selected)

    def decode_latents(self, latents: Any, on_chunk: Callable[[Any], None]) -> int:
        """`decode_video_chunks` of normalized `(1, 24, T, H, W)` latents at any canvas."""
        state = PipelineState()
        state.set("latents", latents)
        return self.decode_video_chunks("fl2va", state, on_chunk)

    def encode_clip(self, pixels: Any, seed: int) -> Any:
        """A `(T, H, W, 3)` uint8 clip's normalized `(1, 24, T', H/16, W/16)` latents: the
        official keyframe encode (`encode_condition`) over the whole clip, posterior sampled
        under `seed`, as `open_h3` encodes a draft video."""
        components = _ScopedPipeline(self._pipes["t2va"], self.components["vae"])
        vae = components.vae
        raw = encode_condition(vae, pixels, components.pixel_mean, components.pixel_std, seed)
        mean = torch.tensor(vae.config.latents_mean, device=raw.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=raw.device).view(1, -1, 1, 1, 1)
        return (raw - mean) / std

    def silence_rows(self, frames: int) -> Any:
        """Packed `(2 * latents, 32)` rows of stereo silence covering `frames` video frames,
        at the posterior mean (`open_h3/pipeline.py` encode_silence)."""
        audio_vae = self.components["audio_vae"]
        device = next(audio_vae.parameters()).device
        samples = round(frames / FPS * self.sample_rate)
        # The encode's WeightNorm weights go with it, as every other audio VAE pass's do: kept,
        # 0.3 GB of them pinned 0.77 GB of Torch's cache through the executor's idle offload, and
        # SeedVR2 2160p found too little room beside it (cr-226).
        with torch.no_grad(), _audio_weight_norm_scope(audio_vae):
            rows = _soundtrack_rows(audio_vae, torch.zeros(AUDIO_CHANNELS, samples, device=device))
        return rows.reshape(-1, rows.shape[-1])

    def export_context(self, state: Any, *, provenance: str) -> AVContext | None:
        """The successor's carry from this request's denoised latents; None if it carries
        nothing."""
        return export_context(
            state.latents, state.audio_latents, state.continuation_delivery, provenance
        )

    def _run(self, task: Task, name: str, state: Any, *, component: str | None = None) -> None:
        workflow = self._workflow(task, state)
        pipe = (
            self._pipes[workflow]
            if component is None
            else _ScopedPipeline(self._pipes[workflow], self.components[component])
        )
        if name in {"vae_encoder", "decode.audio"}:
            # Legacy WeightNorm leaves computed CUDA weights outside parameters
            # and buffers; release them on success or failure before stage eviction.
            # The unchanged pre-hook recomputes from weight_g/weight_v next time.
            with _audio_weight_norm_scope(self.components["audio_vae"]):
                self._run_with(task, pipe, name, state)
        else:
            self._run_with(task, pipe, name, state)

    def _run_with(self, task: Task, pipe: Any, name: str, state: Any) -> None:
        block = self._blocks[self._workflow(task, state)].sub_blocks[name]
        block(pipe, state)

    @staticmethod
    def _workflow(task: Task, state: Any) -> str:
        if (
            _TRUNK[task] == "fl2va"
            and state.get("image") is None
            and state.get("last_image") is None
        ):
            return task.replace("fl2va", "t2va")
        return task


def _diffusers_workflow(workflow: str) -> str:
    """The official block graph a cozy workflow runs: turbo shares its base name's."""
    return workflow.removesuffix("_turbo")


def _trunk_task(task: Task, trunk: Trunk) -> Task:
    if _TRUNK[task] != trunk:
        raise ConformanceError(
            f"{task} is not a {trunk} task", code="artifact_config", fields=["task"]
        )
    return task


class OfficialH3TurboLoRA:
    """The standalone adapter's two roots, independent of the base checkpoint."""

    def __init__(self, config: Config, plans: Mapping[Trunk, TimestepPlan]) -> None:
        if set(config.keys()) != _OVERLAY_CONFIG_SECTIONS:
            raise ConformanceError(
                "a turbo LoRA checkpoint contains exactly fl2va_turbo and ref2va_turbo",
                code="artifact_config",
            )
        self.plans = {trunk: plans[trunk] for trunk in _TRUNKS}
        self.specs = {trunk: _overlay_spec(config, trunk, self.plans[trunk]) for trunk in _TRUNKS}
        overlays = {
            trunk: _build_overlay(self.specs[trunk][0], self.plans[trunk], self.specs[trunk][1])
            for trunk in _TRUNKS
        }
        _validate_dual_topology(overlays, _OVERLAY_COMPONENT)
        self.components = {_OVERLAY_COMPONENT[trunk]: value for trunk, value in overlays.items()}

    def overlay(self, base: OfficialH3Pipeline, trunk: Trunk) -> Overlay:
        config, structure, _layout = base._dit_specs[trunk]
        if structure != "adaln-pruned" or config != self.specs[trunk][0]:
            raise ConformanceError(
                "turbo LoRA and AdaLN-pruned base must have the same DiT configuration",
                code="artifact_config",
            )
        return self.components[_OVERLAY_COMPONENT[trunk]]


def build_h3_turbo_base(config: Config, plans: Mapping[Trunk, TimestepPlan]) -> OfficialH3Pipeline:
    """The ordinary five-root artifact, with its turbo consumers installed before fill."""
    specs = _dit_specs(_artifact_sections(config))
    if any(structure != "adaln-pruned" for _config, structure, _layout in specs.values()):
        raise ConformanceError("turbo requires an AdaLN-pruned base", code="artifact_config")
    pipe = OfficialH3Pipeline(config, plans)
    for name in _DIT_COMPONENT.values():
        pipe.components[name].install_lora_consumers()
    return pipe


def build_h3_turbo_lora(config: Config, plans: Mapping[Trunk, TimestepPlan]) -> OfficialH3TurboLoRA:
    return OfficialH3TurboLoRA(config, plans)


class OfficialH3RegenerateLoRA:
    """Open-H3-Regenerate-2K's refine overlay: the merged rank-576 LoRA and its refine plan's
    AdaLN tables (LMS Sharp's curve folded in by `minimax-h3-tools/regenerate-lora`), served
    over the FL2VA DiT with its own heads."""

    def __init__(self, config: Config, plan: TimestepPlan) -> None:
        # The checkpoint holds one component, `fl2va_regenerate`, so its construction config is
        # that component's own (the Runtime keys a document by component only for several).
        section = dict(config.mapping())
        extension = section.pop("cozy_h3", None)
        if not isinstance(extension, dict):
            raise ConformanceError("missing Regenerate metadata", code="artifact_config")
        extension = dict(extension)
        layout = TableLayout.parse(extension.pop("table_keys", None))
        expected: dict[str, object] = {
            "task": "fl2va",
            "modulation": "adaln-pruned",
            "distillation": "regenerate",
            "lora_rank": _REGENERATE_RANK,
            "lora_alpha": float(_REGENERATE_RANK),
        }
        if not section or extension != expected:
            raise ConformanceError(
                f"artifact config {REGENERATE_COMPONENT!r} is not the rank-576 Regenerate overlay",
                code="artifact_config",
                fields=[REGENERATE_COMPONENT, "cozy_h3"],
            )
        for schedule in plan.schedules:
            layout.require(schedule.video_timesteps, schedule.audio_timesteps)
        self.plan = plan
        self.spec = section
        overlay = Overlay.from_official_config(
            section,
            rank=_REGENERATE_RANK,
            alpha=float(_REGENERATE_RANK),
            bank=REGENERATE_BANK,
            schedule=None,
            table_timesteps=layout.timesteps,
            table_block_keys=layout.block_keys,
            block_table_dtype=torch.bfloat16,
            final_table_dtype=torch.bfloat16,
        )
        self.components = {REGENERATE_COMPONENT: overlay.eval()}

    def overlay(self, base: OfficialH3Pipeline) -> Overlay:
        config, structure, _layout = base._dit_specs["fl2va"]
        if structure != "adaln-pruned" or config != self.spec:
            raise ConformanceError(
                "the Regenerate overlay and the AdaLN-pruned base must have the same DiT configuration",
                code="artifact_config",
            )
        return self.components[REGENERATE_COMPONENT]


def build_h3_regenerate_lora(config: Config, plan: TimestepPlan) -> OfficialH3RegenerateLoRA:
    return OfficialH3RegenerateLoRA(config, plan)


def build_h3_pipeline(config: Config, plans: Mapping[Trunk, TimestepPlan]) -> OfficialH3Pipeline:
    return OfficialH3Pipeline(config, plans)


def _artifact_sections(config: Config) -> Config:
    present = set(config.keys())
    expected = _WEIGHTED_CONFIG_SECTIONS
    if present != expected:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        raise ConformanceError(
            f"artifact config sections differ from the uniform dual FULL contract: "
            f"missing={missing}, unexpected={unexpected}",
            code="artifact_config",
            fields=[*missing, *unexpected],
        )
    return config


def _dit_spec(config: Config, task: Trunk) -> tuple[dict[str, Any], str, TableLayout | None]:
    component = _DIT_COMPONENT[task]
    upstream = config.section(component).mapping()
    extension = upstream.pop("cozy_h3", None)
    if not isinstance(extension, Mapping):
        raise ConformanceError(
            f"artifact config {component!r} has no closed cozy_h3 structure",
            code="artifact_config",
            fields=[component, "cozy_h3"],
        )
    structure = extension.get("modulation")
    if structure not in {"full", "adaln-pruned"}:
        raise ConformanceError(
            f"artifact config {component!r} has unknown modulation structure {structure!r}",
            code="artifact_config",
            fields=[component, "cozy_h3", "modulation"],
        )
    # FULL computes modulation from live weights. Pruned row identities belong to
    # the checkpoint, so adding or reordering table rows needs no package release.
    fields = {"task", "modulation"}
    if structure == "adaln-pruned":
        if "table_keys" not in extension:
            raise ConformanceError(
                f"model component {component!r} is missing its modulation table row labels; "
                "regenerate the lane with minimax-h3-tools/bf16-pruned or fp8-pruned",
                code="artifact_config",
                fields=[component, "cozy_h3", "table_keys"],
            )
        fields.add("table_keys")
    if structure == "adaln-pruned" and "generating_projection_digest" in extension:
        digest = extension["generating_projection_digest"]
        if (
            not isinstance(digest, str)
            or len(digest) != 71
            or not digest.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in digest[7:])
        ):
            raise ConformanceError(
                "AdaLN generating projection binding is not an exact digest",
                code="artifact_config",
                fields=[component, "cozy_h3", "generating_projection_digest"],
            )
        fields.add("generating_projection_digest")
    if set(extension) != fields:
        raise ConformanceError(
            f"artifact config {component!r} has no closed cozy_h3 structure",
            code="artifact_config",
            fields=[component, "cozy_h3"],
        )
    expected: dict[str, str] = {"task": task}
    for name, want in expected.items():
        if extension[name] != want:
            raise ConformanceError(
                f"artifact config {component!r} {name!r} is {extension[name]!r}, expected {want!r}",
                code="artifact_config",
                fields=[component, "cozy_h3", name],
            )
    layout = TableLayout.parse(extension["table_keys"]) if structure == "adaln-pruned" else None
    return upstream, str(structure), layout


def _dit_specs(config: Config) -> dict[Trunk, tuple[dict[str, Any], str, TableLayout | None]]:
    specs = {trunk: _dit_spec(config, trunk) for trunk in _TRUNKS}
    if len({spec[1] for spec in specs.values()}) != 1:
        raise ConformanceError(
            "FL2VA and Ref2VA DiTs use different modulation structures",
            code="artifact_config",
            fields=["transformer", "transformer_ref"],
        )
    if specs["fl2va"][0] != specs["ref2va"][0]:
        raise ConformanceError(
            "FL2VA and Ref2VA DiTs do not carry the same upstream architecture config",
            code="artifact_config",
            fields=["transformer", "transformer_ref"],
        )
    return specs


def _build_dit(config: Mapping[str, Any], structure: str, layout: TableLayout | None) -> Any:
    if structure == "full":
        transformer = MiniMaxH3Transformer3DModel.from_config(dict(config))
    else:
        if layout is None:
            raise ConformanceError(
                "pruned model is missing its table row labels", code="artifact_config"
            )
        transformer = AdaLNPrunedMiniMaxH3Transformer.from_official_config(
            config,
            table_timesteps=layout.timesteps,
            table_block_keys=layout.block_keys,
        )
    return _apply_transformer_dtype(transformer)


def _overlay_spec(
    config: Config, trunk: Trunk, plan: TimestepPlan
) -> tuple[dict[str, Any], TableLayout]:
    """Only the released rank-64 PDD-8 adapter is supported."""
    component = _OVERLAY_COMPONENT[trunk]
    section = config.section(component).mapping()
    extension = section.pop("cozy_h3", None)
    if not isinstance(extension, dict):
        raise ConformanceError("missing PDD metadata", code="artifact_config")
    extension = dict(extension)
    layout = TableLayout.parse(extension.pop("table_keys", None))
    pdd_steps, pdd_block = 32, 4
    expected: dict[str, object] = {
        "task": trunk,
        "modulation": "adaln-pruned",
        "distillation": _DISTILLATION,
        "lora_rank": 64,
        "lora_alpha": 64.0,
        "pdd_num_steps": pdd_steps,
        "pdd_block_size": pdd_block,
    }
    if not section or extension != expected:
        raise ConformanceError(
            f"artifact config {component!r} is not the released rank-64 PDD-8 overlay",
            code="artifact_config",
            fields=[component, "cozy_h3"],
        )
    _validate_plan(plan, trunk, layout)
    if plan.steps != (pdd_steps // pdd_block,):
        raise ConformanceError(
            "the PDD overlay's timestep plan must hold its one distilled schedule",
            code="artifact_config",
        )
    return section, layout


def _build_overlay(
    config: Mapping[str, Any], plan: TimestepPlan, layout: TableLayout
) -> Overlay:
    (schedule,) = plan.schedules
    timesteps, block_keys = layout.timesteps, layout.block_keys
    overlay = Overlay.from_official_config(
        config,
        rank=64,
        alpha=64.0,
        bank=TURBO_BANK,
        schedule=TurboSchedule(schedule.video_timesteps, schedule.audio_timesteps),
        table_timesteps=timesteps,
        table_block_keys=block_keys,
        block_table_dtype=torch.bfloat16,
        final_table_dtype=torch.bfloat16,
    )
    overlay.eval()
    return overlay


def _validate_dual_topology(members: Mapping[Trunk, Any], names: Mapping[Trunk, str]) -> None:
    """Prove one architecture instantiated twice for two independent weight identities."""
    fl2va, ref2va = members["fl2va"], members["ref2va"]
    fields = [names["fl2va"], names["ref2va"]]
    if fl2va is ref2va:
        raise ConformanceError(
            "FL2VA and Ref2VA component names alias one module instance",
            code="artifact_config",
            fields=fields,
        )

    def topology(module: Any) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        return tuple(
            # Row count follows each checkpoint's labels, independently per trunk.
            # The remaining axes still prove the same modulation architecture.
            (
                name,
                tuple(
                    int(value)
                    for value in (
                        tensor.shape[1:]
                        if name == "norm_out.table" or name.endswith(".adaln_proj.table")
                        else tensor.shape
                    )
                ),
                str(tensor.dtype),
            )
            for name, tensor in module.state_dict().items()
        )

    if type(fl2va) is not type(ref2va) or topology(fl2va) != topology(ref2va):
        raise ConformanceError(
            "FL2VA and Ref2VA components do not have one identical class and destination topology",
            code="artifact_config",
            fields=fields,
        )


def _validate_dual_dit_topology(dits: Mapping[Trunk, Any]) -> None:
    _validate_dual_topology(dits, _DIT_COMPONENT)


def _apply_transformer_dtype(transformer: Any) -> Any:
    """Reproduce Diffusers' mixed FULL compute policy on Runtime destinations."""
    # Cast each root directly to its final dtype. In particular, RoPE frequencies
    # are real config-derived buffers: BF16 followed by FP32 cannot restore them.
    for name, component in transformer.named_children():
        dtype = torch.float32 if name in transformer._keep_in_fp32_modules else torch.bfloat16
        component.to(dtype=dtype)
    return transformer.eval()


def _apply_video_vae_dtype(vae: Any, config: Config | None = None) -> Any:
    """Construct decode operands in their checkpoint dtype before Runtime's census.

    Both original FP32 and pre-rounded FP16 matrix weights run under the same FP16
    decode autocast. Loading must preserve their supplied logical dtype and price the
    actual residency, rather than requiring a checkpoint rewrite. Encoder, norms,
    biases, register tokens and config-derived RoPE retain their FP32 contract.
    Config-only construction defaults to the original FP32 architecture.
    """
    config = config or Config({})
    weights = [
        (f"decoder.{name}", parameter)
        for name, parameter in vae.decoder.named_parameters()
        if name.rsplit(".", 1)[-1] == "weight" and parameter.dim() >= 2
    ]
    weights.append(("post_quant_conv.weight", vae.post_quant_conv.weight))
    supported = {"f32": torch.float32, "f16": torch.float16}
    for name, parameter in weights:
        key = f"vae.{name}"
        dtype = config.tensor_dtype(key, default="f32")
        if dtype not in supported:
            raise ConformanceError(
                f"{key} uses unsupported dtype {dtype!r}; expected f32 or f16",
                code="artifact_dtype",
                fields=[key],
            )
        parameter.data = parameter.data.to(dtype=supported[dtype])
    return vae.eval()


def _validate_model_contract(pipe: Any, transformer: Any, video_vae: Any, audio_vae: Any) -> None:
    """Refuse shape-preserving scalar drift in the one official release geometry."""
    contracts = (
        (
            "pipeline",
            pipe.config,
            {"canvas_short_edge": 768, "canvas_max_pixels": 768 * 1344},
        ),
        (
            "transformer",
            transformer.config,
            {
                "num_attention_heads": 56,
                "attention_head_dim": 128,
                "hidden_size": 5376,
                "num_layers": 50,
                "num_refiner_layers": 2,
                "ffn_dim": 14336,
                "in_channels": 24,
                "audio_in_channels": 32,
                "patch_size": PATCH_SIZE,
                "text_dim": 5120,
                "freq_dim": 256,
                "time_embed_hidden_dim": 5376,
                "time_embed_dim": 2688,
                "rope_freq_dim": 16,
                "rope_theta": 10000.0,
                "norm_eps": 1e-5,
                "qk_norm_eps": 1e-5,
                "final_norm_eps": 1e-5,
            },
        ),
        (
            "vae",
            video_vae.config,
            {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 24,
                "spatial_downsample_factors": (2, 2, 2, 2, 1, 1),
                "temporal_downsample_factors": (1, 2, 2, 1, 1, 1),
                "clip_length": FRAMES_PER_CHUNK,
                "token_drop": 3,
            },
        ),
        (
            "audio_vae",
            audio_vae.config,
            {
                "encoder_rates": (2, 4, 4, 5, 5),
                "latent_dim": 2048,
                "latent_channels": 32,
                "decoder_rates": (5, 5, 2, 2, 2, 2, 2),
                "sampling_rate": 32000,
            },
        ),
    )
    for component, config, expected in contracts:
        for name, want in expected.items():
            got = config.get(name)
            if isinstance(want, tuple) and got is not None:
                got = tuple(got)
            if got != want:
                raise ConformanceError(
                    f"official H3 {component} config {name!r} is {got!r}, expected {want!r}",
                    code="artifact_config",
                    fields=[component, name],
                )
    if pipe.config.get("reference_image_short_edge", 2048) != 2048:
        raise ConformanceError(
            "official H3 reference image short edge differs from 2048",
            code="artifact_config",
            fields=["pipeline", "reference_image_short_edge"],
        )
    if (
        video_vae.tokens_chunk_size != LATENTS_PER_CHUNK
        or video_vae.spatial_compression_ratio != VAE_SPATIAL_RATIO
    ):
        raise ConformanceError(
            "official H3 video VAE derived geometry differs from 5-token/16-pixel release",
            code="artifact_config",
            fields=["vae"],
        )
    # These four are hard-coded properties of the installed library, so this can only fire
    # when the DEPENDENCY moves — never on artifact drift, which is why it is not an
    # `artifact_config` refusal. It is the tripwire under `_CEILING_S`: the override is
    # calibrated against a ceiling of exactly 15.0, and a library that restated its bounds
    # must be re-read before we keep overriding them.
    envelope = (pipe.fps, pipe.min_duration, pipe.max_duration, pipe.audio_channels)
    if envelope != (FPS, MIN_DURATION, MAX_DURATION, AUDIO_CHANNELS):
        raise ConformanceError(
            f"installed Diffusers states the H3 clip envelope as {envelope}, expected "
            f"{(FPS, MIN_DURATION, MAX_DURATION, AUDIO_CHANNELS)}",
            code="dependency_drift",
            fields=["pipeline"],
        )


def _audio_tensor(audio: DecodedAudio) -> Any:
    """`(channels, samples)` float32, one copy of the PCM."""
    return torch.from_numpy(
        np.stack([np.frombuffer(channel, dtype="<f4") for channel in audio.pcm_f32le])
    )


def _resample(waveform: Any, rate: int, target: int) -> Any:
    """Bit-identical port of `torchaudio.transforms.Resample(rate, target)` defaults.

    The official pipeline resamples reference audio with torchaudio, which has no release for
    the Torch this Runtime serves, so references reach it already at the audio VAE rate.
    Resampling before the pipeline's duration cut differs only within the filter span there.
    """
    if rate == target:
        return waveform
    gcd = math.gcd(rate, target)
    rate, target = rate // gcd, target // gcd
    base = min(rate, target) * 0.99
    width = math.ceil(6 * rate / base)
    idx = torch.arange(-width, width + rate, dtype=torch.float64)[None, None] / rate
    t = torch.arange(0, -target, -1, dtype=torch.float32)[:, None, None] / target + idx
    t *= base
    t = t.clamp_(-6, 6)
    window = torch.cos(t * math.pi / 6 / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernel *= window * (base / rate)
    channels, length = waveform.shape
    padded = torch.nn.functional.pad(waveform, (width, width + rate))
    out = torch.nn.functional.conv1d(padded[:, None], kernel.to(torch.float32), stride=rate)
    return out.transpose(1, 2).reshape(channels, -1)[:, : math.ceil(target * length / rate)]


@dataclass
class MotionVideo(MiniMaxH3VideoReference):
    """A reference clip whose real frames, `motion`, are already at the size the call uses."""

    motion: Any = None


@dataclass(frozen=True, slots=True)
class ReferenceClip:
    """A clip as the call uses it: `(T, H, W, 3)` uint8 frames on the 24-fps clock at its
    `clip_size`, the source geometry its canvas follows, and its soundtrack if kept."""

    frames: Any
    width: int
    height: int
    duration: Fraction
    soundtrack: Any | None = None
    sample_rate: int = 0


def clip_size(width: int, height: int, *, guide: bool) -> tuple[int, int]:
    """The (height, width) a clip is read at: the canvas its aspect resolves to, which the
    target shares, or for a motion clip that canvas's `motion_size`."""
    canvas = tuple(int(value) for value in resolve_canvas_size(width, height, *CANVAS))
    return canvas if guide else motion_size(*canvas)


def read_reference_video(
    events: Iterable[Any], *, frames: int, audio: bool, guide: bool = False
) -> ReferenceClip:
    """Decode a reference clip once, straight to the size the call uses.

    Each frame shows until the next starts (the last for its own duration) and lands on the
    whole-frame 24-fps clock, resized once (antialiased bicubic) as it arrives to its
    `clip_size`. Only the call's first `frames` are kept
    and no source-resolution frame outlives its event. With `audio`, the soundtrack is kept on
    the video's origin, to the nearest sample of its own clock.
    """
    out: np.ndarray | None = None
    kept = 0
    origin = end = Fraction(0)
    held: tuple[Fraction, np.ndarray | None] | None = None
    first: DecodedVideoFrame | None = None
    pcm: list[list[bytes]] = []
    audio_start: Fraction | None = None
    rate = 0

    def place(picture: np.ndarray | None, stop: Fraction) -> None:
        nonlocal kept
        assert out is not None
        upto = min(_round_fraction((stop - origin) * FPS), frames)
        while picture is not None and kept < upto:
            out[kept] = picture
            kept += 1

    for event in events:
        if isinstance(event, DecodedAudioChunk):
            if audio:
                if audio_start is None:
                    audio_start, rate, pcm = event.start_time, event.sample_rate, [[] for _ in event.pcm_f32le]
                for channel, chunk in zip(pcm, event.pcm_f32le, strict=True):
                    channel.append(chunk)
            continue
        if not isinstance(event, DecodedVideoFrame):
            continue
        start = event.start_time
        if first is None:
            first, origin = event, start
            size = clip_size(event.width, event.height, guide=guide)
            out = np.empty((frames, *size, 3), dtype=np.uint8)
        elif held is not None and start < held[0]:
            raise ConformanceError("reference video presentation time runs backwards",
                                   code="reference_policy")
        if held is not None:
            place(held[1], start)
        picture = None
        if _round_fraction((start - origin) * FPS) < frames:
            assert out is not None
            rgb = torch.frombuffer(bytearray(event.rgb), dtype=torch.uint8).view(event.height, event.width, 3)
            if rgb.shape[:2] != out.shape[1:3]:
                chw = rgb.permute(2, 0, 1)[None]
                rgb = torch.nn.functional.interpolate(chw, size=out.shape[1:3], mode="bicubic", antialias=True)
                rgb = rgb[0].permute(1, 2, 0)
            picture = rgb.numpy()
        held = (start, picture)
        end = start + event.duration * event.time_base
    if first is None or held is None or out is None:
        raise ConformanceError("reference video has no frames", code="reference_policy")
    end = min(end, origin + Fraction(frames, FPS))
    place(held[1], end)
    if kept == 0:
        raise ConformanceError("reference video has no frame on the 24-fps clock", code="reference_policy")
    soundtrack = None
    if audio_start is not None:
        waveform = torch.from_numpy(np.stack([np.frombuffer(b"".join(c), dtype="<f4") for c in pcm]))
        offset = _round_fraction((audio_start - origin) * rate)
        target = _round_fraction((end - origin) * rate)
        lo, hi = max(offset, 0), min(offset + int(waveform.shape[1]), target)
        if lo >= hi:
            raise ConformanceError("reference soundtrack has no samples on its video's timeline",
                                   code="reference_policy")
        soundtrack = torch.zeros((len(pcm), target), dtype=torch.float32)
        soundtrack[:, lo:hi] = waveform[:, lo - offset : hi - offset]
    return ReferenceClip(out[:kept], first.width, first.height, end - origin, soundtrack, rate)


def _round_fraction(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _float_tuple(tensor: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor.detach().float().cpu())


def _float32_digest(values: Sequence[float]) -> str:
    raw = b"".join(struct.pack("<f", value) for value in values)
    return hashlib.sha256(raw).hexdigest()


def _as_float32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def _float_hex(value: float) -> str:
    return _as_float32(value).hex()


def _modulation_class(
    name: str,
    timestep: float,
    modality: str,
    tag: int,
    presence: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "timestep": _float_hex(timestep),
        "modality": modality,
        "modality_tag": tag,
        "presence": presence,
    }
