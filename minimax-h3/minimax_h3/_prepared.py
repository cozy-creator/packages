"""Resident H3 intervals, split only at actual row geometry or attention-policy changes.

Conditioning and initial RNG stay in the ordinary pipeline. One native sequence owns all
segments and advances both video/audio latents without Python decisions between steps.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.models.modeling_utils import get_parameter_dtype

from cozy_runtime import compute
from cozy_runtime.author import CapabilityError, Context, OutputError, Telemetry, attention_scope

from .adaln_pruned import AdaLNPrunedMiniMaxH3Transformer


@dataclass(frozen=True)
class Segment:
    first: int
    end: int
    dense: bool
    unique_rows: int


def segments(row_plan: list[Any], dense_steps: int, *, start: int = 0) -> tuple[Segment, ...]:
    """Keep upstream's variable unique-timestep shapes; never pad or change row semantics."""
    result: list[Segment] = []
    previous = None
    for index in range(start, len(row_plan)):
        times, indices = row_plan[index]
        shape = (index < dense_steps, times.numel(), tuple(indices.shape))
        if result and shape == previous:
            old = result[-1]
            result[-1] = Segment(old.first, index + 1, old.dense, old.unique_rows)
        else:
            result.append(Segment(index, index + 1, shape[0], shape[1]))
        previous = shape
    return tuple(result)


class Euler:
    """Exact MiniMaxH3Scheduler.step order for a static segment of each modality."""

    def __init__(self, scheduler: Any, first: int, end: int, device: Any) -> None:
        self.timesteps = scheduler.timesteps[first:end].to(device=device).clone()
        self.sigmas = scheduler.sigmas[first : end + 1].to(device=device).clone()

    def step(self, index: torch.Tensor, sample: torch.Tensor, velocity: torch.Tensor) -> None:
        # Upstream's loop casts the predicted velocity to float32 before scheduler.step.
        velocity = velocity.float()
        timestep = self.timesteps.index_select(0, index).reshape(())
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        sigma_from_timestep = sigma_from_timestep.reshape((1,) * sample.ndim)
        denoised = sample + sigma_from_timestep * velocity
        dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas.index_select(0, index).reshape(()).to(dtype=dtype)
        following = self.sigmas.index_select(0, index + 1).reshape(()).to(dtype=dtype)
        ratio = following / sigma
        result = ratio * sample.to(dtype=dtype) + (1.0 - ratio) * denoised.to(dtype=dtype)
        sample.copy_(result.to(dtype=sample.dtype))


def prepare(
    transformer: AdaLNPrunedMiniMaxH3Transformer,
    state: Any,
    video_scheduler: Any,
    audio_scheduler: Any,
    layout_at: Callable[[int], Any],
    dense_steps: int,
    *,
    start: int = 0,
) -> Any:
    """Prepare without mutating the caller's latents; Runtime owns every graph dependency."""
    sequence_type = getattr(compute, "PreparedSequence", None)
    if sequence_type is None or not callable(getattr(sequence_type, "record", None)):
        raise CapabilityError(
            "native stage sequences unavailable", code="prepared_graph_ineligible"
        )
    if (
        not isinstance(transformer, AdaLNPrunedMiniMaxH3Transformer)
        or transformer._lora_hooks_installed
    ):
        raise CapabilityError(
            "native H3 requires the base AdaLN tables", code="prepared_graph_ineligible"
        )
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise CapabilityError(
            "native H3 is currently single-rank", code="prepared_graph_ineligible"
        )
    if not state.latents.is_cuda or not state.audio_latents.is_cuda:
        raise CapabilityError("native H3 requires CUDA latents", code="prepared_graph_ineligible")
    # The package's Kitchen producers carry Python previous-step summary state. A
    # repeated graph needs explicit GPU carry before those implementations are eligible.
    from cozy_runtime.internal.producers import Processor

    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if isinstance(processor, Processor):
            from .producer import PRODUCERS

            selected = processor._attention_backend
            active = str(getattr(selected, "value", selected)) == processor.backend
            if active and PRODUCERS.get(processor.producer.name) is processor.producer:
                raise CapabilityError(
                    "H3 Kitchen producer previous-step summaries need native GPU carry",
                    code="prepared_graph_ineligible",
                )
    if state.attention_kwargs:
        raise CapabilityError(
            "native H3 requires fixed base attention inputs", code="prepared_graph_ineligible"
        )
    row_plan = state.row_timestep_plan
    total = len(row_plan)
    if (
        not total
        or type(start) is not int
        or not 0 <= start < total
        or len(video_scheduler.sigmas) != total + 1
        or len(audio_scheduler.sigmas) != total + 1
    ):
        raise CapabilityError(
            "native H3 requires complete dual schedules", code="prepared_graph_ineligible"
        )
    from cozy_runtime.author._model import PLACED

    placed = PLACED.get(transformer)
    if placed is not None:
        binding = placed[0].retain(placed[1], ("",))
        try:
            if not binding.resident():
                raise CapabilityError(
                    "native H3 needs resident weights before segment preparation",
                    code="prepared_graph_ineligible",
                )
        finally:
            binding.close()
    fields = state.denoiser_input_fields
    names = ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")
    if set(fields) != set(names):
        raise CapabilityError(
            "native H3 layout differs from the fixed base contract",
            code="prepared_graph_ineligible",
        )
    token_tags, position_ids, video_indices, audio_indices, text_indices = (
        fields[n] for n in names
    )
    table_rows = []
    # The same validation used by every ordinary forward, once for every exact scheduled
    # row before capture. Unknown timesteps/modalities remain errors, never placeholders.
    for times, indices in row_plan:
        transformer._arm(
            transformer,
            (),
            {
                "timestep": times,
                "timestep_indices": indices,
                "token_tags": token_tags,
                "attention_kwargs": {},
            },
        )
        try:
            table_rows.append(transformer.time_proj.rows(times))
        finally:
            transformer._release()
    with torch.inference_mode(False), torch.no_grad():
        video, audio = state.latents.clone(), state.audio_latents.clone()
        with attention_scope(layout_at(start)):
            rotary = transformer.rope(position_ids)
            text = transformer.token_refiner(
                transformer.context_embedder(
                    state.prompt_embeds.to(get_parameter_dtype(transformer.context_embedder))
                )
            )
    pieces = segments(row_plan, dense_steps, start=start)
    prepared = []
    with torch.inference_mode(False), torch.no_grad():
        for part in pieces:
            prepared.append(
                (
                    part,
                    torch.stack(table_rows[part.first : part.end]),
                    torch.stack([row_plan[i][1] for i in range(part.first, part.end)]),
                    Euler(video_scheduler, part.first, part.end, video.device),
                    Euler(audio_scheduler, part.first, part.end, audio.device),
                )
            )
    with sequence_type.record() as recording:
        plans = []
        for part, rows, indices, video_euler, audio_euler in prepared:
            body = step_body(
                transformer,
                video,
                audio,
                text,
                rotary,
                rows,
                indices,
                token_tags,
                video_indices,
                audio_indices,
                text_indices,
                video_euler,
                audio_euler,
                state.num_condition_video_rows,
                state.num_condition_audio_rows,
            )
            # Policy uses the original global step, even when an ordinary prefix ran.
            # Runtime validates kernel/fill identities for every prepared definition.
            with attention_scope(layout_at(part.first)):
                plan = recording.capture(
                    body,
                    state=(video, audio),
                    steps=part.end - part.first,
                    owners=(
                        transformer,
                        text,
                        *rotary,
                        rows,
                        indices,
                        token_tags,
                        video_indices,
                        audio_indices,
                        text_indices,
                        video_euler.timesteps,
                        video_euler.sigmas,
                        audio_euler.timesteps,
                        audio_euler.sigmas,
                    ),
                    resident_only=True,
                )
            plans.append(plan)
        if not all(plan.valid() for plan in plans):
            raise CapabilityError(
                "H3 weights or selected kernels changed during segment preparation",
                code="prepared_graph_ineligible",
            )
        return recording.finish(outputs=(video, audio)), pieces


def step_body(
    transformer: Any,
    video: torch.Tensor,
    audio: torch.Tensor,
    text: torch.Tensor,
    rotary: Any,
    rows: torch.Tensor,
    indices: torch.Tensor,
    token_tags: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    video_euler: Euler,
    audio_euler: Euler,
    video_prefix: int,
    audio_prefix: int,
) -> Callable[[torch.Tensor], None]:
    def step(index: torch.Tensor) -> None:
        velocity, audio_velocity = transformer.forward_prepared(
            video[None],
            audio[None],
            text,
            rotary,
            rows.index_select(0, index).squeeze(0),
            indices.index_select(0, index).squeeze(0),
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
        )
        video_euler.step(index, video[video_prefix:], velocity[0, video_prefix:])
        audio_euler.step(index, audio[audio_prefix:], audio_velocity[0, audio_prefix:])

    return step


def run(
    transformer: Any,
    state: Any,
    video_scheduler: Any,
    audio_scheduler: Any,
    layout_at: Callable[[int], Any],
    dense_steps: int,
    ctx: Context,
    tel: Telemetry,
    *,
    start: int = 0,
) -> bool:
    began = time.perf_counter()
    try:
        plan, pieces = prepare(
            transformer,
            state,
            video_scheduler,
            audio_scheduler,
            layout_at,
            dense_steps,
            start=start,
        )
    except CapabilityError as error:
        if error.code != "prepared_graph_ineligible":
            raise
        tel.log("eager H3", reason=str(error))
        return False
    except torch.OutOfMemoryError:
        tel.log("eager H3", reason="resident sequence preparation exceeded capacity")
        return False
    tel.metric("request_path.h3_prepare_ms", (time.perf_counter() - began) * 1000)
    tel.log(
        "H3 native segments",
        segments=";".join(
            f"{p.first}:{p.end}:{'dense' if p.dense else 'sparse'}:{p.unique_rows}" for p in pieces
        ),
    )
    try:
        execution = ctx.plan(plan)
        observed = -1
        while True:
            ctx.raise_if_cancelled()
            progress = execution.poll()
            if progress.completed_steps != observed:
                observed = progress.completed_steps
                tel.progress(
                    (start + observed) / len(state.row_timestep_plan),
                    stage="denoise",
                    overall_fraction=0.15 + 0.7 * (start + observed) / len(state.row_timestep_plan),
                )
            if progress.done:
                break
            time.sleep(0.05)
        video, audio = execution.wait()
        if progress.canceled or progress.completed_steps != len(state.row_timestep_plan) - start:
            raise OutputError(
                "native H3 stopped before all scheduled steps", code="incomplete_denoise"
            )
        state.latents, state.audio_latents = video, audio
        tel.metric("request_path.h3_eager_prefix_steps", float(start))
        tel.metric("request_path.h3_native_steps", float(progress.completed_steps))
        tel.metric("request_path.h3_native_submissions", 1.0)
        tel.metric(
            "request_path.h3_host_graph_launches", float(plan.facts()["host_graph_launches"])
        )
        return True
    finally:
        plan.close()
