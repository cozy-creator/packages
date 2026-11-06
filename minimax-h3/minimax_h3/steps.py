"""H3's denoise as an `author.denoise_loop`: FL2VA or ref2va, full or turbo; the Runtime runs it
eagerly or natively.

The step is the official loop step (`MiniMaxH3LoopDenoiser`, then `MiniMaxH3LoopSchedulerStep`)
over tensors alone. It reads its timesteps and sigmas from tables at the step index, and takes
every host decision from its label: its attention layout, how many distinct timesteps its rows
hold, and whether the Kitchen producer's sites reuse the previous
step's key means, which it carries in one slot per site (`producer.Carry`); the last dense step
seeds them. Across ranks the loop runs eagerly: each step has its own layout and the producer
decides as a step loop does. The loop reaches every per-call tensor through its inputs, so the
Runtime keeps its program for the next call of the same key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from cozy_runtime import stepcache
from cozy_runtime.author import AttentionLayout, attention_scope, denoise_loop, producer_sites

from . import producer
from .adaln_pruned import step_invariants
from .overlay import ATTENTION_KWARG, OVERLAY_KWARG


@dataclass(frozen=True, slots=True)
class Label:
    layout: AttentionLayout
    rows: int
    rolling: bool
    seeding: bool


def denoise(
    transformer: Any,
    schedulers: Any,
    state: Any,
    layout_at: Callable[[int], AttentionLayout],
    dense_steps: int,
    on_step: Callable[[int], None],
    *,
    trunk: str,
    selector: Mapping[str, Any] | None = None,
    clips: Any = None,
    cache: stepcache.Policy | None = None,
) -> None:
    """One rank's denoise of the loop's block `state` on its `trunk`'s DiT, its latents advanced in
    place: under the turbo bank `selector` names, its overlay armed before the loop and each
    step's head read from a table at its index, and under the preset's step `cache`. A ref2va's clip rows (`clips`)
    leave before its first sparse step, so its sparse steps are a second loop over the rows left."""
    overlay = (selector or {}).get(OVERLAY_KWARG)
    steps = len(state.row_timestep_plan)
    parallel = getattr(transformer, "_parallel_config", None) is not None
    with torch.no_grad(), step_invariants() as memo:
        sites = () if parallel else producer_sites(transformer)
        slots = tuple(torch.zeros(s.module.heads, 128, device=state.latents.device) for s in sites)
        parts = [range(steps)]
        if clips is not None and dense_steps < steps and bool((clips.frame >= 0).any()):
            parts = [range(dense_steps), range(dense_steps, steps)]
        for part in parts:
            if part.start:
                clips.release(state)
            _loop(
                transformer, schedulers, state, layout_at, dense_steps, on_step, part, memo,
                trunk=trunk, selector=selector, overlay=overlay, sites=sites, slots=slots,
                parallel=parallel, cache=cache,
            )  # fmt: skip
    state.noise_pred = state.audio_noise_pred = None  # the official loop's outputs; none reads them


def _loop(
    transformer: Any,
    schedulers: Any,
    state: Any,
    layout_at: Callable[[int], AttentionLayout],
    dense_steps: int,
    on_step: Callable[[int], None],
    part: range,
    memo: Any,
    *,
    trunk: str,
    selector: Mapping[str, Any] | None,
    overlay: Any,
    sites: Sequence[Any],
    slots: tuple[torch.Tensor, ...],
    parallel: bool,
    cache: stepcache.Policy | None,
) -> None:
    """The steps `part` of `state`'s plan as one `denoise_loop`. Their rows are checked against
    the tables they read, and each step's turbo head found, before it (`_arm`): its steps read
    nothing on the host. A part after the first starts its producer slots unwritten (`producer.present`)."""
    plan = state.row_timestep_plan
    fields = state.denoiser_input_fields
    position_ids, tags = fields["position_ids"], fields["token_tags"]
    device = state.latents.device
    for index in part:  # an uncovered row refuses here, never inside the loop
        transformer._arm(*plan[index], tags, selector)
    transformer._release()
    rotary = memo.get("rope", (position_ids,), lambda: transformer.rope(position_ids))
    producer.present(sites, rotary, tags.numel())
    timesteps = torch.zeros(len(plan), max(t.numel() for t, _ in plan), device=device)
    for index, (times, _) in enumerate(plan):
        timesteps[index, : times.numel()] = times
    indices = torch.stack([rows for _, rows in plan])
    heads = overlay is not None and overlay.heads
    table = torch.tensor(
        [overlay.schedule.step(times) if heads else 0 for times, _ in plan], device=device
    )
    labels = tuple(
        Label(
            layout_at(index if parallel else dense_steps if index >= dense_steps else 0),
            plan[index][0].numel(),
            bool(sites) and index >= max(dense_steps, 1) and index > part.start,
            bool(sites) and index == dense_steps - 1,
        )
        for index in part
    )
    keys = tuple(id(site.module) for site in sites)
    video_rows, audio_rows = state.num_condition_video_rows, state.num_condition_audio_rows
    inputs = (
        state.prompt_embeds, timesteps, indices, tags, position_ids, fields["video_indices"],
        fields["audio_indices"], fields["text_indices"], state.timesteps,
        state.audio_timesteps, schedulers.scheduler.sigmas, schedulers.audio_scheduler.sigmas,
        table,
    )  # fmt: skip

    def step(i: torch.Tensor, video: Any, audio: Any, *rest: Any, label: Label) -> None:
        means, given = rest[: len(keys)], rest[len(keys) :]
        prompt, times, rows, tags, positions, at_video, at_audio, at_text, *schedule = given
        video_t, audio_t, video_sigmas, audio_sigmas, table = schedule
        if heads:
            transformer._engage(overlay, table.index_select(0, i))
        carry = producer.Carry(dict(zip(keys, means, strict=True)), label.rolling, label.seeding)
        with attention_scope(label.layout), producer.carrying(carry) if keys else nullcontext():
            velocity, audio_velocity = transformer(
                hidden_states=video[None],
                audio_hidden_states=audio[None],
                encoder_hidden_states=prompt,
                timestep=times.index_select(0, i)[0, : label.rows],
                timestep_indices=rows.index_select(0, i)[0],
                token_tags=tags,
                position_ids=positions,
                video_indices=at_video,
                audio_indices=at_audio,
                text_indices=at_text,
                return_dict=False,
            )
        velocity, audio_velocity = velocity[0, video_rows:], audio_velocity[0, audio_rows:]
        euler(video[video_rows:], velocity.float(), i, video_t, video_sigmas)
        euler(audio[audio_rows:], audio_velocity.float(), i, audio_t, audio_sigmas)

    if overlay is not None:  # armed before the loop, as `denoise_loop` requires
        transformer._engage(overlay, table[part.start])
    try:
        denoise_loop(
            step,
            part,
            state=(state.latents, state.audio_latents),
            carried=slots,
            inputs=inputs,
            labels=labels,
            key=("minimax-h3.denoise/1", trunk, video_rows, audio_rows),
            on_step=on_step,
            cache=cache,
        )
    finally:
        transformer._release()


def denoise_refine(
    transformer: Any,
    overlay: Any,
    *,
    video: torch.Tensor,
    audio: torch.Tensor,
    prompt_embeds: torch.Tensor,
    fields: dict[str, torch.Tensor],
    plan: list[tuple[torch.Tensor, torch.Tensor]],
    sigmas: torch.Tensor,
    windows: list[tuple[int, int]],
    weights: list[torch.Tensor],
    dense_steps: int,
    layout_at: Callable[[int], AttentionLayout],
    on_step: Callable[[int], None],
) -> None:
    """The Regenerate refine (NVlabs `open_h3/pipeline.py` `refine`) as a `denoise_loop`: Euler on
    the video rows `video` alone, every step's velocity from the FL2VA DiT under `overlay`.

    `windows` are equal row spans of `video`, each laid out as a clip of its own (`fields`: H3's
    positions start at every clip's text), so one layout and one rotary table serve them all;
    `weights` are their per-row blend. One span is the whole canvas. `audio` is the clean conditioning rows one window carries; `plan` each step's
    `(timesteps, timestep_indices)`.
    """
    position_ids, tags = fields["position_ids"], fields["token_tags"]
    device = video.device
    selector = {ATTENTION_KWARG: overlay.bank, OVERLAY_KWARG: overlay}
    with torch.no_grad(), step_invariants() as memo:
        for times, rows in plan:  # an uncovered row refuses here, never inside the loop
            transformer._arm(times, rows, tags, selector)
        transformer._release()
        sites = producer_sites(transformer)
        rotary = memo.get("rope", (position_ids,), lambda: transformer.rope(position_ids))
        producer.present(sites, rotary, tags.numel())
        timesteps = torch.stack([times for times, _ in plan]).to(device)
        indices = torch.stack([rows for _, rows in plan]).to(device)
        # The first `dense_steps` run dense (the preset's), the last of them seeding the Kitchen
        # producer's key means. A key mean carried from a step's last window is no other window's:
        # windows compute their own statistics.
        single = bool(sites) and len(windows) == 1
        labels = tuple(
            Label(layout_at(dense_steps if index >= dense_steps else 0), int(times.numel()),
                  single and index >= max(dense_steps, 1), single and index == dense_steps - 1)  # fmt: skip
            for index, (times, _) in enumerate(plan)
        )
        keys = tuple(id(site.module) for site in sites)
        slots = tuple(torch.zeros(site.module.heads, 128, device=device) for site in sites)

        def step(i: torch.Tensor, rows: Any, *means: Any, label: Label) -> None:
            carry = producer.Carry(dict(zip(keys, means, strict=True)), label.rolling, label.seeding)
            velocity = torch.zeros_like(rows)
            for (start, count), weight in zip(windows, weights, strict=True):
                with attention_scope(label.layout), producer.carrying(carry):
                    predicted, _ = transformer(
                        hidden_states=rows[None, start : start + count],
                        audio_hidden_states=audio[None],
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps.index_select(0, i)[0, : label.rows],
                        timestep_indices=indices.index_select(0, i)[0],
                        token_tags=tags,
                        position_ids=position_ids,
                        video_indices=fields["video_indices"],
                        audio_indices=fields["audio_indices"],
                        text_indices=fields["text_indices"],
                        return_dict=False,
                    )
                velocity[start : start + count].addcmul_(predicted[0].float(), weight[:, None])
            delta = sigmas.index_select(0, i) - sigmas.index_select(0, i + 1)
            rows.add_(velocity * delta.reshape(1, 1))

        transformer._engage(overlay, 0)  # armed before the loop, as `denoise_loop` requires
        try:
            denoise_loop(
                step, range(len(plan)), state=(video,), carried=slots, labels=labels, on_step=on_step
            )
        finally:
            transformer._release()


def euler(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    i: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
) -> None:
    """`MiniMaxH3Scheduler.step` (eta 0) at step `i`, in its order of operations, in place."""
    sigma_from_timestep = 1 - timesteps.index_select(0, i).reshape(()).to(sample.dtype)
    denoised = sample + sigma_from_timestep.reshape((1,) * sample.ndim) * velocity
    dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    sigma = sigmas.index_select(0, i).reshape(()).to(dtype)
    ratio = sigmas.index_select(0, i + 1).reshape(()).to(dtype) / sigma
    sample.copy_((ratio * sample.to(dtype) + (1.0 - ratio) * denoised.to(dtype)).to(sample.dtype))
