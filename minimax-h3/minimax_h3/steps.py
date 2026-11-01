"""FL2VA's denoise as an `author.denoise_loop`: the Runtime runs it eagerly or natively.

The step is the official loop step (`MiniMaxH3LoopDenoiser`, then `MiniMaxH3LoopSchedulerStep`)
over tensors alone. It reads its timesteps and sigmas from tables at the step index, and takes
every host decision from its phase: dense or sparse attention, how many distinct timesteps its
rows hold, and whether the Kitchen producer's sites reuse the previous step's key means, which
it carries in one slot per site (`producer.Carry`); the last dense step seeds them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from cozy_runtime.author import AttentionLayout, attention_scope, denoise_loop, producer_sites

from . import producer
from .adaln_pruned import step_invariants
from .overlay import ATTENTION_KWARG, OVERLAY_KWARG


@dataclass(frozen=True, slots=True)
class Phase:
    sparse: bool
    rows: int
    rolling: bool
    seeding: bool


def denoise_fl2va(
    transformer: Any,
    schedulers: Any,
    state: Any,
    layout_at: Callable[[int], AttentionLayout],
    sol_dense_steps: int,
    on_step: Callable[[int], None],
) -> None:
    """One rank's FL2VA denoise of the loop's block `state`, its latents advanced in place."""
    plan = state.row_timestep_plan
    fields = state.denoiser_input_fields
    position_ids, tags = fields["position_ids"], fields["token_tags"]
    device = state.latents.device
    with torch.no_grad(), step_invariants(checked=True) as memo:
        for times, rows in plan:  # an uncovered row refuses here, never inside the loop
            transformer._arm(times, rows, tags, None)
        transformer._release()
        sites = producer_sites(transformer)
        rotary = memo.get("rope", (position_ids,), lambda: transformer.rope(position_ids))
        producer.present(sites, rotary, tags.numel())
        timesteps = torch.zeros(len(plan), max(t.numel() for t, _ in plan), device=device)
        for index, (times, _) in enumerate(plan):
            timesteps[index, : times.numel()] = times
        indices = torch.stack([rows for _, rows in plan])
        phases = tuple(
            Phase(
                index >= sol_dense_steps,
                times.numel(),
                bool(sites) and index >= max(sol_dense_steps, 1),
                bool(sites) and index == sol_dense_steps - 1,
            )
            for index, (times, _) in enumerate(plan)
        )
        layouts = {sparse: layout_at(sol_dense_steps if sparse else 0) for sparse in (False, True)}
        keys = tuple(id(site.module) for site in sites)
        slots = tuple(torch.zeros(site.module.heads, 128, device=device) for site in sites)
        video_rows, audio_rows = state.num_condition_video_rows, state.num_condition_audio_rows
        video_sigmas, audio_sigmas = schedulers.scheduler.sigmas, schedulers.audio_scheduler.sigmas

        # Read-only values of this call are explicit bindings, never captured through state/fields.
        inputs = (
            state.prompt_embeds, timesteps, indices, tags, position_ids,
            fields["video_indices"], fields["audio_indices"], fields["text_indices"],
            state.timesteps, state.audio_timesteps, video_sigmas, audio_sigmas,
        )
        site_layout = tuple(
            (site.component, site.path, site.producer.name, site.producer.base, int(site.module.heads))
            for site in sites
        )
        phase_key = (
            "minimax-h3.fl2va-phase/1", video_rows, audio_rows,
            phases, layouts[False], layouts[True], site_layout,
        )

        def step(i: torch.Tensor, video: Any, audio: Any, *bound: Any, phase: Phase) -> None:
            means = bound[: len(keys)]
            (
                prompt_embeds, denoiser_times, denoiser_indices, token_tags, positions,
                video_indices, audio_indices, text_indices, video_times, audio_times,
                video_sigma_table, audio_sigma_table,
            ) = bound[len(keys) :]
            carry = producer.Carry(
                dict(zip(keys, means, strict=True)), phase.rolling, phase.seeding
            )
            with attention_scope(layouts[phase.sparse]), producer.carrying(carry):
                velocity, audio_velocity = transformer(
                    hidden_states=video[None],
                    audio_hidden_states=audio[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=denoiser_times.index_select(0, i)[0, : phase.rows],
                    timestep_indices=denoiser_indices.index_select(0, i)[0],
                    token_tags=token_tags,
                    position_ids=positions,
                    video_indices=video_indices,
                    audio_indices=audio_indices,
                    text_indices=text_indices,
                    return_dict=False,
                )
            velocity, audio_velocity = velocity[0, video_rows:], audio_velocity[0, audio_rows:]
            euler(video[video_rows:], velocity.float(), i, video_times, video_sigma_table)
            euler(
                audio[audio_rows:], audio_velocity.float(), i, audio_times, audio_sigma_table
            )

        denoise_loop(
            step,
            range(len(plan)),
            state=(state.latents, state.audio_latents),
            carried=slots,
            inputs=inputs,
            key=phase_key,
            phases=phases,
            on_step=on_step,
        )
    state.noise_pred = state.audio_noise_pred = None  # the official loop's outputs; none reads them


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
    with torch.no_grad(), step_invariants(checked=True, overlay=overlay) as memo:
        for times, rows in plan:  # an uncovered row refuses here, never inside the loop
            transformer._arm(times, rows, tags, selector)
        transformer._release()
        sites = producer_sites(transformer)
        rotary = memo.get("rope", (position_ids,), lambda: transformer.rope(position_ids))
        producer.present(sites, rotary, tags.numel())
        timesteps = torch.stack([times for times, _ in plan]).to(device)
        indices = torch.stack([rows for _, rows in plan]).to(device)
        # A key mean carried from a step's last window is no other window's: windows compute
        # their own statistics. Every step is sparse, so no dense step seeds them.
        rolling = bool(sites) and len(windows) == 1
        phases = tuple(
            Phase(True, int(times.numel()), rolling and index > 0, False)
            for index, (times, _) in enumerate(plan)
        )
        layout = layout_at(0)
        keys = tuple(id(site.module) for site in sites)
        slots = tuple(torch.zeros(site.module.heads, 128, device=device) for site in sites)

        def step(i: torch.Tensor, rows: Any, *means: Any, phase: Phase) -> None:
            carry = producer.Carry(dict(zip(keys, means, strict=True)), phase.rolling)
            velocity = torch.zeros_like(rows)
            for (start, count), weight in zip(windows, weights, strict=True):
                with attention_scope(layout), producer.carrying(carry):
                    predicted, _ = transformer(
                        hidden_states=rows[None, start : start + count],
                        audio_hidden_states=audio[None],
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps.index_select(0, i)[0, : phase.rows],
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

        denoise_loop(step, range(len(plan)), state=(video,), carried=slots, phases=phases, on_step=on_step)


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
