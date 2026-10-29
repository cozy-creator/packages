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

        def step(i: torch.Tensor, video: Any, audio: Any, *means: Any, phase: Phase) -> None:
            carry = producer.Carry(
                dict(zip(keys, means, strict=True)), phase.rolling, phase.seeding
            )
            with attention_scope(layouts[phase.sparse]), producer.carrying(carry):
                velocity, audio_velocity = transformer(
                    hidden_states=video[None],
                    audio_hidden_states=audio[None],
                    encoder_hidden_states=state.prompt_embeds,
                    timestep=timesteps.index_select(0, i)[0, : phase.rows],
                    timestep_indices=indices.index_select(0, i)[0],
                    token_tags=tags,
                    position_ids=position_ids,
                    video_indices=fields["video_indices"],
                    audio_indices=fields["audio_indices"],
                    text_indices=fields["text_indices"],
                    return_dict=False,
                )
            velocity, audio_velocity = velocity[0, video_rows:], audio_velocity[0, audio_rows:]
            euler(video[video_rows:], velocity.float(), i, state.timesteps, video_sigmas)
            euler(
                audio[audio_rows:], audio_velocity.float(), i, state.audio_timesteps, audio_sigmas
            )

        denoise_loop(
            step,
            range(len(plan)),
            state=(state.latents, state.audio_latents),
            carried=slots,
            phases=phases,
            on_step=on_step,
        )
    state.noise_pred = state.audio_noise_pred = None  # the official loop's outputs; none reads them


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
