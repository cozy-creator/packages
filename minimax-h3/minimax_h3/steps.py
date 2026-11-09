"""H3's denoise loops: FL2VA's and Ref2VA's official loop, and the Regenerate refine.

Each step is the official loop step (`MiniMaxH3LoopDenoiser`, then `MiniMaxH3LoopSchedulerStep`):
the DiT's velocities, then each modality's scheduler step. H3 owns its attention layouts, so every
DiT call names its own: dense for the first `sol_dense_steps` steps and the token refiner, sparse
after. The Kitchen producer's sites reuse the previous step's key means; the last dense step seeds
them. A step's state is its DiT call's tensors (`key_means` in, a third output back, the rotary
presentation, the turbo head's index) and never a context around it, so a recording of the call
replays any step of its kind.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from cozy_runtime.accel import stepcache
from cozy_runtime.accel.attention import attention_scope, AttentionLayout, producer_sites

from . import producer
from .overlay import ATTENTION_KWARG, OVERLAY_KWARG, Overlay


def denoise_official(
    transformer: Any,
    schedulers: Any,
    state: Any,
    layout_at: Callable[[int], AttentionLayout],
    sol_dense_steps: int,
    on_step: Callable[[int], object],
    *,
    steps: range | None = None,
    overlay: Overlay | None = None,
    cache: stepcache.Policy | None = None,
) -> None:
    """One rank's denoise of the loop's block `state` on its DiT (FL2VA's or Ref2VA's), its latents
    advanced over `steps` of its plan (all of them by default) by `schedulers`' own steps: under
    the turbo `overlay` (its tables and LoRA, and each step's collapsed heads), its DiT's stack
    under `cache` when a preset names one."""
    plan = state.row_timestep_plan
    steps = range(len(plan)) if steps is None else steps
    fields = state.denoiser_input_fields
    position_ids, tags = fields["position_ids"], fields["token_tags"]
    video, audio = state.latents, state.audio_latents
    video_rows, audio_rows = state.num_condition_video_rows, state.num_condition_audio_rows
    selector = None if overlay is None else {ATTENTION_KWARG: overlay.bank, OVERLAY_KWARG: overlay}
    schedule = None if overlay is None else overlay.schedule  # a turbo bank's evaluations select its heads
    heads = [0 if schedule is None else schedule.step(times) for times, _ in plan]
    with torch.no_grad():
        for times, rows in plan:  # an uncovered row refuses here, never inside the loop
            transformer._arm(times, rows, tags, selector)
        transformer._release()
        sites = producer_sites(transformer)
        rotary = transformer.rope(position_ids)
        presentation = producer.present(sites, rotary, tags.numel())
        head = None
        if overlay is not None:
            transformer._engage(overlay, heads[steps.start])
            if overlay.heads:  # each step's collapsed heads, by index on the device
                head = torch.tensor(heads, device=video.device)
        dense, sparse = layout_at(0), layout_at(sol_dense_steps)
        means: tuple[torch.Tensor | None, ...] | None = None  # each block's key means, once a step left them
        caching = nullcontext() if cache is None else stepcache.running(cache, len(steps), video.device)
        try:
            # The token refiner's attention is a site too: dense, as a dense step's.
            with attention_scope(dense):
                text = transformer.text_rows(state.prompt_embeds)
            with caching as run:
                for i in steps:
                    if run is not None:
                        run.step(i - steps.start)
                    times, rows = plan[i]
                    layout = sparse if i >= sol_dense_steps else dense
                    # A loop's first step measures its own key means: none of its steps left one.
                    rolling = bool(sites) and i >= max(sol_dense_steps, 1) and i > steps.start
                    keeps = bool(sites) and (i == sol_dense_steps - 1 or layout.sparse)
                    with attention_scope(layout):
                        velocity, audio_velocity, *kept = transformer(
                            hidden_states=video[None],
                            audio_hidden_states=audio[None],
                            encoder_hidden_states=state.prompt_embeds,
                            timestep=times,
                            timestep_indices=rows,
                            token_tags=tags,
                            position_ids=position_ids,
                            video_indices=fields["video_indices"],
                            audio_indices=fields["audio_indices"],
                            text_indices=fields["text_indices"],
                            return_dict=False,
                            rotary=rotary,
                            text=text,
                            key_means=means if rolling else None,
                            keep_key_means=keeps,
                            rope_presentation=presentation,
                            head=None if head is None else head[i : i + 1],
                            checked=True,
                            bank=None if overlay is None else overlay.bank,
                        )
                    if kept:
                        means = _kept(means, kept[0])
                    video[video_rows:] = schedulers.scheduler.step(
                        velocity[0, video_rows:].float(), state.timesteps[i], video[video_rows:], return_dict=False
                    )[0]
                    audio[audio_rows:] = schedulers.audio_scheduler.step(
                        audio_velocity[0, audio_rows:].float(),
                        state.audio_timesteps[i],
                        audio[audio_rows:],
                        return_dict=False,
                    )[0]
                    del velocity, audio_velocity
                    on_step(i)
        finally:
            transformer._release()
            producer.end_loop()  # after the last step: it forgets the rotary this loop's calls read
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
    piece: Callable[[], AbstractContextManager[None]],
    weights: list[torch.Tensor],
    dense_steps: int,
    layout_at: Callable[[int], AttentionLayout],
    on_step: Callable[[int], object],
) -> torch.Tensor:
    """The Regenerate refine (NVlabs `open_h3/pipeline.py` `refine`): Euler on the video rows
    `video` alone, every step's velocity from the FL2VA DiT under `overlay`; the refined rows.

    `windows` are equal row spans of `video`, each laid out as a clip of its own (`fields`: H3's
    positions start at every clip's text), so one layout and one rotary table serve them all;
    `weights` are their per-row blend. One span is the whole canvas. `audio` is the clean conditioning rows one window carries; `plan` each step's
    `(timesteps, timestep_indices)`.
    """
    position_ids, tags = fields["position_ids"], fields["token_tags"]
    selector = {ATTENTION_KWARG: overlay.bank, OVERLAY_KWARG: overlay}
    with torch.no_grad():
        for times, rows in plan:  # an uncovered row refuses here, never inside the loop
            transformer._arm(times, rows, tags, selector)
        transformer._release()
        sites = producer_sites(transformer)
        rotary = transformer.rope(position_ids)
        presentation = producer.present(sites, rotary, tags.numel())
        # The first `dense_steps` run dense (the preset's), the last of them seeding the Kitchen
        # producer's key means. A key mean carried from a step's last window is no other window's:
        # windows compute their own statistics.
        single = bool(sites) and len(windows) == 1
        dense, sparse = layout_at(0), layout_at(dense_steps)
        means: tuple[torch.Tensor | None, ...] | None = None
        transformer._engage(overlay, 0)
        try:
            with attention_scope(dense):  # under the overlay, as every step reads it
                text = transformer.text_rows(prompt_embeds)
            for i, (times, rows) in enumerate(plan):
                layout = sparse if i >= dense_steps else dense
                rolling = single and i >= max(dense_steps, 1)
                keeps = single and (i == dense_steps - 1 or layout.sparse)
                velocity = torch.zeros_like(video)
                for (start, count), weight in zip(windows, weights, strict=True):
                    with piece(), attention_scope(layout):
                        predicted, _, *kept = transformer(
                            hidden_states=video[None, start : start + count],
                            audio_hidden_states=audio[None],
                            encoder_hidden_states=prompt_embeds,
                            timestep=times,
                            timestep_indices=rows,
                            token_tags=tags,
                            position_ids=position_ids,
                            video_indices=fields["video_indices"],
                            audio_indices=fields["audio_indices"],
                            text_indices=fields["text_indices"],
                            return_dict=False,
                            rotary=rotary,
                            text=text,
                            key_means=means if rolling else None,
                            keep_key_means=keeps,
                            rope_presentation=presentation,
                            checked=True,
                            bank=overlay.bank,
                        )
                    if kept:
                        means = _kept(means, kept[0])
                    velocity[start : start + count].addcmul_(predicted[0].float(), weight[:, None])
                    del predicted
                video = video + velocity * (sigmas[i] - sigmas[i + 1])
                on_step(i)
        finally:
            transformer._release()
            producer.end_loop()  # after the last step: it forgets the rotary this loop's calls read
    return video


def _kept(
    held: tuple[torch.Tensor | None, ...] | None, new: tuple[torch.Tensor | None, ...]
) -> tuple[torch.Tensor | None, ...]:
    """Each block's newest key means: a block a call left none for keeps its last."""
    if held is None:
        return new
    return tuple(now if now is not None else before for before, now in zip(held, new, strict=True))
