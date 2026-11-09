"""Open-H3-Regenerate-2K (NVlabs `sol-engine` `models/Open-H3-Regenerate-2K`): an H3 latent, upscaled
in latent space (`latent_upscaler`), re-noised to its start sigma and refined at the target canvas
through the FL2VA DiT under the Regenerate overlay; `open_h3/pipeline.py` `upscale` and `refine`.

The refine is text-only T2VA: `[text | clean silent audio | video]`, CFG 1, Euler on video alone.
A clip whose rows do not fit one forward refines in overlapping temporal windows that share each
step's state and blend their velocities (`plan_windows`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from cozy_runtime import rng
from cozy_runtime.accel.attention import AttentionLayout
from cozy_runtime.author import Cut, Windows, windows
from cozy_runtime.author import device as placed_device
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)

from .official import (
    FRAMES_PER_CHUNK,
    LATENTS_PER_CHUNK,
    PATCH_SIZE,
    VAE_SPATIAL_RATIO,
    _EXACT_ATTENTION_PATHS,
    Schedule,
)
from .steps import denoise_refine

#: Output size -> (computed height, width, kept rows): 1080p and 2160p compute whole latent
#: patches (1088, 2176 rows) and keep the centre rows, as `open_h3/pipeline.py` RESOLUTIONS.
TARGETS: dict[str, tuple[int, int, int]] = {
    "768p": (768, 1344, 768),
    "1080p": (1088, 1920, 1080),
    "1440p": (1440, 2560, 1440),
    "2160p": (2176, 3840, 2160),
}
#: The upscaler's trained range ends at 1440p: a larger target goes through it first.
UPSCALER_CEILING = (1440, 2560)
#: NVIDIA's refine forwards (`open_h3/sol_engine_fwd.py` SPARSE, `fast_dit.SparsePlan`): the refine
#: steps that run dense (None: every one), and the Sol `tau` of the sparse ones. A sparse step's every
#: block is sparse, its text and audio rows exact; the token refiner stays dense.
PRESETS: dict[str, tuple[int | None, float]] = {
    "full": (None, 1.0),  # `standard`
    "standard": (1, 1.0),  # `sol_engine`
    "draft": (0, 1.75),  # `sol_engine_fast`
}
#: The VAE posterior sample `infer.py --input-video` draws (`--vae-seed`).
POSTERIOR_SEED = 42
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
AUDIO_CHANNELS = 2
#: Latent frames neighbouring windows share (about one second).
OVERLAP = 7
#: The most video rows one forward has refined at a normal step time (1440p 15 s whole, run 20382;
#: 2160p 15 s in three windows, run 20470): 465k rows (2160p 15 s in two windows on 48 GB, sized
#: from total memory, run 20476) made no progress in an hour.
_MOST_ROWS = 107 * 3600
#: The refine's windows: None, as few as the card's room holds (`_ROW_BYTES` a row); a row count,
#: windows of at most that many rows from the shape alone, the same on every card (`_MOST_ROWS`).
WINDOW_ROWS: int | None = None
#: What a refine holds per packed row at a block's peak: 5.5 bf16 rows of hidden 5376 (its input,
#: residual, norm and attention output are whole; the Runtime cuts the attention and FFN working
#: sets). 2160p 15 s whole needs more than 48 GB (run 20468) and runs in three windows on 24 GB.
_ROW_BYTES = 11 * 5376


def latent_size(resolution: str) -> tuple[int, int]:
    height, width, _ = TARGETS[resolution]
    return height // VAE_SPATIAL_RATIO, width // VAE_SPATIAL_RATIO


def upscale_passes(resolution: str) -> list[tuple[int, int]]:
    """The latent sizes the upscaler produces in order; none for a refine-only 768p pass."""
    if resolution == "768p":
        return []
    target = latent_size(resolution)
    ceiling = (UPSCALER_CEILING[0] // VAE_SPATIAL_RATIO, UPSCALER_CEILING[1] // VAE_SPATIAL_RATIO)
    return [ceiling, target] if target[0] > ceiling[0] else [target]


def frames_of(latent_frames: int) -> int:
    """Video frames of `5n + 2` latent frames on the VAE's `17n + 5` grid."""
    return (latent_frames - 2) // LATENTS_PER_CHUNK * FRAMES_PER_CHUNK + 5


def blend(spans: list[tuple[int, int]], latent_frames: int, rows_per_frame: int, device: Any) -> list[torch.Tensor]:
    """Per-row weights of each span: linear ramps across each overlap, summing to one per row."""
    raw = []
    for index, (start, length) in enumerate(spans):
        weight = torch.ones(length, dtype=torch.float32)
        if index > 0:
            shared = spans[index - 1][0] + spans[index - 1][1] - start
            weight[:shared] = torch.arange(1, shared + 1, dtype=torch.float32) / (shared + 1)
        if index + 1 < len(spans):
            shared = start + length - spans[index + 1][0]
            weight[length - shared :] = torch.arange(shared, 0, -1, dtype=torch.float32) / (shared + 1)
        raw.append(weight)
    total = torch.zeros(latent_frames, dtype=torch.float32)
    for (start, length), weight in zip(spans, raw, strict=True):
        total[start : start + length] += weight
    return [
        (weight / total[start : start + length]).repeat_interleave(rows_per_frame).to(device)
        for (start, length), weight in zip(spans, raw, strict=True)
    ]


def unpatchify(rows: torch.Tensor, latent_frames: int, height: int, width: int) -> torch.Tensor:
    """Video rows back to `(1, 24, T, H, W)`: `patchify_video_latents` inverted."""
    patch_t, patch_h, patch_w = PATCH_SIZE
    channels = rows.shape[-1] // (patch_t * patch_h * patch_w)
    value = rows.reshape(
        1, latent_frames // patch_t, height // patch_h, width // patch_w, channels, patch_t, patch_h, patch_w
    )
    return value.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, channels, latent_frames, height, width)


def plan_windows(shape: tuple[int, ...]) -> Windows:
    """The refine's windows over `(1, 24, T, H, W)` latents' frames, chosen once for its loop: the
    fewest equal `5n + 2` clips sharing at least OVERLAP frames, each starting on the VAE's latent
    chunk grid (a multiple of LATENTS_PER_CHUNK), within `WINDOW_ROWS`. A window is laid out as a
    clip of its own, and H3's rotary clock spaces latent frames 5/3 x (1, 4, 4, 4, 4) from a clip's
    start, so a window off the grid gives each chunk's one-frame latent a four-frame step: it
    flickers at every chunk (run 22567, the middle of three windows, starting at 32)."""
    _, _, latent_frames, height, width = shape
    rows = (height // PATCH_SIZE[1]) * (width // PATCH_SIZE[2])
    cut = Cut(0, overlap=OVERLAP, align=LATENTS_PER_CHUNK)
    return windows("minimax-h3.refine", (latent_frames, rows), cut, per_element_bytes=_ROW_BYTES,
                   most=WINDOW_ROWS, equal=True)  # fmt: skip


def refine(
    transformer: Any,
    overlay: Any,
    *,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_tags: torch.Tensor,
    layout: Windows,
    audio: torch.Tensor,
    schedule: Schedule,
    seed: int,
    quality: str,
    on_step: Callable[[int], None],
) -> torch.Tensor:
    """Refine normalized `(1, 24, T, H, W)` latents at their size over `layout` (`plan_windows`),
    each window carrying the clean `audio` rows of its length; the refined latents. The noise
    over the video rows is the Runtime's (`rng.noise`, drawn on the CPU from `seed`), the same on
    every card.
    """
    device = torch.device(str(placed_device()))
    _, _, latent_frames, height, width = (int(value) for value in latents.shape)
    rows_per_frame = (height // PATCH_SIZE[1]) * (width // PATCH_SIZE[2])
    spans = [(first, stop - first) for first, stop in layout]
    length = spans[0][1]
    low = patchify_video_latents(latents.to(device, torch.float32), PATCH_SIZE)
    noise = rng.noise(low.shape, seed, device=device)
    start = schedule.start
    video = (1 - start) * low + start * noise
    del low, noise

    audio = audio.to(device, torch.bfloat16)
    tags = text_tags.flatten().to("cpu", torch.long)
    if not bool((tags == TEXT_TAG).all()):
        raise ValueError("the refine conditions on text alone: every text row is tagged text")
    values = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=tags,
        num_latent_frames=length,
        latent_height=height,
        latent_width=width,
        num_audio_latents=int(audio.shape[0]) // AUDIO_CHANNELS,
        patch_size=PATCH_SIZE,
        audio_channels=AUDIO_CHANNELS,
        audio_tag=AUDIO_TAG,
        video_tag=VIDEO_TAG,
        keyframe_anchors=(),
    )
    names = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")
    fields = {name: value.to(device) for name, value in zip(names, values[:5], strict=True)}
    plan = [
        MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices=fields["video_indices"].cpu(),
            audio_indices=fields["audio_indices"].cpu(),
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
            num_text_tokens=int(tags.numel()),
            video_timestep=timestep,
            audio_timestep=1.0,
            condition_video_timestep=1.0,
            condition_audio_timestep=1.0,
        )
        for timestep in schedule.video_timesteps
    ]
    dense, tau = PRESETS[quality]
    dense = len(plan) if dense is None else dense
    video = denoise_refine(
        transformer,
        overlay,
        video=video,
        audio=audio,
        prompt_embeds=prompt_embeds.to(device),
        fields=fields,
        plan=[(times.to(device), rows.to(device)) for times, rows in plan],
        sigmas=torch.tensor(schedule.video_sigmas, dtype=torch.float32, device=device),
        windows=[(first * rows_per_frame, length * rows_per_frame) for first, _ in spans],
        piece=layout.piece,
        weights=blend(spans, latent_frames, rows_per_frame, device),
        dense_steps=dense,
        layout_at=lambda step: AttentionLayout(
            live_tokens=int(fields["token_tags"].numel()),
            protected_prefix=int(fields["text_indices"].numel()) + int(fields["audio_indices"].numel()),
            step=step, dense_until_step=dense, dense_paths=("token_refiner",),
            exact_paths=_EXACT_ATTENTION_PATHS, tau=tau,
        ),  # fmt: skip
        on_step=on_step,
    )
    return unpatchify(video, latent_frames, height, width)

