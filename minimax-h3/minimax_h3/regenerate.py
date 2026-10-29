"""Open-H3-Regenerate-2K (NVlabs `sol-engine` `models/Open-H3-Regenerate-2K`): an H3 latent, upscaled
in latent space (`latent_upscaler`), re-noised to its start sigma and refined at the target canvas
through the FL2VA DiT under the Regenerate overlay; `open_h3/pipeline.py` `upscale` and `refine`.

The refine is text-only T2VA: `[text | clean silent audio | video]`, CFG 1, Euler on video alone.
When the whole canvas's rows do not fit the card, it refines overlapping temporal windows that
share each step's state and blend their velocities, so the clip still completes, slower.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
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
    Schedule,
    _attention_layout,
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
#: The VAE posterior sample `infer.py --input-video` draws (`--vae-seed`).
POSTERIOR_SEED = 42
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
AUDIO_CHANNELS = 2
#: Latent frames neighbouring windows share (about one second).
OVERLAP = 7
#: What one DiT block holds per packed row beside its weights: the bf16 residual rows
#: (hidden 5376) and the attention output before `to_out` (56 x 128), whose full buffers the
#: Runtime's cuts do not split, with a fifth spare; and what a card holds beside them (context,
#: the two-block weight window, the head-cut K/V). An approximation the proof bounds.
_ROW_BYTES = 6 * (5376 + 7168) * 2 // 5
_BESIDE = 3 << 30


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


def windows(latent_frames: int, rows_per_frame: int, capacity: int) -> list[tuple[int, int]]:
    """`(first latent frame, latent frames)` spans: the whole clip when its rows fit `capacity`
    bytes, else the fewest equal `5n + 2` windows overlapping by at least OVERLAP that do."""
    fit = max(0, capacity - _BESIDE) // (_ROW_BYTES * rows_per_frame)
    if latent_frames <= fit:
        return [(0, latent_frames)]
    if fit < 2 * OVERLAP + 2:
        raise MemoryError(
            f"one {rows_per_frame}-row latent frame needs {_ROW_BYTES * rows_per_frame} bytes; "
            f"this card holds windows of {fit} frames, too few to refine in overlapping windows"
        )
    count = 2
    while True:
        need = math.ceil((latent_frames + (count - 1) * OVERLAP) / count)
        length = need + (-(need - 2)) % LATENTS_PER_CHUNK
        if length <= fit:
            break
        count += 1
    starts = [round(index * (latent_frames - length) / (count - 1)) for index in range(count)]
    return [(start, length) for start in starts]


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


def plan_windows(shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """The refine's windows for `(1, 24, T, H, W)` latents on this process's card."""
    device = torch.device(str(placed_device()))
    _, _, latent_frames, height, width = shape
    capacity = int(torch.cuda.mem_get_info(device)[1]) if device.type == "cuda" else 1 << 62
    return windows(latent_frames, (height // PATCH_SIZE[1]) * (width // PATCH_SIZE[2]), capacity)


def refine(
    transformer: Any,
    overlay: Any,
    *,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_tags: torch.Tensor,
    spans: list[tuple[int, int]],
    audio: torch.Tensor,
    schedule: Schedule,
    seed: int,
    on_step: Callable[[int], None],
) -> torch.Tensor:
    """Refine normalized `(1, 24, T, H, W)` latents at their size over `spans` (`plan_windows`),
    each window carrying the clean `audio` rows of its length; the refined latents.

    The noise is drawn as `infer.py` draws it (a CUDA generator over the `[1, rows, 96]` video
    rows), so a seed reproduces the reference's refine on the same latent.
    """
    device = torch.device(str(placed_device()))
    _, _, latent_frames, height, width = (int(value) for value in latents.shape)
    rows_per_frame = (height // PATCH_SIZE[1]) * (width // PATCH_SIZE[2])
    length = spans[0][1]
    low = patchify_video_latents(latents.to(device, torch.float32), PATCH_SIZE)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((1, *low.shape), device=device, dtype=torch.float32, generator=generator)[0]
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
    view = SimpleNamespace(denoiser_input_fields=fields, num_condition_video_rows=0)
    denoise_refine(
        transformer,
        overlay,
        video=video,
        audio=audio,
        prompt_embeds=prompt_embeds.to(device),
        fields=fields,
        plan=[(times.to(device), rows.to(device)) for times, rows in plan],
        sigmas=torch.tensor(schedule.video_sigmas, dtype=torch.float32, device=device),
        windows=[(first * rows_per_frame, length * rows_per_frame) for first, _ in spans],
        weights=blend(spans, latent_frames, rows_per_frame, device),
        layout_at=lambda step: _attention_layout(view, step, sol_dense_steps=0),
        on_step=on_step,
    )
    return unpatchify(video, latent_frames, height, width)

