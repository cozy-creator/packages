"""A guide video (`StoryReference.guide`): a source the render follows frame by frame.

It is ComfyUI's `MiniMaxH3AddGuide` at frame 0, which Alissonerdx's edit LoRAs were trained
with, not a reference. The clip is read at the render's canvas and VAE-encoded. Each latent
frame becomes condition rows on the target latent frame it guides: the same rotary time and
grid, noised and held as FL2VA's keyframes are, on every step. A kept soundtrack becomes
condition audio on the target's clock, clean as a reference soundtrack is. Qwen never sees the
clip, and nothing is cut from it.

The rows go in after the official layout and latents, before the timesteps, so any trunk takes
them. They are the first condition rows, so everything after them (references, a carried
context) keeps its order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
)

#: The `PipelineState` keys of a call's guide: its clip, then its encoded rows.
GUIDE = "exact_guide"
GUIDE_LATENTS = "exact_guide_latents"


@dataclass(frozen=True, slots=True)
class GuideClip:
    """`(T, H, W, 3)` uint8 frames at the render's canvas, and a stereo soundtrack at the audio
    VAE's rate, if kept."""

    frames: Any
    audio: torch.Tensor | None = None


def guide_rows(state: Any, video: torch.Tensor, audio: torch.Tensor | None) -> None:
    """Pack the guide's rows in front of every other condition row of `state`'s layout.

    `video` is the noised, patchified guide (rows of whole target latent frames), `audio` its
    channel-major soundtrack rows (`(channels, latents, width)`), each at most the target's.
    """
    positions, tags = state.position_ids, state.token_tags
    video_rows, audio_rows = state.video_indices, state.audio_indices
    text = int(state.text_indices.numel())
    target_video = video_rows[state.num_condition_video_rows :]
    target_audio = audio_rows[state.num_condition_audio_rows :]
    count_video = int(video.shape[0])
    if count_video > target_video.numel():
        raise ValueError("a guide video has more frames than the render")
    guide_positions = [positions[target_video[:count_video]]]
    guide_tags = [tags[target_video[:count_video]]]
    count_audio = 0
    if audio is not None:
        channels, latents, width = (int(n) for n in audio.shape)
        target = positions[target_audio].view(channels, -1, 3)
        if latents > target.shape[1]:
            raise ValueError("a guide video's soundtrack is longer than the render's")
        count_audio = channels * latents
        guide_positions.append(target[:, :latents].reshape(-1, 3))
        guide_tags.append(tags[target_audio[:count_audio]])
    count = count_video + count_audio

    def moved(indices: torch.Tensor) -> torch.Tensor:
        """Every row after the text, `count` places on."""
        return indices + count * (indices >= text)

    state.set("position_ids", torch.cat([positions[:text], *guide_positions, positions[text:]]))
    state.set("token_tags", torch.cat([tags[:text], *guide_tags, tags[text:]]))
    device = video_rows.device
    state.set("video_indices", torch.cat([
        torch.arange(text, text + count_video, device=device), moved(video_rows)
    ]))
    state.set("audio_indices", torch.cat([
        torch.arange(text + count_video, text + count, device=device), moved(audio_rows)
    ]))
    state.set("num_condition_video_rows", state.num_condition_video_rows + count_video)
    state.set("latents", torch.cat([video.to(state.latents), state.latents]))
    if audio is not None:
        packed = audio.reshape(count_audio, width).to(state.audio_latents)
        state.set("audio_latents", torch.cat([packed, state.audio_latents]))
        state.set("num_condition_audio_rows", state.num_condition_audio_rows + count_audio)


#: The `PipelineState` keys of a Ref2VA call's keyframe anchors: `(image, "first" | "last")`
#: pairs at the render's canvas, then their encoded latents. Not Diffusers' FL2VA
#: `keyframe_anchors`, which FL2VA's own keyframes fill (run 20657).
ANCHORS = "cozy_ref2va_keyframes"
ANCHOR_LATENTS = "cozy_ref2va_keyframe_latents"


def anchor_rows(state: Any, video: torch.Tensor, *, last: bool) -> None:
    """One keyframe's rows (one latent frame, noised and patchified) in front of every other
    condition row, held on every step, as `guide_rows` packs a guide: Ref2VA's first and last
    frames (`StoryReference` `frame`). They take the target's frame grid at the rotary time
    the official FL2VA layout gives that keyframe: the target's start, or its last pixel frame
    (ComfyUI's Add Guide at frame -1 alike)."""
    positions, tags = state.position_ids, state.token_tags
    video_rows = state.video_indices
    text = int(state.text_indices.numel())
    target = video_rows[state.num_condition_video_rows :]
    count = int(video.shape[0])
    if count > target.numel() or target.numel() % count:
        raise ValueError("a keyframe's rows are not one of the render's latent frames")
    grid = positions[target[:count]].clone()
    if last:
        spans = np.ones(target.numel() // count, dtype=np.float64) * _ROPE_FRAME_RESCALE
        for offset, frames in enumerate(_ROPE_FRAMES_PER_LATENT):
            spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= frames
        grid[:, 0] += float(spans.sum()) - _ROPE_FRAME_RESCALE

    def moved(indices: torch.Tensor) -> torch.Tensor:
        return indices + count * (indices >= text)

    state.set("position_ids", torch.cat([positions[:text], grid, positions[text:]]))
    state.set("token_tags", torch.cat([tags[:text], tags[target[:count]], tags[text:]]))
    device = video_rows.device
    state.set("video_indices", torch.cat([
        torch.arange(text, text + count, device=device), moved(video_rows)
    ]))
    state.set("audio_indices", moved(state.audio_indices))
    state.set("num_condition_video_rows", state.num_condition_video_rows + count)
    state.set("latents", torch.cat([video.to(state.latents), state.latents]))
