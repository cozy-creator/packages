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

import torch

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
