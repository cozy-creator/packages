"""A video reference at motion-guide size.

At the official sizing (24 fps, the target's canvas) a 12 s clip beside a 15 s render adds
~95k rows (82,656 latent, 12,096 vision) to ~109k: 5.5x the denoise of the same render without
it on an H100 (runs 10437, 10446). Three cuts bring it to ~1.25x and keep the motion and the
camera (h3a-094, owner's choice 2026-10-17):

- Each clip is read straight to half the edge of the canvas the official normalization would
  put it on (`official.read_reference_video`), so its latent rows and its vision tokens shrink
  ~4-fold. The layout puts the smaller grid on the
  same aspect-normalized rotary extent, as it does an image reference at its own size.
- Every second latent frame of a clip leaves the sequence at its official rotary position:
  the kept rows are exactly the encoded ones, and the target sits where it always does.
- The clip's remaining latent rows guide only the dense steps, where staging and motion are
  decided, and leave before the first sparse one, which then runs at no-reference length.

- Over a continued piece's carried picture head the clip has no rows: the carried rows hold the
  head, and the clip guides from the first frame after it.

Image references, standalone audio and a continuation's carried rows are untouched. A guide
video is never a reference (`guide_rows`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from .continuation import ContextReferenceLayout

CANVAS_DIVISOR = 2
FRAME_STRIDE = 2
#: The VAE's 16-pixel cells in 2x2 patches, and Qwen's 16-pixel patches merged 2x2.
_GRID = 32
_LAYOUT_FIELDS = {
    "position_ids",
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
    "prompt_embeds",
}


def motion_size(height: int, width: int) -> tuple[int, int]:
    """A clip's (height, width) once its canvas-sized frames shrink to guide motion."""
    small_height, small_width = (
        max(_GRID, round(edge / CANVAS_DIVISOR / _GRID) * _GRID) for edge in (height, width)
    )
    return small_height, small_width


def public_video(reference: Any) -> bool:
    return reference.kind == "video" and not isinstance(reference, ContextReferenceLayout)


class ClipRows:
    """Each packed video row's latent frame within its public clip (-1 for every other row),
    followed through each drop. The references' rows follow `start` guide rows."""

    def __init__(
        self,
        block_state: Any,
        references: Sequence[Any],
        condition_latents: Sequence[Any],
        patch: Sequence[int],
        *,
        start: int = 0,
    ) -> None:
        _, patch_h, patch_w = patch
        self.frame = torch.full((block_state.video_indices.numel(),), -1, dtype=torch.int64)
        offset, latents = start, iter(condition_latents)
        for reference in references:
            if reference.kind not in ("image", "video"):
                continue
            frames, height, width = (int(n) for n in next(latents).shape[2:5])
            per_frame = (height // patch_h) * (width // patch_w)
            if public_video(reference):
                span = slice(offset, offset + frames * per_frame)
                self.frame[span] = torch.arange(frames).repeat_interleave(per_frame)
            offset += frames * per_frame
        if next(latents, None) is not None or offset != block_state.num_condition_video_rows:
            raise ValueError("video reference rows disagree with the packed layout")

    def thin(self, block_state: Any, head: int = 0) -> None:
        """Before the first evaluation: keep every `FRAME_STRIDE`-th latent frame of each clip, past
        a continued piece's carried picture head (its first `head` latent frames). There the carried
        rows, on the target's own positions, are the picture: a clip frame there pulled the
        regenerated head to the clip's phase whenever the previous piece ended out of it, and the
        delivered frames cut away from the previous piece's (h3a-097, runs 25838 and 25471). The
        kept frames hold their positions, so the clip stays 1:1 with the frames after the head."""
        self._drop(block_state, lambda frame: (frame % FRAME_STRIDE != 0) | (frame < head))

    def release(self, block_state: Any) -> None:
        """Before the first sparse evaluation: every clip's latent rows leave."""
        self._drop(block_state, lambda frame: torch.ones_like(frame, dtype=torch.bool))

    def _drop(
        self, block_state: Any, dropped_frame: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        dropped = (self.frame >= 0) & dropped_frame(self.frame)
        if not bool(dropped.any()):
            return
        video_indices = block_state.video_indices
        keep = torch.ones(
            block_state.token_tags.numel(), dtype=torch.bool, device=video_indices.device
        )
        keep[video_indices[dropped.to(video_indices.device)]] = False
        select_rows(block_state, keep)
        self.frame = self.frame[~dropped]


def select_rows(block_state: Any, keep: torch.Tensor) -> None:
    """Keep the packed rows `keep` marks, renumbered in order, with every per-row field."""
    fields = dict(block_state.denoiser_input_fields)
    if set(fields) != _LAYOUT_FIELDS:
        raise ValueError(f"unexpected packed-layout fields {sorted(fields)}")
    position = torch.cumsum(keep, 0) - 1
    kept = {}
    for name in ("video_indices", "audio_indices", "text_indices"):
        rows = fields[name]
        kept[name] = keep[rows]
        fields[name] = position[rows[kept[name]]]
    fields["position_ids"] = fields["position_ids"][keep]
    fields["token_tags"] = fields["token_tags"][keep]
    video, audio, text = kept["video_indices"], kept["audio_indices"], kept["text_indices"]
    block_state.num_condition_video_rows = int(video[: block_state.num_condition_video_rows].sum())
    block_state.num_condition_audio_rows = int(audio[: block_state.num_condition_audio_rows].sum())
    block_state.latents = block_state.latents[video]
    block_state.audio_latents = block_state.audio_latents[audio]
    if not bool(text.all()):
        fields["prompt_embeds"] = fields["prompt_embeds"][:, text]
    block_state.row_timestep_plan = [
        (timesteps, indices[keep]) for timesteps, indices in block_state.row_timestep_plan
    ]
    for name, value in fields.items():
        setattr(block_state, name, value)
    block_state.denoiser_input_fields = fields
