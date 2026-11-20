"""Inference-only prototype for retaining same-location Plate guide patches."""

from __future__ import annotations

import torch
from torch.nn.functional import max_pool2d


def _patch_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Relative L1 difference between clean patch features."""
    difference = (left - right).abs_().mean(dim=-1)
    magnitude = left.abs().mean(dim=-1) + right.abs().mean(dim=-1)
    return difference / magnitude.clamp_min_(1e-6)


def plate_patch_offsets(
    video: torch.Tensor,
    grid: tuple[int, int, int],
    *,
    first_prunable: int,
    omit_frames: int,
) -> torch.Tensor:
    """Return ordered offsets for the same row budget as omitting `omit_frames` whole frames.

    A rotating set of anchors keeps each location at least every third latent frame and
    retains a patch in every overlapping 2×2 region. The caller supplies the protected
    head (including any continuation join). The final frame is always kept.
    """
    frames, height, width = grid
    rows = frames * height * width
    available = max(0, frames - first_prunable - 1)
    if video.ndim != 2 or video.shape[0] != rows or min(grid) < 1:
        raise ValueError("Plate patches must be packed in the original time, height, width order")
    if not 1 <= first_prunable <= frames or not 0 <= omit_frames <= available // 2:
        raise ValueError("Plate patch budget exceeds the available interior frames")
    device = video.device
    budget = omit_frames * height * width
    if budget == 0:
        return torch.arange(rows, device=device)

    patches = video.reshape(frames, height, width, -1).float()
    adjacent = _patch_distance(patches[:-1], patches[1:])
    motion = adjacent[:-1].maximum(adjacent[1:])
    motion = max_pool2d(motion[None], kernel_size=3, stride=1, padding=1)[0]

    times = torch.arange(1, frames - 1, device=device)[:, None, None]
    y = torch.arange(height, device=device)[None, :, None]
    x = torch.arange(width, device=device)[None, None, :]
    since_anchor = (times - y - x).remainder(3)
    # A candidate one frame after an anchor already compares with it in `adjacent`.
    # For candidates two frames out, compare with the last guaranteed kept patch too.
    drift = torch.zeros_like(motion)
    drift[1:] = _patch_distance(patches[2:-1], patches[:-3])
    check_drift = (since_anchor == 2) & (times > first_prunable)
    motion = motion.maximum(drift.masked_fill_(~check_drift, 0))

    detail = torch.zeros_like(motion)
    interior = patches[1:-1]
    vertical = _patch_distance(interior[:, 1:], interior[:, :-1])
    detail[:, 1:] = vertical
    detail[:, :-1] = detail[:, :-1].maximum(vertical)
    horizontal = _patch_distance(interior[:, :, 1:], interior[:, :, :-1])
    detail[:, :, 1:] = detail[:, :, 1:].maximum(horizontal)
    detail[:, :, :-1] = detail[:, :, :-1].maximum(horizontal)
    score = 2 * motion + detail
    score.masked_fill_((since_anchor == 0) | (times < first_prunable), float("inf"))

    # Rank only the interior and spread ties throughout equal-score candidates. In a
    # static shot this distributes omissions over time rather than front-loading them.
    scores = score.flatten()
    threshold = scores.kthvalue(budget).values
    omitted = scores < threshold
    tied = (scores == threshold).nonzero(as_tuple=True)[0]
    remaining = budget - int(omitted.count_nonzero())
    ranks = (torch.arange(remaining, device=device) * tied.numel() + tied.numel() // 2) // remaining
    omitted[tied[ranks]] = True

    keep = torch.ones(grid, dtype=torch.bool, device=device)
    keep[1:-1] = ~omitted.reshape_as(score)
    return keep.flatten().nonzero(as_tuple=True)[0]
