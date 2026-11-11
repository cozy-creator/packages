"""Takes the 16 px cell pattern out of H3's decoded pixels.

The VAE's decoder is a ViT whose linear head paints each latent token's 16x16 patch, whatever
produced the latents (ComfyUI's and NVIDIA's decodes of the released VAE do this too). Two
traces of it tile the picture at the cell's period:

- a fixed grey tile, added where texture is strong: its size grows with the square of the local
  texture, up to about 3 of 255 in rock or foliage and nothing in flat sky;
- less fine texture at the cell's edges than at its middle, by 5-15%.

Both are measured from the decode itself and removed from every frame: the tile subtracted and
the high-pass multiplied flat. Nothing is blurred. This runs on the decode's own canvas, where
the period is exact and the phase is zero; a later resize or crop only moves the grid.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F

#: The high-pass of the review's pattern measure: a Gaussian of sigma 1.2 px, radius 4.
_SIGMA, _RADIUS = 1.2, 4
_LUMA = (0.2126, 0.7152, 0.0722)
#: Every 4th frame of a piece measures, adding to the decode's running totals, so a piece's
#: content cannot pulse across a join.
_MEASURE_EVERY = 4
#: A long straight edge sits at one position in the cell for its whole length. Counted whole it
#: would draw the grid this removes, so a pixel counts for at most this many times the mean.
_OUTLIER = 4.0
#: The most one position's texture may move: a stray edge in a nearly flat picture stays small.
_GAIN = (0.75, 1.33)
#: Local texture counts up to this (the strong rock and foliage the tile was measured on), so
#: the square law never scales the tile past its measured size beside a hard edge.
_MOST_TEXTURE = 12 / 255


class CellEqualizer:
    """One decode's filter over its `(t, 3, H, W)` float pieces in `[0, 1]`, in decode order."""

    def __init__(self, period: int) -> None:
        self._period = period
        # Zero until the first piece's sums broadcast them to the tile's (p, p).
        self._energy: torch.Tensor | float = 0.0
        self._tile: torch.Tensor | float = 0.0
        self._weight: torch.Tensor | float = 0.0

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """`frames`, in place, without the pattern; a piece with non-finite values passes
        through for the caller's integrity check to find."""
        if not bool(torch.isfinite(frames).all()):
            return frames
        kernel = _kernel(frames)
        luma = torch.tensor(_LUMA, device=frames.device, dtype=frames.dtype).view(3, 1, 1)
        for frame in frames[:: _MEASURE_EVERY]:
            self._measure(((frame - _blur(frame, kernel)) * luma).sum(0))
        tile, gain = self._tables(frames)
        for frame in frames:
            high = frame - _blur(frame, kernel)
            local = _local(((high * luma).sum(0)).abs(), self._period)
            frame += high.mul_(gain - 1) - local.square_().mul_(tile)
        return frames

    def _measure(self, high: torch.Tensor) -> None:
        p = self._period
        height, width = high.shape
        size = height // p, p, width // p, p
        mag = high.abs()
        square = _local(mag, p).square_()
        limit = _OUTLIER * mag.mean()
        self._energy += mag.clamp_max(limit).view(size).mean((0, 2))
        self._tile += (high.clamp(-limit, limit) * square).view(size).mean((0, 2))
        self._weight += square.square_().view(size).mean((0, 2))

    def _tables(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._period
        count = frames.shape[-2] // p, frames.shape[-1] // p
        energy = cast(torch.Tensor, self._energy)  # `__call__` measured this piece first
        rows = energy.mean(1, keepdim=True).clamp_min(1e-6)
        columns = energy.mean(0, keepdim=True).clamp_min(1e-6)
        gain = (rows.mean() / rows * (columns.mean() / columns)).clamp(*_GAIN)
        tile = self._tile / cast(torch.Tensor, self._weight).clamp_min(1e-12)
        tile = tile - tile.mean()
        return tile.repeat(count), gain.repeat(count)


def _local(magnitude: torch.Tensor, period: int) -> torch.Tensor:
    """Texture around each pixel: `magnitude`'s mean per cell, spread back over the picture."""
    means = F.avg_pool2d(magnitude[None, None], period).clamp_max(_MOST_TEXTURE)
    return F.interpolate(means, magnitude.shape, mode="bilinear", align_corners=False)[0, 0]


def _kernel(like: torch.Tensor) -> torch.Tensor:
    x = torch.arange(-_RADIUS, _RADIUS + 1, device=like.device, dtype=like.dtype)
    weights = torch.exp(-0.5 * (x / _SIGMA) ** 2)
    return weights / weights.sum()


def _blur(frame: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """`(3, H, W)` under a separable Gaussian, edges reflected."""
    size = kernel.numel()
    value = F.pad(frame[None], (_RADIUS, _RADIUS, _RADIUS, _RADIUS), mode="reflect")
    value = F.conv2d(value, kernel.view(1, 1, size, 1).expand(3, 1, size, 1), groups=3)
    return F.conv2d(value, kernel.view(1, 1, 1, size).expand(3, 1, 1, size), groups=3)[0]
