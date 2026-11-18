"""A cutout (se-154): Qwen-Image-2.1 renders the subject on a plain white backdrop, BiRefNet's matte is the alpha, and each
edge pixel's colour is unmixed from the backdrop behind it, so the white leaves no fringe on dark hair. Qwen's own alpha
needs its model card's words: other words paint a checkerboard on Turbo (run PRO6-29)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from birefnet.model import BackgroundRemovalModel
from cozy_runtime.author import Adjustments, Telemetry
from PIL import Image

#: Matte values this close to 0 or 1 are its noise, and snap.
_SNAP = 0.02
#: The backdrop is averaged on a grid of this many pixels, filled in under the subject, then smoothed back up.
_CELL = 16
#: A backdrop farther than this from its smooth estimate on average (0-1) wasn't plain: a pattern, props, a checkerboard.
BUSY = 0.06
#: A matte covering less of the picture than this found no subject.
EMPTY = 0.002


def backdrop(rgb: torch.Tensor, clear: torch.Tensor) -> torch.Tensor:
    """The backdrop behind each pixel of `rgb` (3,H,W): the picture where `clear` (H,W) is set, averaged per cell, the
    cells under the subject filled from around them, and smoothed back up to the picture's size."""
    h, w = clear.shape
    pad = (0, math.ceil(w / _CELL) * _CELL - w, 0, math.ceil(h / _CELL) * _CELL - h)
    weight = F.pad(clear.float()[None, None], pad)
    total = F.avg_pool2d(F.pad(rgb[None], pad, mode="replicate") * weight, _CELL)
    share = F.avg_pool2d(weight, _CELL)
    for _ in range(sum(share.shape[-2:])):
        if bool((share > 0).all()):
            break
        empty = share == 0
        total = torch.where(empty, F.avg_pool2d(total, 3, 1, 1, count_include_pad=False), total)
        share = torch.where(empty, F.avg_pool2d(share, 3, 1, 1, count_include_pad=False), share)
    grid = F.avg_pool2d(F.pad(total / share.clamp_min(1e-6), (1, 1, 1, 1), mode="replicate"), 3, 1)
    up = F.interpolate(grid, size=(pad[3] + h, pad[1] + w), mode="bilinear", align_corners=False)
    return up[0, :, :h, :w]


def cut(image: Image.Image, matte: BackgroundRemovalModel, adj: Adjustments, tel: Telemetry) -> Image.Image:
    """`image`, rendered on a plain backdrop, cut out by `matte`: RGBA, or `image` as it is when the matte found nothing."""
    alpha = torch.from_numpy(np.asarray(matte.mask(image), dtype=np.float32) / 255)
    alpha = torch.where(alpha < _SNAP, 0.0, torch.where(alpha > 1 - _SNAP, 1.0, alpha))
    covered = float(alpha.mean())
    if covered < EMPTY:
        adj.warn("prompt", "the matte found no subject to cut out, so the picture keeps its backdrop")
        tel.log("cutout", covered=round(covered, 4))
        return image
    rgb = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
    clear = alpha == 0
    if not bool(clear.any()):
        tel.log("cutout", covered=round(covered, 4))
        return image.convert("RGBA")
    behind = backdrop(rgb, clear)
    busy = float((rgb - behind).abs().mean(0)[clear].mean())
    edge = (alpha > 0) & (alpha < 1)
    fore = torch.where(edge, ((rgb - (1 - alpha) * behind) / alpha.clamp_min(_SNAP)).clamp(0, 1), rgb)
    plain = behind[:, clear].median(dim=1).values
    tel.log("cutout", covered=round(covered, 4), edge=round(float(edge.float().mean()), 4), busy=round(busy, 4),
            backdrop="#" + "".join(f"{round(float(c) * 255):02x}" for c in plain))
    if busy > BUSY:
        adj.warn("prompt", "the backdrop Qwen Image 2.1 drew wasn't plain, so the cutout's edges may be rough: try another seed")
    pixels = (torch.cat([fore, alpha[None]]) * 255).round().to(torch.uint8).permute(1, 2, 0)
    return Image.fromarray(pixels.contiguous().numpy())
