"""An edit keeps the picture it edits wherever it left it unchanged (se-153).

Qwen redraws the whole canvas on every edit, so what the request leaves alone still comes back
resampled: softer, slightly recoloured and sometimes a little zoomed. Chained edits compound it
(Paul, 11-17: "if you use images as references, like A -> B -> C and keep editing them, it
introduces noise"). After the decode, the output is aligned to the picture, a smooth change mask
is taken from their difference, and the picture's own pixels go back where nothing changed. A
global change (relight, style, black and white) changes most of the canvas or shifts its colour,
and is left as the model made it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torch.nn import functional as F

#: The ratio the output may differ from the picture's before its framing counts as changed.
RATIO_TOLERANCE = 0.02
#: The longest side the alignment and the mask are measured at.
MEASURE_SIDE = 512
#: The scales the output may be off by, searched about its centre (Qwen's small zoom).
SCALES = tuple(1 + step / 400 for step in range(-12, 13))
#: How much closer an alignment must bring the output before it is moved at all.
ALIGNED_GAIN = 0.9
#: Colour difference (CIE76 on blurred Lab) above which a region is the edit's.
CHANGED_DELTA_E = 10.0
#: Below it a pixel fits the colour correction.
STILL_DELTA_E = 6.0
#: Past this fraction of the canvas changed, the edit is global and nothing is kept.
GLOBAL_FRACTION = 0.85
#: A colour shift this large over what stayed is an intended relight, and nothing is kept.
GLOBAL_SHIFT = 8.0
#: The changed mask grows by this fraction of the long side, then feathers by half of it.
GROW = 0.015


@dataclass(frozen=True)
class Kept:
    """What `keep_unchanged` did, for the run's log."""

    fraction: float  # of the canvas, the picture's own pixels
    scale: float
    shift: tuple[float, float]  # px, x and y
    skipped: str = ""


def _lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0, 1] (3, H, W) as CIE Lab (D65)."""
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp_min(0) ** 2.4)
    m = rgb.new_tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = torch.einsum("ij,jhw->ihw", m, linear) / rgb.new_tensor([0.95047, 1.0, 1.08883]).view(3, 1, 1)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-12) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """A separable Gaussian over (C, H, W), edges replicated."""
    radius = max(1, int(3 * sigma))
    taps = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(taps**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    c = x.shape[0]
    y = F.pad(x[None], (radius, radius, 0, 0), mode="replicate")
    y = F.conv2d(y, kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    y = F.pad(y, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(y, kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)[0]


def _small(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    s = MEASURE_SIDE / max(h, w)
    return F.interpolate(x[None], size=(max(8, round(h * s)), max(8, round(w * s))), mode="area")[0]


def _warp(x: torch.Tensor, scale: float, tx: float, ty: float, mode: str = "bilinear") -> torch.Tensor:
    """`x` sampled at scale·q + t (normalized coordinates): what the output shows at q of the picture's frame."""
    if scale == 1.0 and tx == 0.0 and ty == 0.0:
        return x
    theta = x.new_tensor([[scale, 0.0, tx], [0.0, scale, ty]])[None]
    grid = F.affine_grid(theta, [1, *x.shape], align_corners=False)
    return F.grid_sample(x[None], grid, mode=mode, padding_mode="border", align_corners=False)[0]


def _edges(gray: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude, windowed: what phase correlation matches, steady under recolouring."""
    gx = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 1))
    gy = F.pad(gray[:, 1:, :] - gray[:, :-1, :], (0, 0, 0, 1))
    e = (gx**2 + gy**2).sqrt()[0]
    window = torch.outer(torch.hann_window(e.shape[0], device=e.device), torch.hann_window(e.shape[1], device=e.device))
    return (e - e.mean()) * window


def _correlate(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Phase correlation: the shift (dy, dx px) taking `b` onto `a`, and its peak."""
    cross = torch.fft.rfft2(a) * torch.fft.rfft2(b).conj()
    surface = torch.fft.irfft2(cross / cross.abs().clamp_min(1e-9), s=a.shape)
    h, w = a.shape
    y, x = divmod(int(surface.argmax()), w)

    def vertex(before: float, at: float, after: float) -> float:  # the parabola's, to a fraction of a pixel
        bend = before - 2 * at + after
        return 0.0 if bend >= 0 else 0.5 * (before - after) / bend

    dy = y + vertex(float(surface[y - 1, x]), float(surface[y, x]), float(surface[(y + 1) % h, x]))
    dx = x + vertex(float(surface[y, x - 1]), float(surface[y, x]), float(surface[y, (x + 1) % w]))
    return float(surface[y, x]), (dy - h if dy > h / 2 else dy), (dx - w if dx > w / 2 else dx)


def _align(picture: torch.Tensor, output: torch.Tensor) -> tuple[float, float, float]:
    """The scale about the centre and shift (normalized) that bring the output onto the picture."""
    reference = _edges(picture.mean(0, keepdim=True))
    best = (-1.0, 1.0, 0.0, 0.0)
    for scale in SCALES:
        peak, dy, dx = _correlate(reference, _edges(_warp(output, scale, 0.0, 0.0).mean(0, keepdim=True)))
        if peak > best[0]:
            h, w = reference.shape
            # The output shows the picture's q at q - d px; sampled at scale·q, that is -scale·d, normalized.
            best = (peak, scale, -2 * dx / w * scale, -2 * dy / h * scale)
    return best[1], best[2], best[3]


def keep_unchanged(picture: Image.Image, decoded: torch.Tensor) -> tuple[torch.Tensor, Kept]:
    """`decoded` (1, 3, H, W in [-1, 1], the decoder's) with `picture`'s own pixels wherever the edit left it unchanged."""
    _, channels, height, width = decoded.shape
    if channels != 3:
        return decoded, Kept(0.0, 1.0, (0.0, 0.0), "the output has transparency")
    if picture.mode in ("RGBA", "LA", "PA") and picture.getextrema()[-1][0] < 255:
        return decoded, Kept(0.0, 1.0, (0.0, 0.0), "the picture has transparency")
    if abs(picture.width / picture.height / (width / height) - 1) > RATIO_TOLERANCE:
        return decoded, Kept(0.0, 1.0, (0.0, 0.0), "the framing changed")
    device = decoded.device
    resized = picture.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    source = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8).view(height, width, 3)
    source = source.to(device).permute(2, 0, 1).float() / 255
    output = (decoded[0].float() / 2 + 0.5).clamp(0, 1)

    small_source, small_output = _small(source), _small(output)
    scale, tx, ty = _align(small_source, small_output)
    sigma = 2.0
    lab_source = _lab(_blur(small_source, sigma))
    moved = _warp(small_output, scale, tx, ty)
    difference = (_lab(_blur(moved, sigma)) - lab_source).norm(dim=0)
    plain = (_lab(_blur(small_output, sigma)) - lab_source).norm(dim=0)
    if difference.mean() > ALIGNED_GAIN * plain.mean():  # not clearly better aligned: left in place
        scale, tx, ty, moved, difference = 1.0, 0.0, 0.0, small_output, plain
    shift = (tx * width / 2, ty * height / 2)
    if float((difference > CHANGED_DELTA_E).float().mean()) > GLOBAL_FRACTION:
        return decoded, Kept(0.0, scale, shift, "the edit changed most of the picture")

    # The colour drift over what stayed: a gain and offset per channel, fit where it stayed.
    still = difference < STILL_DELTA_E
    if int(still.sum()) < still.numel() // 20:
        return decoded, Kept(0.0, scale, shift, "too little of the picture stayed to match")
    x, y = moved[:, still], small_source[:, still]
    gain = ((x - x.mean(1, keepdim=True)) * (y - y.mean(1, keepdim=True))).mean(1) / x.var(1, unbiased=False).clamp_min(1e-6)
    gain = gain.clamp(0.8, 1.25)
    offset = y.mean(1) - gain * x.mean(1)
    corrected = (moved * gain.view(3, 1, 1) + offset.view(3, 1, 1)).clamp(0, 1)
    drift = float((_lab(corrected.mean((1, 2), keepdim=True)) - _lab(moved.mean((1, 2), keepdim=True))).norm())
    if drift > GLOBAL_SHIFT:
        return decoded, Kept(0.0, scale, shift, f"the colour moved by {drift:.0f} ΔE, a relight")
    changed = ((_lab(_blur(corrected, sigma)) - lab_source).norm(dim=0) > CHANGED_DELTA_E).float()[None]

    # Grown, so the edit's own edge is the model's, then feathered; at the canvas's size.
    grow = max(1, round(GROW * MEASURE_SIDE))
    changed = F.max_pool2d(changed[None], 2 * grow + 1, stride=1, padding=grow)[0]
    changed = _blur(changed, grow / 2).clamp(0, 1)
    changed = F.interpolate(changed[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    edited = _warp(output, scale, tx, ty, "bicubic") * gain.view(3, 1, 1) + offset.view(3, 1, 1)
    kept = source * (1 - changed) + edited.clamp(0, 1) * changed
    # F32, so a kept pixel quantizes back to the picture's own byte.
    return (kept * 2 - 1)[None], Kept(float(1 - changed.mean()), scale, shift)
