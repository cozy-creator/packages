"""Probe (cl-1210 sound joins, never merged): how alike two renderings of a join's carried second are, and what a cross-fade
of them does. `a` is segment k's delivered sound over the second its successor carried (its tail), `b` the successor's own
rendering of it (its head), aligned on the shot's sample clock, `(channels, samples)`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Wave = npt.NDArray[np.float32]
FADES_MS = (0, 10, 25, 50, 100)
BANDS = np.geomspace(50.0, 16000.0, 25)


def _mono(wave: Wave) -> npt.NDArray[np.float64]:
    return wave.astype(np.float64).mean(axis=0)


def _corr(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    x, y = x - x.mean(), y - y.mean()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _xcorr(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], most: int) -> tuple[float, int]:
    """The best normalized correlation of `y` against `x` within `most` samples either way, and its lag (samples)."""
    best, lag = -2.0, 0
    for shift in range(-most, most + 1):
        a = x[max(0, shift): len(x) + min(0, shift)]
        b = y[max(0, -shift): len(y) + min(0, -shift)]
        value = _corr(a, b)
        if value > best:
            best, lag = value, shift
    return best, lag


def _bands(x: npt.NDArray[np.float64], rate: int) -> npt.NDArray[np.float64]:
    """Power per log-spaced band (dB), Hann frames of 2048 averaged."""
    size = 2048
    frames = [x[at: at + size] * np.hanning(size) for at in range(0, len(x) - size + 1, size // 2)] or [np.pad(x, (0, size - len(x)))]
    power = np.mean([np.abs(np.fft.rfft(frame)) ** 2 for frame in frames], axis=0)
    freqs = np.fft.rfftfreq(size, 1 / rate)
    # Bands holding no bin at this resolution are left out (NaN), as are those 40 dB under the loudest: no reading there.
    levels = np.asarray([power[(freqs >= lo) & (freqs < hi)].sum() if ((freqs >= lo) & (freqs < hi)).any() else np.nan
                         for lo, hi in zip(BANDS[:-1], BANDS[1:], strict=True)])
    db = 10 * np.log10(levels + 1e-12)
    db[db < np.nanmax(db) - 40] = np.nan
    return db


def _energy(x: npt.NDArray[np.float64], frame: int) -> npt.NDArray[np.float64]:
    count = len(x) // frame
    return (x[: count * frame].reshape(count, frame) ** 2).mean(axis=1)


def seam(a: Wave, b: Wave, rate: int) -> dict[str, str]:
    """How alike the two renderings are, and each fade at the overlap's middle (0: a switch)."""
    x, y = _mono(a), _mono(b)
    n = min(len(x), len(y))
    x, y = x[-n:], y[-n:]
    mid = slice(n // 4, 3 * n // 4)
    xm, ym = x[mid], y[mid]
    best, lag = _xcorr(xm, ym, rate // 50)
    rms = lambda v: 20 * np.log10(np.sqrt(np.mean(v**2)) + 1e-12)  # noqa: E731
    env = rate // 100
    spectral = _bands(xm, rate) - _bands(ym, rate)
    out = {
        "overlap_ms": f"{1000 * n / rate:.1f}",
        "corr": f"{_corr(x, y):.3f}",
        "corr_mid": f"{_corr(xm, ym):.3f}",
        "xcorr_mid": f"{best:.3f}",
        "xcorr_lag_ms": f"{1000 * lag / rate:.2f}",
        "envelope_corr_mid": f"{_corr(np.sqrt(_energy(xm, env)), np.sqrt(_energy(ym, env))):.3f}",
        "level_db_mid": f"{rms(xm):.1f}/{rms(ym):.1f}",
        "band_diff_db_mean": f"{np.nanmean(np.abs(spectral)):.1f}",
        "band_diff_db_max": f"{np.nanmax(np.abs(spectral)):.1f}",
    }
    # A sample step's scale: the 99.9th percentile of both renderings' steps over the overlap.
    scale = max(float(np.percentile(np.abs(np.diff(np.concatenate([x, y]))), 99.9)), 1e-6)
    m = n // 2
    for fade_ms in FADES_MS:
        width = int(rate * fade_ms / 1000)
        lo, hi = m - width // 2, m - width // 2 + width
        t = np.clip((np.arange(n) - lo) / max(width, 1), 0.0, 1.0) if width else (np.arange(n) >= m).astype(np.float64)
        wa, wb = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
        z = wa * x + wb * y
        near = slice(max(0, lo - rate // 200), min(n - 1, hi + rate // 200))
        step = float(np.abs(np.diff(z)[near]).max()) / scale
        tag = f"fade{fade_ms}"
        out[f"{tag}_step"] = f"{step:.2f}"
        if width:
            frame = max(rate // 200, 1)
            region = slice(lo, hi)
            got = _energy(z[region], frame)
            # Two renderings that don't cancel keep their weighted energies; a dip is phasing.
            want = _energy(wa[region] * x[region], frame) + _energy(wb[region] * y[region], frame)
            ratio = 10 * np.log10((got + 1e-12) / (want + 1e-12))
            out[f"{tag}_energy_db"] = f"{ratio.min():.1f}/{ratio.max():.1f}"
            if width >= 2048:
                notch = _bands(z[region], rate) - 10 * np.log10(
                    (10 ** (_bands(wa[region] * x[region], rate) / 10) + 10 ** (_bands(wb[region] * y[region], rate) / 10)) + 1e-12)
                out[f"{tag}_band_dip_db"] = f"{np.nanmin(notch):.1f}"
    return out


def reach(found: npt.NDArray[np.float64], rate: int) -> dict[str, str]:
    """The decoder's right context in effect: for each cut of n ticks, how far before it the decode changed (ms, at 1e-4
    and 1e-3), and the largest change."""
    fields = {}
    for n, _length, wide, narrow, most in found.tolist():
        fields[f"cut{int(n)}"] = f"{1000 * wide / rate:.1f}/{1000 * narrow / rate:.1f}/{most:.4f}"
    return fields
