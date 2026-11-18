"""Probe (cl-1210 sound joins, never merged): how alike two renderings of the same stretch are, and the step a join makes.
Waves are `(channels, samples)`. Every result is a few short strings, so a log line holds at most 8 fields."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Wave = npt.NDArray[np.float32]


def _db(v: npt.NDArray[np.float64]) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(v**2)) + 1e-12))


def alike(own: Wave, theirs: Wave) -> dict[str, str]:
    """`theirs` against `own` over the same samples: correlation, the largest difference, and each one's level."""
    n = min(int(own.shape[1]), int(theirs.shape[1]))
    x, y = own[:, :n].astype(np.float64), theirs[:, :n].astype(np.float64)
    a, b = x.mean(axis=0) - x.mean(), y.mean(axis=0) - y.mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"corr": f"{corr:.4f}", "max_diff": f"{float(np.abs(x - y).max()):.5f}", "rms_db_own": f"{_db(x):.1f}",
            "rms_db_join": f"{_db(y):.1f}", "samples": str(n)}  # fmt: skip


def step(wave: Wave, at: int, rate: int, *, wide: int = 0) -> str:
    """The largest sample step within `wide` samples and 5 ms either side of `at`, over the 99.9th percentile of the
    second around it: a click scores past 1."""
    x = wave.astype(np.float64)
    near = slice(max(1, at - wide // 2 - rate // 200), min(int(x.shape[1]), at + wide // 2 + rate // 200))
    around = slice(max(1, at - rate // 2), min(int(x.shape[1]), at + rate // 2))
    jumps = np.abs(np.diff(x, axis=1))
    scale = max(float(np.percentile(jumps[:, around.start - 1 : around.stop - 1], 99.9)), 1e-9)
    return f"{float(jumps[:, near.start - 1 : near.stop - 1].max()) / scale:.2f}"
