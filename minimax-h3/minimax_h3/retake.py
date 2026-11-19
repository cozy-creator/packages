"""A range of a take made again (`retake`): where its window lands on H3's grid, how it ends, and its segments.

The take is cut at the range's edges and only the middle is made again (Paul, 11-19). The window carries in the
take's last 22 frames before it, with their sound, as any continuation does. At its end it lands on the take: on a
shot change within a chunk of the asked end, where nothing needs to continue, else on the take's next 39 frames and
their sound, held over the sample's tail (the double form at both ends, se-177 arm B). A window longer than one sample
holds chains segments, the suffix only on the last.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .continuation import MAX_FRAMES, MIN_FRAMES, Carry

FPS = 24
CHUNK = 17
#: The frames a redo goes on from: Paul, 11-19, "condition on many frames (like 22) using motion conditioning".
PREFIX_FRAMES = 22
#: The kept frames after a window, held with their sound: 39 frames are 65 audio ticks, room for the second of sound.
SUFFIX_FRAMES = 39
#: How far a range's end moves to land on a shot change: one VAE chunk, as far as the grid moves it anyway.
SNAP = CHUNK
#: The frames a window ending on a shot change samples past it, at least: the sound past the join crossfades it.
TAIL = 4
#: A later segment's most new frames: 22 carried + 340 = 362.
_MOST = MAX_FRAMES - PREFIX_FRAMES
#: A segment's fewest: its sample is at least 5 s.
_LEAST = MIN_FRAMES - PREFIX_FRAMES


@dataclass(frozen=True)
class Window:
    """The frames a redo makes again, `[start, end)` on the take's 24-fps clock, and how it ends: on its `suffix` (the
    take's next frames, held), on a shot change (`cut`), or at the take's end (neither)."""

    start: int
    end: int
    suffix: bool
    cut: bool
    #: Each segment's new frames, in order.
    lengths: tuple[int, ...]
    warnings: tuple[str, ...]

    def plans(self, sound: bool) -> list[tuple[int, Carry, Carry, Carry, int]]:
        """Each segment's `ContinuationPlan` fields: its new frames, its carry (the take's frames and, with `sound`, its
        second of sound; then the previous segment's), its successor's, its suffix and its cut tail."""
        carry, first = Carry(PREFIX_FRAMES, True), Carry(PREFIX_FRAMES, sound)
        plans = []
        for index, frames in enumerate(self.lengths):
            last = index + 1 == len(self.lengths)
            suffix = Carry(SUFFIX_FRAMES, sound) if last and self.suffix else Carry()
            tail = 0 if not last or self.suffix else max(TAIL, _LEAST - frames)
            plans.append((frames, carry if index else first, Carry() if last else carry, suffix, tail))
        return plans


def _seconds(frames: int) -> str:
    return f"{frames / FPS:.2f} s"


def window(start: int, end: int, length: int, *, phased: bool, cut: int | None) -> Window:
    """The window for a range `[start, end)` of a take of `length` frames. `phased`: the take's latents carry it, so the
    window starts on their `17k + 5` grid and its suffix on a chunk. `cut`: a shot change, the first frame of the next
    shot, within `SNAP` of `end`. Each edge moves outward, never in, and says by how much."""
    said: list[str] = []
    if start < PREFIX_FRAMES:
        said.append(f"a redo goes on from {PREFIX_FRAMES} frames before it, so it starts at {_seconds(PREFIX_FRAMES)}, "
                    f"not {_seconds(start)}")  # fmt: skip
    a = CHUNK * max(1, (start - 5) // CHUNK) + 5 if phased else max(PREFIX_FRAMES, start)
    if a != start and start >= PREFIX_FRAMES:
        said.append(f"the redo starts at {_seconds(a)}, on H3's grid, {_seconds(start - a)} before the {_seconds(start)} "
                    "asked")  # fmt: skip
    if cut is not None and cut > a:
        if cut != end:
            said.append(f"the redo ends on the shot change at {_seconds(cut)}, not {_seconds(end)}")
        return Window(a, cut, False, True, _split(cut - a, 1, _MOST - TAIL), tuple(said))
    # Phased, the kept frames after start on a chunk; any other video, the window is the length the grid takes.
    b = -(-max(end, a + 1) // CHUNK) * CHUNK if phased else a + max(end - a, 1) + (12 - max(end - a, 1)) % CHUNK
    # One sample holds the carry, at least 63 new frames and the suffix: 22 + 63 + 39 = 124, 5 s.
    least = MIN_FRAMES - PREFIX_FRAMES - SUFFIX_FRAMES
    while b - a < least:
        b += CHUNK
    if b + SUFFIX_FRAMES > length:
        said.append(f"the redo ends within {_seconds(SUFFIX_FRAMES)} of the take's end, so it runs on to the end, "
                    f"{_seconds(length)}")  # fmt: skip
        return Window(a, length, False, False, _split(length - a, 1, _MOST - TAIL), tuple(said))
    if b != end:
        said.append(f"the redo ends at {_seconds(b)}, on H3's grid, {_seconds(b - end)} after the {_seconds(end)} asked")
    return Window(a, b, True, False, _split(b - a, least, _MOST - SUFFIX_FRAMES), tuple(said))


def _split(total: int, least: int, most: int) -> tuple[int, ...]:
    """A window's new frames as segments: each before the last a whole number of chunks, its sample `22 + 17m` on the
    grid (102-340 new frames), so the last keeps the window's place on the grid; the last `least`-`most`. The fewest
    segments, as even as the chunks allow."""
    if total <= most:
        return (total,)
    count = 2 + (total - most - 1) // _MOST
    each = min(_MOST, max(_LEAST, CHUNK * round(total / count / CHUNK)))
    while total - (count - 1) * each > most and each + CHUNK <= _MOST:
        each += CHUNK
    while total - (count - 1) * each < least and each - CHUNK >= _LEAST:
        each -= CHUNK
    return (*([each] * (count - 1)), total - (count - 1) * each)


#: A frame pair is a shot change when its pictures differ by this much (mean absolute difference, 0-1, of the frames
#: shrunk to `_SIDE`) and its colours by this much (total variation of 16-bin channel histograms, 0-1), each at least
#: `_OUTLIER` times the median pair's around it: a cut replaces the picture and its colours at once; motion, even fast,
#: moves the picture for many frames running and its colours little. A missed cut only holds the next frames instead.
_PICTURE = 0.08
_COLOUR = 0.1
_OUTLIER = 4.0
_SIDE = 64


def _measures(frame: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A frame shrunk by block means to `_SIDE` on its long side, in 0-1, and its colour histogram."""
    factor = max(1, max(frame.shape[:2]) // _SIDE)
    height, width = frame.shape[0] // factor * factor, frame.shape[1] // factor * factor
    small = frame[:height, :width].reshape(height // factor, factor, width // factor, factor, 3).mean(axis=(1, 3)) / 255
    counts = np.concatenate([np.bincount(frame[..., channel].ravel() >> 4, minlength=16) for channel in range(3)])
    return small, counts / counts.sum()


def shot_changes(frames: Iterable[npt.NDArray[np.uint8]]) -> list[int]:
    """The index of each frame among `frames` (consecutive `(h, w, 3)` RGB8 pictures, each read once) that opens a new
    shot."""
    measured = [_measures(frame) for frame in frames]
    if len(measured) < 3:
        return []
    picture = np.array([np.abs(b[0] - a[0]).mean() for a, b in zip(measured, measured[1:])])
    colour = np.array([0.5 * np.abs(b[1] - a[1]).sum() for a, b in zip(measured, measured[1:])])
    return [
        index + 1 for index in range(len(picture))
        if picture[index] >= max(_PICTURE, _OUTLIER * float(np.median(picture)))
        and colour[index] >= max(_COLOUR, _OUTLIER * float(np.median(colour)))
    ]  # fmt: skip
