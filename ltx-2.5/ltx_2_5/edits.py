"""One canvas for continuing a take and for redoing a range of it: frames and sound held in place, the rest made.

A pickup is `[carry | new]`: the take's last 25 frames and their sound held, upstream's long-video carry. A retake is
`[context | range | context]`: the frames on both sides held and the mask widened 0.5 s into them, so the model makes
the hand-over (LTX Desktop's retake and extend, which is a retake on a longer canvas). A range longer than a canvas
runs as pieces left to right, each holding the frames made before it. The result goes back over the take as Desktop and
Lightricks' API stitch it: the take's own frames outside, a linear blend of at most 16 frames just outside the edit; the
sound crossfades over 40 ms, equal power (upstream's chunk seam). A shot change near an edge is where the edit ends.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FPS = 24
#: Frames in one canvas: RoPE's 20 s at 24 fps, `8k + 1`.
CANVAS = 481
#: What a pickup holds: upstream's long-video carry (ltx_pipelines `--chunk-carry-frames`), a causal start.
CARRY = 25
#: The least context a range holds on a side that has any: LTX Desktop's floor at 24 fps (`window.py`:
#: ⌈(0.3 + 0.5) s x fps⌉ + 7 + 8 + 1).
CONTEXT = 36
#: How far the mask widens into a range's context (Desktop `MASK_DELTA_SECONDS`, 0.5 s).
WIDEN = 12
#: The blend onto the take's own frames just outside an edit (Desktop and ltxv-api `MAX_FADE_FRAMES`).
FADE = 16
#: The sound's equal-power crossfade at a seam, longer than the vocoder's receptive field (upstream's chunk seam).
CROSSFADE_S = 0.04
#: A shot change this many frames from an edge, either side, is where the edit ends: the frames a canvas makes past
#: an edge (the widened mask and a latent's rounding).
NEAR = WIDEN + 7
#: A frame pair is a shot change when its pictures differ by this much (mean absolute difference, 0-1, of the frames
#: shrunk to `_SIDE`) and its colours by this much (total variation of 16-bin channel histograms, 0-1), each at least
#: `_OUTLIER` times the median pair's around it: a cut replaces the picture and its colours at once. Motion, even a whip
#: pan, moves the picture for frames running, so a pair beside another that moved is not one; and a flash comes back,
#: so the change must still hold `FLASH` frames on either side. A missed cut only holds the next frames instead.
#: MiniMax H3's retake uses the same rule and numbers.
_PICTURE = 0.08
_COLOUR = 0.1
_OUTLIER = 4.0
_SIDE = 64
#: The longest flash (a strobe, lightning, a muzzle flash) a shot change outlasts, in frames.
FLASH = 4

type Frame = npt.NDArray[np.uint8]
type Waveform = npt.NDArray[np.float32]
type Side = Literal["context", "inside", "none"]


@dataclass(frozen=True, slots=True)
class Piece:
    """One canvas over take frames `[first, first + frames)` (`8k + 1`) that makes `[start, end)` again. It holds the
    take's frames before `start` and those from `end` up to `limit`; past `limit` (a shot change, the take's end, or
    frames not made yet) it holds nothing and makes every frame, dropping what lies past `end`."""

    first: int
    frames: int
    start: int
    end: int
    limit: int
    #: The mask widens into held context (a retake), or stops at the edit (a pickup's carry, as upstream's chunks).
    widen: bool

    @property
    def before(self) -> int:
        """Frames held before the edit."""
        return self.start - self.first

    @property
    def after(self) -> int:
        """Frames held after the edit."""
        return max(0, min(self.first + self.frames, self.limit) - self.end)

    def span(self) -> tuple[float, float]:
        """The canvas seconds made again: the edit, widened into context, and everything past it when nothing after
        it is held."""
        start = max(0, self.before - (WIDEN if self.widen and self.before else 0))
        end = min(self.frames, self.end - self.first + (WIDEN if self.widen else 0)) if self.after else self.frames
        return start / FPS, end / FPS


def aligned_down(frames: int) -> int:
    return frames - (frames - 1) % 8 if frames >= 1 else 1


def pickup(done: int, new: int) -> Piece:
    """A take continued from its frame `done` for `new` frames, holding its last frames (upstream's 25, fewer for a
    shorter take): the new part is a whole number of latents, so the canvas is `8k + 1`."""
    carry = aligned_down(min(CARRY, done))
    new = min(8 * math.ceil(max(1, new) / 8), CANVAS - carry)
    return Piece(first=done - carry, frames=carry + new, start=done, end=done + new, limit=done, widen=False)


def retake(start: int, end: int, *, before: int, after: int) -> list[Piece]:
    """Canvases that redo take frames `[start, end)`, which have `before` frames to hold before them and `after`
    after (none at a shot change or the take's ends). One canvas fills with context split evenly (LTX Desktop's
    window); a range too long for one, with at least `CONTEXT` a side, runs as equal pieces left to right, each but the
    last holding only what comes before it."""
    lo, hi = start - before, end + after
    room = CANVAS - min(before, CONTEXT) - min(after, CONTEXT)
    count = 1 if end - start <= room else math.ceil((end - start) / (CANVAS - 2 * CONTEXT))
    size = math.ceil((end - start) / count)
    pieces = []
    for index in range(count):
        a, b = start + index * size, min(end, start + (index + 1) * size)
        pieces.append(_fit(a, b, lo, hi if index == count - 1 else b))
    return pieces


def _fit(start: int, end: int, lo: int, hi: int) -> Piece:
    """A canvas making `[start, end)`, holding what it can of `[lo, start)` and `[end, hi)`: Desktop's even split
    with the rest to whichever side has it, then `8k + 1` by growing context, else trimming it, else padding the end."""
    width, have_before, have_after = end - start, start - lo, hi - end
    spare = CANVAS - width
    before, after = min(spare // 2, have_before), min(spare // 2, have_after)
    after += min(spare - before - after, have_after - after)
    before += min(spare - before - after, have_before - before)
    total = before + width + after
    grow = (8 - (total - 1) % 8) % 8
    if grow and total + grow <= CANVAS and have_after - after + have_before - before >= grow:
        more = min(grow, have_after - after)
        after, before = after + more, before + grow - more
    else:
        trim = (total - 1) % 8
        less = min(trim, after)
        after, before = after - less, before - min(trim - less, before)
    frames = before + width + after
    if (frames - 1) % 8 or frames < 9:
        # Nothing left to trim: frames made past the end and dropped (H3's plates do the same).
        after, frames = 0, max(9, frames + (8 - (frames - 1) % 8) % 8)
    return Piece(first=start - before, frames=frames, start=start, end=end, limit=end + after, widen=True)


def _measures(frame: Frame) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A frame shrunk by block means to `_SIDE` on its long side, in 0-1, and its colour histogram."""
    factor = max(1, max(frame.shape[:2]) // _SIDE)
    height, width = frame.shape[0] // factor * factor, frame.shape[1] // factor * factor
    small = frame[:height, :width].reshape(height // factor, factor, width // factor, factor, 3).mean(axis=(1, 3)) / 255
    counts = np.concatenate([np.bincount(frame[..., channel].ravel() >> 4, minlength=16) for channel in range(3)])
    return small, counts / counts.sum()


def shot_changes(frames: Iterable[Frame]) -> list[int]:
    """The index of each frame among `frames` (consecutive `(h, w, 3)` RGB8 pictures, each read once) that opens a new
    shot."""
    measured = [_measures(frame) for frame in frames]
    if len(measured) < 3:
        return []

    def picture(a: int, b: int) -> float:
        return float(np.abs(measured[b][0] - measured[a][0]).mean())

    def colour(a: int, b: int) -> float:
        return float(0.5 * np.abs(measured[b][1] - measured[a][1]).sum())

    last = len(measured) - 1
    moved = [picture(index, index + 1) for index in range(last)]
    least = max(_PICTURE, _OUTLIER * float(np.median(moved)))
    least_colour = max(_COLOUR, _OUTLIER * float(np.median([colour(index, index + 1) for index in range(last)])))

    def changed(a: int, b: int) -> bool:
        return picture(a, b) >= least and colour(a, b) >= least_colour

    return [
        index + 1 for index in range(last)
        if changed(index, index + 1)
        and all(moved[beside] < least for beside in (index - 1, index + 1) if 0 <= beside < last)
        and changed(max(0, index - FLASH), index + 1) and changed(index, min(last, index + 1 + FLASH))
    ]  # fmt: skip


def nearest(found: Sequence[int], edge: int) -> int | None:
    """The cut within `NEAR` frames of `edge`, the closest; None when there is none."""
    close = [cut for cut in found if abs(cut - edge) <= NEAR]
    return min(close, key=lambda cut: (abs(cut - edge), cut)) if close else None


def weight(index: int, piece: Piece) -> float:
    """How much of `piece`'s frame shows over the take's at take frame `index`: 1 inside the edit, a linear fade
    across at most `FADE` held frames just outside it (Desktop's `generated_weight`), 0 elsewhere."""
    fade_in, fade_out = min(FADE, piece.before), min(FADE, piece.after)
    if piece.start - fade_in <= index < piece.start:
        return (index - piece.start + fade_in) / fade_in
    if piece.start <= index < piece.end:
        return 1.0
    if piece.end <= index < piece.end + fade_out:
        return (piece.end + fade_out - index - 1) / fade_out
    return 0.0


def mix(own: Frame, made: Frame, amount: float) -> Frame:
    """`amount` of `made` over `own`, in RGB (Desktop's `blend_rgb`)."""
    if amount >= 1.0:
        return made
    mixed = own.astype(np.float32) * (1.0 - amount) + made.astype(np.float32) * amount
    return np.clip(mixed, 0, 255).astype(np.uint8)


class Edit:
    """The take's frames as an edit leaves them, at the canvas size: each frame a canvas made, and how much of it
    shows over the take's own frame. Every frame outside is the take's, as decoded."""

    def __init__(self) -> None:
        self.frames: dict[int, Frame] = {}
        self.weights: dict[int, float] = {}

    def put(self, piece: Piece, made: npt.NDArray[np.uint8]) -> None:
        """`made`, a canvas's decoded frames, over the edit and its blend. A seam between two pieces blends the
        later one over the earlier one's frames."""
        for index in range(piece.start - min(FADE, piece.before),
                           min(piece.end + min(FADE, piece.after), piece.first + len(made))):
            amount, frame = weight(index, piece), made[index - piece.first]
            if self.weights.get(index) == 1.0 and amount < 1.0:
                self.frames[index] = mix(self.frames[index], frame, amount)
            else:
                self.frames[index], self.weights[index] = frame.copy(), amount

    def at(self, index: int, own: Frame) -> Frame:
        """Take frame `index` as the edit leaves it, `own` being the take's frame at the canvas size."""
        if index not in self.frames:
            return own
        return mix(own, self.frames[index], self.weights[index])


def crossfade(out: Waveform, into: Waveform) -> Waveform:
    """Equal power across the samples: `out` falls as a cosine, `into` rises as a sine."""
    turn = (np.arange(out.shape[-1], dtype=np.float64) + 0.5) / max(1, out.shape[-1]) * (np.pi / 2)
    faded: Waveform = (out * np.cos(turn) + into * np.sin(turn)).astype(np.float32)
    return faded


def splice(sound: Waveform, made: Waveform, at: int, start: int, end: int, *, lead: Side, tail: Side,
           fade: int) -> None:
    """`made`, whose first sample is `sound`'s `at`, over `sound[:, start:end]` in place, crossfaded over `fade`
    samples at each edge: in the held context beside it, else just inside the edit (a shot change), else not at all
    (the take's ends)."""
    stop = min(sound.shape[1], at + made.shape[1])

    def put(lo: int, hi: int, fade_in: bool | None) -> None:
        lo, hi = max(lo, at, 0), min(hi, stop)
        if hi <= lo:
            return
        new, old = made[:, lo - at:hi - at], sound[:, lo:hi]
        sound[:, lo:hi] = new if fade_in is None else crossfade(old, new) if fade_in else crossfade(new, old)

    a, b = start, end
    if lead == "context":
        put(start - fade, start, True)
    elif lead == "inside":
        put(start, start + fade, True)
        a = start + fade
    if tail == "context":
        put(end, end + fade, False)
    elif tail == "inside":
        put(end - fade, end, False)
        b = end - fade
    put(a, b, None)
