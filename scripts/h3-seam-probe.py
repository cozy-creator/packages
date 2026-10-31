#!/usr/bin/env python3
"""Seam metrics for a MiniMax-H3 shot's piece joins, read from the delivered MP4.

Per join: audio continuity (waveform NCC across the join, relative to the same NCC at
neighbouring in-segment points), loudness and noise-floor steps (K-weighted, LUFS-like),
held frames near the join, the picture jump (cut), and luma/chroma/sharpness steps.
Per segment: the same picture/sound levels, so drift down a chain shows as a trend.

    minimax-h3/.venv/bin/python scripts/h3-seam-probe.py ~/.cozy/outputs/paul-minimax-h3/5430-video.mp4
    ... VIDEO --run 5430          joins from `cozy run show 5430 --json --full` (default: <run>-video.mp4)
    ... VIDEO --joins 192,480     frame index of each segment's first frame
    ... VIDEO --untrimmed A.mp4 B.mp4   per join: the next segment's output before its context
                                  replay is trimmed; scores replay fidelity of the context audio
    ... VIDEO --json FILE|- --frames DIR
    ... --self-test               synthetic hard cut vs continuous split

Columns (per join; single joins are noisy, compare medians and the calibration line):
  cont.25 cont1s  waveform NCC across the join / NCC at in-segment points beside it, 0.25 s and
                  1 s windows: ~1 continuous; ~0.4 a tail spliced onto another segment of the same
                  piece (a "cover band"); - near silence
  ncc1s lag_ms    raw NCC peak (1 s windows) and its lag
  dLU dFloor      K-weighted level 250 ms after minus before; 5th-percentile 50 ms level, 2 s each side
  cut             frame difference at the join / typical in its segments: ~1 smooth, >3 a visible jump
  hold motion     longest held run (MAD < --hold-mad) within 2 s; motion 1 s after / before
  dY dCb dCr      mean over 12 frames after minus before (0-255); sharp: log2 Laplacian-variance ratio
  replay lag_ms   context audio vs the next segment's regenerated copy: ~1 faithful, low = cover band
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import av
import numpy as np
from PIL import Image

STAT_NAMES = ("y", "cb", "cr", "y_sd", "cb_sd", "cr_sd", "sharp", "motion")
VISUAL_WINDOW = 12  # frames each side for level steps
HOLD_SPAN = 48  # frames each side searched for held frames
FLOOR_SPAN_S = 2.0
SILENT_DBFS = -60.0  # NCC on near-silence measures codec noise; such windows score n/a
# BS.1770 K-weighting (48 kHz design); evaluated at physical frequency for any rate.
K_STAGES = (
    ((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585)),
    ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621)),
)


@dataclass
class Media:
    fps: float
    stats: np.ndarray  # [frames, len(STAT_NAMES)]
    audio: np.ndarray  # [channels, samples] float32
    sample_rate: int
    stills: dict[int, Image.Image] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return len(self.stats)

    def col(self, name: str) -> np.ndarray:
        return self.stats[:, STAT_NAMES.index(name)]


@dataclass
class Join:
    index: int
    frame: int
    time_s: float
    continuity_250ms: float
    continuity_1s: float
    ncc_1s: float
    lag_ms: float
    loudness_pre: float
    loudness_post: float
    loudness_step: float
    loudness_step_1s: float
    floor_pre: float
    floor_post: float
    floor_step: float
    cut: float
    hold_frames: int
    hold_offset: int
    motion_ratio: float
    y_step: float
    cb_step: float
    cr_step: float
    sharp_log2: float
    y_frame_z: float
    chroma_frame_z: float
    replay_ncc: float | None = None
    replay_lag_ms: float | None = None


@dataclass
class Segment:
    index: int
    start: int
    frames: int
    y: float
    cb: float
    cr: float
    chroma_sd: float
    sharp: float
    motion: float
    held_pct: float
    loudness: float
    floor: float


@dataclass
class Report:
    video: str
    fps: float
    frames: int
    sample_rate: int
    joins: list[Join]
    segments: list[Segment]
    calibration: dict[str, float]
    trend: dict[str, float]


def read_media(path: Path, keep: frozenset[int] | set[int] = frozenset()) -> Media:
    rows: list[tuple[float, ...]] = []
    stills: dict[int, Image.Image] = {}
    prev: np.ndarray | None = None
    with av.open(str(path)) as box:
        video_stream = box.streams.video[0]
        video_stream.codec_context.thread_count = 1
        assert video_stream.average_rate is not None, "video has no average frame rate"
        fps = float(video_stream.average_rate)
        for n, video_frame in enumerate(box.decode(video_stream)):
            planes = video_frame.to_ndarray(format="yuv420p")
            h, w = video_frame.height, video_frame.width
            y = planes[:h].astype(np.float32)
            cb = planes[h : h + h // 4].reshape(h // 2, w // 2)
            cr = planes[h + h // 4 :].reshape(h // 2, w // 2)
            lap = 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
            small = y[: h // 4 * 4, : w // 4 * 4].reshape(h // 4, 4, w // 4, 4).mean((1, 3))
            motion = math.nan if prev is None else float(np.abs(small - prev).mean())
            prev = small
            rows.append(tuple(float(v) for v in (y.mean(), cb.mean(), cr.mean(), y.std(), cb.std(), cr.std(), lap.var(), motion)))
            if n in keep:
                to_image = cast(Callable[[], Image.Image], video_frame.to_image)
                stills[n] = to_image().resize((w // 4, h // 4))
    with av.open(str(path)) as box:
        audio_stream = box.streams.audio[0]
        rate = audio_stream.codec_context.sample_rate
        chunks: list[np.ndarray] = []
        start: int | None = None
        for audio_frame in box.decode(audio_stream):
            pcm = audio_frame.to_ndarray().astype(np.float32)
            if not audio_frame.format.is_planar:
                pcm = pcm.reshape(-1, len(audio_frame.layout.channels)).T
            if start is None:
                if audio_frame.pts is None:
                    start = 0
                else:
                    assert audio_frame.time_base is not None, "audio PTS has no time base"
                    start = round(float(audio_frame.pts * audio_frame.time_base) * rate)
            chunks.append(pcm)
        audio = np.concatenate(chunks, axis=1)
        if start and start > 0:
            audio = np.pad(audio, ((0, 0), (start, 0)))
        elif start and start < 0:
            audio = audio[:, -start:]
    return Media(fps, np.array(rows, dtype=np.float64), audio, rate, stills)


def run_joins(run: int, fps: float) -> list[int]:
    shown = subprocess.run(
        ["cozy", "run", "show", str(run), "--json", "--full"], check=True, capture_output=True, text=True
    )
    ends = sorted(
        {
            round(e["payload"]["duration_us"] * fps / 1e6)
            for e in json.loads(shown.stdout)["events"]
            if e["type"] == "output_item.delta"
            and e["payload"].get("output") == "video"
            and "duration_us" in e["payload"]
        }
    )
    if not ends:
        raise SystemExit(f"run {run} records no per-segment video revisions; pass --joins")
    return ends[:-1]


def k_weighted(audio: np.ndarray, rate: int) -> np.ndarray:
    n = audio.shape[1]
    z = np.exp(1j * 2 * np.pi * np.fft.rfftfreq(n, 1 / rate) / 48000)
    response = np.ones_like(z)
    for b, a in K_STAGES:
        response *= np.polyval(b[::-1], 1 / z) / np.polyval(a[::-1], 1 / z)
    return np.fft.irfft(np.fft.rfft(audio, axis=1) * response, n, axis=1).astype(np.float32)


def lufs(kw: np.ndarray) -> float:
    power = float((kw.astype(np.float64) ** 2).mean(axis=1).sum()) if kw.size else 0.0
    return -0.691 + 10 * math.log10(power) if power > 0 else -math.inf


def gated_lufs(kw: np.ndarray, rate: int) -> float:
    block, hop = int(0.4 * rate), int(0.1 * rate)
    levels = np.array([lufs(kw[:, s : s + block]) for s in range(0, kw.shape[1] - block + 1, hop)])
    levels = levels[levels > -70]
    if not len(levels):
        return -math.inf
    def mean(v: np.ndarray) -> float:
        return -0.691 + 10 * math.log10(float(np.mean(10 ** ((v + 0.691) / 10))))

    return mean(levels[levels > mean(levels) - 10])


def floor_db(kw: np.ndarray, rate: int) -> float:
    block = int(0.05 * rate)
    levels = [lufs(kw[:, s : s + block]) for s in range(0, kw.shape[1] - block + 1, block)]
    return float(np.percentile(levels, 5)) if levels else math.nan


def ncc(pre: np.ndarray, post: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Peak of the normalized cross-correlation; lag > 0: post repeats pre that many samples later."""
    pre, post = pre - pre.mean(), post - post.mean()
    norm = math.sqrt(float((pre**2).sum() * (post**2).sum()))
    if norm == 0 or 10 * math.log10(norm / math.sqrt(len(pre) * len(post)) + 1e-30) < SILENT_DBFS:
        return math.nan, 0
    n = len(pre) + len(post)
    c = np.fft.irfft(np.conj(np.fft.rfft(pre, n)) * np.fft.rfft(post, n), n)
    c = np.concatenate([c[-max_lag:], c[: max_lag + 1]]) / norm
    k = int(np.argmax(c))
    return float(c[k]), k - max_lag


def continuity(mono: np.ndarray, s: int, w: int) -> tuple[float, float, int]:
    """NCC across s, divided by the mean NCC at s-w and s+w (same material, no join): ~1 continuous."""
    if s - 2 * w < 0 or s + 2 * w > len(mono):
        return math.nan, math.nan, 0
    peak, lag = ncc(mono[s - w : s], mono[s : s + w], w // 2)
    controls = [ncc(mono[t - w : t], mono[t : t + w], w // 2)[0] for t in (s - w, s + w)]
    control = np.nanmean(controls) if any(math.isfinite(c) for c in controls) else 0.0
    return peak / max(float(control), 0.05), peak, lag


def longest_hold(motion: np.ndarray, lo: int, hi: int, threshold: float) -> tuple[int, int]:
    best, best_at, run = 0, 0, 0
    for i in range(lo, hi):
        run = run + 1 if motion[i] < threshold else 0
        if run > best:
            best, best_at = run, i - run + 1
    return best, best_at


def window_mean(x: np.ndarray, lo: int, hi: int) -> float:
    return float(np.nanmean(x[max(lo, 0) : min(hi, len(x))]))


def analyse(
    label: str,
    media: Media,
    joins: list[int],
    hold_mad: float,
    untrimmed: list[Media] | None = None,
    context_frames: int = 22,
) -> Report:
    fps, rate = media.fps, media.sample_rate
    bounds = [0, *joins, media.frames]
    inside = np.ones(media.frames, bool)
    inside[[0, *joins]] = False
    mono = media.audio.mean(0)
    kw = k_weighted(media.audio, rate)
    motion = media.col("motion")
    w1, wq, span = rate, rate // 4, int(FLOOR_SPAN_S * rate)

    def sample(frame: int) -> int:
        return round(frame * rate / fps)

    def level(lo: int, hi: int) -> float:
        return lufs(kw[:, max(lo, 0) : hi])

    def step(name: str, j: int) -> float:
        x, v = media.col(name), VISUAL_WINDOW
        return window_mean(x, j, j + v) - window_mean(x, j - v, j)

    def frame_z(name: str, j: int, lo: int, hi: int) -> float:
        x = media.col(name)
        steps = np.abs(np.diff(x[lo:hi]))[inside[lo + 1 : hi]]
        return abs(float(x[j] - x[j - 1])) / max(float(np.percentile(steps, 95)), 1e-6)

    out: list[Join] = []
    for i, j in enumerate(joins):
        lo, hi = bounds[i], bounds[i + 2]
        s, v = sample(j), VISUAL_WINDOW
        c1, peak, lag = continuity(mono, s, w1)
        cq, _, _ = continuity(mono, s, wq)
        held, at = longest_hold(motion, max(j - HOLD_SPAN, 1), min(j + HOLD_SPAN, media.frames), hold_mad)
        chroma = math.hypot(frame_z("cb", j, lo, hi), frame_z("cr", j, lo, hi))
        sharp = media.col("sharp")
        typical = float(np.nanmedian(motion[lo:hi][inside[lo:hi]]))
        lp, lq = level(s - wq, s), level(s, s + wq)
        fp, fq = floor_db(kw[:, max(s - span, 0) : s], rate), floor_db(kw[:, s : s + span], rate)
        join = Join(
            index=i + 1, frame=j, time_s=round(j / fps, 3),
            continuity_250ms=cq, continuity_1s=c1, ncc_1s=peak, lag_ms=1000 * lag / rate,
            loudness_pre=lp, loudness_post=lq, loudness_step=lq - lp,
            loudness_step_1s=level(s, s + w1) - level(s - w1, s),
            floor_pre=fp, floor_post=fq, floor_step=fq - fp,
            cut=float(motion[j] / max(typical, 1e-6)), hold_frames=held, hold_offset=at - j if held else 0,
            motion_ratio=window_mean(motion, j + 1, j + 2 * v + 1) / max(window_mean(motion, j - 2 * v, j), 1e-6),
            y_step=step("y", j), cb_step=step("cb", j), cr_step=step("cr", j),
            sharp_log2=math.log2(window_mean(sharp, j, j + v) / window_mean(sharp, j - v, j)),
            y_frame_z=frame_z("y", j, lo, hi), chroma_frame_z=chroma,
        )
        if untrimmed:
            seg = untrimmed[i]
            n = round(context_frames * rate / fps)
            replay = seg.audio.mean(0)[:n]
            join.replay_ncc, replay_lag = ncc(mono[s - n : s], replay, round(0.05 * rate))
            join.replay_lag_ms = 1000 * replay_lag / rate
        out.append(join)

    segments: list[Segment] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        col = {name: float(np.nanmedian(media.col(name)[lo:hi])) for name in STAT_NAMES}
        moving = motion[lo + 1 : hi]
        segments.append(Segment(
            index=i + 1, start=lo, frames=hi - lo, y=col["y"], cb=col["cb"], cr=col["cr"],
            chroma_sd=math.hypot(col["cb_sd"], col["cr_sd"]), sharp=col["sharp"], motion=col["motion"],
            held_pct=100 * float((moving < hold_mad).mean()) if len(moving) else 0.0,
            loudness=gated_lufs(kw[:, sample(lo) : sample(hi)], rate),
            floor=floor_db(kw[:, sample(lo) : sample(hi)], rate),
        ))
    return Report(
        label, fps, media.frames, rate, out, segments,
        calibrate(mono, kw, bounds, rate, fps), trend(out, segments),
    )


def calibrate(mono: np.ndarray, kw: np.ndarray, bounds: list[int], rate: int, fps: float) -> dict[str, float]:
    """The file's own material: points inside segments (true continuation) and tails spliced
    onto non-adjacent segment heads (same piece, not a continuation)."""
    result: dict[str, float] = {}
    starts = [round(b * rate / fps) for b in bounds]
    q = rate // 4
    steps = [
        abs(lufs(kw[:, s : s + q]) - lufs(kw[:, s - q : s]))
        for a, b in itertools.pairwise(starts)
        for s in range(a + q, b - q + 1, q)
    ]
    result["abs_loudness_step_in_segment_median"] = float(np.nanmedian(np.where(np.isinf(steps), np.nan, steps)))
    for label, w in (("250ms", rate // 4), ("1s", rate)):
        inner = [
            continuity(mono, s, w)[0]
            for a, b in itertools.pairwise(starts)
            for s in range(a + 2 * w, b - 2 * w + 1, rate)
        ]
        spliced = [
            continuity(np.concatenate([mono[t - 2 * w : t], mono[h : h + 2 * w]]), 2 * w, w)[0]
            for i, t in enumerate(starts[1:-1])
            for k, h in enumerate(starts[1:-1])
            if k != i and t >= 2 * w and h + 2 * w <= len(mono)
        ]
        result[f"continuity_{label}_in_segment_median"] = float(np.nanmedian(inner)) if inner else math.nan
        result[f"continuity_{label}_spliced_median"] = float(np.nanmedian(spliced)) if spliced else math.nan
        result[f"n_{label}_in_segment"], result[f"n_{label}_spliced"] = len(inner), len(spliced)
    return result


def slope(values: list[float]) -> float:
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    ok = np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else math.nan


def trend(joins: list[Join], segments: list[Segment]) -> dict[str, float]:
    seg = {k: slope([getattr(s, k) for s in segments]) for k in ("y", "chroma_sd", "loudness", "floor")}
    seg["sharp_log2"] = slope([math.log2(s.sharp) for s in segments])
    per_join = {f"join_{k}": slope([getattr(j, k) for j in joins]) for k in ("continuity_1s", "cut", "y_step")}
    return {**{f"segment_{k}": v for k, v in seg.items()}, **per_join}


def finite(value: object) -> object:
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, float | np.floating):
        return round(float(value), 4) if math.isfinite(value) else None
    return value


def fmt(x: float | None, spec: str) -> str:
    if x is None or not math.isfinite(x):
        return "-".rjust(int("".join(c for c in spec.split(".")[0] if c.isdigit()) or 0))
    return format(x, spec)


def print_report(r: Report) -> None:
    print(f"{r.video}: {r.frames} frames @ {r.fps:g} fps, {len(r.joins)} joins, audio {r.sample_rate} Hz")
    print(" join  frame   time  cont.25 cont1s  ncc1s  lag_ms   dLU  dFloor   cut hold  motion   dY    dCb   dCr  sharp"
          + ("  replay lag_ms" if r.joins and r.joins[0].replay_ncc is not None else ""))
    for j in r.joins:
        line = (f" {j.index:4d} {j.frame:6d} {j.time_s:6.1f}  {fmt(j.continuity_250ms, '6.2f')} "
                f"{fmt(j.continuity_1s, '6.2f')} {fmt(j.ncc_1s, '6.2f')} {fmt(j.lag_ms, '+7.1f')} "
                f"{fmt(j.loudness_step, '+5.1f')} {fmt(j.floor_step, '+6.1f')} {fmt(j.cut, '5.1f')} "
                f"{j.hold_frames:4d} {fmt(j.motion_ratio, '6.2f')} {j.y_step:+6.1f} {j.cb_step:+5.1f} "
                f"{j.cr_step:+5.1f} {j.sharp_log2:+5.2f}")
        if j.replay_ncc is not None:
            line += f"  {fmt(j.replay_ncc, '6.2f')} {fmt(j.replay_lag_ms, '+6.1f')}"
        print(line)
    print("  seg  start frames     Y     Cb     Cr chroma  sharp motion held%   LUFS  floor")
    for s in r.segments:
        print(f" {s.index:4d} {s.start:6d} {s.frames:6d} {s.y:5.1f} {s.cb:6.1f} {s.cr:6.1f} {s.chroma_sd:6.2f} "
              f"{s.sharp:6.1f} {s.motion:6.2f} {s.held_pct:5.1f} {fmt(s.loudness, '6.1f')} {fmt(s.floor, '6.1f')}")
    def median(key: str) -> float:
        return float(np.nanmedian([getattr(j, key) for j in r.joins])) if r.joins else math.nan

    print("median over joins: " + ", ".join(
        f"{k} {fmt(median(k), '.2f')}" for k in ("continuity_250ms", "continuity_1s", "floor_step", "cut"))
        + f", |dLU| {fmt(float(np.nanmedian([abs(j.loudness_step) for j in r.joins])) if r.joins else math.nan, '.1f')}")
    c = r.calibration
    print(f"calibration (this file): continuity in-segment {fmt(c['continuity_250ms_in_segment_median'], '.2f')}"
          f" / {fmt(c['continuity_1s_in_segment_median'], '.2f')} (0.25 s / 1 s), spliced non-adjacent segments "
          f"{fmt(c['continuity_250ms_spliced_median'], '.2f')} / {fmt(c['continuity_1s_spliced_median'], '.2f')}; "
          f"in-segment |dLU| median {fmt(c['abs_loudness_step_in_segment_median'], '.1f')}")
    print("trend per segment: " + ", ".join(f"{k.removeprefix('segment_')} {fmt(v, '+.3f')}"
                                           for k, v in r.trend.items() if k.startswith("segment_"))
          + "; per join: " + ", ".join(f"{k.removeprefix('join_')} {fmt(v, '+.3f')}"
                                       for k, v in r.trend.items() if k.startswith("join_")))


def write_stills(media: Media, joins: list[int], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, j in enumerate(joins):
        frames = [media.stills[f] for f in still_frames(j, media.frames) if f in media.stills]
        strip = Image.new("RGB", (sum(f.width for f in frames), frames[0].height))
        x = 0
        for f in frames:
            strip.paste(f, (x, 0))
            x += f.width
        strip.save(directory / f"join-{i + 1:02d}-f{j}.jpg", quality=85)


def still_frames(j: int, total: int) -> list[int]:
    return [f for f in (j - VISUAL_WINDOW, j - 1, j, j + 1, j + VISUAL_WINDOW) if 0 <= f < total]


def encode(path: Path, pictures: list[np.ndarray], audio: np.ndarray, fps: int, rate: int) -> None:
    with av.open(str(path), "w") as box:
        video = box.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = pictures[0].shape[1], pictures[0].shape[0], "yuv420p"
        sound = box.add_stream("aac", rate=rate, layout="stereo")
        for p in pictures:
            box.mux(video.encode(av.VideoFrame.from_ndarray(p, format="rgb24")))
        box.mux(video.encode())
        for s in range(0, audio.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio[:, s : s + 1024]), format="fltp", layout="stereo")
            frame.sample_rate = rate
            box.mux(sound.encode(frame))
        box.mux(sound.encode())


def synthetic(seed: int, frames: int, fps: int, rate: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    coarse = Image.fromarray(rng.integers(0, 256, (12, 40, 3), dtype=np.uint8))
    texture = np.asarray(coarse.resize((640, 192), Image.Resampling.BICUBIC))
    tint = rng.uniform(0.5, 1.0, 3)
    pictures = [(texture[:, 2 * f : 2 * f + 320] * tint).astype(np.uint8) for f in range(frames)]
    t = np.arange(round(frames * rate / fps)) / rate
    pitch = rng.uniform(150, 400)
    tone = sum(np.sin(2 * np.pi * pitch * h * t + rng.uniform(0, 6)) / h for h in (1, 2, 3, 5))
    wave = 0.2 * tone * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(1, 3) * t)) + 0.02 * rng.standard_normal(len(t))
    return pictures, np.stack([wave, wave]).astype(np.float32)


def self_test(hold_mad: float) -> None:
    fps, rate, half = 24, 32000, 72
    a_pics, a_wave = synthetic(1, 2 * half, fps, rate)
    b_pics, b_wave = synthetic(2, 2 * half, fps, rate)
    cut_at = round(half * rate / fps)
    cases = {
        "continuous split": (a_pics, a_wave),
        "hard cut": (a_pics[:half] + b_pics[half:], np.concatenate([a_wave[:, :cut_at], b_wave[:, cut_at:]], 1)),
    }
    with tempfile.TemporaryDirectory() as tmp:
        result = {}
        for name, (pictures, wave) in cases.items():
            path = Path(tmp) / f"{name.replace(' ', '-')}.mp4"
            encode(path, pictures, wave, fps, rate)
            join = analyse(name, read_media(path), [half], hold_mad).joins[0]
            result[name] = join
            print(f"{name:17s} continuity {join.continuity_250ms:.2f} / {join.continuity_1s:.2f}  "
                  f"ncc {join.ncc_1s:.2f}  cut {join.cut:.1f}  dY {join.y_step:+.1f}")
    cont, cut = result["continuous split"], result["hard cut"]
    ok = cont.continuity_1s > 0.8 and cut.continuity_1s < 0.4 and cont.cut < 2 and cut.cut > 3
    print("self-test", "passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--joins", help="comma-separated first frame of segments 2..N")
    parser.add_argument("--run", type=int, help="read joins from this run's per-segment video revisions")
    parser.add_argument("--untrimmed", nargs="+", type=Path, help="per join: next segment's untrimmed output")
    parser.add_argument("--context-frames", type=int, default=22)
    parser.add_argument("--hold-mad", type=float, default=0.8,
                        help="held frame: mean |luma step| at 1/4 resolution below this (0-255)")
    parser.add_argument("--frames", type=Path, help="write a frame strip per join to this directory")
    parser.add_argument("--json", metavar="FILE", help="also write the report as JSON ('-': stdout only)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.hold_mad)
    if args.video is None:
        parser.error("video is required")
    with av.open(str(args.video)) as box:
        average_rate = box.streams.video[0].average_rate
        assert average_rate is not None, "video has no average frame rate"
        fps = float(average_rate)
    if args.joins:
        joins = sorted(int(x) for x in args.joins.split(","))
    else:
        named = re.fullmatch(r"(\d+)-video\.mp4", args.video.name)
        run = args.run or (int(named.group(1)) if named else None)
        if run is None:
            parser.error("pass --joins or --run (or a <run>-video.mp4 file)")
        joins = run_joins(run, fps)
    keep = {f for j in joins for f in still_frames(j, 1 << 30)} if args.frames else set()
    media = read_media(args.video, keep)
    if joins and not (joins[0] > 0 and joins[-1] < media.frames):
        raise SystemExit(f"joins {joins} fall outside {media.frames} frames")
    untrimmed = [read_media(p) for p in args.untrimmed] if args.untrimmed else None
    if untrimmed and len(untrimmed) != len(joins):
        raise SystemExit(f"--untrimmed needs one file per join ({len(joins)})")
    report = analyse(str(args.video), media, joins, args.hold_mad, untrimmed, args.context_frames)
    if args.frames:
        write_stills(media, joins, args.frames)
    if args.json:
        text = json.dumps(finite(asdict(report)), indent=1)
        if args.json == "-":
            print(text)
        else:
            Path(args.json).write_text(text + "\n")
    if args.json != "-":
        print_report(report)


if __name__ == "__main__":
    main()
