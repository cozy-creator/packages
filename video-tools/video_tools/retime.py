"""Render one shared source-to-output time map for picture and sound."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise

import numpy as np
from cozy_runtime.author import InvalidRequest
from cozy_runtime.media import (
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedVideoFrame,
)
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class TimeMap:
    """Sampled monotone map. Its endpoints are the source and output durations."""

    def __init__(self, source: FloatArray, output: FloatArray):
        self.source = source
        self.output = output

    def time(self, source_second: float) -> float:
        return float(np.interp(source_second, self.source, self.output))

    def original(self, times: FloatArray) -> FloatArray:
        return np.interp(times, self.output, self.source)


def _keyframes(
    points: Sequence[tuple[float, float]],
    duration: float,
    default: float,
) -> list[tuple[float, float]]:
    ordered = sorted(points)
    if not ordered or ordered[0][0] > 0:
        ordered.insert(0, (0.0, default))
    if any(not 0 <= source <= duration for source, _ in ordered) or any(
        a[0] >= b[0] for a, b in pairwise(ordered)
    ):
        raise InvalidRequest(
            "Choose different source times within the video.", fields=["keyframes"]
        )
    return ordered


def remap(points: Sequence[tuple[float, float]], duration: float) -> TimeMap:
    keyframes = _keyframes(points, duration, 0.0)
    if keyframes[0][1] != 0.0 or any(a[1] >= b[1] for a, b in pairwise(keyframes)):
        raise InvalidRequest(
            "Output times must start at zero and increase in source order.", fields=["keyframes"]
        )
    last_source, last_output = keyframes[-1]
    if last_source < duration:
        keyframes.append((duration, last_output + duration - last_source))
    samples = np.asarray(keyframes, dtype=np.float64)
    return TimeMap(samples[:, 0], samples[:, 1])


def ramp(points: Sequence[tuple[float, float]], duration: float) -> TimeMap:
    keyframes = _keyframes(points, duration, 1.0)
    if keyframes[-1][0] < duration:
        keyframes.append((duration, keyframes[-1][1]))
    # Speed follows a symmetric smoothstep in output time between each pair of points.
    # Its integral sets the source position, and its mean gives the exact new duration.
    part = np.linspace(0.0, 1.0, 257, dtype=np.float64)[1:]
    antiderivative = part**3 - 0.5 * part**4
    sources = [np.zeros(1, dtype=np.float64)]
    outputs = [np.zeros(1, dtype=np.float64)]
    cursor = 0.0
    for (first, initial), (last, final) in pairwise(keyframes):
        elapsed = (last - first) / ((initial + final) / 2)
        original = first + elapsed * (initial * part + (final - initial) * antiderivative)
        original[-1] = last
        sources.append(original)
        outputs.append(cursor + elapsed * part)
        cursor += elapsed
    return TimeMap(np.concatenate(sources), np.concatenate(outputs))


def _audio(
    data: DecodedAudioChunk | None,
    fmt: DecodedAudioFormat,
    start: int,
    stop: int,
    curve: TimeMap,
    origin: Fraction,
) -> Iterator[DecodedAudioChunk]:
    rate = fmt.sample_rate
    source = np.arange(data.sample_count if data else 0, dtype=np.float64)
    channels = tuple(np.frombuffer(block, dtype="<f4") for block in data.pcm_f32le) if data else ()
    offset = float(data.start_time - origin) if data else 0.0
    for pos in range(start, stop, 16384):
        end = min(stop, pos + 16384)
        count = end - pos
        if data is None:
            pcm = (bytes(count * 4),) * fmt.channels
        else:
            samples = (curve.original(np.arange(pos, end, dtype=np.float64) / rate) - offset) * rate
            pcm = tuple(
                np.interp(samples, source, channel).astype("<f4").tobytes() for channel in channels
            )
        yield DecodedAudioChunk(
            fmt.channels,
            count,
            rate,
            fmt.channel_layout,
            fmt.channel_names,
            pcm,
            pos,
            fmt.time_base,
        )


def retimed(
    events: Iterator[DecodedMediaEvent], curve: TimeMap, checkpoint: Callable[[], None]
) -> Iterator[DecodedMediaEvent]:
    header = next(events, None)
    if not isinstance(header, DecodedMediaHeader) or header.video is None:
        raise InvalidRequest("The input has no video stream.", fields=["video"])
    rate = header.video.nominal_frame_rate or Fraction(24)
    video = replace(header.video, time_base=1 / rate, nominal_frame_rate=rate)
    audio = (
        None
        if header.audio is None
        else replace(header.audio, time_base=Fraction(1, header.audio.sample_rate))
    )
    yield DecodedMediaHeader(video, audio)

    total_frames = max(1, round(float(curve.output[-1]) * rate))
    total_samples = round(total_frames * audio.sample_rate / rate) if audio else 0
    origin: Fraction | None = None
    previous: DecodedVideoFrame | None = None
    pending: list[DecodedAudioChunk] = []
    next_frame = 0
    next_sample = 0

    def sound(chunk: DecodedAudioChunk) -> Iterator[DecodedAudioChunk]:
        nonlocal next_sample
        assert origin is not None and audio is not None
        rate = audio.sample_rate
        start = min(total_samples, round(curve.time(float(chunk.start_time - origin)) * rate))
        stop = min(
            total_samples,
            round(curve.time(float(chunk.start_time + chunk.duration - origin)) * rate),
        )
        for encoded in _audio(None, audio, next_sample, start, curve, origin):
            checkpoint()
            yield encoded
        for encoded in _audio(chunk, audio, max(start, next_sample), stop, curve, origin):
            checkpoint()
            yield encoded
        next_sample = max(next_sample, stop)

    for event in events:
        checkpoint()
        if isinstance(event, DecodedAudioChunk):
            if origin is None:
                pending.append(event)
            else:
                yield from sound(event)
        elif isinstance(event, DecodedVideoFrame):
            if origin is None:
                origin = event.start_time
                for chunk in pending:
                    yield from sound(chunk)
                pending.clear()
            elif previous is not None:
                boundary = curve.time(float(event.start_time - origin)) * rate
                while next_frame < total_frames and next_frame < boundary - 1e-8:
                    checkpoint()
                    yield replace(previous, pts=next_frame, duration=1, time_base=video.time_base)
                    next_frame += 1
            previous = event

    if origin is None or previous is None:
        raise InvalidRequest("The input has no video frames.", fields=["video"])
    while next_frame < total_frames:
        checkpoint()
        yield replace(previous, pts=next_frame, duration=1, time_base=video.time_base)
        next_frame += 1
    if audio is not None:
        for encoded in _audio(None, audio, next_sample, total_samples, curve, origin):
            checkpoint()
            yield encoded
