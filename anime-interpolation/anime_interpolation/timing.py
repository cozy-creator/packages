"""Fill held drawings on the original presentation clock; never interpolate across cuts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from fractions import Fraction

import numpy as np
import numpy.typing as npt
from cozy_runtime.author import InvalidRequest
from cozy_runtime.media import DecodedMediaEvent, DecodedMediaHeader, DecodedVideoFrame

Interpolate = Callable[[DecodedVideoFrame, DecodedVideoFrame, float], bytes]
Thumbnail = npt.NDArray[np.int16]

# Fill drawings exposed for about 1/8 second, including animation on threes at 23.976 fps.
# The independent count bound caps look-ahead at high input rates.
_MAX_HOLD = Fraction(1001, 8000)
_MAX_HELD_FRAMES = 8


def thumbnail(frame: DecodedVideoFrame) -> Thumbnail:
    # A regular sample retains edits to smaller foreground objects without allocating a full-size
    # difference image. Signed arithmetic avoids folding darkening into brightening on subtraction.
    rows = np.linspace(0, frame.height - 1, min(96, frame.height), dtype=np.intp)
    columns = np.linspace(0, frame.width - 1, min(128, frame.width), dtype=np.intp)
    image = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    return image[rows[:, None], columns].astype(np.int16)


def compare(left: Thumbnail, right: Thumbnail) -> tuple[bool, bool]:
    difference = np.abs(left - right).max(axis=2)
    # Lossy copies of the same drawing can differ slightly. The pixel threshold also prevents a
    # small moving character from being ignored just because most of the background is static.
    held = bool(
        difference.mean() < 2 and np.count_nonzero(difference > 12) <= difference.size / 500
    )
    cut = bool(difference.mean() >= 56 and np.count_nonzero(difference > 36) >= difference.size / 2)
    return held, cut


def span(
    frames: list[DecodedVideoFrame],
    end: DecodedVideoFrame | None,
    interpolate: Interpolate,
    double: bool,
) -> Iterator[DecodedVideoFrame]:
    start = frames[0]
    for index, frame in enumerate(frames):
        if end is not None and index:
            fraction = (frame.start_time - start.start_time) / (end.start_time - start.start_time)
            rgb = interpolate(start, end, float(fraction))
        else:
            rgb = frame.rgb
        if not double:
            yield replace(frame, rgb=rgb)
            continue

        yield replace(frame, rgb=rgb, pts=frame.pts * 2, time_base=frame.time_base / 2)
        halfway = frame.start_time + frame.duration * frame.time_base / 2
        if end is not None and halfway < end.start_time:
            fraction = (halfway - start.start_time) / (end.start_time - start.start_time)
            rgb = interpolate(start, end, float(fraction))
        yield replace(
            frame, rgb=rgb, pts=frame.pts * 2 + frame.duration, time_base=frame.time_base / 2
        )


def transform(
    source: Iterator[DecodedMediaEvent],
    interpolate: Interpolate,
    *,
    double: bool,
) -> Iterator[DecodedMediaEvent]:
    header = next(source, None)
    if not isinstance(header, DecodedMediaHeader) or header.video is None:
        raise InvalidRequest("The input needs a video stream", code="video_required")
    video = header.video
    rate = video.nominal_frame_rate
    if rate is None or not 0 < rate * (2 if double else 1) < 1000:
        raise InvalidRequest(
            "The input needs a supported constant frame rate", code="video_frame_rate"
        )
    if double:
        header = replace(
            header, video=replace(video, time_base=video.time_base / 2, nominal_frame_rate=rate * 2)
        )
    yield header

    frames: list[DecodedVideoFrame] = []
    anchor: Thumbnail | None = None
    first: Fraction | None = None
    long_hold = False
    count = 0
    clock_tolerance = min(video.time_base, 1 / (rate * 10))
    for event in source:
        if not isinstance(event, DecodedVideoFrame):
            yield event
            continue
        first = event.start_time if first is None else first
        # Runtime writes a fixed output clock. Refuse timing it would change, allowing one input
        # tick for rounded PTS in a coarser container clock such as millisecond Matroska.
        expected = first + count / rate
        end_time = event.start_time + event.duration * event.time_base
        if (
            abs(event.start_time - expected) > clock_tolerance
            or abs(end_time - expected - 1 / rate) > clock_tolerance
        ):
            raise InvalidRequest(
                "Anime Frame Interpolation needs a constant-rate video to preserve its timing",
                code="video_variable_frame_rate",
            )
        count += 1
        current = thumbnail(event)
        if anchor is not None:
            held, cut = compare(anchor, current)
            if held:
                frames.append(event)
                if (
                    long_hold
                    or len(frames) > _MAX_HELD_FRAMES
                    or end_time - frames[0].start_time > _MAX_HOLD + clock_tolerance
                ):
                    yield from span(frames, None, interpolate, double)
                    frames.clear()
                    long_hold = True
                continue
            if frames:
                fill = (
                    not cut
                    and not long_hold
                    and event.start_time - frames[0].start_time <= _MAX_HOLD + clock_tolerance
                )
                yield from span(frames, event if fill else None, interpolate, double)
        frames = [event]
        anchor = current
        long_hold = False
    if frames:
        yield from span(frames, None, interpolate, double)
