"""Video-in, video-out tools for preparing Plates."""

from __future__ import annotations

from typing import Annotated

import msgspec
from cozy_runtime.author import App, Context, Outputs, Telemetry
from cozy_runtime.fields import FileBound, Label, Range
from cozy_runtime.media import VideoFile

from .retime import ramp, remap, retimed
from .stabilize import stabilized, transforms

app = App()

SourceVideo = Annotated[VideoFile, FileBound(max_bytes=1 << 40, max_decoded_bytes=1 << 40)]


class StabilizeInput(msgspec.Struct, kw_only=True):
    video: SourceVideo
    crop_to_fit: Annotated[bool, Label("Crop to fit")] = False
    smoothing: Annotated[float, Label("Smoothing"), Range(0.1, 3.0, unit="s")] = 0.5


class VideoOutput(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 40, media_types=("video/mp4",))]


@app.tool(label="Stabilize Video", named="stabilize")
def stabilize(ctx: Context, payload: StabilizeInput, out: Outputs, tel: Telemetry) -> VideoOutput:
    """Reduce camera shake while keeping the framing and soundtrack.

    Crop to fit zooms in enough to cover the borders. By default, exposed edges are black.
    """
    probe = payload.video.probe()
    rate = float(probe.frame_rate or 24)
    with tel.stage("analyze camera motion"), payload.video.stream(audio=False) as source:
        corrections = transforms(
            iter(source),
            round(rate * payload.smoothing),
            payload.crop_to_fit,
            ctx.raise_if_cancelled,
        )

    with tel.stage("stabilize video"), payload.video.stream() as source:
        frames = stabilized(iter(source), corrections, ctx.raise_if_cancelled)
        # Keep source presentation timing when the MP4 writer emits a constant-rate file.
        clock = remap((), float(probe.duration))
        saved = out.save_video_stream(retimed(frames, clock, ctx.raise_if_cancelled))
    return VideoOutput(saved.video)


class TimePoint(msgspec.Struct, kw_only=True):
    source_time: Annotated[float, Label("Source time"), Range(0.0, 86400.0, unit="s")]
    output_time: Annotated[float, Label("Output time"), Range(0.0, 86400.0, unit="s")]


class SpeedPoint(msgspec.Struct, kw_only=True):
    source_time: Annotated[float, Label("Source time"), Range(0.0, 86400.0, unit="s")]
    speed: Annotated[float, Label("Speed"), Range(0.125, 8.0, unit="x")] = 1.0


class RemapInput(msgspec.Struct, kw_only=True):
    video: SourceVideo
    keyframes: Annotated[list[TimePoint], Label("Time points")] = msgspec.field(
        default_factory=list
    )


class RampInput(msgspec.Struct, kw_only=True):
    video: SourceVideo
    keyframes: Annotated[list[SpeedPoint], Label("Speed points")] = msgspec.field(
        default_factory=list
    )


@app.tool(label="Time Remap", named="time-remap")
def time_remap(ctx: Context, payload: RemapInput, out: Outputs) -> VideoOutput:
    """Place chosen source moments at new output times.

    The video starts at zero. Between time points the speed is constant. After the last point,
    the remaining source plays at normal speed. The soundtrack is retimed with the picture.
    """
    duration = float(payload.video.probe().duration)
    curve = remap([(point.source_time, point.output_time) for point in payload.keyframes], duration)
    with payload.video.stream() as source:
        return VideoOutput(
            out.save_video_stream(retimed(iter(source), curve, ctx.raise_if_cancelled)).video
        )


@app.tool(label="Speed Ramp", named="speed-ramp")
def speed_ramp(ctx: Context, payload: RampInput, out: Outputs) -> VideoOutput:
    """Ease between playback speeds at chosen source times and render the new timing.

    The video starts at 1x unless you set its first point at zero. After the last point, its
    speed stays constant. The soundtrack is retimed with the picture.
    """
    duration = float(payload.video.probe().duration)
    curve = ramp([(point.source_time, point.speed) for point in payload.keyframes], duration)
    with payload.video.stream() as source:
        return VideoOutput(
            out.save_video_stream(retimed(iter(source), curve, ctx.raise_if_cancelled)).video
        )
