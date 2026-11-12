"""Upscale videos 4x with RelayVSR while retaining their presentation clock and soundtrack."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum
from typing import Annotated

import msgspec
import torch
import torch.nn.functional as F
from cozy_runtime import rng
from cozy_runtime.author import App, Context, Outputs, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, Ratios, Shape
from cozy_runtime.media import DecodedAudioChunk, DecodedMediaHeader, DecodedVideoFrame, VideoFile

from .flash import Condition, History
from .model import RelayModel, frame_tensor

app = App()


class Resolution(StrEnum):
    HD = "1080p"
    QHD = "1440p"
    UHD = "2160p"


_RESOLUTIONS: dict[Resolution, tuple[int, int]] = {
    Resolution.HD: (1920, 1080),
    Resolution.QHD: (2560, 1440),
    Resolution.UHD: (3840, 2160),
}


class UpscaleInput(msgspec.Struct, kw_only=True):
    #: `events`' stream.
    video: Annotated[VideoFile, FileBound(max_decoded_bytes=2 << 30)]
    #: `events`: the canvas's height (`output_sizes`).
    resolution: Annotated[Resolution, Shape(pixels=_RESOLUTIONS)] = Resolution.HD
    #: `upscale`: the keyframes' noise.
    seed: Seed = 42  # type: ignore[assignment]  # a literal: the interface is read from source (Seed is an int NewType)
    #: `output_sizes`: the box's shape, a letterboxed 16:9 canvas; unset, the source's shape.
    aspect_ratio: Annotated[AspectRatio, Ratios("16:9")] | None = None
    #: `events`: a keyframe every `window_size - 1` frames (upstream's `--window-size`).
    window_size: Annotated[int, msgspec.Meta(ge=2, le=64), Advanced()] = 16


class UpscaleOutput(msgspec.Struct):
    video: Annotated[
        VideoFile,
        FileBound(
            # TensorD's output allowance reads this (256 MiB where none is declared); the disk is the limit.
            max_bytes=1 << 40,
            max_decoded_bytes=32 << 30,
            media_types=("video/mp4",),
        ),
    ]
    frame_count: int
    fps: float
    warnings: list[str] = msgspec.field(default_factory=list)


def output_sizes(
    width: int, height: int, edge: int, aspect_ratio: AspectRatio | None
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The size fitted in the `edge`-high 16:9 canvas, and the encoder's: that canvas, or the size."""
    canvas = (edge * 16 // 9, edge)
    scale = min(canvas[0] / width, canvas[1] / height)
    fitted = (max(2, 2 * round(width * scale / 2)), max(2, 2 * round(height * scale / 2)))
    return fitted, fitted if aspect_ratio is None else canvas


def finish(x: torch.Tensor, width: int, height: int, fitted: tuple[int, int], canvas: tuple[int, int]) -> bytes:
    """A padded 4x frame in [-1, 1] as `canvas` RGB8: cropped to 4x the source, area-scaled to
    `fitted` (bicubic where that is larger), centred on black."""
    x = x[..., : 4 * height, : 4 * width].float()
    if (x.shape[-1], x.shape[-2]) != fitted:
        mode = "area" if fitted[0] < x.shape[-1] else "bicubic"
        x = F.interpolate(x, size=(fitted[1], fitted[0]), mode=mode)
    pixels = x.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    left, top = (canvas[0] - fitted[0]) // 2, (canvas[1] - fitted[1]) // 2
    pixels = F.pad(pixels, (left, canvas[0] - fitted[0] - left, top, canvas[1] - fitted[1] - top))
    return pixels[0].permute(1, 2, 0).contiguous().cpu().numpy().tobytes()


@app.tool(
    label="Upscale",
    named="video",
    defaults={"model": "kopper/relayvsr@0.2.0"},
    memoize=True,
)
def upscale(
    ctx: Context,
    payload: UpscaleInput,
    model: RelayModel,
    out: Outputs,
    tel: Telemetry,
) -> UpscaleOutput:
    """A video upscaled 4x by RelayVSR, then fitted to `resolution`.

    Wan2.2-TI2V-5B, distilled to one step, makes a reference latent every `window_size - 1`
    frames; a 32M-parameter decoder upscales every frame between two of them as a residual over
    bicubic, streaming. A video already that big comes back as it is."""
    probe = payload.video.probe()
    edge = int(payload.resolution.value[:-1])
    fitted, canvas = output_sizes(probe.width, probe.height, edge, payload.aspect_ratio)
    warnings: list[str] = []
    if fitted[0] <= probe.width and canvas == fitted:
        warnings.append(f"this video is {probe.width}x{probe.height}, already {payload.resolution}: it comes back as it is")
        with payload.video.stream() as stream:
            saved = out.save_video_stream(iter(stream))
        return UpscaleOutput(saved.video, saved.frame_count, float(saved.frame_rate), warnings)

    model.hold()
    device, dtype = torch.device(str(model.device)), model.dtype
    total = round(probe.duration * probe.frame_rate) if probe.frame_rate else 0
    generator = rng.generator(payload.seed)
    ring = model.ring()
    count = 0

    def events() -> Iterator[DecodedMediaHeader | DecodedVideoFrame | DecodedAudioChunk]:
        nonlocal count
        # A window is its start keyframe and the frames up to the next one: those are held (on the
        # device) until that keyframe is made, then each is upscaled and streamed, followed by the
        # sound up to its end. The start of every window but the first was its predecessor's end,
        # and is decoded again only to seed the decoder's history.
        audio: deque[DecodedAudioChunk] = deque()
        start: tuple[DecodedVideoFrame, torch.Tensor, Condition] | None = None
        pending: list[tuple[DecodedVideoFrame, torch.Tensor]] = []
        first = True

        def keyframe(tensor: torch.Tensor) -> Condition:
            return model.condition(model.keyframe(tensor, ring, generator))

        def window() -> Iterator[DecodedVideoFrame | DecodedAudioChunk]:
            nonlocal start, pending, first, count
            assert start is not None
            begin, tensor, condition = start
            end = keyframe(pending[-1][1]) if pending else condition
            history: History = ()
            for offset, (frame, x) in enumerate([(begin, tensor), *pending]):
                ctx.raise_if_cancelled()
                upscaled, history = model.decode(x, condition, end, history)
                if offset == 0 and not first:
                    continue
                count += 1
                rgb = finish(upscaled, frame.width, frame.height, fitted, canvas)
                yield replace(frame, width=canvas[0], height=canvas[1], rgb=rgb)
                if total:
                    ctx.progress(min(count, total) - 1, total)
                stop = (frame.pts + frame.duration) * frame.time_base
                while audio and audio[0].start_time < stop:
                    yield audio.popleft()
            first = False
            if pending:
                start = (*pending[-1], end)
            pending = []

        with tel.stage("upscaling frames"), payload.video.stream() as stream:
            for event in stream:
                ctx.raise_if_cancelled()
                if isinstance(event, DecodedMediaHeader):
                    if event.video is None:
                        raise ValueError("RelayVSR requires a video stream")
                    yield replace(event, video=replace(event.video, width=canvas[0], height=canvas[1]))
                elif isinstance(event, DecodedVideoFrame):
                    tensor = frame_tensor(event.rgb, event.width, event.height, device, dtype)
                    if start is None:
                        start = (event, tensor, keyframe(tensor))
                    else:
                        pending.append((event, tensor))
                        if len(pending) == payload.window_size - 1:
                            yield from window()
                else:
                    audio.append(event)
            if pending or first and start is not None:
                yield from window()
            yield from audio

    try:
        saved = out.save_video_stream(events())
    finally:
        ring.close()
    if saved.frame_count != count:
        raise RuntimeError("video encoder changed the upscaled frame count")
    return UpscaleOutput(saved.video, count, float(saved.frame_rate), warnings)
