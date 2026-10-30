"""Restore videos with SeedVR2 while retaining their presentation clock and soundtrack."""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum
from typing import Annotated, Literal

import msgspec
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as vision
from cozy_runtime.author import (
    App,
    FileBound,
    Context,
    DecodedAudioChunk,
    DecodedMediaHeader,
    DecodedVideoFrame,
    Outputs,
    Shape,
    Telemetry,
    VideoFile,
)
from .color import Correction
from .frames import extend
from .model import RestorationModel, restore as restore_window
from . import quantize as quantize  # its recipes
from .upstream.temporal import blend_overlapping_frames

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


class Tuning(msgspec.Struct, kw_only=True):
    """The ComfyUI SeedVR2 Video Upscaler node's controls, under its names (README: Tuning)."""

    color_correction: Correction = "lab"
    input_noise_scale: Annotated[float, msgspec.Meta(ge=0, le=1)] = 0.0
    latent_noise_scale: Annotated[float, msgspec.Meta(ge=0, le=1)] = 0.0
    batch_size: Annotated[int, msgspec.Meta(ge=1)] | None = None
    uniform_batch_size: bool = False
    temporal_overlap: Annotated[int, msgspec.Meta(ge=0, le=16)] = 0
    prepend_frames: Annotated[int, msgspec.Meta(ge=0, le=32)] = 0

    def __post_init__(self) -> None:
        if self.batch_size is not None and self.batch_size % 4 != 1:
            raise ValueError(
                f"batch_size must be 4n+1 (1, 5, 9, 13, 17, 21, ...), got {self.batch_size}"
            )


class UpscaleInput(Tuning, kw_only=True):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, max_decoded_bytes=2 << 30)]
    resolution: Annotated[Resolution, Shape(pixels=_RESOLUTIONS)] = Resolution.HD
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] = 42
    aspect_ratio: Literal["source", "16:9"] = "source"


class UpscaleOutput(msgspec.Struct):
    video: Annotated[
        VideoFile,
        FileBound(
            max_bytes=1 << 30,
            max_decoded_bytes=32 << 30,
            media_types=("video/mp4",),
        ),
    ]
    frame_count: int
    fps: float


def output_sizes(
    width: int, height: int, edge: int, aspect_ratio: Literal["source", "16:9"]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Restore at the fitted size; the encoder receives the requested even-sized canvas."""
    canvas = (edge * 16 // 9, edge)
    scale = min(canvas[0] / width, canvas[1] / height)
    restored = (max(2, 2 * round(width * scale / 2)), max(2, 2 * round(height * scale / 2)))
    return restored, canvas if aspect_ratio == "16:9" else restored


@app.entrypoint(
    defaults={"model": "bytedance-seed/seedvr2-7b@0.1.0"},
    memoize=True,
)
def upscale(
    ctx: Context,
    payload: UpscaleInput,
    model: RestorationModel,
    out: Outputs,
    tel: Telemetry,
) -> UpscaleOutput:
    """Restore with SeedVR2 7B: the whole clip as one window, or windows of at most `batch_size`
    frames that share `temporal_overlap`. The Runtime cuts the DiT's work to fit the device."""
    size, overlap = payload.batch_size or sys.maxsize, payload.temporal_overlap
    if overlap >= size:
        tel.log(
            "temporal_overlap is not below batch_size; using 0, as upstream does", level="warning"
        )
        overlap = 0
    count = 0

    def restore(batch: list[DecodedVideoFrame], width: int, height: int) -> list[Image.Image]:
        frames = [Image.frombytes("RGB", (f.width, f.height), f.rgb) for f in batch]
        if payload.uniform_batch_size and payload.batch_size:
            frames = extend(frames, size - len(frames))
        with tel.stage("restoring video window"):
            restored = restore_window(
                model,
                frames,
                width,
                height,
                payload.seed,
                tel,
                correction=payload.color_correction,
                input_noise=payload.input_noise_scale,
                latent_noise=payload.latent_noise_scale,
            )
        return restored[: len(batch)]

    def events() -> Iterator[DecodedMediaHeader | DecodedVideoFrame | DecodedAudioChunk]:
        nonlocal count
        # Only one batch and the previous batch's restored tail are held; the media writer
        # streams completed frames, each followed by the sound up to its end (the writer
        # interleaves both tracks within a bounded buffer). Prepended mirror frames are
        # restored, then dropped.
        pending: list[DecodedVideoFrame] = []
        audio: deque[DecodedAudioChunk] = deque()
        tail: list[Image.Image] = []
        skip = payload.prepend_frames
        prepended = skip == 0
        width = height = 0
        canvas = (0, 0)

        def flush(last: bool) -> Iterator[DecodedVideoFrame]:
            nonlocal tail, count, pending, skip
            ctx.raise_if_cancelled()
            batch = pending[:size]
            if tail and len(batch) == len(tail):
                restored = tail
            else:
                restored = restore(batch, width, height)
                if tail:
                    previous = torch.stack(
                        [vision.to_tensor(frame).permute(1, 2, 0) for frame in tail]
                    )
                    current = torch.stack(
                        [
                            vision.to_tensor(frame).permute(1, 2, 0)
                            for frame in restored[: len(tail)]
                        ]
                    )
                    blended = blend_overlapping_frames(previous, current, len(tail))
                    restored[: len(tail)] = [
                        vision.to_pil_image(frame.permute(2, 0, 1)) for frame in blended
                    ]
            final = last and len(pending) <= size
            emit = len(restored) if final else len(restored) - overlap
            for original, image in zip(batch[:emit], restored[:emit], strict=True):
                if skip:
                    skip -= 1
                    continue
                count += 1
                image = ImageOps.pad(image, canvas, method=Image.Resampling.LANCZOS, color="black")
                yield replace(original, width=canvas[0], height=canvas[1], rgb=image.tobytes())
                end = (original.pts + original.duration) * original.time_base
                while audio and audio[0].start_time < end:
                    yield audio.popleft()
            tail = [] if final else restored[emit:]
            pending = [] if final else pending[emit:]

        with payload.video.stream() as stream:
            for event in stream:
                ctx.raise_if_cancelled()
                if isinstance(event, DecodedMediaHeader):
                    if event.video is None:
                        raise ValueError("SeedVR2 requires a video stream")
                    edge = int(payload.resolution.value[:-1])
                    (width, height), canvas = output_sizes(
                        event.video.width, event.video.height, edge, payload.aspect_ratio
                    )
                    yield replace(
                        event, video=replace(event.video, width=canvas[0], height=canvas[1])
                    )
                elif isinstance(event, DecodedVideoFrame):
                    pending.append(event)
                    if not prepended and len(pending) > payload.prepend_frames:
                        pending, prepended = (
                            extend(pending, payload.prepend_frames, prepend=True),
                            True,
                        )
                    while prepended and len(pending) >= size:
                        yield from flush(False)
                else:
                    audio.append(event)
            if not prepended and pending:
                pending = extend(pending, payload.prepend_frames, prepend=True)
            while pending:
                yield from flush(True)
            yield from audio

    saved = out.save_video_stream(events())
    if saved.frame_count != count:
        raise RuntimeError("video encoder changed the restored frame count")
    return UpscaleOutput(saved.video, count, float(saved.frame_rate))
