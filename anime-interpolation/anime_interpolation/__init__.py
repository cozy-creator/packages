"""Anime frame interpolation for short held drawings and doubled playback rates."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import msgspec
from cozy_runtime.author import App, Outputs
from cozy_runtime.fields import FileBound
from cozy_runtime.media import VideoFile

from .model import Rife
from .timing import transform

app = App()


class FrameRate(StrEnum):
    KeepSource = "Keep source frame rate"
    Double = "Double source frame rate"


class UpscaleInput(msgspec.Struct, kw_only=True):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 40, max_decoded_bytes=2 << 30)]
    frame_rate: FrameRate = FrameRate.KeepSource


class UpscaleOutput(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 40, media_types=("video/mp4",))]


@app.tool(
    label="Anime Frame Interpolation",
    named="video",
    defaults={"model": "hzwer/practical-rife@0.1.0"},
    memoize=True,
)
def upscale(payload: UpscaleInput, model: Rife, out: Outputs) -> UpscaleOutput:
    """Smooth short held drawings while preserving cuts, long holds, and the soundtrack.

    Keep the source frame rate or double it, for example from 24 fps to 48 fps.
    """
    with payload.video.stream() as source:
        saved = out.save_video_stream(
            transform(
                iter(source), model.interpolate, double=payload.frame_rate == FrameRate.Double
            )
        )
    return UpscaleOutput(saved.video)
