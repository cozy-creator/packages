"""The soundtrack's refusals, before any frame is encoded."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from cozy_runtime.author import OutputError, Telemetry

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class MediaFacts:
    """The requested clock: the frame count and rate a soundtrack must agree with."""

    frames: int
    fps: int
    sample_rate: int

    @property
    def duration(self) -> Fraction:
        return Fraction(self.frames, self.fps)

    @property
    def av_tolerance(self) -> Fraction:
        return Fraction(1, self.fps)

    def av_agrees(self, audio_seconds: Fraction) -> bool:
        return abs(audio_seconds - self.duration) <= self.av_tolerance


def refuse_before_encode(
    *,
    waveform: torch.Tensor,
    audio_nonfinite_fraction: float,
    requested: MediaFacts,
    tel: Telemetry,
) -> None:
    """Refuse a malformed or non-finite soundtrack, or one off the video clock, before a
    frame is decoded or encoded: everything here is known once the audio has landed.
    Non-finite video is refused by the decode-to-encode stream at the chunk it appears in.
    """
    if waveform.ndim != 2:
        raise OutputError(
            f"the audio decode has shape {tuple(waveform.shape)}, expected (channels, samples)",
            code="output_integrity",
        )
    tel.metric("audio_nonfinite_fraction", round(audio_nonfinite_fraction, 6))
    if audio_nonfinite_fraction:
        raise OutputError(
            "the official H3 audio decode produced non-finite values", code="output_integrity"
        )
    channels, samples = (int(value) for value in waveform.shape)
    if channels not in (1, 2):
        raise OutputError(
            f"the generated soundtrack has {channels} channels, expected mono or stereo",
            code="output_integrity",
        )
    audio_duration = Fraction(samples, requested.sample_rate)
    tel.metric("audio_seconds", round(float(audio_duration), 4))
    if not requested.av_agrees(audio_duration):
        raise OutputError(
            f"the generated soundtrack is {float(audio_duration):.3f}s against "
            f"{float(requested.duration):.3f}s of video",
            code="output_integrity",
        )
