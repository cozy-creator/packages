"""A shot's pieces joined as they land: each piece's H.264 packets copied, and its soundtrack,
which starts where the previous piece's ended (continuation cuts it so), kept whole on one
sample clock. No picture is decoded or encoded."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction

from cozy_runtime.author import DecodedAudioChunk, DecodedAudioFormat, OutputError, Outputs, SavedVideo, VideoFile, VideoJoin

from minimax_h3.official import FPS


class PieceJoin:
    """The pieces so far as one video; each `append` returns it, publishable as it grows."""

    def __init__(self, out: Outputs, check: Callable[[], None]) -> None:
        self.out, self.check = out, check
        self.join: VideoJoin | None = None
        self.audio: DecodedAudioFormat | None = None
        self.pieces = self.frames = self.samples = 0

    def append(self, video: VideoFile, frames: int, *, last: bool) -> VideoFile:
        """Join one piece of `frames` planned frames."""
        sound = video.probe().soundtrack
        if sound is None:
            raise OutputError(f"piece {self.pieces + 1} has no soundtrack", code="shot_frames")
        rate = sound.sample_rate
        audio = DecodedAudioFormat(channels=sound.channels, sample_rate=rate, channel_layout=sound.channel_layout,
                                   channel_names=sound.channel_names, time_base=Fraction(1, rate))  # fmt: skip
        if self.join is None:
            self.audio, self.join = audio, self.out.join_video(audio)
        elif audio != self.audio:
            raise OutputError(f"piece {self.pieces + 1}'s soundtrack differs from the first's", code="shot_frames")
        self.pieces += 1

        def soundtrack(copied: int) -> Iterator[DecodedAudioChunk]:
            if copied != frames:
                raise OutputError(f"piece {self.pieces} has {copied} frames, not its planned {frames}",
                                  code="shot_frames")  # fmt: skip
            with video.stream(pictures=False) as stream:
                for event in stream:
                    self.check()
                    if isinstance(event, DecodedAudioChunk):
                        yield DecodedAudioChunk(
                            channels=event.channels, sample_count=event.sample_count, sample_rate=rate,
                            channel_layout=event.channel_layout, channel_names=event.channel_names,
                            pcm_f32le=event.pcm_f32le, pts=self.samples, time_base=audio.time_base,
                        )  # fmt: skip
                        self.samples += event.sample_count
            self.frames += frames
            drift = self.samples - round(Fraction(self.frames * rate, FPS))
            if abs(drift) > rate // FPS:
                raise OutputError(f"after piece {self.pieces} the sound is {drift} samples off the picture",
                                  code="shot_frames")  # fmt: skip

        return self.join.append(video, soundtrack, last=last)

    def finish(self) -> SavedVideo:
        """The joined video: the last revision when the last append said so."""
        if self.join is None:
            raise OutputError("no piece was joined", code="shot_frames")
        return self.join.finish()
