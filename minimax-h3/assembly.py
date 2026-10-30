"""One fixed, CPU-only long-video assembler over Runtime media events.

This is not an editor. It accepts a streamed list of completed shots, drops the first frame of
each one marked as overlapping the previous one, chooses segment audio or one exclusive master track, and
commits one deterministic MP4 through Runtime's streaming sink. Cuts with segment audio
copy the shots' H.264 packets unchanged and encode only the joined soundtrack.

It is H3's own module rather than a separate project (h3a-024): assembly is the second half
of `long_form`, and it adds no dependency the package
does not already carry — `cozy-runtime[media]` is already H3's decode and encode surface. As
a separate package it carried its own Runtime pin, nobody restamped it, and it rotted out of
reach at `cozy-runtime>=0.2.13` while the daemon moved to 0.10.0.

It declares NO model slot, so an attempt of it is CPU-only, and it is `@invocable`, so
`long_form` may one day await it as an ordinary self-call child instead of the caller
driving the second step. It holds no decoded segment: `save_video_stream` consumes the
generator, so exactly one frame is in memory at a time.
"""

from __future__ import annotations

import hashlib
import math
from array import array
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
from fractions import Fraction
from threading import Event
from types import TracebackType
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import (
    AssetBound,
    AudioAsset,
    Context,
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedVideoFormat,
    DecodedVideoFrame,
    InvalidRequest,
    OutputError,
    Outputs,
    SavedVideo,
    Telemetry,
    ThreadPoolExecutor,
    VideoAsset,
    VideoJoin,
    invocable,
)

MAX_INPUT_BYTES = 256 << 20
MAX_EVENT_BYTES = 32 << 20
LEVEL_WINDOW_SECONDS = Fraction(1, 2)
GAIN_RIDE_SECONDS = Fraction(2, 1)
MAX_GAIN_DB = 9.0
PEAK_CEILING = 0.999
FRAME_RATE = Fraction(24)


ShotVideo = Annotated[
    VideoAsset,
    AssetBound(max_bytes=MAX_INPUT_BYTES, max_decoded_bytes=MAX_EVENT_BYTES, media_types=("video/mp4",)),
]


class Shot(msgspec.Struct):
    """A video whose first frame repeats the previous video's last frame
    (`first_frame_overlaps_previous`): that frame and its stretch of sound are dropped once."""

    video: ShotVideo
    first_frame_overlaps_previous: bool = False


class AssembleVideoRequest(msgspec.Struct):
    """The shots in order. Every frame is kept unless a shot overlaps the previous one; with no
    overlap and no master track the packets are copied, not re-encoded. (Each is an object: a
    union of an asset and a struct does not decode, run 20669.)"""

    videos: Annotated[list[Shot], msgspec.Meta(min_length=1)]
    master_audio: Annotated[
        AudioAsset | None,
        AssetBound(max_bytes=MAX_INPUT_BYTES, max_decoded_bytes=MAX_EVENT_BYTES),
    ] = None


class SegmentReceipt(msgspec.Struct):
    digest: str
    size_bytes: int
    media_type: str
    source_frames: int
    selected_frames: int
    replay_frame_digest: str | None
    segment_soundtrack_samples: int
    selected_audio_samples: int
    selected_audio_source_start_sample: int
    selected_audio_source_stop_sample: int
    audio_trimmed_samples: int
    audio_padded_samples: int
    head_gain_db: float
    tail_gain_db: float
    head_gain_clamped: bool
    tail_gain_clamped: bool


class AssembleVideoResponse(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",), max_bytes=MAX_INPUT_BYTES)]
    segments: list[SegmentReceipt]
    audio_mode: Literal["segments", "master"]
    master_audio_digest: str | None
    source_frames: int
    output_frames: int
    replay_frames_removed: int
    submitted_audio_samples: int
    decoded_audio_samples: int
    audio_trimmed_samples: int
    audio_padded_samples: int
    audio_sample_rate: int
    audio_codec_frame_samples: int
    av_endpoint_delta_samples: int
    global_gain_db: float
    video_codec: str
    video_profile: str | None
    audio_codec: str
    audio_profile: str | None
    audio_channels: int
    width: int
    height: int
    frame_rate_numerator: int
    frame_rate_denominator: int
    pixel_aspect_ratio_numerator: int
    pixel_aspect_ratio_denominator: int
    color_primaries: int
    color_transfer: int
    color_matrix: int
    color_range: int


@dataclass(frozen=True, slots=True)
class _Scan:
    asset: VideoAsset
    header: DecodedMediaHeader
    frames: int
    audio_samples: int
    audio_head: tuple[bytes, ...]
    audio_tail: tuple[bytes, ...]
    audio_peak: float
    audio_head_peak: float
    first_frame_digest: str


@dataclass(frozen=True, slots=True)
class _AudioScan:
    header: DecodedAudioFormat
    samples: int


@dataclass(frozen=True, slots=True)
class _Gains:
    head_db: float = 0.0
    tail_db: float = 0.0
    head_clamped: bool = False
    tail_clamped: bool = False


@dataclass(frozen=True, slots=True)
class _AudioSelection:
    source_start: int
    selected: int
    from_source: int
    trimmed: int
    padded: int


def _nearest(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _rms(channels: tuple[bytes, ...], *, start: int = 0, samples: int | None = None) -> float:
    """Window RMS over f32 PCM, summed in C by `math.sumprod` rather than per sample.

    The gain ride calls this over 48 kHz windows; a Python-level loop over individual
    floats was the assembly path's largest CPU cost and bought nothing.
    """
    total = 0.0
    count = 0
    for raw in channels:
        values = array("f")
        values.frombytes(raw)
        stop = len(values) if samples is None else min(len(values), start + samples)
        window = values[start:stop]
        total += math.sumprod(window, window)
        count += len(window)
    return math.sqrt(total / count) if count else 0.0


def _scan_video(asset: VideoAsset, check: Callable[[], None], *, pictures: bool = True) -> _Scan:
    """Clocks and bounded audio windows; without `pictures` only the soundtrack is decoded
    (a packet copy's Runtime join checks the frame clock itself)."""
    header: DecodedMediaHeader | None = None
    frames = 0
    audio_samples = 0
    head: list[bytearray] = []
    tail: list[deque[float]] = []
    audio_peak = 0.0
    head_peak = 0.0
    head_peak_samples = 0
    first_frame_digest = ""
    head_limit = 0
    with asset.stream(pictures=pictures) as stream:
        for event in stream:
            check()
            if isinstance(event, DecodedMediaHeader):
                if pictures and (event.video is None or event.video.nominal_frame_rate is None):
                    raise InvalidRequest(
                        "video has no fixed nominal frame rate", code="invalid_request"
                    )
                header = event
                if event.audio is not None:
                    window = _nearest(LEVEL_WINDOW_SECONDS * event.audio.sample_rate)
                    ride_samples = _nearest(GAIN_RIDE_SECONDS * event.audio.sample_rate)
                    # Retain one extra frame so later-shot RMS can begin after the replay trim.
                    trim = (
                        _nearest(
                            Fraction(event.audio.sample_rate, 1) / event.video.nominal_frame_rate
                        )
                        if event.video is not None and event.video.nominal_frame_rate
                        else 0
                    )
                    head_peak_samples = ride_samples + trim
                    head_limit = window + trim
                    head = [bytearray() for _ in range(event.audio.channels)]
                    tail = [deque(maxlen=window) for _ in range(event.audio.channels)]
                continue
            if isinstance(event, DecodedVideoFrame):
                assert header is not None and header.video is not None
                rate = header.video.nominal_frame_rate
                assert rate is not None
                if (
                    event.start_time != Fraction(frames, 1) / rate
                    or event.duration * event.time_base != 1 / rate
                ):
                    raise InvalidRequest(
                        "video is not on the fixed zero-based frame clock",
                        code="invalid_request",
                    )
                if not first_frame_digest:
                    first_frame_digest = "sha256:" + hashlib.sha256(event.rgb).hexdigest()
                frames += 1
                continue
            assert header is not None and header.audio is not None
            if event.start_time != Fraction(audio_samples, header.audio.sample_rate):
                raise InvalidRequest(
                    "soundtrack is not on the fixed zero-based sample clock",
                    code="invalid_request",
                )
            peak = 0.0
            values_by_channel: list[array[float]] = []
            for raw in event.pcm_f32le:
                values = array("f")
                values.frombytes(raw)
                values_by_channel.append(values)
                peak = max(peak, abs(min(values, default=0.0)), abs(max(values, default=0.0)))
            chunk_start = audio_samples
            audio_peak = max(audio_peak, peak)
            if chunk_start < head_peak_samples:
                head_peak = max(head_peak, peak)
            audio_samples += event.sample_count
            for channel, raw in enumerate(event.pcm_f32le):
                if len(head[channel]) < head_limit * 4:
                    head[channel] += raw[: head_limit * 4 - len(head[channel])]
                tail[channel].extend(values_by_channel[channel])
    if header is None or (pictures and (header.video is None or frames == 0)):
        raise InvalidRequest("video decoded no usable stream", code="invalid_request")
    return _Scan(
        asset=asset,
        header=header,
        frames=frames,
        audio_samples=audio_samples,
        audio_head=tuple(bytes(channel) for channel in head),
        audio_tail=tuple(array("f", channel).tobytes() for channel in tail),
        audio_peak=audio_peak,
        audio_head_peak=head_peak,
        first_frame_digest=first_frame_digest,
    )


class ScanAhead:
    """Validate completed shots on one worker while the caller awaits later children.

    Futures retain only bounded audio windows and clocks, never decoded videos. Assets
    remain attempt-owned and are read again by the final encoder, including its normal
    byte-grant checks. Leaving the scope always joins the worker before attempt cleanup.
    """

    def __init__(self, check: Callable[[], None], *, pictures: bool = True) -> None:
        self._check = check
        self.pictures = pictures
        self._stopped = Event()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-scan")
        self._pending: list[Future[_Scan]] = []

    def __enter__(self) -> ScanAhead:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stopped.set()
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _check_active(self) -> None:
        self._check()
        if self._stopped.is_set():
            raise RuntimeError("assembly scan was abandoned")

    def add(self, asset: VideoAsset) -> None:
        self._check_active()
        # Surface an earlier validation failure before accepting more work.
        for pending in self._pending:
            if pending.done():
                pending.result()
        self._pending.append(
            self._pool.submit(_scan_video, asset, self._check_active, pictures=self.pictures)
        )

    def result(self, index: int) -> _Scan:
        """One added shot's scan, once it is done."""
        self._check_active()
        return self._pending[index].result()

    def finish(self, videos: Sequence[VideoAsset]) -> list[_Scan]:
        self._check_active()
        scans = [pending.result() for pending in self._pending]
        if len(scans) != len(videos) or any(
            scan.asset is not asset for scan, asset in zip(scans, videos, strict=True)
        ):
            raise InvalidRequest("assembly scan inputs changed", code="invalid_request")
        return scans


class CutJoin:
    """The cut join, made incremental. A segment's H.264 packets and its stretch of the
    soundtrack join one CMAF video when the segment lands, and the video joined so far is a
    revision the run publishes; the last revision is the joined video, byte for byte. A
    segment's peak gain is its own, so a published fragment never changes when a later
    segment is loud; it equals the one-shot join's unless a segment clips.

    `continuous` joins segments whose sound each starts where the previous one's ended (H3
    continuation cuts it so): every soundtrack is kept whole, and only the running clock
    must stay within a frame of the picture."""

    def __init__(
        self, out: Outputs, check: Callable[[], None], *, continuous: bool = False
    ) -> None:
        self.out, self.check = out, check
        self.continuous = continuous
        self.join: VideoJoin | None = None
        self.audio: DecodedAudioFormat | None = None
        self.scans: list[_Scan] = []
        self.output_audio = 0

    def append(self, scan: _Scan, frames: int, *, last: bool) -> VideoAsset:
        """Join one segment of `frames` planned frames; the video joined so far."""
        audio = scan.header.audio
        if audio is None or (self.audio is not None and not _same_audio(self.audio, audio)):
            raise InvalidRequest(
                "segment-audio assembly requires one common soundtrack on every video",
                code="invalid_request",
            )
        if self.join is None:
            self.audio, self.join = audio, self.out.join_video(_output_audio(audio))

        def soundtrack(copied: int) -> Iterator[DecodedMediaEvent]:
            if copied != frames:
                raise OutputError(
                    f"segment {len(self.scans) + 1} has {copied} frames, not its planned {frames}",
                    code="shot_frames",
                )
            self.scans.append(replace(scan, frames=copied))
            if self.continuous:
                selection = self._whole(scan, audio)
            else:
                tolerance = self.out.video_audio_frame_samples
                selection = _segment_selections(
                    self.scans, FRAME_RATE, audio, tolerance, [False] * len(self.scans)
                )[-1]
            gain = _global_gain(self.scans[-1:], [selection], [_Gains()], audio)
            segment = _SegmentAudio(audio, selection, _Gains(), gain, self.output_audio)
            with scan.asset.stream(pictures=False) as stream:
                for event in stream:
                    self.check()
                    if isinstance(event, DecodedAudioChunk) and (chunk := segment.take(event)):
                        yield chunk
            if (padding := segment.finish(self.scans[-1])) is not None:
                yield padding
            self.output_audio = segment.pts

        return self.join.append(scan.asset, soundtrack, last=last)

    def _whole(self, scan: _Scan, audio: DecodedAudioFormat) -> _AudioSelection:
        frames = sum(joined.frames for joined in self.scans)
        clock = _nearest(Fraction(frames * audio.sample_rate, 1) / FRAME_RATE)
        drift = self.output_audio + scan.audio_samples - clock
        if abs(drift) > _nearest(Fraction(audio.sample_rate, 1) / FRAME_RATE):
            raise InvalidRequest(
                f"segment {len(self.scans)} leaves the soundtrack {drift} samples off the "
                "picture",
                code="invalid_request",
            )
        return _AudioSelection(0, scan.audio_samples, scan.audio_samples, 0, 0)

    def finish(self) -> SavedVideo:
        """The joined video: the last revision when the last append said so."""
        if self.join is None:
            raise OutputError("no segment was joined", code="shot_frames")
        return self.join.finish()


def _scan_audio(asset: AudioAsset, check: Callable[[], None]) -> _AudioScan:
    header: DecodedAudioFormat | None = None
    samples = 0
    with asset.stream() as stream:
        for event in stream:
            check()
            if isinstance(event, DecodedMediaHeader):
                if event.audio is None or event.video is not None:
                    raise InvalidRequest(
                        "master audio has no audio-only header", code="invalid_request"
                    )
                header = event.audio
            else:
                assert isinstance(event, DecodedAudioChunk)
                assert header is not None
                if event.start_time != Fraction(samples, header.sample_rate):
                    raise InvalidRequest(
                        "master audio is not on a zero-based sample clock",
                        code="invalid_request",
                    )
                samples += event.sample_count
    if header is None or samples == 0:
        raise InvalidRequest("master audio decoded no samples", code="invalid_request")
    return _AudioScan(header, samples)


def _same_video(left: DecodedVideoFormat, right: DecodedVideoFormat) -> bool:
    return (
        left.width,
        left.height,
        left.time_base,
        left.pixel_aspect_ratio,
        left.nominal_frame_rate,
        left.color_primaries,
        left.color_transfer,
        left.color_matrix,
        left.color_range,
    ) == (
        right.width,
        right.height,
        right.time_base,
        right.pixel_aspect_ratio,
        right.nominal_frame_rate,
        right.color_primaries,
        right.color_transfer,
        right.color_matrix,
        right.color_range,
    )


def _same_audio(left: DecodedAudioFormat, right: DecodedAudioFormat) -> bool:
    return (
        left.channels,
        left.sample_rate,
        left.channel_layout,
        left.channel_names,
    ) == (
        right.channels,
        right.sample_rate,
        right.channel_layout,
        right.channel_names,
    )


def _segment_selections(
    scans: list[_Scan],
    rate: Fraction,
    audio: DecodedAudioFormat,
    tolerance: int,
    overlaps: Sequence[bool],
) -> list[_AudioSelection]:
    selections: list[_AudioSelection] = []
    output_frames = 0
    output_audio = 0
    for index, scan in enumerate(scans):
        expected_source = _nearest(Fraction(scan.frames * audio.sample_rate, 1) / rate)
        drift = scan.audio_samples - expected_source
        if abs(drift) > tolerance:
            raise InvalidRequest(
                f"segment {index + 1} soundtrack differs from its frame clock by {drift} samples",
                code="invalid_request",
            )
        selected_frames = scan.frames - int(overlaps[index])
        if selected_frames <= 0:
            raise InvalidRequest(
                f"segment {index + 1} has no frame after the fixed replay trim",
                code="invalid_request",
            )
        cumulative = _nearest(
            Fraction((output_frames + selected_frames) * audio.sample_rate, 1) / rate
        )
        selected = cumulative - output_audio
        source_start = (
            _nearest(Fraction(audio.sample_rate, 1) / rate) if overlaps[index] else 0
        )
        from_source = max(0, min(selected, scan.audio_samples - source_start))
        selections.append(
            _AudioSelection(
                source_start=source_start,
                selected=selected,
                from_source=from_source,
                trimmed=scan.audio_samples - from_source,
                padded=selected - from_source,
            )
        )
        output_frames += selected_frames
        output_audio = cumulative
    return selections


def _seam_gains(
    scans: list[_Scan], selections: list[_AudioSelection], audio: DecodedAudioFormat,
    overlaps: Sequence[bool],
) -> list[_Gains]:
    """Level-matched seams where a shot continues the previous one (it overlaps it)."""
    gains = [_Gains() for _ in scans]
    window = _nearest(LEVEL_WINDOW_SECONDS * audio.sample_rate)
    for index in range(len(scans) - 1):
        if not overlaps[index + 1]:
            continue
        left = _rms(scans[index].audio_tail, samples=window)
        right = _rms(
            scans[index + 1].audio_head,
            start=selections[index + 1].source_start,
            samples=window,
        )
        if left <= 0 or right <= 0:
            continue
        target = math.sqrt(left * right)
        raw_left_db = 20 * math.log10(target / left)
        raw_right_db = 20 * math.log10(target / right)
        left_db = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, raw_left_db))
        right_db = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, raw_right_db))
        prior = gains[index]
        gains[index] = _Gains(
            prior.head_db,
            left_db,
            prior.head_clamped,
            left_db != raw_left_db,
        )
        following = gains[index + 1]
        gains[index + 1] = _Gains(
            right_db,
            following.tail_db,
            right_db != raw_right_db,
            following.tail_clamped,
        )
    return gains


def _global_gain(
    scans: list[_Scan],
    selections: list[_AudioSelection],
    gains: list[_Gains],
    audio: DecodedAudioFormat,
) -> float:
    peak = 0.0
    for scan, selection, gain in zip(scans, selections, gains, strict=True):
        positive_head = max(0.0, gain.head_db)
        positive_tail = max(0.0, gain.tail_db)
        peak = max(
            peak,
            scan.audio_peak,
            scan.audio_head_peak * 10 ** (positive_head / 20),
            scan.audio_peak * 10 ** (positive_tail / 20),
        )
        ride = _nearest(GAIN_RIDE_SECONDS * audio.sample_rate)
        if selection.selected < 2 * ride:
            peak = max(peak, scan.audio_peak * 10 ** ((positive_head + positive_tail) / 20))
    return min(1.0, PEAK_CEILING / peak) if peak else 1.0


def _gain_pcm(
    pcm: tuple[bytes, ...],
    *,
    start: int,
    total: int,
    rate: int,
    gains: _Gains,
    global_gain: float,
) -> tuple[bytes, ...]:
    ride = max(1, _nearest(GAIN_RIDE_SECONDS * rate))
    # Most samples lie between the two seam rides. Preserve their exact source bytes
    # when no global attenuation is needed, including the all-silent H3 soundtrack.
    count = len(pcm[0]) // 4
    if global_gain == 1.0 and (
        (not gains.head_db or start >= ride)
        and (not gains.tail_db or start + count <= max(0, total - ride))
    ):
        return pcm
    transformed: list[bytes] = []
    for raw in pcm:
        values = array("f")
        values.frombytes(raw)
        for offset, value in enumerate(values):
            position = start + offset
            db = 0.0
            if gains.head_db and position < ride:
                phase = position / max(1, ride - 1)
                db += gains.head_db * (1 + math.cos(math.pi * phase)) / 2
            tail_start = max(0, total - ride)
            if gains.tail_db and position >= tail_start:
                phase = (position - tail_start) / max(1, ride - 1)
                db += gains.tail_db * (1 - math.cos(math.pi * phase)) / 2
            values[offset] = float(value) * (10 ** (db / 20)) * global_gain
        transformed.append(values.tobytes())
    return tuple(transformed)


def _slice_pcm(event: DecodedAudioChunk, start: int, stop: int) -> tuple[bytes, ...]:
    return tuple(channel[start * 4 : stop * 4] for channel in event.pcm_f32le)


def _output_audio(audio: DecodedAudioFormat) -> DecodedAudioFormat:
    return DecodedAudioFormat(
        channels=audio.channels,
        sample_rate=audio.sample_rate,
        channel_layout=audio.channel_layout,
        channel_names=audio.channel_names,
        time_base=Fraction(1, audio.sample_rate),
    )


def _output_header(video: DecodedVideoFormat, audio: DecodedAudioFormat) -> DecodedMediaHeader:
    assert video.nominal_frame_rate is not None
    return DecodedMediaHeader(
        video=DecodedVideoFormat(
            width=video.width,
            height=video.height,
            time_base=1 / video.nominal_frame_rate,
            pixel_aspect_ratio=video.pixel_aspect_ratio,
            nominal_frame_rate=video.nominal_frame_rate,
            color_primaries=video.color_primaries,
            color_transfer=video.color_transfer,
            color_matrix=video.color_matrix,
            color_range=video.color_range,
        ),
        audio=_output_audio(audio),
    )


class _SegmentAudio:
    """One segment's selected soundtrack on the output sample clock: trim, pad, seam gain."""

    def __init__(
        self,
        audio: DecodedAudioFormat,
        selection: _AudioSelection,
        gains: _Gains,
        global_gain: float,
        start: int,
    ) -> None:
        self.audio, self.selection, self.gains = audio, selection, gains
        self.global_gain = global_gain
        self.pts = start
        self.source = 0
        self.emitted = 0

    def take(self, event: DecodedAudioChunk) -> DecodedAudioChunk | None:
        chunk_start = self.source
        self.source += event.sample_count
        selection = self.selection
        take_start = max(selection.source_start, chunk_start)
        take_end = min(selection.source_start + selection.from_source, self.source)
        if take_start >= take_end:
            return None
        pcm = _gain_pcm(
            _slice_pcm(event, take_start - chunk_start, take_end - chunk_start),
            start=self.emitted,
            total=selection.selected,
            rate=self.audio.sample_rate,
            gains=self.gains,
            global_gain=self.global_gain,
        )
        self.emitted += take_end - take_start
        return self._chunk(pcm, take_end - take_start)

    def finish(self, scan: _Scan) -> DecodedAudioChunk | None:
        """Refuse a soundtrack that changed since its scan; then the clock's padding."""
        if self.source != scan.audio_samples or self.emitted != self.selection.from_source:
            raise InvalidRequest("media changed between scan and assembly", code="invalid_request")
        padded = self.selection.padded
        if not padded:
            return None
        return self._chunk(tuple(bytes(padded * 4) for _ in range(self.audio.channels)), padded)

    def _chunk(self, pcm: tuple[bytes, ...], count: int) -> DecodedAudioChunk:
        audio = self.audio
        chunk = DecodedAudioChunk(
            channels=audio.channels,
            sample_count=count,
            sample_rate=audio.sample_rate,
            channel_layout=audio.channel_layout,
            channel_names=audio.channel_names,
            pcm_f32le=pcm,
            pts=self.pts,
            time_base=Fraction(1, audio.sample_rate),
        )
        self.pts += count
        return chunk


def _segment_events(
    scans: list[_Scan],
    selections: list[_AudioSelection],
    gains: list[_Gains],
    *,
    global_gain: float,
    check: Callable[[], None],
    on_frame: Callable[[int], None],
    overlaps: Sequence[bool],
) -> Iterator[DecodedMediaEvent]:
    """Decoded frames and the selected soundtrack, for the encoder."""
    video = scans[0].header.video
    audio = scans[0].header.audio
    assert video is not None and video.nominal_frame_rate is not None and audio is not None
    yield _output_header(video, audio)
    output_frame = 0
    output_audio = 0
    for index, (scan, selection) in enumerate(zip(scans, selections, strict=True)):
        segment = _SegmentAudio(audio, selection, gains[index], global_gain, output_audio)
        seen_frames = 0
        with scan.asset.stream() as stream:
            for event in stream:
                check()
                if isinstance(event, DecodedMediaHeader):
                    continue
                if isinstance(event, DecodedVideoFrame):
                    if overlaps[index] and seen_frames == 0:
                        seen_frames += 1
                        continue
                    yield DecodedVideoFrame(
                        width=event.width,
                        height=event.height,
                        rgb=event.rgb,
                        pts=output_frame,
                        duration=1,
                        time_base=1 / video.nominal_frame_rate,
                        pixel_aspect_ratio=event.pixel_aspect_ratio,
                        color_primaries=event.color_primaries,
                        color_transfer=event.color_transfer,
                        color_matrix=event.color_matrix,
                        color_range=event.color_range,
                    )
                    on_frame(output_frame)
                    output_frame += 1
                    seen_frames += 1
                    continue
                if (chunk := segment.take(event)) is not None:
                    yield chunk
        if seen_frames != scan.frames:
            raise InvalidRequest("media changed between scan and assembly", code="invalid_request")
        if (padding := segment.finish(scan)) is not None:
            yield padding
        output_audio = segment.pts


def _soundtrack(
    scans: list[_Scan],
    selections: list[_AudioSelection],
    gains: list[_Gains],
    *,
    global_gain: float,
    check: Callable[[], None],
) -> Iterator[DecodedMediaEvent]:
    """Only the selected soundtrack, read without decoding a picture, for a packet copy."""
    audio = scans[0].header.audio
    assert audio is not None
    yield DecodedMediaHeader(video=None, audio=_output_audio(audio))
    output_audio = 0
    for scan, selection, gain in zip(scans, selections, gains, strict=True):
        segment = _SegmentAudio(audio, selection, gain, global_gain, output_audio)
        with scan.asset.stream(pictures=False) as stream:
            for event in stream:
                check()
                if isinstance(event, DecodedAudioChunk) and (chunk := segment.take(event)):
                    yield chunk
        if (padding := segment.finish(scan)) is not None:
            yield padding
        output_audio = segment.pts


def _master_events(
    scans: list[_Scan],
    master: AudioAsset,
    master_scan: _AudioScan,
    *,
    target_audio: int,
    padding: int,
    check: Callable[[], None],
    on_frame: Callable[[int], None],
    overlaps: Sequence[bool],
) -> Iterator[DecodedMediaEvent]:
    video = scans[0].header.video
    assert video is not None and video.nominal_frame_rate is not None
    yield _output_header(video, master_scan.header)
    output_frame = 0
    output_audio = 0
    source_audio = 0
    with master.stream() as audio_stream:
        audio_header = next(audio_stream)
        assert isinstance(audio_header, DecodedMediaHeader)
        audio_iter = iter(audio_stream)
        next_audio = next(audio_iter, None)
        for index, scan in enumerate(scans):
            seen_frames = 0
            with scan.asset.stream() as video_stream:
                for event in video_stream:
                    check()
                    if not isinstance(event, DecodedVideoFrame):
                        continue
                    if overlaps[index] and seen_frames == 0:
                        seen_frames += 1
                        continue
                    frame_time = Fraction(output_frame, 1) / video.nominal_frame_rate
                    while (
                        isinstance(next_audio, DecodedAudioChunk)
                        and next_audio.start_time < frame_time
                    ):
                        source_audio += next_audio.sample_count
                        if output_audio < target_audio:
                            count = min(target_audio - output_audio, next_audio.sample_count)
                            yield DecodedAudioChunk(
                                channels=master_scan.header.channels,
                                sample_count=count,
                                sample_rate=master_scan.header.sample_rate,
                                channel_layout=master_scan.header.channel_layout,
                                channel_names=master_scan.header.channel_names,
                                pcm_f32le=_slice_pcm(next_audio, 0, count),
                                pts=output_audio,
                                time_base=Fraction(1, master_scan.header.sample_rate),
                            )
                            output_audio += count
                        next_audio = next(audio_iter, None)
                    yield DecodedVideoFrame(
                        width=event.width,
                        height=event.height,
                        rgb=event.rgb,
                        pts=output_frame,
                        duration=1,
                        time_base=1 / video.nominal_frame_rate,
                        pixel_aspect_ratio=event.pixel_aspect_ratio,
                        color_primaries=event.color_primaries,
                        color_transfer=event.color_transfer,
                        color_matrix=event.color_matrix,
                        color_range=event.color_range,
                    )
                    on_frame(output_frame)
                    output_frame += 1
                    seen_frames += 1
            if seen_frames != scan.frames:
                raise InvalidRequest(
                    "media changed between scan and assembly", code="invalid_request"
                )
        while isinstance(next_audio, DecodedAudioChunk):
            check()
            source_audio += next_audio.sample_count
            if output_audio < target_audio:
                count = min(target_audio - output_audio, next_audio.sample_count)
                yield DecodedAudioChunk(
                    channels=master_scan.header.channels,
                    sample_count=count,
                    sample_rate=master_scan.header.sample_rate,
                    channel_layout=master_scan.header.channel_layout,
                    channel_names=master_scan.header.channel_names,
                    pcm_f32le=_slice_pcm(next_audio, 0, count),
                    pts=output_audio,
                    time_base=Fraction(1, master_scan.header.sample_rate),
                )
                output_audio += count
            next_audio = next(audio_iter, None)
    if source_audio != master_scan.samples:
        raise InvalidRequest("master audio changed between scans", code="invalid_request")
    missing = target_audio - output_audio
    if missing != padding:
        raise InvalidRequest("master audio changed between scans", code="invalid_request")
    if missing:
        yield DecodedAudioChunk(
            channels=master_scan.header.channels,
            sample_count=missing,
            sample_rate=master_scan.header.sample_rate,
            channel_layout=master_scan.header.channel_layout,
            channel_names=master_scan.header.channel_names,
            pcm_f32le=tuple(bytes(missing * 4) for _ in range(master_scan.header.channels)),
            pts=output_audio,
            time_base=Fraction(1, master_scan.header.sample_rate),
        )


@invocable
async def assemble_video(
    ctx: Context,
    *,
    payload: AssembleVideoRequest,
    out: Outputs,
    tel: Telemetry,
) -> AssembleVideoResponse:
    """Join one or more completed shots into one deterministic MP4.

    Registered by `h3` as a CPU function (`accelerator=False`): assembly is run-to-completion
    CPU work with no model slot. The videos ride as asset
    references, so a caller names `long_form`'s reported digests and nothing re-uploads.
    """
    return assemble(payload, out=out, tel=tel, check=ctx.raise_if_cancelled)


def assemble(
    payload: AssembleVideoRequest,
    *,
    out: Outputs,
    tel: Telemetry,
    check: Callable[[], None],
    scanned: ScanAhead | None = None,
) -> AssembleVideoResponse:
    """Run inside the current admitted attempt; the result owns its output handle.

    Cuts under segment audio copy the shots' H.264 packets and encode only the soundtrack;
    a replay-trimmed join, a master track or unjoinable tracks decode and encode.
    """
    if not payload.videos:
        raise InvalidRequest("assembly needs at least one video", code="invalid_request")
    videos = [item.video for item in payload.videos]
    overlaps = [item.first_frame_overlaps_previous for item in payload.videos]
    if overlaps[0]:
        raise InvalidRequest("the first video overlaps no previous one", code="invalid_request",
                             fields=["payload.videos[0].first_frame_overlaps_previous"])
    tolerance = out.video_audio_frame_samples
    copy = not any(overlaps) and payload.master_audio is None
    pictures = scanned.pictures if scanned is not None else not copy
    if not (pictures or copy):
        raise InvalidRequest("this join needs scans of its pictures", code="invalid_request")
    scan_step = tel.step_callback(len(payload.videos), stage="scan", overall_range=(0.00, 0.10))
    scans = scanned.finish(videos) if scanned is not None else []
    if scanned is None:
        for index, item in enumerate(videos):
            scans.append(_scan_video(item, check, pictures=pictures))
            scan_step(index)
    else:
        scan_step(len(scans) - 1)
    master_samples = audio_trimmed = audio_padded = 0
    gains = [_Gains() for _ in scans]
    global_gain = 1.0
    selections: list[_AudioSelection] = []
    saved: SavedVideo | None = None
    audio_mode: Literal["segments", "master"] = "segments"
    master_audio_digest = None
    audio_format = scans[0].header.audio
    if payload.master_audio is None and (
        audio_format is None
        or any(
            scan.header.audio is None or not _same_audio(audio_format, scan.header.audio)
            for scan in scans
        )
    ):
        raise InvalidRequest(
            "segment-audio assembly requires one common soundtrack on every video",
            code="invalid_request",
        )
    if copy:
        assert audio_format is not None
        joined = audio_format

        def soundtrack(frames: Sequence[int]) -> Iterator[DecodedMediaEvent]:
            nonlocal global_gain
            # Runtime proved these frame counts on the packets' own clock.
            scans[:] = [replace(scan, frames=n) for scan, n in zip(scans, frames, strict=True)]
            selections[:] = _segment_selections(
                scans, FRAME_RATE, joined, tolerance, overlaps
            )
            global_gain = _global_gain(scans, selections, gains, joined)
            return _soundtrack(scans, selections, gains, global_gain=global_gain, check=check)

        try:
            with tel.stage("join", overall_range=(0.10, 1.00)):
                saved = out.save_video_concat(videos, soundtrack)
        except OutputError as refused:
            if refused.code != "video_copy_incompatible":
                raise
            tel.log("assembly re-encodes unjoinable tracks", reason=str(refused)[:256])
            if not pictures:
                scans = [_scan_video(item, check) for item in videos]
    video_format = scans[0].header.video
    if saved is None:
        assert video_format is not None and video_format.nominal_frame_rate is not None
        if video_format.nominal_frame_rate != FRAME_RATE:
            raise InvalidRequest(
                f"video frame rate is {video_format.nominal_frame_rate}, expected 24",
                code="invalid_request",
            )
        if any(
            scan.header.video is None or not _same_video(video_format, scan.header.video)
            for scan in scans
        ):
            raise InvalidRequest("video formats differ between segments", code="invalid_request")
    source_frames = sum(scan.frames for scan in scans)
    replay_frames = sum(overlaps)
    output_frames = source_frames - replay_frames
    on_frame = tel.step_callback(output_frames, stage="assemble", overall_range=(0.10, 1.00))
    if saved is None and payload.master_audio is None:
        assert audio_format is not None
        selections = _segment_selections(
            scans, FRAME_RATE, audio_format, tolerance, overlaps
        )
        if any(overlaps):
            gains = _seam_gains(scans, selections, audio_format, overlaps)
        global_gain = _global_gain(scans, selections, gains, audio_format)
        saved = out.save_video_stream(
            _segment_events(
                scans,
                selections,
                gains,
                global_gain=global_gain,
                overlaps=overlaps,
                check=check,
                on_frame=on_frame,
            )
        )
    elif saved is None:
        assert payload.master_audio is not None
        master_scan = _scan_audio(payload.master_audio, check)
        target_audio = _nearest(
            Fraction(output_frames * master_scan.header.sample_rate, 1) / FRAME_RATE
        )
        master_drift = master_scan.samples - target_audio
        if abs(master_drift) > tolerance:
            raise InvalidRequest(
                f"master audio differs from the final frame clock by {master_drift} samples",
                code="invalid_request",
            )
        saved = out.save_video_stream(
            _master_events(
                scans,
                payload.master_audio,
                master_scan,
                target_audio=target_audio,
                padding=max(0, -master_drift),
                overlaps=overlaps,
                check=check,
                on_frame=on_frame,
            )
        )
        audio_mode = "master"
        master_audio_digest = payload.master_audio.digest
        audio_format = master_scan.header
        master_samples = master_scan.samples
        audio_trimmed, audio_padded = max(0, master_drift), max(0, -master_drift)
    if selections:
        audio_trimmed = sum(selection.trimmed for selection in selections)
        audio_padded = sum(selection.padded for selection in selections)
    assert saved is not None and audio_format is not None

    target_audio = _nearest(Fraction(output_frames * audio_format.sample_rate, 1) / FRAME_RATE)
    audio_facts = saved.audio
    if audio_facts is None:
        raise InvalidRequest("encoded output has no soundtrack", code="output_integrity")
    # Geometry and colour are Runtime's own probe check; the clock is this module's.
    if saved.frame_rate != FRAME_RATE:
        raise InvalidRequest(
            f"video frame rate is {saved.frame_rate}, expected 24", code="invalid_request"
        )
    if (
        saved.frame_count != output_frames
        or audio_facts.channels != audio_format.channels
        or audio_facts.sample_rate != audio_format.sample_rate
        or audio_facts.codec_frame_samples != tolerance
        or saved.submitted_audio_samples != target_audio
    ):
        raise InvalidRequest(
            "encoded output probe disagrees with the selected frame/sample clock",
            code="output_integrity",
        )

    segments: list[SegmentReceipt] = []
    cumulative_frames = 0
    prior_audio = 0
    for index, (scan, gain) in enumerate(zip(scans, gains, strict=True)):
        selected_frames = scan.frames - int(overlaps[index])
        cumulative_frames += selected_frames
        cumulative_audio = _nearest(
            Fraction(cumulative_frames * audio_format.sample_rate, 1) / FRAME_RATE
        )
        segment_audio = cumulative_audio - prior_audio
        selection = selections[index] if selections else None
        if selection is not None:
            source_start = selection.source_start
            source_stop = source_start + selection.from_source
            segment_trimmed = selection.trimmed
            segment_padded = selection.padded
        else:
            source_start = min(prior_audio, master_samples)
            source_stop = max(source_start, min(cumulative_audio, master_samples))
            segment_trimmed = 0
            segment_padded = segment_audio - max(0, source_stop - source_start)
        prior_audio = cumulative_audio
        segments.append(
            SegmentReceipt(
                digest=scan.asset.digest,
                size_bytes=scan.asset.size_bytes,
                media_type=scan.asset.media_type,
                source_frames=scan.frames,
                selected_frames=selected_frames,
                replay_frame_digest=scan.first_frame_digest if overlaps[index] else None,
                segment_soundtrack_samples=scan.audio_samples,
                selected_audio_samples=segment_audio,
                selected_audio_source_start_sample=source_start,
                selected_audio_source_stop_sample=source_stop,
                audio_trimmed_samples=segment_trimmed,
                audio_padded_samples=segment_padded,
                head_gain_db=gain.head_db,
                tail_gain_db=gain.tail_db,
                head_gain_clamped=gain.head_clamped,
                tail_gain_clamped=gain.tail_clamped,
            )
        )

    return AssembleVideoResponse(
        video=saved.video,
        segments=segments,
        audio_mode=audio_mode,
        master_audio_digest=master_audio_digest,
        source_frames=source_frames,
        output_frames=output_frames,
        replay_frames_removed=replay_frames,
        submitted_audio_samples=saved.submitted_audio_samples,
        decoded_audio_samples=audio_facts.decoded_samples,
        audio_trimmed_samples=audio_trimmed,
        audio_padded_samples=audio_padded,
        audio_sample_rate=audio_format.sample_rate,
        audio_codec_frame_samples=audio_facts.codec_frame_samples,
        av_endpoint_delta_samples=audio_facts.decoded_samples - target_audio,
        global_gain_db=20 * math.log10(global_gain) if global_gain > 0 else -120.0,
        video_codec=saved.video_codec,
        video_profile=saved.video_profile,
        audio_codec=audio_facts.codec,
        audio_profile=audio_facts.profile,
        audio_channels=audio_facts.channels,
        width=saved.width,
        height=saved.height,
        frame_rate_numerator=saved.frame_rate.numerator,
        frame_rate_denominator=saved.frame_rate.denominator,
        pixel_aspect_ratio_numerator=saved.pixel_aspect_ratio.numerator,
        pixel_aspect_ratio_denominator=saved.pixel_aspect_ratio.denominator,
        color_primaries=saved.color_primaries,
        color_transfer=saved.color_transfer,
        color_matrix=saved.color_matrix,
        color_range=saved.color_range,
    )
