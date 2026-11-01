"""MiniMax-H3 as two node functions over one dual-DiT construction: `shot` renders one
continuous take of any length in pieces, `upscale` takes a whole shot above 768p.

Runtime reads typed file handles and encodes outputs. Official Diffusers owns H3 presentation,
conditioning, AdaLN, solver, and decode math. This module validates the product contract,
stages weighted roots, and joins those two boundaries.
"""

from __future__ import annotations

import hashlib
import math
import queue
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import partial
from itertools import accumulate
from typing import Annotated, Any, Literal, cast

import msgspec
import numpy as np
import torch
from cozy_runtime.author import (
    App,
    AudioFile,
    AudioProbe,
    ChildCallError,
    Context,
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaHeader,
    DecodedVideoFormat,
    DecodedVideoFrame,
    File,
    FileBound,
    FileLimits,
    Files,
    Image,
    ImageFile,
    ImagePreparation,
    InvalidRequest,
    OutputError,
    Outputs,
    SavedVideo,
    Shape,
    Telemetry,
    ThreadPoolExecutor,
    UnsupportedInput,
    VideoFile,
    VideoProbe,
    invocable,
    prefetch,
)
from msgspec.structs import replace
from PIL import Image as PILImage
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from gates import MediaFacts, refuse_before_encode
from minimax_h3 import regenerate as refine_math
from minimax_h3.continuation import (
    AVContext,
    Carry,
    ContinuationPlan,
    RenderProvenance,
    context_provenance,
    decode_context,
    encode_context,
    provenance,
    video_tokens,
)
from minimax_h3.guide_rows import GuideClip
from minimax_h3.model import (
    H3LatentUpscaler as H3LatentUpscaler,
    H3Model as H3Model,
    H3RegenerateLoRA as H3RegenerateLoRA,
    H3TurboBase as H3TurboBase,
    H3TurboLoRA as H3TurboLoRA,
    condition_references,
)
from minimax_h3.official import (
    FPS,
    MAX_AUDIO_REFERENCES,
    MAX_CONDITIONER_VISION_TOKENS,
    MAX_FRAMES,
    MAX_IMAGE_REFERENCES,
    MIN_FRAMES,
    MAX_REFERENCES,
    MAX_VIDEO_REFERENCES,
    REFERENCE_IMAGE_SHORT_EDGE,
    OfficialH3Pipeline,
    ReferencePolicyFacts,
    ScheduleFacts,
    Task,
    TimestepPlan,
    Trunk,
    assert_duration_envelope,
    denoise_rows,
    frames_for,
    reference_image_vision_tokens,
    read_reference_video,
    reference_video_vision_tokens,
    validate_reference_policy,
)
from seedvr2 import Resolution as RestorationResolution
from seedvr2 import UpscaleOutput as RestorationOutput
from seedvr2 import upscale as restore_video
from story import (
    AUDIO_READ_BYTES,
    CLIPS,
    FILE,
    IMAGE_READ_BYTES,
    REFERENCE_AUDIO,
    REFERENCE_VIDEO,
    VIDEO_MAX_BYTES,
    VIDEO_READ_BYTES,
    PieceCall,
    StoryReference,
    encoder_prompt,
    piece_call,
    validate_references,
    voices,
)

app = App()

_MIB = 1 << 20
#: Standalone audio references share one 15-second budget; each is 2-15 s (`REFERENCE_AUDIO`).
_MAX_AUDIO_SECONDS = Fraction(15, 1)

Reference = ImageFile | VideoFile | AudioFile
#: One reference of each kind, with its retained-read bound. A Ref2VA piece's `assets` declare
#: the preparations the shot's own fields do, so Runtime records a truncation on the call the
#: client made.
_REFERENCE = (
    Annotated[ImageFile, FileBound(max_decoded_bytes=IMAGE_READ_BYTES)]
    | Annotated[
        VideoFile, FileBound(max_bytes=VIDEO_MAX_BYTES, max_decoded_bytes=VIDEO_READ_BYTES)
    ]
    | Annotated[AudioFile, FileBound(max_decoded_bytes=AUDIO_READ_BYTES)]
)
ReferenceFiles = Annotated[
    Files[_REFERENCE],
    FileLimits(
        images=MAX_IMAGE_REFERENCES,
        videos=MAX_VIDEO_REFERENCES,
        audio=MAX_AUDIO_REFERENCES,
        total=MAX_REFERENCES,
    ),
    ImagePreparation(max_edge=8192, max_pixels=16_777_216),
    REFERENCE_VIDEO,
    REFERENCE_AUDIO,
    msgspec.Meta(min_length=1),
]


class VideoUse(msgspec.Struct, frozen=True):
    """How a call uses one of its videos: its sound (`use_audio`), whether its frames are
    anchored in the output (`anchor_video`), and the frame of it, on the 24-fps clock, the
    call's window starts at (`shot_windows`)."""

    audio: bool = False
    anchor: bool = False
    start: Annotated[int, msgspec.Meta(ge=0)] = 0


def video_uses(references: Sequence[StoryReference], starts: Mapping[str, int]) -> tuple[VideoUse, ...]:
    """Each of a call's videos, in packed order, from its window's `starts`."""
    return tuple(
        VideoUse(ref.use_audio, ref.kind == "anchor_video", starts[ref.name])
        for ref in references if ref.kind in CLIPS
    )


def _call_assets(
    files: dict[str, Reference], references: Sequence[StoryReference]
) -> Files[Reference]:
    """One H3 call's references in packed order, labeled `Picture|Audio|Video N` per kind."""
    counts = {"image": 0, "audio": 0, "video": 0}
    labeled: list[Reference] = []
    for reference in references:
        file = files[reference.name]
        counts[file.kind] += 1
        label = {"image": "Picture", "audio": "Audio", "video": "Video"}[file.kind]
        labeled.append(file.with_label(f"{label} {counts[file.kind]}"))
    return Files(labeled)


DEFAULT_REFERENCE_IMAGE_SHORT_EDGE = 1024
_SHORT_EDGE_LADDER = (2048, 1536, 1024, 768, 512, 256)
_REFERENCE_FIDELITY_EDGES = {"low": 256, "medium": 1024, "high": REFERENCE_IMAGE_SHORT_EDGE}
# Length is the request's largest cost lever: the DiT attends over ONE packed sequence whose
# rows scale with the frame count, and attention is quadratic in it. Every whole second in
# the envelope is served, and every generation call must name its length (owner ruling
# 2026-09-27): there is no default a caller could pay for without choosing it.
# Declared, because the wire IS the declaration: `describe` reads this file and runs none of
# it (#713). `assert_duration_envelope` refuses unless the official 17n+5 snap and the frame
# envelope serve exactly these whole seconds, so the pair below cannot drift from the
# geometry it advertises. The ceiling is 15: 362 frames is the top of the `17n + 5` grid a
# 15-second model reaches, and delivering it needs the scoped ceiling in `official.py`
# (se-053), because upstream states its own bound in seconds and the grid has no 360.
MIN_DURATION_S = 5
MAX_DURATION_S = 15
assert_duration_envelope((MIN_DURATION_S, MAX_DURATION_S))
# Frames size the DiT's sequence and every scope's scratch, so they are the request's demand
# axis: staged admission reuses a scope's measured peak only for the same frame count.
DurationSeconds = Annotated[
    int,
    msgspec.Meta(
        ge=MIN_DURATION_S,
        le=MAX_DURATION_S,
        description=(
            "Required. Clip length in whole seconds, 5 to 15 (15 is the full-length "
            "ceiling), snapped up to the video VAE's own 17n+5 frame grid; shorter is "
            "quadratically faster."
        ),
    ),
    Shape(frames={s: frames_for(s) for s in range(MIN_DURATION_S, MAX_DURATION_S + 1)}),
]
#: A piece's new seconds. Its memory is sized by its `frames`, not by these alone.
PieceSeconds = Annotated[int, msgspec.Meta(ge=MIN_DURATION_S, le=MAX_DURATION_S)]


class Resolution(Enum):
    """The output size: native 768p, or above it through `upscale.model`."""

    NATIVE = "768p"
    FULL_HD = "1080p"
    QHD = "1440p"
    UHD = "2160p"


#: The refine starts the Regenerate overlay serves (`efficient-large-model/open-h3-regenerate-2k@0.1.0`'s plan).
REGENERATE_STRENGTHS = (0.7, 0.8, 0.909)
_SEEDVR2_FIELDS = (
    "color_correction", "input_noise_scale", "latent_noise_scale", "batch_size",
    "uniform_batch_size", "temporal_overlap", "prepend_frames",
)  # fmt: skip


class Upscale(msgspec.Struct, kw_only=True):
    """How `resolution` is reached (UPSCALE.md).

    `model`:
    - `seedvr2` (default): SeedVR2 7B restores the decoded 768p original. The 7:4 frame is
      letterboxed into 16:9. The other fields are its ComfyUI controls (seedvr2 README, Tuning);
      unset ones keep SeedVR2's defaults.
    - `regenerate-2k` (NVIDIA's Open-H3-Regenerate-2K): the video's 768p latent is upscaled in
      latent space and re-rendered at `resolution` by H3 in three steps (two at 0.8 and 0.7). It fills
      16:9: the 7:4 frame is stretched 1.6% horizontally, as NVIDIA trained it, where SeedVR2
      letterboxes. At `768p` it is a refine-only pass at the native size.

    `strength` (regenerate-2k): how much of the source the refine redraws, as its start noise:
    0.909 (default, NVIDIA's), 0.8 or 0.7. Lower keeps more of the source (faces, on-screen text)
    and is faster; higher invents more detail. Small text is redrawn at every strength and can
    become other letters; SeedVR2 keeps it.
    """

    model: Literal["seedvr2", "regenerate-2k"] = "seedvr2"
    strength: float | None = None
    color_correction: (
        Literal["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"] | None
    ) = None
    input_noise_scale: Annotated[float, msgspec.Meta(ge=0, le=1)] | None = None
    latent_noise_scale: Annotated[float, msgspec.Meta(ge=0, le=1)] | None = None
    batch_size: Annotated[int, msgspec.Meta(ge=1)] | None = None
    uniform_batch_size: bool | None = None
    temporal_overlap: Annotated[int, msgspec.Meta(ge=0, le=16)] | None = None
    prepend_frames: Annotated[int, msgspec.Meta(ge=0, le=32)] | None = None

    def __post_init__(self) -> None:
        if self.batch_size is not None and self.batch_size % 4 != 1:
            raise ValueError(
                f"batch_size must be 4n+1 (1, 5, 9, 13, 17, 21, ...), got {self.batch_size}"
            )
        if self.strength is not None and round(self.strength, 3) not in REGENERATE_STRENGTHS:
            raise ValueError(
                f"strength is one of {', '.join(map(str, REGENERATE_STRENGTHS))}, got {self.strength}"
            )

    def ignored(self) -> list[str]:
        """Warnings for the fields the chosen method does not read."""
        if self.model == "regenerate-2k":
            unused = [name for name in _SEEDVR2_FIELDS if getattr(self, name) is not None]
            return [f"upscale.{name} applies to seedvr2 only" for name in unused]
        return ["upscale.strength applies to regenerate-2k only"] if self.strength is not None else []

    @property
    def start(self) -> float:
        return REGENERATE_STRENGTHS[-1] if self.strength is None else self.strength


class H3VideoOutput(msgspec.Struct):
    """The catalog wire shape. Checkpoint, plan, geometry, and digest facts remain
    attempt observations (se-012): they ride Telemetry, never the customer result."""

    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, media_types=("video/mp4",))]
    warnings: list[str]


def _reference_policy(assets: ReferenceFiles) -> ReferencePolicyFacts:
    kinds = [assets.info(index).kind for index in range(len(assets))]
    try:
        return validate_reference_policy(kinds)
    except ValueError as exc:
        raise UnsupportedInput(str(exc), code="reference_policy", fields=["assets"]) from exc


class ImageSizing(msgspec.Struct, frozen=True):
    """One image reference's fidelity: its explicit short edge, or the auto-sized default."""

    field: str
    width: int
    height: int
    requested: int | None
    resolved: int = 0

    @property
    def tokens(self) -> int:
        return reference_image_vision_tokens(self.width, self.height, self.resolved)


class ReferenceSizing(msgspec.Struct, frozen=True):
    images: tuple[ImageSizing, ...]
    video_tokens: int
    default: int

    @property
    def edges(self) -> list[int]:
        return [image.resolved for image in self.images]

    @property
    def total(self) -> int:
        return self.video_tokens + sum(image.tokens for image in self.images)

    @property
    def summary(self) -> str:
        """Per image: field, requested (or ``default``) -> resolved short edge, tokens."""
        return "; ".join(
            f"{image.field} {image.requested or 'default'}->{image.resolved} {image.tokens} tokens"
            for image in self.images
        )


def resolve_reference_sizing(
    images: Sequence[ImageSizing], *, default: int, video_tokens: int
) -> ReferenceSizing:
    """An explicit fidelity is never changed. The images without one share the package
    default and step down the ladder together until the total fits the budget; auto-sizing
    never rises above the default."""

    def sized(edge: int) -> ReferenceSizing:
        resolved = tuple(replace(image, resolved=image.requested or edge) for image in images)
        return ReferenceSizing(resolved, video_tokens, default)

    ladder = [default, *(rung for rung in _SHORT_EDGE_LADDER if rung < default)]
    for edge in ladder:
        sizing = sized(edge)
        if sizing.total <= MAX_CONDITIONER_VISION_TOKENS:
            return sizing
    floor = sized(ladder[-1])
    explicit = [image for image in floor.images if image.requested]
    raise UnsupportedInput(
        f"reference vision presentation needs {floor.total} tokens even with every default-sized "
        f"image at {ladder[-1]} px ({video_tokens} from videos, "
        f"{sum(image.tokens for image in explicit)} from the explicit short edges "
        f"{[image.requested for image in explicit]}); this release admits at most "
        f"{MAX_CONDITIONER_VISION_TOKENS}",
        code="reference_policy",
        fields=["assets"],
    )


def assets_to_h3_refs(
    references: Files[Reference],
    *,
    pipe: OfficialH3Pipeline,
    frames: int,
    videos: Sequence[VideoUse] = (),
    edges: Sequence[int] = (),
) -> tuple[list[Any], ReferenceSizing, GuideClip | None]:
    """Check every reference from its header and size the vision budget, then read each.

    A video is read over its window: from its `videos` entry's `start` (per video in packed
    order), the call's `frames` at most, which is all the official pipeline keeps of it, and
    budgeted at that length. It streams straight to the size it is used at on the 24-fps clock
    (`read_reference_video`); its soundtrack is decoded only where its entry says so. An anchor
    video is not a reference (`guide_rows`): Qwen never sees it.
    """
    audio_duration = Fraction(0)
    video_tokens = 0
    images: list[ImageSizing] = []
    use = iter(videos)
    uses = [
        next(use, VideoUse()) if isinstance(reference, VideoFile) else VideoUse()
        for reference in references
    ]
    keeps = [item.audio for item in uses]
    if any(item.anchor for item in uses) and all(
        item.anchor or isinstance(reference, AudioFile)
        for item, reference in zip(uses, references, strict=True)
    ):
        raise InvalidRequest(
            "an anchor_video needs a picture or another video beside it in the call",
            code="reference_policy",
            fields=["references"],
        )
    for index, reference in enumerate(references):
        info = references.info(index)
        field = info.id
        if isinstance(reference, ImageFile):
            image = reference.probe()
            _validate_ratio(image.width, image.height, field)
            edge = edges[len(images)] if len(images) < len(edges) else 0
            requested = edge or (
                None if info.fidelity == "auto" else _REFERENCE_FIDELITY_EDGES[info.fidelity]
            )
            images.append(ImageSizing(field, image.width, image.height, requested))
        elif isinstance(reference, VideoFile):
            video = reference.probe()
            keeps[index] = keeps[index] and video.soundtrack is not None
            _validate_video(video, field, sound=keeps[index])
            if not uses[index].anchor:
                window = min(Fraction(frames, FPS), video.duration - Fraction(uses[index].start, FPS))
                video_tokens += reference_video_vision_tokens(video.width, video.height, window)
        else:
            audio = reference.probe()
            _validate_audio(audio, field)
            audio_duration += audio.duration
            _validate_audio_aggregate(audio_duration)
    sizing = resolve_reference_sizing(
        images, default=DEFAULT_REFERENCE_IMAGE_SHORT_EDGE, video_tokens=video_tokens
    )
    prepared = []
    guide = None
    for index, reference in enumerate(references):
        if isinstance(reference, ImageFile):
            prepared.append(pipe.image_reference(reference.read()))
        elif isinstance(reference, VideoFile):
            start = uses[index].start
            with reference.stream(start=Fraction(start, FPS), end=Fraction(start + frames, FPS),
                                  audio=keeps[index]) as events:  # fmt: skip
                clip = read_reference_video(
                    events, frames=frames, audio=keeps[index], guide=uses[index].anchor
                )
            if uses[index].anchor:
                guide = pipe.guide_clip(clip)
            else:
                prepared.append(pipe.video_reference(clip))
        else:
            prepared.append(pipe.audio_reference(reference.read()))
    return prepared, sizing, guide


def _validate_ratio(width: int, height: int, field: str) -> None:
    if width > 4 * height or height > 4 * width:
        raise UnsupportedInput(
            f"{field} must have an aspect ratio between 1:4 and 4:1, got {width}x{height}",
            code="reference_policy",
            fields=[field],
        )


def _validate_audio(audio: AudioProbe, field: str) -> None:
    _validate_audio_channels(audio.channels, field)


def _validate_video(video: VideoProbe, field: str, *, sound: bool) -> None:
    _validate_ratio(video.width, video.height, field)
    if video.pixel_aspect_ratio != 1:
        raise UnsupportedInput(
            f"{field} must use square pixels, got {video.pixel_aspect_ratio}",
            code="reference_policy",
            fields=[field],
        )
    # Frame and soundtrack clock alignment belong to the model's video_reference.
    if sound and video.soundtrack is not None:
        _validate_audio_channels(video.soundtrack.channels, field)


def _validate_audio_aggregate(duration: Fraction) -> None:
    """The standalone-audio modality cap. A video's soundtrack belongs to that video and
    never consumes this budget; videos have no aggregate cap."""
    if duration > _MAX_AUDIO_SECONDS:
        raise InvalidRequest(
            "standalone audio references total more than 15 seconds",
            code="reference_policy",
            fields=["assets"],
        )


def _validate_audio_channels(channels: int, field: str) -> None:
    if channels not in (1, 2):
        raise UnsupportedInput(
            f"{field} audio must be mono or stereo, got {channels} channels",
            code="reference_policy",
            fields=[field],
        )


def _finish(
    model: H3Model,
    task: Task,
    state: Any,
    schedule: ScheduleFacts,
    *,
    duration_s: int,
    out: Outputs,
    tel: Telemetry,
    cancel: Any,
    completed: Callable[[Any], None] | None = None,
    delivered_frames: int | None = None,
    sampled_frames: int | None = None,
    ended: Callable[[Any], None] | None = None,
    earlier: _Earlier | None = None,
) -> H3VideoOutput:
    """Decode and encode the render, with `earlier` pieces before it as one video."""
    cancel()
    if completed is not None:
        # The successor's carry: a slice of the denoised latents, before either decode.
        completed(state)
    frames = frames_for(duration_s) if delivered_frames is None else delivered_frames
    waveform, sample_rate = _soundtrack(model, task, state, frames=frames, tel=tel, cancel=cancel)
    decode = partial(model.decode_video, task, state)
    if earlier is not None:
        # The whole shot as one latent: the VAE cross-fades its chunks across every join,
        # where decoding piece by piece steps the picture (run 20751: luma +3.7 at a join).
        latents = torch.cat([earlier.latents.to(state.latents.device), state.latents[:, :, earlier.head :]], dim=2)
        decode = partial(model.decode_latents, latents)
        frames += earlier.frames
        waveform = torch.cat([earlier.waveform, waveform], dim=1)

    # The encoder runs under the decode (h3a-017): each chunk is quantized on its device
    # and landed in the host buffer while a worker feeds the runtime's streaming MP4 sink.
    with tel.stage("decode_video", overall_range=(0.90, 0.97)):
        stream = _VideoStream(
            torch,
            out,
            frames=frames,
            waveform=waveform,
            sample_rate=sample_rate,
            cancel=cancel,
            progress=tel.step_callback(frames, stage="decode_video", overall_range=(0.90, 0.97)),
        )
        try:
            decode(on_chunk=stream.push)
        except BaseException:
            stream.abandon()
            raise
    try:
        saved = stream.finish(lambda share: tel.progress(share, stage="encode_video"))
    except BaseException:
        stream.abandon()
        raise
    _, height, width, _ = (int(value) for value in stream.pixels.shape)
    if ended is not None:
        ended(stream.pixels)

    tel.log(
        "h3 output geometry",
        width=width,
        height=height,
        frames=frames,
        fps=FPS,
        requested_duration_s=duration_s,
        duration_seconds=round(frames / FPS, 3),
        denoise_rows=denoise_rows(
            frames if sampled_frames is None else sampled_frames, height, width
        ),
        sample_rate=sample_rate,
    )
    if sampled_frames is not None:
        tel.log(
            "h3 continuation window",
            sampled_frames=sampled_frames,
            delivered_frames=frames,
            trimmed_frames=sampled_frames - frames,
        )
    tel.log(
        "h3 schedule facts",
        timestep_plan_digest=schedule.timestep_plan_digest,
        video_sigma_digest=schedule.video_sigma_digest,
        audio_sigma_digest=schedule.audio_sigma_digest,
        video_timestep_digest=schedule.video_timestep_digest,
        audio_timestep_digest=schedule.audio_timestep_digest,
        sigma_grid_points=schedule.sigma_grid_points,
        transformer_evaluations=schedule.transformer_evaluations,
    )
    tel.log(
        "h3 container facts",
        video_codec=saved.video_codec,
        frame_count=saved.frame_count,
        frame_rate=str(saved.frame_rate),
        color_matrix=saved.color_matrix,
        color_range=saved.color_range,
        audio_codec=saved.audio.codec if saved.audio is not None else "",
        audio_decoded_samples=saved.audio.decoded_samples if saved.audio is not None else 0,
        video_bytes=saved.video.size_bytes,
    )
    return H3VideoOutput(video=saved.video, warnings=[])


def _soundtrack(
    model: H3Model, task: Task, state: Any, *, frames: int, tel: Telemetry, cancel: Any
) -> tuple[Any, int]:
    """The render's decoded `(channels, samples)` soundtrack, checked against its clock."""
    with tel.stage("decode_audio", overall_range=(0.85, 0.90)):
        audio, sample_rate = model.decode_audio(task, state)
        release_rate = model.pipe.sample_rate
        if audio.ndim != 3 or int(audio.shape[0]) != 1 or sample_rate != release_rate:
            raise OutputError(
                f"official H3 audio decode returned shape {tuple(audio.shape)} at {sample_rate}Hz; "
                f"the release clock is {release_rate}Hz",
                code="output_integrity",
            )
        waveform = audio[0].to(torch.float32).contiguous().cpu()
    cancel()
    refuse_before_encode(
        waveform=waveform,
        audio_nonfinite_fraction=_nonfinite_fraction(torch, waveform),
        requested=MediaFacts(frames=frames, fps=FPS, sample_rate=sample_rate),
        tel=tel,
    )
    return waveform, sample_rate


#: Frames per handoff slice: 8 frames of 1344x768 are 100 MB of fp32 source and 25 MB of
#: RGB8, so a slice's device temporaries stay small and its hash and encode hide under the
#: decode of the next chunk.
_RGB8_CHUNK_FRAMES = 8
#: Generated frames are square-pixel RGB at 24 fps, tagged BT.709 limited-range in the
#: container — the runtime's own HD rule, so a player converts them as the encoder did.
_VIDEO_COLOR = {"color_primaries": 1, "color_transfer": 1, "color_matrix": 1, "color_range": 1}
_AUDIO_LAYOUTS = {1: ("mono", ("FC",)), 2: ("stereo", ("FL", "FR"))}


def _land(torch: Any, chunk: Any, landed: Any) -> None:
    """Quantize one `(t, 3, H, W)` float slice on its device, straight into `landed`, its
    `(t, H, W, 3)` slice of the host buffer — no per-slice host temporary."""
    chunk.clamp_(0, 1).mul_(255).round_()
    landed.copy_(chunk.to(torch.uint8).permute(0, 2, 3, 1).contiguous())


class _VideoStream:
    """The decode-to-encode handoff (h3a-017).

    The decode thread refuses non-finite values, quantizes each chunk on its device and
    lands it in its slice of ONE host RGB8 buffer; a worker thread feeds the slices in order
    to the runtime's streaming MP4 sink while the next chunk decodes. The soundtrack, decoded
    first, is interleaved behind the frames it covers. Each frame the sink takes is told
    (`finish`), and the sink's end, its error included, ends the wait for it.
    """

    def __init__(
        self,
        torch: Any,
        out: Outputs,
        *,
        frames: int,
        waveform: Any | None,
        sample_rate: int,
        cancel: Any,
        progress: Callable[[int], None],
    ) -> None:
        """`waveform` is the `(channels, samples)` soundtrack, or None for a silent video."""
        self._torch = torch
        self._out = out
        self._frames = frames
        self._waveform = waveform
        self._sample_rate = sample_rate
        self._cancel = cancel
        self._progress = progress
        self._slices: queue.SimpleQueue[tuple[int, int] | None] = queue.SimpleQueue()
        #: each frame the sink took, then None at its end
        self._taken: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._encode: Future[SavedVideo] | None = None
        self._abandoned = False
        self.pixels: Any = None
        self.landed = 0

    def push(self, chunk: Any) -> None:
        """One `(t, 3, H, W)` float chunk on its device, in decode order."""
        self._cancel()
        if self._encode is not None and self._encode.done():
            self._encode.result()
        torch = self._torch
        frames, channels, height, width = (int(value) for value in chunk.shape)
        bad = _nonfinite_fraction(torch, chunk)
        if bad:
            raise OutputError(
                f"the official H3 video decode produced non-finite values ({bad:.6f} of frames "
                f"{self.landed}..{self.landed + frames})",
                code="output_integrity",
            )
        if self.pixels is None and channels == 3:
            self.pixels = torch.empty(
                (self._frames, height, width, 3), dtype=torch.uint8, device="cpu"
            )
            self._encode = self._pool.submit(self._out.save_video_stream, self._events())
        if (
            self.pixels is None
            or channels != 3
            or (height, width) != tuple(self.pixels.shape[1:3])
            or self.landed + frames > self._frames
        ):
            raise OutputError(
                f"official H3 decode produced a {channels}x{height}x{width} chunk of {frames} "
                f"frames at frame {self.landed}; the request is {self._frames} frames of "
                f"3x{height}x{width}",
                code="output_integrity",
            )
        for start in range(0, frames, _RGB8_CHUNK_FRAMES):
            stop = self.landed + min(_RGB8_CHUNK_FRAMES, frames - start)
            landed = self.pixels[self.landed : stop]
            _land(torch, chunk[start : start + len(landed)], landed)
            self._slices.put((self.landed, self.landed + len(landed)))
            self.landed += len(landed)
        self._progress(self.landed - 1)

    def finish(self, progress: Callable[[float], None]) -> SavedVideo:
        """Close the stream once every requested frame has landed; the sink's probed asset,
        `progress` told the share of frames it took meanwhile. A sink that wrote fewer frames
        fails the run."""
        if self.landed != self._frames or self._encode is None:
            raise OutputError(
                f"official H3 decode returned {self.landed} frames, expected {self._frames}",
                code="output_integrity",
            )
        self._slices.put(None)
        self._encode.add_done_callback(lambda _: self._taken.put(None))
        try:
            while (index := self._taken.get()) is not None:
                if self._taken.empty():  # the newest only
                    progress((index + 1) / self._frames)
            saved = self._encode.result()
        finally:
            self._pool.shutdown(wait=False)
        if saved.frame_count != self._frames:
            raise OutputError(
                f"the video encoder wrote {saved.frame_count} of {self._frames} frames",
                code="output_integrity",
            )
        return saved

    def abandon(self) -> None:
        """A failed decode ends the stream so the sink aborts and unlinks its partial file."""
        self._abandoned = True
        self._slices.put(None)
        if self._encode is not None:
            with suppress(Exception):
                self._encode.result()
        self._pool.shutdown(wait=False)

    def _events(self) -> Iterator[DecodedMediaHeader | DecodedVideoFrame | DecodedAudioChunk]:
        _, height, width, _ = (int(value) for value in self.pixels.shape)
        channels, samples = (0, 0) if self._waveform is None else (int(v) for v in self._waveform.shape)
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=width,
                height=height,
                time_base=Fraction(1, FPS),
                pixel_aspect_ratio=Fraction(1),
                nominal_frame_rate=Fraction(FPS),
                **_VIDEO_COLOR,
            ),
            audio=None
            if self._waveform is None
            else DecodedAudioFormat(
                channels=channels,
                sample_rate=self._sample_rate,
                channel_layout=_AUDIO_LAYOUTS[channels][0],
                channel_names=_AUDIO_LAYOUTS[channels][1],
                time_base=Fraction(1, self._sample_rate),
            ),
        )
        submitted = 0
        while (item := self._slices.get()) is not None:
            start, stop = item
            landed = self.pixels[start:stop]
            for index in range(start, stop):
                self._check_encoding()
                yield DecodedVideoFrame(
                    width=width,
                    height=height,
                    rgb=bytes(landed[index - start].numpy()),
                    pts=index,
                    duration=1,
                    time_base=Fraction(1, FPS),
                    pixel_aspect_ratio=Fraction(1),
                    **_VIDEO_COLOR,
                )
                self._taken.put(index)
            covered = min(samples, stop * self._sample_rate // FPS)
            yield from self._audio(submitted, covered)
            submitted = covered
        self._check_encoding()
        yield from self._audio(submitted, samples)

    def _check_encoding(self) -> None:
        if self._abandoned:
            raise OutputError("the decode abandoned the video stream", code="output_integrity")
        self._cancel()

    def _audio(self, start: int, stop: int) -> Iterator[DecodedAudioChunk]:
        if stop <= start:
            return
        channels = int(self._waveform.shape[0])
        yield DecodedAudioChunk(
            channels=channels,
            sample_count=stop - start,
            sample_rate=self._sample_rate,
            channel_layout=_AUDIO_LAYOUTS[channels][0],
            channel_names=_AUDIO_LAYOUTS[channels][1],
            pcm_f32le=tuple(
                self._waveform[channel, start:stop].numpy().tobytes() for channel in range(channels)
            ),
            pts=start,
            time_base=Fraction(1, self._sample_rate),
        )


def _nonfinite_fraction(torch: Any, value: Any) -> float:
    """Count non-finite values with at most roughly 32 MiB of temporary mask."""
    if value.ndim < 2:
        chunks = (value,)
    else:
        values_per_column = value.numel() // int(value.shape[1])
        columns = max(1, (32 * _MIB) // max(1, values_per_column))
        chunks = value.split(columns, dim=1)
    bad = sum(int(torch.count_nonzero(~torch.isfinite(chunk))) for chunk in chunks)
    return float(bad / int(value.numel()))


#: The releases every H3 callable defaults to; the machine picks each lane from the GPU.
_MODEL = "minimaxai/minimax-h3@1.0.0-rc.7"
_TURBO_LORA = "alibaba-pai/minimax-h3-turbo-lora@1.0.1"
_REGENERATE_LORA = "efficient-large-model/open-h3-regenerate-2k@0.1.0"
_LATENT_UPSCALER = "lbh-123-ai/h3-latent-upscaler@0.1.0"


#: The components each task runs. A machine downloads these first and the rest behind the
#: run, so a pod's first request pays for one DiT, not two.
_FL2VA = ("text_encoder", "transformer", "vae", "audio_vae")
_REF2VA = ("text_encoder", "transformer_ref", "vae", "audio_vae")


#: The output canvas the Regenerate route computes (`regenerate.TARGETS`): the demand axis beside
#: the frames.
_REGENERATE_PIXELS = {
    Resolution.NATIVE: (1344, 768),
    Resolution.FULL_HD: (1920, 1088),
    Resolution.QHD: (2560, 1440),
    Resolution.UHD: (3840, 2176),
}
RegenerateResolution = Annotated[Resolution, Shape(pixels=_REGENERATE_PIXELS)]


class RegenerateInput(msgspec.Struct, kw_only=True):
    """`upscale`'s Regenerate route (`regenerate_shot`): the video's 768p frames encoded,
    upscaled in latent space and refined by H3 at `resolution` from `prompt`, its soundtrack
    kept. The 7:4 frame fills 16:9, stretched 1.6% horizontally; `768p` refines only."""

    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, max_decoded_bytes=2 << 30)]
    prompt: str
    resolution: RegenerateResolution = Resolution.QHD
    strength: float = 0.909
    seed: int | None = None

    def __post_init__(self) -> None:
        if round(self.strength, 3) not in REGENERATE_STRENGTHS:
            raise ValueError(
                f"strength is one of {', '.join(map(str, REGENERATE_STRENGTHS))}, got {self.strength}"
            )


class RegenerateOutput(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, media_types=("video/mp4",))]
    warnings: list[str]


#: The canvas the refine's latent is encoded at: H3's native 1344x768.
_DRAFT_SIZE = (768, 1344)


def _read_draft(video: VideoFile) -> tuple[Any, Any | None, int, int, list[str]]:
    """A video of any length as `(frames on the VAE grid, 768, 1344, 3)` uint8, each frame
    cover-scaled and centre-cropped as it decodes into one buffer sized from the header, its
    last frame held up to the grid; its `(channels, samples)` soundtrack or None, the sample
    rate, and how many frames it delivers."""
    probe = video.probe()
    if probe.frame_rate != FPS:
        raise InvalidRequest(f"regenerate takes 24 fps video, not {probe.frame_rate}", fields=["video"])
    expected = round(probe.duration * FPS)
    pixels = torch.empty((expected + 17, *_DRAFT_SIZE, 3), dtype=torch.uint8)
    pcm: list[list[Any]] = []
    count = width = height = rate = 0
    with video.stream() as stream:
        for event in stream:
            if isinstance(event, DecodedMediaHeader):
                if event.video is None:
                    raise InvalidRequest("regenerate needs a video stream", fields=["video"])
                width, height = event.video.width, event.video.height
            elif isinstance(event, DecodedVideoFrame):
                if count == len(pixels):
                    raise InvalidRequest(f"the video decodes past its header's {expected} frames", fields=["video"])
                frame = torch.frombuffer(bytearray(event.rgb), dtype=torch.uint8)
                pixels[count] = _cover(frame.reshape(height, width, 3))
                count += 1
            else:
                rate = event.sample_rate
                pcm.append([np.frombuffer(data, dtype=np.float32) for data in event.pcm_f32le])
    if count < MIN_FRAMES:
        raise InvalidRequest(f"regenerate takes at least {MIN_FRAMES} frames (5 s at 24 fps), not {count}",
                             fields=["video"])  # fmt: skip
    grid = count + (-(count - 5)) % 17
    if grid > len(pixels):
        pixels = torch.cat([pixels, pixels[-1:].expand(grid - len(pixels), -1, -1, -1)])
    pixels = pixels[:grid]
    pixels[count:] = pixels[count - 1]
    warnings = [] if (height, width) == _DRAFT_SIZE else [
        f"the {width}x{height} video was cover-scaled to 1344x768 for the refine"
    ]  # fmt: skip
    waveform = None
    if pcm:
        waveform = torch.from_numpy(np.stack([np.concatenate(c) for c in zip(*pcm, strict=True)]))
        if int(waveform.shape[0]) not in _AUDIO_LAYOUTS:
            raise InvalidRequest("regenerate keeps mono or stereo soundtracks", fields=["video"])
    return pixels, waveform, rate, count, warnings


def _cover(frame: Any) -> Any:
    """One `(h, w, 3)` uint8 frame cover-scaled (bilinear) and centre-cropped to 1344x768."""
    height, width = (int(value) for value in frame.shape[:2])
    if (height, width) == _DRAFT_SIZE:
        return frame
    scale = max(_DRAFT_SIZE[0] / height, _DRAFT_SIZE[1] / width)
    size = (max(_DRAFT_SIZE[0], round(height * scale)), max(_DRAFT_SIZE[1], round(width * scale)))
    image = torch.nn.functional.interpolate(
        frame.permute(2, 0, 1)[None].float(), size=size, mode="bilinear", align_corners=False
    )[0]
    top, left = (size[0] - _DRAFT_SIZE[0]) // 2, (size[1] - _DRAFT_SIZE[1]) // 2
    crop = image[:, top : top + _DRAFT_SIZE[0], left : left + _DRAFT_SIZE[1]]
    return crop.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)


def _plan(model: H3Model, turbo_lora: H3TurboLoRA | None, trunk: Trunk) -> TimestepPlan:
    """The schedules a call may run: its turbo LoRA's when it samples with one, else its model's."""
    return (model.pipe if turbo_lora is None else turbo_lora.pipe).plans[trunk]


def _render_keyframes(
    ctx: Context,
    task: Task,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    prompt: str,
    seed: int | None,
    duration_s: int,
    steps: int | None,
    first: Image | None,
    last: Image | None,
    turbo_lora: H3TurboLoRA | None = None,
    delivery: ContinuationPlan | None = None,
    completed: Callable[[Any], None] | None = None,
    ended: Callable[[Any], None] | None = None,
) -> H3VideoOutput:
    """An FL2VA render: text, and a first and/or last keyframe. `delivery` and `completed`
    hand a successor its carry, and `ended` its last frame (`fl2va_piece`)."""
    plan = _plan(model, turbo_lora, "fl2va")
    steps = plan.schedule(steps).transformer_evaluations
    keyframes = ("vae",) if first is not None or last is not None else ()
    dit = "transformer" if turbo_lora is None else {"transformer", "fl2va_turbo"}
    ctx.plan("text_encoder", *keyframes, dit, "audio_vae", "vae")
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=seed)
    notes = _retargeted(view.adapters, "transformer")
    log_prompt(tel, "h3 encoder prompt", prompt)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        state = model.pipe.start_fl2va(
            prompt=prompt,
            first_frame=first,
            last_frame=last,
            generator=model.pipe.generator(view.generator),
            steps=steps,
            frames=frames_for(duration_s),
            task=task,
            plan=plan,
            delivery=delivery,
        )
    with tel.stage("condition_text", overall_range=(0.03, 0.08)):
        model.condition_text(task, state)
    if first is not None or last is not None:
        with tel.stage("condition_media", overall_range=(0.08, 0.15)):
            model.condition_fl2va_media(task, state)
    if turbo_lora is None:
        sample = model.sample_fl2va
    else:
        assert isinstance(model, H3TurboBase)
        sample = partial(model.sample_fl2va_turbo, turbo_lora=turbo_lora)
    with tel.stage("denoise", overall_range=(0.15, 0.85)):
        schedule = sample(
            state,
            on_step=tel.step_callback(steps, stage="denoise", overall_range=(0.15, 0.85)),
            cancel=ctx.raise_if_cancelled,
        )
    result = _finish(
        model, task, state, schedule, duration_s=duration_s, out=out, tel=tel,
        cancel=ctx.raise_if_cancelled, completed=completed, ended=ended,
        delivered_frames=None if delivery is None else delivery.delivered_frames,
        sampled_frames=None if delivery is None else delivery.sample_frames,
    )  # fmt: skip
    return H3VideoOutput(result.video, notes + result.warnings)


def _references_to_video(
    ctx: Context,
    task: Task,
    payload: Ref2VAInput,
    assets: ReferenceFiles,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    turbo_lora: H3TurboLoRA | None,
    context: AVContext | None,
    delivery: ContinuationPlan,
    expected_context_provenance: str,
    completed: Callable[[Any], None] | None,
    anchors: Sequence[tuple[Image, Literal["first", "last"]]],
    ended: Callable[[Any], None],
    earlier: _Earlier | None = None,
) -> H3VideoOutput:
    plan = _plan(model, turbo_lora, "ref2va")
    steps = plan.schedule(payload.steps).transformer_evaluations
    dit = "transformer_ref" if turbo_lora is None else {"transformer_ref", "ref2va_turbo"}
    ctx.plan("text_encoder", {"vae", "audio_vae"}, dit, "audio_vae", "vae")
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=payload.seed)
    notes = _retargeted(view.adapters, "transformer_ref")
    if payload.frames != held_frames(delivery, payload.videos):
        raise InvalidRequest(
            f"a {payload.duration_s} s piece with these references holds "
            f"{held_frames(delivery, payload.videos)} frames, not {payload.frames}",
            fields=["payload.frames"],
        )
    frames = delivery.sample_frames
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        references, sizing, guide = assets_to_h3_refs(
            assets, pipe=model.pipe, frames=frames, videos=payload.videos, edges=payload.picture_edges
        )
        tel.log(
            "h3 reference sizing",
            images=sizing.summary,
            video_tokens=sizing.video_tokens,
            default=sizing.default,
            total=sizing.total,
            budget=MAX_CONDITIONER_VISION_TOKENS,
        )
        prompt = encoder_prompt(task, payload.prompt)
        log_prompt(tel, "h3 encoder prompt", prompt)
        state = model.pipe.start_ref2va(
            prompt=prompt,
            references=references,
            generator=model.pipe.generator(view.generator),
            steps=steps,
            frames=frames,
            context=context,
            delivery=delivery,
            expected_context_provenance=expected_context_provenance,
            reference_image_short_edges=sizing.edges,
            task=task,
            plan=plan,
            guide=guide,
            anchors=anchors,
        )
        for index, reference in enumerate(state.normalized_references):
            if reference.kind == "image":
                info = assets.info(index)
                width, height = reference.image.size
                tel.log(
                    "h3 reference resolution",
                    input_id=info.id,
                    label=info.label,
                    fidelity=info.fidelity,
                    width=width,
                    height=height,
                )
    # Text and references at once where the text encoder lives on another GPU.
    with tel.stage("condition", overall_range=(0.03, 0.15)):
        condition_references(model, task, state, tel=tel)
    if turbo_lora is None:
        sample = model.sample_ref2va
    else:
        assert isinstance(model, H3TurboBase)
        sample = partial(model.sample_ref2va_turbo, turbo_lora=turbo_lora)
    with tel.stage("denoise", overall_range=(0.15, 0.85)):
        schedule = sample(
            state,
            on_step=tel.step_callback(steps, stage="denoise", overall_range=(0.15, 0.85)),
            cancel=ctx.raise_if_cancelled,
        )
    result = _finish(
        model, task, state, schedule, duration_s=payload.duration_s, out=out, tel=tel,
        cancel=ctx.raise_if_cancelled, completed=completed, ended=ended,
        delivered_frames=delivery.delivered_frames, sampled_frames=delivery.sample_frames, earlier=earlier,
    )  # fmt: skip
    return H3VideoOutput(result.video, notes + result.warnings)


def _keep_frames(
    out: Outputs, exports: Sequence[int], last: list[ImageFile], exported: list[ImageFile], pixels: Any
) -> None:
    """A call's delivered `(frames, H, W, 3)` RGB8: its last frame, and the `exports` it returns."""
    last.append(out.save_image(PILImage.fromarray(pixels[-1].numpy())))
    exported.extend(out.save_image(PILImage.fromarray(pixels[index].numpy())) for index in exports)


def _anchors(
    first: ImageFile | None, last: ImageFile | None
) -> list[tuple[Image, Literal["first", "last"]]]:
    """A Ref2VA call's keyframes, read: anchored as ComfyUI's Add Guide anchors them."""
    pairs: list[tuple[ImageFile | None, Literal["first", "last"]]] = [(first, "first"), (last, "last")]
    return [(image, at) for asset, at in pairs if (image := _frame(asset, f"{at}_frame")) is not None]


# --- pieces ----------------------------------------------------------------------------

ContextFrames = Literal[0, 22, 39, 56]


ContextFile = Annotated[
    File, FileBound(max_bytes=16 << 20, media_types=("application/octet-stream",))
]
#: A piece's delivered video latents (`_Pieces`), about 3 MB a second at 768p.
LatentFile = Annotated[File, FileBound(max_bytes=256 << 20, media_types=("application/octet-stream",))]
PieceVideo = Annotated[VideoFile, FileBound(max_bytes=1 << 30, max_decoded_bytes=32 << 20, media_types=("video/mp4",))]


def held_frames(plan: ContinuationPlan, videos: Sequence[VideoUse] = ()) -> int:
    """The frames a call's DiT holds: its sampled window, plus the carried picture again as
    condition rows (the double form; single-copy was tested and rejected, 2026-10-10), plus
    an anchor video's frames, at most the window's again."""
    anchored = any(video.anchor for video in videos)
    return plan.sample_frames * (1 + anchored) + plan.carried.frames


class Ref2VAInput(msgspec.Struct):
    prompt: str
    seed: int
    duration_s: PieceSeconds
    steps: int | None
    turbo: bool
    #: `held_frames` of this piece: the shape Runtime banks its memory by. A continuation
    #: holds more than its seconds say, so a 12 s piece after a 6 s one ran on the 6 s
    #: peak and ran out on GPU 1 (run 1560). Any other value is refused.
    frames: int
    #: How the piece uses each of its videos, in packed order.
    videos: tuple[VideoUse, ...]
    #: What this piece carries from its predecessor, and what its successor carries.
    context_frames: ContextFrames = 0
    context_audio: bool = False
    next_context_frames: ContextFrames = 0
    next_context_audio: bool = False
    previous_provenance: RenderProvenance | None = None
    context: ContextFile | None = None
    #: Keyframes, anchored as ComfyUI's Add Guide anchors them (`anchor_rows`).
    first_frame: FrameImage = None
    last_frame: FrameImage = None
    #: Delivered frames to return (`PieceOutput.frames`), by index.
    exports: tuple[int, ...] = ()
    #: Each picture's short edge in packed order; 0 sizes it as usual.
    picture_edges: tuple[int, ...] = ()
    #: Return the delivered latents (`PieceOutput.latents`).
    latents: bool = False
    #: A shot's last piece: the earlier pieces' latents and videos, decoded with it as one.
    earlier_latents: tuple[LatentFile, ...] = ()
    earlier_videos: tuple[PieceVideo, ...] = ()


class PieceOutput(msgspec.Struct):
    #: The piece's video; a shot's last piece returns the whole shot.
    video: PieceVideo
    warnings: list[str]
    provenance: RenderProvenance
    # A fixed output slot: Runtime grants result outputs by field path, never inside a union.
    # A piece with no successor returns it empty instead of exporting any window.
    context: ContextFile
    #: The piece's last delivered frame. A child's result is a granted input of the parent,
    #: so it may feed the next piece (a picture the parent saved itself may not, run 20651).
    last_frame: ImageFile
    #: The delivered frames the call's `exports` asks for, in order.
    frames: list[ImageFile]
    #: Its delivered video latents when the call asks (`latents`), else empty: a shot's last
    #: piece decodes them with its own as one.
    latents: LatentFile


@dataclass(frozen=True)
class _Earlier:
    """A shot's pieces before its last: their delivered latents end to end, their sound end to
    end (each starts where the last ended, as continuation cut it), and their frames; and the
    last piece's head to leave out, whole VAE cycles, so its latents follow in phase."""

    latents: Any
    waveform: Any
    frames: int
    head: int

    @classmethod
    def read(cls, latents: Sequence[File], videos: Sequence[VideoFile], head: int) -> _Earlier:
        parts = []
        for video in videos:
            with video.stream(pictures=False) as stream:
                for event in stream:
                    if isinstance(event, DecodedAudioChunk):
                        parts.append(np.stack([np.frombuffer(pcm, dtype=np.float32) for pcm in event.pcm_f32le]))
        joined = torch.cat([load_tensors(file.read_bytes())["latents"] for file in latents], dim=2)
        # 5k + 2 latents decode to 17k + 5 frames (`continuation.video_tokens`).
        frames = (int(joined.shape[2]) - 2) // 5 * 17 + 5
        return cls(joined, torch.from_numpy(np.concatenate(parts, axis=1)), frames, head)


class _Pieces:
    """What a piece hands on once it has denoised, before its decode: its successor's carry
    and, when asked, its delivered latents."""

    def __init__(self, out: Outputs, model: H3Model, plan: ContinuationPlan, observed: RenderProvenance,
                 latents: bool) -> None:  # fmt: skip
        self.out, self.model, self.plan, self.observed, self.latents = out, model, plan, observed, latents
        self.saved: dict[str, File] = {}

    def __call__(self, state: Any) -> None:
        if self.plan.successor:
            carried = self.model.pipe.export_context(state, provenance=context_provenance(self.observed))
            assert carried is not None
            self.saved["context"] = self.out.save_bytes(encode_context(carried), media_type="application/octet-stream")
        if self.latents:
            # The head is whole VAE cycles (`continuation`), so what follows it joins the
            # previous piece's latents in phase.
            head = video_tokens(self.plan.carried.frames) if self.plan.carried.frames else 0
            delivered = state.latents[:, :, head:].detach().float().cpu().contiguous()
            self.saved["latents"] = self.out.save_bytes(save_tensors({"latents": delivered}),
                                                       media_type="application/octet-stream")  # fmt: skip

    @property
    def wanted(self) -> bool:
        return bool(self.plan.successor) or self.latents

    def file(self, name: str) -> File:
        return self.saved.get(name) or self.out.save_bytes(b"", media_type="application/octet-stream")


def _sampling(payload: FL2VAInput | Ref2VAInput, turbo_lora: H3TurboLoRA) -> H3TurboLoRA | None:
    """The piece's sampler over the one shared construction: the PDD adapter, or none."""
    if payload.turbo and payload.steps is not None:
        raise InvalidRequest("a turbo piece runs its LoRA's one schedule; it takes no steps")
    return turbo_lora if payload.turbo else None


@invocable(
    defaults={"base_model": _MODEL, "turbo_lora": _TURBO_LORA},
    components={"base_model": _REF2VA, "turbo_lora": ("ref2va_turbo",)},
    memoize=True,
)
async def ref2va_piece(
    ctx: Context,
    *,
    payload: Ref2VAInput,
    assets: ReferenceFiles,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> PieceOutput:
    """A piece on Ref2VA: its references, the context it carries in, and its successor's."""
    lora = _sampling(payload, turbo_lora)
    observed = provenance(base_model.checkpoint_ref, "" if lora is None else lora.checkpoint_ref)
    plan = ContinuationPlan(
        payload.duration_s,
        Carry(int(payload.context_frames), payload.context_audio),
        Carry(int(payload.next_context_frames), payload.next_context_audio),
    )
    context: AVContext | None = None
    if plan.carried:
        if payload.context is None:
            raise InvalidRequest("a carried picture or sound needs the previous context", fields=["payload.context"])
        context = decode_context(payload.context.read_bytes())
    _reference_policy(assets)
    handed = _Pieces(out, base_model, plan, observed, payload.latents)
    earlier = None
    if payload.earlier_latents:
        head = video_tokens(plan.carried.frames) if plan.carried.frames else 0
        earlier = _Earlier.read(payload.earlier_latents, payload.earlier_videos, head)
    ended: list[ImageFile] = []
    exported: list[ImageFile] = []
    skip = 0 if earlier is None else earlier.frames
    piece = _references_to_video(
        ctx, "ref2va" if lora is None else "ref2va_turbo", payload, assets, base_model, out, tel,
        turbo_lora=lora, context=context, delivery=plan,
        expected_context_provenance=context_provenance(payload.previous_provenance or observed),
        completed=handed if handed.wanted else None,
        anchors=_anchors(payload.first_frame, payload.last_frame),
        ended=partial(_keep_frames, out, tuple(skip + index for index in payload.exports), ended, exported),
        earlier=earlier,
    )  # fmt: skip
    return PieceOutput(piece.video, list(piece.warnings), observed, handed.file("context"), ended[0], exported,
                       handed.file("latents"))  # fmt: skip


app.child_task(ref2va_piece)


def log_prompt(tel: Telemetry, event: str, prompt: str, **fields: str | int) -> None:
    """The text's digest and ends: a log value holds 400 characters, not a whole prompt."""
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    tel.log(event, level="info", digest=digest, chars=len(prompt), head=prompt[:400], tail=prompt[-400:], **fields)




#: The full model's step counts; turbo always runs its LoRA's 8.
FullSteps = Literal[30, 40, 50]
#: A keyframe the shot supplies (`keyframes.first`/`last`).
FrameImage = Annotated[
    ImageFile | None,
    FileBound(
        max_bytes=64 << 20,
        max_decoded_bytes=256 << 20,
        media_types=("image/png", "image/jpeg", "image/webp"),
    ),
]


class FL2VAInput(msgspec.Struct, kw_only=True):
    """One FL2VA piece: its text, keyframes and length, and what its successor carries."""

    prompt: str
    seed: int
    duration_s: DurationSeconds
    steps: int | None
    turbo: bool
    first_frame: FrameImage = None
    last_frame: FrameImage = None
    next_context_frames: ContextFrames = 0
    next_context_audio: bool = False
    #: Delivered frames to return (`PieceOutput.frames`), by index.
    exports: tuple[int, ...] = ()
    #: Return the delivered latents (`PieceOutput.latents`).
    latents: bool = False


def _frame(asset: ImageFile | None, field: str) -> Image | None:
    if asset is None:
        return None
    image = asset.read()
    _validate_ratio(image.width, image.height, field)
    return image


def _retargeted(adapters: Sequence[Any], dit: str) -> list[str]:
    """A line for each caller LoRA trained on the other transformer: Runtime applies it to this
    one (as ComfyUI would load it), and says so once per LoRA (`shot` keeps one)."""
    other = {"transformer": "FL2VA", "transformer_ref": "Ref2VA"}
    return [
        f"LoRA {adapter.ref} was trained on {adapter.source_component} ({other.get(adapter.source_component, '?')}) "
        f"and is applied to {dit} ({other[dit]})"
        for adapter in adapters
        if adapter.source_component in other and adapter.source_component != dit
    ]


@invocable(
    defaults={"base_model": _MODEL, "turbo_lora": _TURBO_LORA},
    components={"base_model": _FL2VA, "turbo_lora": ("fl2va_turbo",)},
    memoize=True,
)
async def fl2va_piece(
    ctx: Context,
    *,
    payload: FL2VAInput,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> PieceOutput:
    """A piece on FL2VA: its text and its first and/or last keyframe, or text alone."""
    lora = _sampling(payload, turbo_lora)
    observed = provenance(base_model.checkpoint_ref, "" if lora is None else lora.checkpoint_ref)
    plan = ContinuationPlan(
        payload.duration_s, Carry(),
        Carry(int(payload.next_context_frames), payload.next_context_audio),
    )  # fmt: skip
    handed = _Pieces(out, base_model, plan, observed, payload.latents)
    ended: list[ImageFile] = []
    exported: list[ImageFile] = []
    shot = _render_keyframes(
        ctx, "fl2va" if lora is None else "fl2va_turbo", base_model, out, tel,
        prompt=encoder_prompt("fl2va", payload.prompt), seed=payload.seed,
        duration_s=payload.duration_s, steps=payload.steps,
        first=_frame(payload.first_frame, "first_frame"), last=_frame(payload.last_frame, "last_frame"),
        turbo_lora=lora, delivery=plan if plan.successor else None,
        completed=handed if handed.wanted else None,
        ended=partial(_keep_frames, out, payload.exports, ended, exported),
    )  # fmt: skip
    return PieceOutput(shot.video, list(shot.warnings), observed, handed.file("context"), ended[0], exported,
                       handed.file("latents"))  # fmt: skip


app.child_task(fl2va_piece)


@invocable(
    defaults={"base_model": _MODEL, "regenerate_lora": _REGENERATE_LORA, "latent_upscaler": _LATENT_UPSCALER},
    components={"base_model": _FL2VA, "regenerate_lora": ("fl2va_regenerate",), "latent_upscaler": ("upscaler",)},
    memoize=True,
)
async def regenerate_shot(
    ctx: Context,
    *,
    payload: RegenerateInput,
    base_model: H3TurboBase,
    regenerate_lora: H3RegenerateLoRA,
    latent_upscaler: H3LatentUpscaler,
    out: Outputs,
    tel: Telemetry,
) -> RegenerateOutput:
    """`upscale.model: regenerate-2k` over a whole shot: its 768p latent upscaled, refined at
    `resolution` in overlapping windows where the card needs them, and decoded with its
    soundtrack."""
    ctx.plan("vae", "text_encoder", "audio_vae", "upscaler", {"transformer", "fl2va_regenerate"}, "vae")
    cancel = ctx.raise_if_cancelled
    with tel.stage("read_video", overall_range=(0.00, 0.03)):
        pixels, waveform, sample_rate, frames, warnings = _read_draft(payload.video)
    with tel.stage("encode_video", overall_range=(0.03, 0.08)):
        latents = base_model.encode_clip(pixels)
    del pixels
    resolution = payload.resolution.value
    with tel.stage("upscale_latent", overall_range=(0.08, 0.10)):
        for height, width in refine_math.upscale_passes(resolution):
            latents = latent_upscaler.upscale(latents, height=height, width=width)
    cancel()
    with tel.stage("condition_text", overall_range=(0.10, 0.14)):
        text = base_model.encode_prompt(encoder_prompt("fl2va", payload.prompt), frames_for(MIN_DURATION_S))
    spans = refine_math.plan_windows(tuple(int(value) for value in latents.shape))
    schedule = regenerate_lora.pipe.plan.refined(payload.strength)
    steps = schedule.transformer_evaluations
    seed = payload.seed if payload.seed is not None else int.from_bytes(
        hashlib.sha256(f"{ctx.request_id}/h3/regenerate".encode()).digest()[:4], "big")  # fmt: skip
    tel.log("h3 regenerate", resolution=resolution, strength=payload.strength, start_sigma=schedule.start,
            transformer_evaluations=steps, latent="x".join(str(int(value)) for value in latents.shape[2:]),
            windows=len(spans), window_latent_frames=spans[0][1])  # fmt: skip
    with tel.stage("refine", overall_range=(0.14, 0.85)):
        refined = base_model.refine(
            latents, overlay=regenerate_lora, prompt_embeds=text[0], text_tags=text[1], spans=spans,
            audio=base_model.silence(refine_math.frames_of(spans[0][1])), schedule=schedule, seed=seed,
            on_step=tel.step_callback(steps, stage="refine", overall_range=(0.14, 0.85)),
        )  # fmt: skip
    cancel()
    height, _, keep = refine_math.TARGETS[resolution]
    top = (height - keep) // 2
    with tel.stage("decode_video", overall_range=(0.85, 0.97)):
        stream = _VideoStream(
            torch, out, frames=frames, waveform=waveform, sample_rate=sample_rate, cancel=cancel,
            progress=tel.step_callback(frames, stage="decode_video", overall_range=(0.85, 0.97)),
        )  # fmt: skip
        delivered = 0

        def push(chunk: Any) -> None:
            nonlocal delivered
            take = min(int(chunk.shape[0]), frames - delivered)
            if take > 0:
                stream.push(chunk[:take, :, top : top + keep])
                delivered += take

        try:
            base_model.decode_latents(refined, on_chunk=push)
        except BaseException:
            stream.abandon()
            raise
    try:
        saved = stream.finish(lambda share: tel.progress(share, stage="encode_video"))
    except BaseException:
        stream.abandon()
        raise
    tel.log("h3 container facts", video_codec=saved.video_codec, frame_count=saved.frame_count,
            frame_rate=str(saved.frame_rate), video_bytes=saved.video.size_bytes)  # fmt: skip
    ctx.release_gpus()
    return RegenerateOutput(saved.video, warnings)


app.child_task(regenerate_shot)


# --- shot: one continuous take of any length (`video/shot`) ------------------------------
# The project names the pictures a shot uses and renders them; a shot here renders one take.
# Its pieces of at most 15 s each continue the last from its carried latents, never decoded.
# A piece is an ordinary serving child memoized on its full inputs (the previous piece's context
# included), so a resubmitted shot skips the pieces it already finished. The CPU parent joins
# each piece's packets as it lands and publishes the video so far.

ShotImage = Annotated[
    ImageFile,
    FileBound(max_bytes=64 << 20, max_decoded_bytes=256 << 20,
              media_types=("image/png", "image/jpeg", "image/webp")),
]  # fmt: skip
ShotSound = Annotated[AudioFile, FileBound(max_bytes=256 << 20, max_decoded_bytes=AUDIO_READ_BYTES), REFERENCE_AUDIO]
ShotFootage = Annotated[
    VideoFile, FileBound(max_bytes=VIDEO_MAX_BYTES, max_decoded_bytes=VIDEO_READ_BYTES), REFERENCE_VIDEO
]


class Beat(msgspec.Struct):
    """From `at` seconds into the shot, `prompt` is what happens."""

    at: Annotated[float, msgspec.Meta(ge=0)]
    prompt: str


class ShotAsset(msgspec.Struct, kw_only=True):
    """A picture of a character, a prop, or the environment (the shot's setup plate). Every one
    is used, named in the text or not; `description` joins its definition, `voice` is a
    character's timbre."""

    name: str
    kind: Literal["character", "prop", "environment"]
    image: ShotImage
    description: str = ""
    voice: ShotSound | None = None


class ShotAudio(msgspec.Struct):
    name: str
    audio: ShotSound


class ReferenceFootage(msgspec.Struct, kw_only=True):
    """A clip whose motion and camera the shot performs, second for second; its sound only
    with `use_audio`."""

    name: str
    video: ShotFootage
    use_audio: bool = False


class Plate(msgspec.Struct):
    """A clip the shot follows frame by frame (Add Guide rows), across all its pieces."""

    video: ShotFootage


class Keyframes(msgspec.Struct, kw_only=True):
    first: FrameImage = None
    last: FrameImage = None


class WorldState(msgspec.Struct, kw_only=True):
    """How a place stood: a soft picture of the set, read at 512 px, and `layout`, one line of
    who and what is where."""

    name: str
    image: ShotImage
    layout: str = ""


class ShotRequest(msgspec.Struct, kw_only=True):
    """One continuous shot of any length (README.md). `writes` asks for frames of it: `first`,
    `last` or seconds in. `turbo` and `steps` are this package's sampling: the turbo LoRA's
    eight steps, or the full model's 30, 40 or 50."""

    prompt: str
    duration_s: Annotated[float, msgspec.Meta(
        gt=0, extra={"max_piece_s": 15},
        description="Any length; rendered in pieces of at most 15 s (max_piece_s), so a longer shot needs beats.",
    )]  # fmt: skip
    beats: list[Beat] = []
    seed: int | None = None
    style: str = ""
    soundscape: str = ""
    music: str = "N/A"
    assets: list[ShotAsset] = []
    audio: list[ShotAudio] = []
    reference_footage: list[ReferenceFootage] = []
    plate: Plate | None = None
    keyframes: Keyframes | None = None
    world_state: list[WorldState] = []
    writes: list[Literal["first", "last"] | Annotated[float, msgspec.Meta(ge=0)]] = []
    turbo: bool = True
    steps: FullSteps | None = None


class ShotResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, media_types=("video/mp4",))]
    #: One per `writes` entry, in order.
    frames: list[ImageFile]
    warnings: list[str]


#: What each later piece carries from the one before: 22 frames of picture and one second of sound.
_CARRY = Carry(22, True)
#: A later piece's new seconds: its carried frames share the 362-frame window, so 14.
_MAX_NEXT_S = (MAX_FRAMES - _CARRY.head_frames) // FPS
#: A world state's picture is read at this short edge: the full picture opens the shot on a
#: copy of it (run 20661, first-frame SSIM 0.871), 256 px keeps the set but not who stands
#: where (run 20689); its `layout` line carries the positions.
_WORLD_STATE_EDGE = 512
#: The first piece's closing sentence, and each later piece's opening one. A pinned head
#: repeats a camera move in every piece it continues, so later pieces hold the framing.
_ONE_SHOT = "One continuous shot with no cuts."
_CONTINUED = "The same shot continues without a cut; the camera holds its framing."
#: A sentence that moves the camera, left out of a continued piece.
_CAMERA_MOVE = re.compile(
    r"\b(?:camera|pans?|panning|tilts?|tilting|zooms?|zooming|dolly|dollies|crane|orbits?|push(?:es)? in|pulls? (?:back|out))\b",
    re.I,
)
#: A pinned head delays a new action about 2.4 s, so no beat starts from 2 s before a piece's
#: edge to 2.5 s after it, where the pieces can avoid it.
_BEFORE_EDGE_S, _AFTER_EDGE_S = 2.0, 2.5
#: A field that ends a text's description (MiniMax's FL2VA lines, Ref2VA's sections), and a
#: description's own heading.
_AFTER = re.compile(r"^[ \t]*(?:audio|overall[ _-]*soundscape|non[ _-]*diegetic[ _-]*music)[ \t]*:", re.I | re.M)
_DESCRIPTION = re.compile(r"^[ \t]*(?:integrated[ _-]*multimodal|detailed)[ _-]*description[ \t]*:[ \t]*\n?", re.I)


def _delivered(seconds: int, first: bool) -> int:
    """The frames a piece of `seconds` new seconds delivers on H3's 17n+5 grid."""
    return ContinuationPlan(seconds, Carry() if first else _CARRY).delivered_frames


def plan_pieces(duration_s: float, beats: Iterable[float]) -> list[int]:
    """A shot's pieces in whole seconds: the fewest that reach `duration_s` (the first at most
    15 s, each later one at most 14, none under 5). Within that count: the fewest edges in a
    beat's window (`_BEFORE_EDGE_S` before it to `_AFTER_EDGE_S` after it), so each piece cuts
    between beats; then the least overshoot, in whole seconds; then each piece as long as it
    can be (Paul, 10-30: a longer generation is more consistent); then the least overshoot in
    frames."""
    target = max(1, math.ceil(duration_s * FPS - 1e-6))
    count = 1 + max(0, math.ceil((target - _delivered(MAX_DURATION_S, True)) / _delivered(_MAX_NEXT_S, False)))
    marks = [at * FPS for at in beats]
    # Each end, in frames, the pieces before the last reach: its score (minus the edges in a
    # beat's window, the lengths) and those lengths.
    best: dict[int, tuple[int, list[int]]] = {0: (0, [])}
    for index in range(count - 1):
        reached: dict[int, tuple[int, list[int]]] = {}
        for start, (clear, lengths) in best.items():
            for length in range(MIN_DURATION_S, (MAX_DURATION_S if index == 0 else _MAX_NEXT_S) + 1):
                end = start + _delivered(length, index == 0)
                hit = any(end - _BEFORE_EDGE_S * FPS <= mark <= end + _AFTER_EDGE_S * FPS for mark in marks)
                score = (clear - hit, [*lengths, length])
                if end < target and (end not in reached or score > reached[end]):
                    reached[end] = score
        best = reached
    longest = MAX_DURATION_S if count == 1 else _MAX_NEXT_S
    plans = [
        (clear, -math.ceil((end - target) / FPS), lengths, -end, [*lengths, length])
        for start, (clear, lengths) in best.items()
        for length in range(MIN_DURATION_S, longest + 1)
        if (end := start + _delivered(length, count == 1)) >= target
        and all(start + _delivered(shorter, count == 1) < target for shorter in range(MIN_DURATION_S, length))
    ]
    return max(plans)[-1]


def piece_texts(prompt: str, beats: Sequence[Beat], starts: Sequence[float]) -> list[str]:
    """Each piece's text, one `[Shot 1]` paragraph with no cut times (H3's `[Shot N] At
    MM:SS.mmm` marks a cut; its guide writes in-shot change as plain English): the shot's
    text, the beat it is in (continuing, when that beat began before it) and the beats it
    reaches, in order. The first piece ends saying it is one continuous shot; a later one
    opens saying it continues and the camera holds its framing, and leaves out the sentences
    that move the camera. `starts` are the pieces' start times. The text ends the prompt's
    description, ahead of any audio or music field it has."""
    prompt = prompt.strip()
    head = found.group(0) if (found := _DESCRIPTION.match(prompt)) else ""
    after = _AFTER.search(prompt, len(head))
    body = prompt[len(head) : after.start() if after else None].strip()
    tail = prompt[after.start() :].strip() if after else ""
    timed = sorted(((beat.at, beat.prompt.strip()) for beat in beats), key=lambda beat: beat[0])
    texts = []
    for index, (start, end) in enumerate(zip(starts, [*starts[1:], math.inf], strict=True)):
        parts = [body]
        if current := [beat for beat in timed if beat[0] <= start]:
            at, text = current[-1]
            parts.append(f"Continuing: {text}" if at < start else text)
        parts += [f"Then {_lower(text)}" for at, text in timed if start < at < end]
        if index:
            parts = [_CONTINUED, *(" ".join(_sentences(part, still=True)) for part in parts)]
        else:
            parts.append(_ONE_SHOT)
        text = head + "[Shot 1] " + " ".join(part for part in parts if part)
        texts.append(f"{text}\n\n{tail}" if tail else text)
    return texts


def shot_windows(
    references: Sequence[StoryReference], spans: Sequence[tuple[int, int]]
) -> tuple[list[dict[str, int]], list[str]]:
    """Where each piece's window of each clip starts, and the warnings. A piece reads the shot
    frames it samples, its carried head included (`spans`: first frame, count), 1:1 from the
    clip's first frame on the 24-fps clock, so a plate stays frame-aligned across an edge. A
    piece takes the tail of a clip that ends in its window, and none of one that leaves it
    less than `REFERENCE_VIDEO`'s minimum."""
    least = math.ceil(Fraction(REFERENCE_VIDEO.min_seconds or 0) * FPS)
    windows: list[dict[str, int]] = [{} for _ in spans]
    total = spans[-1][0] + spans[-1][1]
    warnings = []
    for ref in references:
        if ref.kind not in CLIPS:
            continue
        assert ref.video is not None
        length = math.floor(ref.video.probe().duration * FPS + Fraction(1, 2))
        for window, (start, frames) in zip(windows, spans, strict=True):
            if min(start + frames, length) - start >= least:
                window[ref.name] = start
        if ref.kind == "video" and (span := sum(ref.name in window for window in windows)) > 1:
            warnings.append(f"<{ref.name}> guides {span} pieces, each from its own span: put each beat where the clip "
                            "shows it, or the shot runs ahead of the clip")  # fmt: skip
        if length < total:
            warnings.append(f"<{ref.name}> ends {length / FPS:.1f} s into this {total / FPS:.1f} s shot, "
                            "which runs on without it")  # fmt: skip
        elif length > total:
            warnings.append(f"<{ref.name}> is {length / FPS:.1f} s long; the shot uses its first {total / FPS:.1f} s")
    return windows, warnings


def _sentences(text: str, *, still: bool) -> list[str]:
    """`text`'s sentences; with `still`, those that move the camera are left out."""
    found = re.findall(r"[^.!?]+[.!?]*", text)
    return [s.strip() for s in found if s.strip() and not (still and _CAMERA_MOVE.search(s))]


def _lower(text: str) -> str:
    """A sentence continuing another: its first letter lower-cased, unless it opens a name."""
    return text[:1].lower() + text[1:] if text[:1].isupper() and not text[1:2].isupper() and not text.startswith("<") else text


def _fl2va_text(text: str, payload: ShotRequest) -> str:
    """An FL2VA piece's text: its own, opening with the style, with the soundscape as its `audio`
    line and the music as its `non_diegetic_music` line where it has none."""
    if (style := payload.style.strip()) and style.casefold() not in text.casefold():
        head = found.group(0) if (found := _DESCRIPTION.match(text)) else ""
        text = f"{head}{style.rstrip('.')}. {text[len(head):].lstrip()}"
    lines = []
    if payload.soundscape.strip() and not re.search(r"^\s*audio\s*:", text, re.I | re.M):
        lines.append(f"audio: {payload.soundscape.strip()}")
    if "non_diegetic_music" not in text.casefold():
        lines.append(f"non_diegetic_music: {payload.music.strip() or 'N/A'}")
    return "\n\n".join([text.rstrip(), *lines])


def _shot_references(payload: ShotRequest) -> tuple[list[StoryReference], list[str]]:
    """Every picture, sound and clip the shot attaches, and the warnings reading them gives."""
    refs = [
        StoryReference(name=asset.name, kind="background" if asset.kind == "environment" else asset.kind,
                       description=asset.description, image=asset.image, voice=asset.voice)
        for asset in payload.assets
    ]  # fmt: skip
    refs += [StoryReference(name=sound.name, kind="audio", audio=sound.audio) for sound in payload.audio]
    warnings = []
    for clip in payload.reference_footage:
        silent = clip.use_audio and clip.video.probe().soundtrack is None
        if silent:
            warnings.append(f"<{clip.name}> has no audio track; use_audio ignored")
        refs.append(StoryReference(name=clip.name, kind="video", video=clip.video, use_audio=clip.use_audio and not silent))
    refs += [StoryReference(name=state.name, kind="world_state", image=state.image, description=state.layout)
             for state in payload.world_state]  # fmt: skip
    if payload.plate is not None:
        refs.append(StoryReference(name="Plate", kind="anchor_video", video=payload.plate.video))
    validate_references(refs)
    return refs, warnings


def piece_seed(seed: int | None, request_id: str, index: int) -> int:
    """The shot's seed, one more per piece; or chosen once per request, so a retried attempt
    keeps it."""
    if seed is not None:
        return seed + index
    return int.from_bytes(hashlib.sha256(f"{request_id}/h3/shot/{index}".encode()).digest()[:4], "big")


async def shot(ctx: Context, payload: ShotRequest, out: Outputs, tel: Telemetry) -> ShotResult:
    """One continuous shot (`video/shot`): its pieces rendered in order, each continuing the
    last, joined into one video published as it grows; and the frames its `writes` ask for, each
    taken by the piece that delivers it. The run's model stack (`--lora`, `model.base_model`)
    serves every piece."""
    ctx.raise_if_cancelled()
    pieces = plan_pieces(payload.duration_s, [beat.at for beat in payload.beats])
    if len(pieces) > 1 and not payload.beats:
        raise InvalidRequest(f"a shot longer than {MAX_DURATION_S} s needs beats: say what happens when "
                             f"in this {payload.duration_s:g} s shot", fields=["beats"])  # fmt: skip
    keyframes = payload.keyframes or Keyframes()
    if payload.plate is not None and keyframes.first is not None:
        raise InvalidRequest("a plate sets the shot's opening; it takes no first keyframe", fields=["keyframes.first"])
    references, warnings = _shot_references(payload)
    if payload.turbo and payload.steps is not None:
        warnings.append(f"turbo runs its LoRA's 8 steps; steps {payload.steps} ignored")
    steps = None if payload.turbo else payload.steps
    count = len(pieces)
    plans = [ContinuationPlan(length, _CARRY if index else Carry(), _CARRY if index + 1 < count else Carry())
             for index, length in enumerate(pieces)]  # fmt: skip
    *starts, total = accumulate((plan.delivered_frames for plan in plans), initial=0)
    windows, notes = shot_windows(references, [(start - plan.prefix_frames, plan.sample_frames)
                                               for start, plan in zip(starts, plans, strict=True)])  # fmt: skip
    warnings += notes
    exports: dict[int, list[int]] = {}
    order: list[tuple[int, int]] = []
    for write in payload.writes:
        at = 0 if write == "first" else total - 1 if write == "last" else min(total - 1, round(write * FPS))
        piece = max(index for index, start in enumerate(starts) if start <= at)
        order.append((piece, len(exports.setdefault(piece, []))))
        exports[piece].append(at - starts[piece])
    style, sound, music = payload.style, payload.soundscape, payload.music
    # A continued piece's place is the previous piece's last frame, with the setup plate beside it as an
    # attribute_transfer world state: the plate as the place pulls a shot that has moved from its framing back
    # with a cut (run 20996), and the last frame alone lets the place drift (run 21955).
    places = [ref.name for ref in references if ref.kind == "background"]
    plates = [StoryReference(name=f"{ref.name} plate", kind="world_state", image=ref.image,
                             description=f"It is <{ref.name}>'s setup plate: its architecture and materials, not its framing.")
              for ref in references if ref.kind == "background"]  # fmt: skip
    forced = frozenset(ref.name for ref in [*references, *plates])
    files: dict[str, File | None] = {ref.name: getattr(ref, FILE[ref.kind]) for ref in voices([*references, *plates])[0]}
    calls: list[PieceCall] = []
    for index, text in enumerate(piece_texts(payload.prompt, payload.beats, [start / FPS for start in starts])):
        own = [ref for ref in references if ref.kind not in CLIPS or ref.name in windows[index]]
        # A continued piece a clip guides, with no place to carry it, starts from the previous piece's last frame
        # as a world state: the clip's motion goes on from there, never cutting back to the clip's own moment
        # (run 24603 cut where it had run ahead of its clip).
        anchor = [StoryReference(name=f"Piece {index} end", kind="world_state",
                                 description="This piece starts from it: who and what is where now.")
                  ] if index and not places and any(ref.kind == "video" for ref in own) else []  # fmt: skip
        call = piece_call(text, [*own, *plates, *anchor] if index else own, style=style, soundscape=sound,
                          music=music, index=index, forced=forced | {ref.name for ref in anchor})  # fmt: skip
        if not call.references and index:
            # Ref2VA carries the picture and needs a picture: the previous piece's last frame.
            end = StoryReference(name=f"Piece {index} end", kind="world_state")
            call = piece_call(text, [end], style=style, soundscape=sound, music=music, index=index,
                              forced=frozenset({end.name}))  # fmt: skip
        calls.append(call if call.references else PieceCall(_fl2va_text(text, payload), [], call.warnings))
        warnings += call.warnings
    for edge in starts[1:]:
        late = [beat.at for beat in payload.beats if edge / FPS - _BEFORE_EDGE_S <= beat.at <= edge / FPS + _AFTER_EDGE_S]
        if late:
            warnings.append(f"a piece starts at {edge / FPS:.1f} s, near the beat at {late[0]:g} s, which may start "
                            "about 2 s late; spacing the beats further apart avoids it")  # fmt: skip
    tel.log("h3 shot pieces", pieces=count, seconds=",".join(map(str, pieces)),
            boundaries_s=",".join(f"{start / FPS:.2f}" for start in starts[1:]), delivered_frames=total,
            clip_windows=";".join(",".join(f"{name}@{at}" for name, at in w.items()) for w in windows))  # fmt: skip
    if not calls[0].references and any(call.references for call in calls[1:]):
        # Only a text-only first piece (FL2VA) runs ahead of a Ref2VA one. A hint the shot never
        # uses loads on after it ends and holds the GPU from the next run (0.2.0 gate 41159884913:
        # h3-5s, then upscale waited 3 h 50 min).
        prefetch(ref2va_piece)
    work = [plan.sample_frames for plan in plans]
    seeds = [piece_seed(payload.seed, ctx.request_id, index) for index in range(count)]
    several = count > 1
    context, expected = None, None
    returned: list[PieceOutput] = []
    for index, (call, plan) in enumerate(zip(calls, plans, strict=True)):
        ctx.raise_if_cancelled()
        first = keyframes.first if index == 0 else None
        last = keyframes.last if index + 1 == count else None
        span = (sum(work[:index]) / sum(work), sum(work[: index + 1]) / sum(work))
        try:
            with tel.scope(f"Piece {index + 1} of {count}", overall_range=span):
                log_prompt(tel, "h3 piece prompt", call.prompt, piece=index + 1,
                           references=", ".join(ref.name for ref in call.references))  # fmt: skip
                if call.references:
                    uses = video_uses(call.references, windows[index])
                    result = await ref2va_piece(  # type: ignore[call-arg]
                        payload=Ref2VAInput(
                            prompt=call.prompt, seed=seeds[index], duration_s=pieces[index], steps=steps,
                            turbo=payload.turbo, frames=held_frames(plan, uses), videos=uses,
                            previous_provenance=expected, context=context if plan.carried else None,
                            context_frames=cast(ContextFrames, plan.carried.frames),
                            context_audio=plan.carried.audio,
                            next_context_frames=cast(ContextFrames, plan.successor.frames),
                            next_context_audio=plan.successor.audio, first_frame=first, last_frame=last,
                            exports=tuple(exports.get(index, ())), latents=several and index + 1 < count,
                            earlier_latents=tuple(r.latents for r in returned) if several and index + 1 == count else (),
                            earlier_videos=tuple(r.video for r in returned) if several and index + 1 == count else (),
                            picture_edges=tuple(_WORLD_STATE_EDGE if ref.kind == "world_state" else 0
                                                for ref in call.references if FILE[ref.kind] == "image"),
                        ), assets=_call_assets(files | ({place: returned[-1].last_frame for place in places} if index else {}),
                                               call.references),
                    )  # fmt: skip
                else:
                    result = await fl2va_piece(  # type: ignore[call-arg]
                        payload=FL2VAInput(
                            prompt=call.prompt, seed=seeds[index], duration_s=pieces[index], steps=steps,
                            turbo=payload.turbo, first_frame=first, last_frame=last,
                            next_context_frames=cast(ContextFrames, plan.successor.frames),
                            next_context_audio=plan.successor.audio, exports=tuple(exports.get(index, ())),
                            latents=several,
                        ),
                    )  # fmt: skip
        except ChildCallError as failure:
            # Concurrent caller cancellation must not become a successful partial result.
            ctx.raise_if_cancelled()
            raise OutputError(f"piece {index + 1} of {count} failed ({failure.code}): {str(failure)[:512]}",
                              code="piece_failed") from failure  # fmt: skip
        expected = result.provenance
        context = result.context if plan.successor else None
        returned.append(result)
        warnings += result.warnings
        files[f"Piece {index + 1} end"] = result.last_frame
        out.publish("video", result.video, label="Video" if index + 1 == count else f"Piece {index + 1} of {count}")
    ctx.release_gpus()
    tel.log("h3 shot", pieces=count, delivered_frames=total, turbo=payload.turbo)
    return ShotResult(video=returned[-1].video, frames=[returned[piece].frames[slot] for piece, slot in order],
                      warnings=list(dict.fromkeys(warnings)))  # fmt: skip


# --- upscale: one whole shot (`video/upscale`) ---------------------------------------------


class UpscaleInput(msgspec.Struct, kw_only=True):
    """A shot taken to `resolution` (`video/upscale`, UPSCALE.md): `upscale.model` `seedvr2`
    (default) restores its frames; `regenerate-2k` re-renders it with H3 from `prompt`, the text
    it was made from. Its soundtrack is kept."""

    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, max_decoded_bytes=2 << 30)]
    resolution: RegenerateResolution = Resolution.QHD
    upscale: Upscale | None = None
    prompt: str = ""
    seed: int | None = None


class UpscaleOutput(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, media_types=("video/mp4",))]


async def upscale(ctx: Context, payload: UpscaleInput, out: Outputs, tel: Telemetry) -> UpscaleOutput:
    """One whole shot upscaled; the fields its model ignores are logged."""
    method = payload.upscale or Upscale()
    warnings = method.ignored()
    seed = payload.seed if payload.seed is not None else int.from_bytes(
        hashlib.sha256(f"{ctx.request_id}/h3/upscale".encode()).digest()[:4], "big")  # fmt: skip
    if method.model == "regenerate-2k":
        if not payload.prompt.strip():
            raise InvalidRequest("regenerate-2k re-renders the video from the prompt it was made from",
                                 fields=["prompt"])  # fmt: skip
        call = RegenerateInput(video=payload.video, prompt=payload.prompt, resolution=payload.resolution,
                               strength=method.start, seed=seed)  # fmt: skip
        with tel.scope(f"Regenerate at {payload.resolution.value}"):
            result = await cast(Callable[..., Awaitable[RegenerateOutput]], regenerate_shot)(payload=call)
        tel.log("h3 upscale", model=method.model, warnings="; ".join(warnings + result.warnings))
        return UpscaleOutput(result.video)
    if payload.resolution is Resolution.NATIVE:
        raise InvalidRequest("seedvr2 restores above 768p; 768p is regenerate-2k's refine-only pass",
                             fields=["resolution"])  # fmt: skip
    probe = payload.video.probe()
    restore = cast(Callable[..., Awaitable[RestorationOutput]], restore_video)
    with tel.scope(f"SeedVR2 upscale to {payload.resolution.value}"):
        restored = await restore(
            video=payload.video, resolution=RestorationResolution(payload.resolution.value), seed=seed,
            aspect_ratio="16:9" if (probe.width, probe.height) == (1344, 768) else "source",
            **{k: getattr(method, k) for k in _SEEDVR2_FIELDS if getattr(method, k) is not None},
        )  # fmt: skip
    tel.log("h3 upscale", model=method.model, warnings="; ".join(warnings))
    return UpscaleOutput(restored.video)


app.entrypoint(shot, emits_media=True, accelerator=False)
app.entrypoint(upscale, emits_media=True, accelerator=False)
