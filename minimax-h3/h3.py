"""MiniMax-H3's two official serving actions over one dual-DiT construction.

Runtime reads typed asset handles and encodes outputs. Official Diffusers owns H3 presentation,
conditioning, AdaLN, solver, and decode math. This module validates the product contract,
stages weighted roots, and joins those two boundaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import queue
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import partial
from typing import Annotated, Any, Literal, cast

import msgspec
import numpy as np
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    AssetLimits,
    Assets,
    AudioAsset,
    AudioProbe,
    ChildCallError,
    Context,
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaHeader,
    DecodedVideoFormat,
    DecodedVideoFrame,
    FileAsset,
    Image,
    ImageAsset,
    ImagePreparation,
    ImageResult,
    InvalidRequest,
    OutputError,
    Outputs,
    SavedVideo,
    Shape,
    Telemetry,
    ThreadPoolExecutor,
    UnsupportedInput,
    VideoAsset,
    VideoProbe,
    invocable,
    prefetch,
)
from msgspec.structs import asdict, replace

import describe
from assembly import CutJoin, ScanAhead, assemble_video
from gates import MediaFacts, refuse_before_encode
from long_form_state import (
    RenderProvenance, compatible, compatible_base, context_provenance, provenance,
)
from minimax_h3.continuation import (
    AVContext,
    Carry,
    ContinuationPlan,
    decode_context,
    encode_context,
)
from minimax_h3.guide_rows import GuideClip
from minimax_h3 import regenerate as refine_math
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
from qwen_image_2 import generate as generate_reference
from birefnet import RemoveBackgroundOutput
from birefnet import remove_background
from cozy_runtime.vlm import AskPrompt, AskResult
from qwen3_5 import ask
from seedvr2 import Resolution as RestorationResolution
from seedvr2 import UpscaleOutput as RestorationOutput
from seedvr2 import upscale as restore_video
from story import (
    AUDIO_READ_BYTES,
    FILE,
    IMAGE_READ_BYTES,
    REFERENCE_AUDIO,
    REFERENCE_VIDEO,
    VIDEO_MAX_BYTES,
    VIDEO_READ_BYTES,
    WRITER_TOKENS,
    SegmentCall,
    StoryReference,
    apply_written,
    compile_segments,
    encoder_prompt,
    image_prompt,
    opening_cut_warnings,
    reference_seed,
    resolve_reference_images,
    validate_references,
    writer_request,
)
from text_states import encode_text

app = App()

_MIB = 1 << 20
#: Standalone audio references share one 15-second budget; each is 2-15 s (`REFERENCE_AUDIO`).
_MAX_AUDIO_SECONDS = Fraction(15, 1)

Reference = ImageAsset | VideoAsset | AudioAsset
#: One reference of each kind, with its retained-read bound. Every input carrying references
#: (the render children's and direct ref2va's `assets`, and `StoryReference`) declares the same
#: preparations, so Runtime records the truncation on the call the client made.
_REFERENCE = (
    Annotated[ImageAsset, AssetBound(max_decoded_bytes=IMAGE_READ_BYTES)]
    | Annotated[
        VideoAsset, AssetBound(max_bytes=VIDEO_MAX_BYTES, max_decoded_bytes=VIDEO_READ_BYTES)
    ]
    | Annotated[AudioAsset, AssetBound(max_decoded_bytes=AUDIO_READ_BYTES)]
)
ReferenceAssets = Annotated[
    Assets[_REFERENCE],
    AssetLimits(
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
# Direct ref2va takes either attached assets or named `references`; the wire field is required,
# so a references-only request sends `"assets": []`.
DirectReferenceAssets = Annotated[
    Assets[_REFERENCE],
    AssetLimits(
        images=MAX_IMAGE_REFERENCES,
        videos=MAX_VIDEO_REFERENCES,
        audio=MAX_AUDIO_REFERENCES,
        total=MAX_REFERENCES,
    ),
    ImagePreparation(max_edge=8192, max_pixels=16_777_216),
    REFERENCE_VIDEO,
    REFERENCE_AUDIO,
]


class VideoUse(msgspec.Struct, frozen=True):
    """How a call uses one of its videos: its sound (`use_audio`), and whether the output
    follows it frame by frame (`guide`)."""

    audio: bool = False
    guide: bool = False


def video_uses(references: Sequence[StoryReference]) -> tuple[VideoUse, ...]:
    """Each of a call's videos, in packed order."""
    return tuple(VideoUse(ref.use_audio, ref.guide) for ref in references if ref.kind == "video")


def _call_assets(
    files: dict[str, Reference], references: Sequence[StoryReference]
) -> Assets[Reference]:
    """One H3 call's references in packed order, labeled `Picture|Audio|Video N` per kind."""
    counts = {"image": 0, "audio": 0, "video": 0}
    labeled: list[Reference] = []
    for reference in references:
        file = files[reference.name]
        counts[file.kind] += 1
        label = {"image": "Picture", "audio": "Audio", "video": "Video"}[file.kind]
        labeled.append(file.with_label(f"{label} {counts[file.kind]}"))
    return Assets(labeled)


def truncation_warnings(videos: Iterable[tuple[str, VideoAsset]]) -> list[str]:
    """A line for each reference video Runtime cut to the first 15 s `REFERENCE_VIDEO` keeps."""
    lines = []
    for name, video in videos:
        probe = video.probe()
        if probe.source_duration > probe.duration:
            lines.append(
                f"Reference video {name} is {float(probe.source_duration):.1f} s long; "
                f"H3 used its first {float(probe.duration):g} s."
            )
    return lines


def named_videos(
    references: Iterable[StoryReference], calls: Iterable[SegmentCall]
) -> list[tuple[str, VideoAsset]]:
    """The video references some call attaches, by name."""
    used = {ref.name for call in calls for ref in call.references}
    return [
        (f"<{ref.name}>", ref.video)
        for ref in references
        if ref.video is not None and ref.name in used
    ]


KeyframeAssets = Annotated[
    Assets[Annotated[ImageAsset, AssetBound(max_decoded_bytes=256 << 20)]],
    AssetLimits(images=2),
]
DEFAULT_REFERENCE_IMAGE_SHORT_EDGE = 1024
_SHORT_EDGE_LADDER = (2048, 1536, 1024, 768, 512, 256)
_REFERENCE_FIDELITY_EDGES = {"low": 256, "medium": 1024, "high": REFERENCE_IMAGE_SHORT_EDGE}
# The model's timestep plans own the step counts: a request names one of them or none (the
# fewest), and a turbo function always runs its LoRA's one schedule (`TimestepPlan.schedule`).
Steps = Annotated[
    int,
    msgspec.Meta(
        ge=1,
        description=(
            "Denoise steps (transformer evaluations): a count the model's timestep plan "
            "serves; absent, its fewest. Fewer is faster."
        ),
    ),
]
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
#: A segment's new seconds. Its memory is sized by its `frames`, not by these alone.
SegmentSeconds = Annotated[int, msgspec.Meta(ge=MIN_DURATION_S, le=MAX_DURATION_S)]


class ClipInput(msgspec.Struct, kw_only=True):
    """One H3 generation: a turbo call's request and every long_form segment.

    `prompt` is any text, sent verbatim, after each reference's `<Picture N>`, `<Video N>`
    or `<Audio N>` label (numbered per kind in input order; a video with sound also takes the
    next `<Audio N>`). Text that never names non_diegetic_music gains an `N/A` music line.
    """

    prompt: str
    seed: int | None = None
    duration_s: DurationSeconds


class StandardClipInput(ClipInput, kw_only=True):
    """`ClipInput` with the standard calls' denoise steps."""

    steps: Steps | None = None


class ReferenceClipInput(ClipInput, kw_only=True):
    """A reference render call: how it uses each of its videos, in packed order."""

    duration_s: SegmentSeconds
    #: `held_frames` of this call: the shape Runtime banks its memory by.
    frames: int
    videos: tuple[VideoUse, ...]


class ReferenceStandardClipInput(ReferenceClipInput, kw_only=True):
    steps: Steps | None = None


class Resolution(Enum):
    """The output size: native 768p, or above it through `upscale.method`."""

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

    `method`:
    - `seedvr2` (default): SeedVR2 7B restores the decoded 768p original. The 7:4 frame is
      letterboxed into 16:9. The other fields are its ComfyUI controls (seedvr2 README, Tuning);
      unset ones keep SeedVR2's defaults.
    - `regenerate` (NVIDIA's Open-H3-Regenerate-2K): the render's own 768p latent is upscaled in
      latent space and re-rendered at `resolution` by H3 in three steps (two at 0.8 and 0.7). It fills
      16:9: the 7:4 frame is stretched 1.6% horizontally, as NVIDIA trained it, where SeedVR2
      letterboxes. At `768p` it is a refine-only pass at the native size.

    `strength` (regenerate): how much of the source the refine redraws, as its start noise:
    0.909 (default, NVIDIA's), 0.8 or 0.7. Lower keeps more of the source (faces, on-screen text)
    and is faster; higher invents more detail. Small text is redrawn at every strength and can
    become other letters; SeedVR2 keeps it.
    """

    method: Literal["seedvr2", "regenerate"] = "seedvr2"
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
        if self.method == "regenerate":
            unused = [name for name in _SEEDVR2_FIELDS if getattr(self, name) is not None]
            return [f"upscale.{name} applies to seedvr2 only" for name in unused]
        return ["upscale.strength applies to regenerate only"] if self.strength is not None else []

    @property
    def start(self) -> float:
        return REGENERATE_STRENGTHS[-1] if self.strength is None else self.strength


class RenderInput(ClipInput, kw_only=True):
    """`upscale` picks how `resolution` is reached and passes its controls (UPSCALE.md)."""

    resolution: Resolution = Resolution.NATIVE
    upscale: Upscale | None = None


class StandardRenderInput(RenderInput, kw_only=True):
    steps: Steps | None = None


class ReferenceRenderInput(RenderInput, kw_only=True):
    """Direct ref2va request using the same named references as a ``long_form`` segment."""

    references: list[StoryReference] | None = None
    style: str = ""
    clean_references: bool = True
    describe_references: bool = True


class ReferenceStandardRenderInput(ReferenceRenderInput, kw_only=True):
    steps: Steps | None = None


class H3VideoOutput(msgspec.Struct):
    """The catalog wire shape. Checkpoint, plan, geometry, and digest facts remain
    attempt observations (se-012): they ride Telemetry, never the customer result."""

    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, media_types=("video/mp4",))]
    warnings: list[str]


class RenderOutput(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",), max_bytes=1 << 30)]
    originals: Annotated[list[VideoAsset], AssetBound(media_types=("video/mp4",), max_bytes=1 << 30)]
    warnings: list[str]


class ReferenceRenderOutput(RenderOutput):
    """Direct ref2va also returns the references it prepared or generated, each published as it lands."""

    references: list[ImageAsset] = msgspec.field(default_factory=list)


def preflight_reference_media(
    payload: StandardClipInput, assets: ReferenceAssets
) -> ReferencePolicyFacts:
    """Refuse cross-field count errors before Runtime hydrates a single asset."""
    del payload
    return _reference_policy(assets)


def preflight_reference_media_turbo(
    payload: ClipInput, assets: ReferenceAssets
) -> ReferencePolicyFacts:
    del payload
    return _reference_policy(assets)


def _reference_policy(assets: ReferenceAssets) -> ReferencePolicyFacts:
    kinds = [assets.info(index).kind for index in range(len(assets))]
    try:
        return validate_reference_policy(kinds)
    except ValueError as exc:
        raise UnsupportedInput(str(exc), code="reference_policy", fields=["assets"]) from exc


def _keyframe_roles(assets: KeyframeAssets) -> tuple[int | None, int | None]:
    """Explicit first/last labels choose roles; other images fill remaining roles in order."""
    named = {
        assets.info(index).label: index
        for index in range(len(assets))
        if assets.info(index).label in {"first", "last"}
    }
    remaining = iter(
        index for index in range(len(assets)) if assets.info(index).label not in {"first", "last"}
    )
    first = named["first"] if "first" in named else next(remaining, None)
    last = named["last"] if "last" in named else next(remaining, None)
    if next(remaining, None) is not None:
        raise InvalidRequest("FL2VA accepts at most two images", fields=["assets"])
    return first, last


def _keyframe_image(
    assets: KeyframeAssets,
    index: int | None,
    *,
    field: str,
) -> Image | None:
    if index is None:
        return None
    image = assets[index].read()
    _validate_ratio(image.width, image.height, field)
    return image


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
    references: Assets[Reference],
    *,
    pipe: OfficialH3Pipeline,
    frames: int,
    videos: Sequence[VideoUse] = (),
) -> tuple[list[Any], ReferenceSizing, GuideClip | None]:
    """Check every reference from its header and size the vision budget, then read each.

    Durations are Runtime's (`REFERENCE_VIDEO`/`REFERENCE_AUDIO`): `probe().duration` is the
    length a read returns, so a truncated clip is budgeted at 15 s. A video streams straight to
    the size it is used at on the 24-fps clock (`read_reference_video`), keeping only the
    call's `frames`, which is all the official pipeline keeps of it; its soundtrack is decoded
    only where its `videos` entry (per video in packed order) says so. A guide video is not a
    reference (`guide_rows`): Qwen never sees it.
    """
    audio_duration = Fraction(0)
    video_tokens = 0
    images: list[ImageSizing] = []
    use = iter(videos)
    uses = [
        next(use, VideoUse()) if isinstance(reference, VideoAsset) else VideoUse()
        for reference in references
    ]
    keeps = [item.audio for item in uses]
    if any(item.guide for item in uses) and all(
        item.guide or isinstance(reference, AudioAsset)
        for item, reference in zip(uses, references, strict=True)
    ):
        raise InvalidRequest(
            "a guide video needs a picture or another video beside it in the call",
            code="reference_policy",
            fields=["references"],
        )
    for index, reference in enumerate(references):
        info = references.info(index)
        field = info.id
        if isinstance(reference, ImageAsset):
            image = reference.probe()
            _validate_ratio(image.width, image.height, field)
            requested = (
                None if info.fidelity == "auto" else _REFERENCE_FIDELITY_EDGES[info.fidelity]
            )
            images.append(ImageSizing(field, image.width, image.height, requested))
        elif isinstance(reference, VideoAsset):
            video = reference.probe()
            keeps[index] = keeps[index] and video.soundtrack is not None
            _validate_video(video, field, sound=keeps[index])
            if not uses[index].guide:
                video_tokens += reference_video_vision_tokens(
                    video.width, video.height, video.duration
                )
        else:
            audio = reference.probe()
            _validate_audio(audio, field)
            audio_duration += audio.duration
            _validate_audio_aggregate(audio_duration)
    sizing = resolve_reference_sizing(
        images, default=DEFAULT_REFERENCE_IMAGE_SHORT_EDGE, video_tokens=video_tokens
    )
    window = Fraction(frames, FPS)
    prepared = []
    guide = None
    for index, reference in enumerate(references):
        if isinstance(reference, ImageAsset):
            prepared.append(pipe.image_reference(reference.read()))
        elif isinstance(reference, VideoAsset):
            with reference.stream(end=window, audio=keeps[index]) as events:
                clip = read_reference_video(
                    events, frames=frames, audio=keeps[index], guide=uses[index].guide
                )
            if uses[index].guide:
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
) -> H3VideoOutput:

    cancel()
    if completed is not None:
        # The successor's carry: a slice of the denoised latents, before either decode.
        completed(state)
    frames = frames_for(duration_s) if delivered_frames is None else delivered_frames
    waveform, sample_rate = _soundtrack(model, task, state, frames=frames, tel=tel, cancel=cancel)

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
            model.decode_video(task, state, on_chunk=stream.push)
        except BaseException:
            stream.abandon()
            raise
    try:
        saved = stream.finish(lambda share: tel.progress(share, stage="encode_video"))
    except BaseException:
        stream.abandon()
        raise
    _, height, width, _ = (int(value) for value in stream.pixels.shape)

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


@dataclass(frozen=True)
class RegenerateTarget:
    """What a render call hands its finished latent to: the Regenerate refine at `resolution`."""

    overlay: H3RegenerateLoRA
    upscaler: H3LatentUpscaler
    resolution: str
    strength: float
    seed: int
    #: The request's prompt, re-encoded as text alone when the render's conditioning is not.
    prompt: str


def _finish_regenerated(
    model: H3Model,
    task: Task,
    state: Any,
    schedule: ScheduleFacts,
    *,
    target: RegenerateTarget,
    duration_s: int,
    out: Outputs,
    tel: Telemetry,
    cancel: Any,
) -> H3VideoOutput:
    """`_finish` for a render that regenerates: its own denoised latent goes straight to the
    upscaler and the refine, and is never decoded at 768p. A text-only FL2VA render's
    conditioning is the refine's; any other is re-encoded as text alone."""
    cancel()
    frames = frames_for(duration_s)
    waveform, sample_rate = _soundtrack(model, task, state, frames=frames, tel=tel, cancel=cancel)
    tags = state.text_token_tags
    text = (state.prompt_embeds, tags) if bool((tags == 1).all()) else None
    tel.log("h3 render schedule", timestep_plan_digest=schedule.timestep_plan_digest,
            transformer_evaluations=schedule.transformer_evaluations)  # fmt: skip
    assert isinstance(model, H3TurboBase)
    return _regenerated(
        model, target, latents=state.latents, prompt=encoder_prompt("fl2va", target.prompt), text=text,
        waveform=waveform, sample_rate=sample_rate, frames=frames, span=(0.90, 0.99),
        out=out, tel=tel, cancel=cancel,
    )  # fmt: skip


def _regenerated(
    model: H3TurboBase,
    target: RegenerateTarget,
    *,
    latents: Any,
    prompt: str,
    text: tuple[Any, Any] | None,
    waveform: Any | None,
    sample_rate: int,
    frames: int,
    span: tuple[float, float],
    out: Outputs,
    tel: Telemetry,
    cancel: Any,
) -> H3VideoOutput:
    """Normalized 768p latents to the delivered `frames` at `target.resolution`: upscale, the
    windowed refine and the official decode at the target canvas, streamed with `waveform`."""
    low, high = span

    def at(share: float) -> float:
        return low + (high - low) * share

    with tel.stage("upscale_latent", overall_range=(at(0.0), at(0.03))):
        for height, width in refine_math.upscale_passes(target.resolution):
            latents = target.upscaler.upscale(latents, height=height, width=width)
    cancel()
    if text is None:
        with tel.stage("condition_text", overall_range=(at(0.03), at(0.08))):
            text = model.encode_prompt(prompt, frames_for(MIN_DURATION_S))
    spans = refine_math.plan_windows(tuple(int(value) for value in latents.shape))
    schedule = target.overlay.pipe.plan.refined(target.strength)
    steps = schedule.transformer_evaluations
    tel.log(
        "h3 regenerate",
        resolution=target.resolution,
        strength=target.strength,
        start_sigma=schedule.start,
        transformer_evaluations=steps,
        latent="x".join(str(int(value)) for value in latents.shape[2:]),
        windows=len(spans),
        window_latent_frames=spans[0][1],
    )
    audio = model.silence(refine_math.frames_of(spans[0][1]))
    with tel.stage("refine", overall_range=(at(0.08), at(0.85))):
        refined = model.refine(
            latents,
            overlay=target.overlay,
            prompt_embeds=text[0],
            text_tags=text[1],
            spans=spans,
            audio=audio,
            schedule=schedule,
            seed=target.seed,
            on_step=tel.step_callback(steps, stage="refine", overall_range=(at(0.08), at(0.85))),
        )
    cancel()
    height, _, keep = refine_math.TARGETS[target.resolution]
    top = (height - keep) // 2
    with tel.stage("decode_video", overall_range=(at(0.85), at(0.97))):
        stream = _VideoStream(
            torch, out, frames=frames, waveform=waveform, sample_rate=sample_rate, cancel=cancel,
            progress=tel.step_callback(frames, stage="decode_video", overall_range=(at(0.85), at(0.97))),
        )  # fmt: skip
        delivered = 0

        def push(chunk: Any) -> None:
            nonlocal delivered
            take = min(int(chunk.shape[0]), frames - delivered)
            if take > 0:
                stream.push(chunk[:take, :, top : top + keep])
                delivered += take

        try:
            model.decode_latents(refined, on_chunk=push)
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
    return H3VideoOutput(video=saved.video, warnings=[])


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


def _rgb8(torch: Any, decoded: Any) -> tuple[Any, str]:
    """A whole `(1, T, 3, H, W)` float decode to ONE host RGB8 buffer and its digest: the
    slice-for-slice reference the streamed handoff below is held to."""
    source = decoded[0]
    frames, _, height, width = (int(value) for value in source.shape)
    pixels = torch.empty((frames, height, width, 3), dtype=torch.uint8, device="cpu")
    digest = hashlib.sha256()
    for start in range(0, frames, _RGB8_CHUNK_FRAMES):
        landed = pixels[start : start + _RGB8_CHUNK_FRAMES]
        _land(torch, source[start : start + len(landed)], landed)
        digest.update(landed.numpy())
    return pixels, digest.hexdigest()


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
_MODEL = "minimaxai/minimax-h3@1.0.0-rc.6"
_TURBO_LORA = "alibaba-pai/minimax-h3-turbo-lora@1.0.1"
_REGENERATE_LORA = "efficient-large-model/open-h3-regenerate-2k@0.1.0"
_LATENT_UPSCALER = "lbh-123-ai/h3-latent-upscaler@0.1.0"


#: The components each task runs. A machine downloads these first and the rest behind the
#: run, so a pod's first request pays for one DiT, not two.
_FL2VA = ("text_encoder", "transformer", "vae", "audio_vae")
_REF2VA = ("text_encoder", "transformer_ref", "vae", "audio_vae")
#: A Ref2VA render that regenerates also runs the FL2VA DiT, NVIDIA's refine trunk.
_REF2VA_REGENERATE = ("text_encoder", "transformer_ref", "vae", "audio_vae", "transformer")

app.entrypoint(defaults={"model": _MODEL}, components={"model": ("text_encoder",)})(
    encode_text
)


@invocable(
    defaults={"model": _MODEL},
    components={"model": _FL2VA},
    memoize=True,
)
async def generate_fl2va(
    ctx: Context,
    *,
    payload: StandardClipInput,
    assets: KeyframeAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    return _keyframes_to_video(ctx, "fl2va", payload, assets, model, out, tel, steps=payload.steps)


@invocable(
    defaults={"base_model": _MODEL, "turbo_lora": _TURBO_LORA},
    components={"base_model": _FL2VA, "turbo_lora": ("fl2va_turbo",)},
    memoize=True,
)
async def generate_fl2va_turbo(
    ctx: Context,
    *,
    payload: ClipInput,
    assets: KeyframeAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    """`fl2va` under PDD-8: the same keyframes and prompt, eight transformer evaluations."""
    return _keyframes_to_video(
        ctx,
        "fl2va_turbo",
        payload,
        assets,
        base_model,
        out,
        tel,
        steps=None,
        turbo_lora=turbo_lora,
    )


@invocable(
    defaults={"model": _MODEL},
    components={"model": _REF2VA},
    memoize=True,
)
async def generate_ref2va(
    ctx: Context,
    *,
    payload: ReferenceStandardClipInput,
    assets: ReferenceAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    _reference_policy(assets)
    return _references_to_video(
        ctx, "ref2va", payload, assets, model, out, tel, steps=payload.steps
    )


@invocable(
    defaults={"base_model": _MODEL, "turbo_lora": _TURBO_LORA},
    components={"base_model": _REF2VA, "turbo_lora": ("ref2va_turbo",)},
    memoize=True,
)
async def generate_ref2va_turbo(
    ctx: Context,
    *,
    payload: ReferenceClipInput,
    assets: ReferenceAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    """`ref2va` under PDD-8: the same references and prompt, eight transformer evaluations."""
    _reference_policy(assets)
    return _references_to_video(
        ctx,
        "ref2va_turbo",
        payload,
        assets,
        base_model,
        out,
        tel,
        steps=None,
        turbo_lora=turbo_lora,
    )



app.entrypoint(internal=True)(generate_fl2va)
app.entrypoint(internal=True)(generate_fl2va_turbo)
app.entrypoint(internal=True)(generate_ref2va)
app.entrypoint(internal=True)(generate_ref2va_turbo)


#: The output canvas the Regenerate route computes (`regenerate.TARGETS`): the demand axis beside
#: the frames.
_REGENERATE_PIXELS = {
    Resolution.NATIVE: (1344, 768),
    Resolution.FULL_HD: (1920, 1088),
    Resolution.QHD: (2560, 1440),
    Resolution.UHD: (3840, 2176),
}
RegenerateResolution = Annotated[Resolution, Shape(pixels=_REGENERATE_PIXELS)]


class RegenerateKeyframesInput(StandardClipInput, kw_only=True):
    """An FL2VA render whose finished latent the Regenerate route takes to `resolution`."""

    task: Literal["fl2va", "fl2va_turbo"]
    resolution: RegenerateResolution
    strength: float


class RegenerateReferencesInput(ReferenceStandardClipInput, kw_only=True):
    """A Ref2VA render whose finished latent the Regenerate route takes to `resolution`."""

    task: Literal["ref2va", "ref2va_turbo"]
    resolution: RegenerateResolution
    strength: float


def _target(payload: RegenerateKeyframesInput | RegenerateReferencesInput, overlay: H3RegenerateLoRA,
            upscaler: H3LatentUpscaler) -> RegenerateTarget:  # fmt: skip
    assert payload.seed is not None
    return RegenerateTarget(overlay, upscaler, payload.resolution.value, payload.strength,
                            payload.seed, payload.prompt)  # fmt: skip


@invocable(
    defaults={
        "base_model": _MODEL,
        "turbo_lora": _TURBO_LORA,
        "regenerate_lora": _REGENERATE_LORA,
        "latent_upscaler": _LATENT_UPSCALER,
    },
    components={
        "base_model": _FL2VA,
        "turbo_lora": ("fl2va_turbo",),
        "regenerate_lora": ("fl2va_regenerate",),
        "latent_upscaler": ("upscaler",),
    },
    memoize=True,
)
async def regenerate_fl2va(
    ctx: Context,
    *,
    payload: RegenerateKeyframesInput,
    assets: KeyframeAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    regenerate_lora: H3RegenerateLoRA,
    latent_upscaler: H3LatentUpscaler,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    """`fl2va` or `fl2va_turbo` whose finished 768p latent goes straight to the Regenerate refine,
    never decoded at 768p."""
    turbo = payload.task == "fl2va_turbo"
    return _keyframes_to_video(
        ctx, payload.task, payload, assets, base_model, out, tel,
        steps=None if turbo else payload.steps, turbo_lora=turbo_lora if turbo else None,
        regenerate=_target(payload, regenerate_lora, latent_upscaler),
    )  # fmt: skip


@invocable(
    defaults={
        "base_model": _MODEL,
        "turbo_lora": _TURBO_LORA,
        "regenerate_lora": _REGENERATE_LORA,
        "latent_upscaler": _LATENT_UPSCALER,
    },
    components={
        "base_model": _REF2VA_REGENERATE,
        "turbo_lora": ("ref2va_turbo",),
        "regenerate_lora": ("fl2va_regenerate",),
        "latent_upscaler": ("upscaler",),
    },
    memoize=True,
)
async def regenerate_ref2va(
    ctx: Context,
    *,
    payload: RegenerateReferencesInput,
    assets: ReferenceAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    regenerate_lora: H3RegenerateLoRA,
    latent_upscaler: H3LatentUpscaler,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    """`ref2va` or `ref2va_turbo` whose finished 768p latent the Regenerate refine takes on the
    FL2VA DiT (NVIDIA's refine trunk), conditioned on the prompt as text alone."""
    _reference_policy(assets)
    turbo = payload.task == "ref2va_turbo"
    return _references_to_video(
        ctx, payload.task, payload, assets, base_model, out, tel,
        steps=None if turbo else payload.steps, turbo_lora=turbo_lora if turbo else None,
        regenerate=_target(payload, regenerate_lora, latent_upscaler),
    )  # fmt: skip


class RegenerateInput(msgspec.Struct, kw_only=True):
    """Re-render an existing H3 video at `resolution` with NVIDIA's Open-H3-Regenerate-2K: its
    768p frames are encoded, upscaled in latent space and refined by H3 at the target, and its
    soundtrack is kept. `prompt` is the one the video was made from; `strength` as in `upscale`
    (UPSCALE.md). The 7:4 frame fills 16:9, stretched 1.6% horizontally; `768p` refines only."""

    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, max_decoded_bytes=2 << 30)]
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
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, media_types=("video/mp4",))]
    warnings: list[str]


def regenerate(
    ctx: Context,
    payload: RegenerateInput,
    base_model: H3TurboBase,
    regenerate_lora: H3RegenerateLoRA,
    latent_upscaler: H3LatentUpscaler,
    out: Outputs,
    tel: Telemetry,
) -> RegenerateOutput:
    """The Regenerate route on a video: the same refine a render's `upscale.method: regenerate`
    runs, from the latent this video encodes to."""
    ctx.plan("vae", "text_encoder", "audio_vae", "upscaler", {"transformer", "fl2va_regenerate"}, "vae")
    with tel.stage("read_video", overall_range=(0.00, 0.03)):
        pixels, waveform, sample_rate, frames, warnings = _read_draft(payload.video)
    with tel.stage("encode_video", overall_range=(0.03, 0.08)):
        latents = base_model.encode_clip(pixels)
    del pixels
    seed = payload.seed if payload.seed is not None else int.from_bytes(
        hashlib.sha256(f"{ctx.request_id}/h3/regenerate".encode()).digest()[:4], "big")  # fmt: skip
    target = RegenerateTarget(regenerate_lora, latent_upscaler, payload.resolution.value,
                              payload.strength, seed, payload.prompt)  # fmt: skip
    result = _regenerated(
        base_model, target, latents=latents, prompt=encoder_prompt("fl2va", payload.prompt),
        text=None, waveform=waveform, sample_rate=sample_rate, frames=frames, span=(0.08, 0.99),
        out=out, tel=tel, cancel=ctx.raise_if_cancelled,
    )  # fmt: skip
    ctx.release_gpus()
    return RegenerateOutput(result.video, warnings)


#: The canvas the refine's latent is encoded at: H3's native 1344x768.
_DRAFT_SIZE = (768, 1344)


def _read_draft(video: VideoAsset) -> tuple[Any, Any | None, int, int, list[str]]:
    """A video's frames as `(frames on the VAE grid, 768, 1344, 3)` uint8 (cover-scaled and
    centre-cropped, its last frame held up to the grid), its `(channels, samples)` soundtrack or
    None, the sample rate, and how many frames it delivers."""
    frames: list[bytes] = []
    pcm: list[list[Any]] = []
    width = height = rate = 0
    warnings: list[str] = []
    with video.stream() as stream:
        for event in stream:
            if isinstance(event, DecodedMediaHeader):
                if event.video is None:
                    raise InvalidRequest("regenerate needs a video stream", fields=["video"])
                width, height = event.video.width, event.video.height
                fps = event.video.nominal_frame_rate
                if fps is None or fps != FPS:
                    raise InvalidRequest(f"regenerate takes 24 fps video, not {fps}", fields=["video"])
            elif isinstance(event, DecodedVideoFrame):
                frames.append(event.rgb)
            else:
                rate = event.sample_rate
                channels = [np.frombuffer(data, dtype=np.float32) for data in event.pcm_f32le]
                pcm.append(channels)
    count = len(frames)
    if not MIN_FRAMES <= count <= MAX_FRAMES:
        raise InvalidRequest(
            f"regenerate takes {MIN_FRAMES} to {MAX_FRAMES} frames (5 to 15 s at 24 fps), not {count}",
            fields=["video"],
        )
    grid = count + (-(count - 5)) % 17
    pixels = torch.empty((grid, *_DRAFT_SIZE, 3), dtype=torch.uint8)
    for index in range(grid):
        frame = torch.frombuffer(bytearray(frames[min(index, count - 1)]), dtype=torch.uint8)
        pixels[index] = _cover(frame.reshape(height, width, 3))
    if (height, width) != _DRAFT_SIZE:
        warnings.append(f"the {width}x{height} video was cover-scaled to 1344x768 for the refine")
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


app.entrypoint(internal=True)(regenerate_fl2va)
app.entrypoint(internal=True)(regenerate_ref2va)
app.entrypoint(
    defaults={
        "base_model": _MODEL,
        "regenerate_lora": _REGENERATE_LORA,
        "latent_upscaler": _LATENT_UPSCALER,
    },
    components={
        "base_model": _FL2VA,
        "regenerate_lora": ("fl2va_regenerate",),
        "latent_upscaler": ("upscaler",),
    },
    memoize=True,
)(regenerate)


async def _render_workflow(
    ctx: Context, payload: RenderInput, assets: KeyframeAssets | ReferenceAssets,
    out: Outputs, tel: Telemetry, *, task: Task, steps: int | None,
    videos: tuple[VideoUse, ...] = (),
) -> RenderOutput:
    seed = shot_seed(payload, ctx.request_id, 0)
    turbo = task.endswith("_turbo")
    fields: dict[str, Any] = {"prompt": payload.prompt, "seed": seed, "duration_s": payload.duration_s}
    if not turbo:
        fields["steps"] = steps
    request = (
        (ReferenceClipInput if turbo else ReferenceStandardClipInput)(
            **fields,
            frames=held_frames(ContinuationPlan(payload.duration_s), videos),
            videos=videos,
        )
        if task.startswith("ref2va")
        else (ClipInput if turbo else StandardClipInput)(**fields)
    )
    upscale = payload.upscale or Upscale()
    ignored = upscale.ignored()
    if upscale.method == "regenerate":
        # The render's own latent is the refine's input: one call, no 768p decode or re-encode.
        child = regenerate_ref2va if task.startswith("ref2va") else regenerate_fl2va
        call = (RegenerateReferencesInput if task.startswith("ref2va") else RegenerateKeyframesInput)(
            **asdict(request), task=task, resolution=payload.resolution, strength=upscale.start,
        )  # fmt: skip
        with tel.scope(f"Regenerate at {payload.resolution.value}"):
            result = await cast(Callable[..., Awaitable[H3VideoOutput]], child)(payload=call, assets=assets)
        ctx.release_gpus()
        return RenderOutput(result.video, [], result.warnings + ignored)
    functions = {"fl2va": generate_fl2va, "fl2va_turbo": generate_fl2va_turbo,
                 "ref2va": generate_ref2va, "ref2va_turbo": generate_ref2va_turbo}
    generate = cast(Callable[..., Awaitable[H3VideoOutput]], functions[task])
    with tel.scope("Generating original 768p video"):
        source = await generate(payload=request, assets=assets)
    if payload.resolution is Resolution.NATIVE:
        ctx.release_gpus()
        ignored = ["upscale settings apply only above 768p"] if payload.upscale else []
        return RenderOutput(source.video, [], source.warnings + ignored)
    out.publish("originals", source.video, label="Original 768p video")
    restore = cast(Callable[..., Awaitable[RestorationOutput]], restore_video)
    with tel.scope(f"SeedVR2 upscale to {payload.resolution.value}"):
        result = await restore(
            video=source.video, resolution=RestorationResolution(payload.resolution.value),
            seed=seed, aspect_ratio="16:9",
            **{k: getattr(upscale, k) for k in _SEEDVR2_FIELDS if getattr(upscale, k) is not None},
        )
    ctx.release_gpus()
    return RenderOutput(result.video, [source.video], source.warnings + ignored)


async def fl2va(ctx: Context, payload: StandardRenderInput, assets: KeyframeAssets,
                out: Outputs, tel: Telemetry) -> RenderOutput:
    return await _render_workflow(ctx, payload, assets, out, tel, task="fl2va", steps=payload.steps)


async def fl2va_turbo(ctx: Context, payload: RenderInput, assets: KeyframeAssets,
                      out: Outputs, tel: Telemetry) -> RenderOutput:
    return await _render_workflow(ctx, payload, assets, out, tel, task="fl2va_turbo", steps=None)


async def _direct_reference_assets(
    ctx: Context,
    payload: ReferenceRenderInput,
    assets: DirectReferenceAssets,
    out: Outputs,
    tel: Telemetry,
) -> tuple[ReferenceRenderInput, ReferenceAssets, tuple[VideoUse, ...], list[ImageAsset], list[str]]:
    """Resolve named references once for a direct call, and how it uses each video.

    ``assets`` is the prepared handoff used by long-form children.  Named references are
    prepared here, then compiled exactly like one long-form segment.
    """
    if payload.references is None:
        if not assets:
            raise InvalidRequest("ref2va requires references or assets", fields=["references"])
        return payload, assets, (), [], []
    if assets:
        raise InvalidRequest("send references or assets, not both", fields=["references", "assets"])
    if not payload.references:
        raise InvalidRequest("references must contain at least one item", fields=["references"])
    validate_references(payload.references)
    shown: list[ImageAsset] = []
    images, references, warnings, clips = await _create_references(
        ctx,
        payload.references,
        tel,
        out,
        shown,
        style=payload.style,
        prepare_characters=payload.clean_references,
        describe_characters=payload.describe_references,
    )
    prompts, written = await _write_prompts([payload.prompt], references, clips, payload.style, "", tel)
    warnings += written
    call = compile_segments(prompts, references, style=payload.style)[0]
    for warning in call.warnings:
        tel.log("h3 reference prompt", level="warning", warning=warning)
    labeled = _call_assets(images, call.references)
    return (replace(payload, prompt=call.prompt), labeled, video_uses(call.references), shown,
            warnings + call.warnings)


def _direct_truncations(payload: ReferenceRenderInput, resolved: Assets[Reference]) -> list[str]:
    """Truncation lines for the videos a direct call attaches: by label (or input id), or by
    reference name."""
    attached = [(index, item) for index, item in enumerate(resolved) if isinstance(item, VideoAsset)]
    if payload.references is None:
        return truncation_warnings(
            (resolved.info(index).label or resolved.info(index).id, item) for index, item in attached
        )
    sent = {item.ref for _, item in attached}
    return truncation_warnings(
        (f"<{ref.name}>", ref.video)
        for ref in payload.references
        if ref.video is not None and ref.video.ref in sent
    )


async def ref2va(ctx: Context, payload: ReferenceStandardRenderInput, out: Outputs, tel: Telemetry,
                 assets: DirectReferenceAssets) -> ReferenceRenderOutput:
    request, resolved, videos, shown, warnings = await _direct_reference_assets(ctx, payload, assets, out, tel)
    truncated = _direct_truncations(payload, resolved)
    result = await _render_workflow(ctx, request, resolved, out, tel, task="ref2va", steps=payload.steps,
                                    videos=videos)
    return ReferenceRenderOutput(
        result.video, result.originals, truncated + warnings + result.warnings, shown
    )


async def ref2va_turbo(ctx: Context, payload: ReferenceRenderInput, out: Outputs, tel: Telemetry,
                       assets: DirectReferenceAssets) -> ReferenceRenderOutput:
    request, resolved, videos, shown, warnings = await _direct_reference_assets(ctx, payload, assets, out, tel)
    truncated = _direct_truncations(payload, resolved)
    result = await _render_workflow(ctx, request, resolved, out, tel, task="ref2va_turbo", steps=None,
                                    videos=videos)
    return ReferenceRenderOutput(
        result.video, result.originals, truncated + warnings + result.warnings, shown
    )


app.entrypoint(fl2va, emits_media=True, accelerator=False)
app.entrypoint(fl2va_turbo, emits_media=True, accelerator=False)
app.entrypoint(ref2va, emits_media=True, accelerator=False)
app.entrypoint(ref2va_turbo, emits_media=True, accelerator=False)



def _keyframes_to_video(
    ctx: Context,
    task: Task,
    payload: ClipInput,
    assets: KeyframeAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    steps: int | None,
    turbo_lora: H3TurboLoRA | None = None,
    regenerate: RegenerateTarget | None = None,
) -> H3VideoOutput:
    ctx.raise_if_cancelled()
    first_index, last_index = _keyframe_roles(assets)
    return _render_keyframes(
        ctx,
        task,
        model,
        out,
        tel,
        prompt=encoder_prompt(task, payload.prompt),
        seed=payload.seed,
        duration_s=payload.duration_s,
        steps=steps,
        first=_keyframe_image(assets, first_index, field="first"),
        last=_keyframe_image(assets, last_index, field="last"),
        turbo_lora=turbo_lora,
        regenerate=regenerate,
    )


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
    regenerate: RegenerateTarget | None = None,
) -> H3VideoOutput:
    """Shared FL2VA rendering for direct requests and chained asset handoffs."""
    plan = _plan(model, turbo_lora, "fl2va")
    steps = plan.schedule(steps).transformer_evaluations
    keyframes = ("vae",) if first is not None or last is not None else ()
    dit = "transformer" if turbo_lora is None else {"transformer", "fl2va_turbo"}
    ctx.plan("text_encoder", *keyframes, dit, *_decode_plan(regenerate))
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=seed)
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
    if regenerate is not None:
        return _finish_regenerated(
            model, task, state, schedule, target=regenerate, duration_s=duration_s, out=out,
            tel=tel, cancel=ctx.raise_if_cancelled,
        )  # fmt: skip
    return _finish(
        model,
        task,
        state,
        schedule,
        duration_s=duration_s,
        out=out,
        tel=tel,
        cancel=ctx.raise_if_cancelled,
    )


def _decode_plan(regenerate: RegenerateTarget | None) -> tuple[Any, ...]:
    """The components a render runs after its denoise: its decodes, or the Regenerate route's."""
    if regenerate is None:
        return ("audio_vae", "vae")
    return ("audio_vae", "upscaler", {"transformer", "fl2va_regenerate"}, "vae")


def _references_to_video(
    ctx: Context,
    task: Task,
    payload: ReferenceClipInput,
    assets: ReferenceAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    steps: int | None,
    turbo_lora: H3TurboLoRA | None = None,
    context: AVContext | None = None,
    delivery: ContinuationPlan | None = None,
    expected_context_provenance: str | None = None,
    completed: Callable[[Any], None] | None = None,
    regenerate: RegenerateTarget | None = None,
) -> H3VideoOutput:
    plan = _plan(model, turbo_lora, "ref2va")
    steps = plan.schedule(steps).transformer_evaluations
    dit = "transformer_ref" if turbo_lora is None else {"transformer_ref", "ref2va_turbo"}
    ctx.plan("text_encoder", {"vae", "audio_vae"}, dit, *_decode_plan(regenerate))
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=payload.seed)
    window = ContinuationPlan(payload.duration_s) if delivery is None else delivery
    if payload.frames != held_frames(window, payload.videos):
        raise InvalidRequest(
            f"a {payload.duration_s} s call with these references holds "
            f"{held_frames(window, payload.videos)} frames, not {payload.frames}",
            fields=["payload.frames"],
        )
    frames = window.sample_frames
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        references, sizing, guide = assets_to_h3_refs(
            assets, pipe=model.pipe, frames=frames, videos=payload.videos
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
    if regenerate is not None:
        if context is not None or delivery is not None or completed is not None:
            raise InvalidRequest("a continued segment cannot regenerate", fields=["upscale"])
        return _finish_regenerated(
            model, task, state, schedule, target=regenerate, duration_s=payload.duration_s,
            out=out, tel=tel, cancel=ctx.raise_if_cancelled,
        )  # fmt: skip
    return _finish(
        model,
        task,
        state,
        schedule,
        duration_s=payload.duration_s,
        out=out,
        tel=tel,
        cancel=ctx.raise_if_cancelled,
        completed=completed,
        delivered_frames=None if delivery is None else delivery.delivered_frames,
        sampled_frames=None if delivery is None else delivery.sample_frames,
    )


# --- long-form composition -------------------------------------------------------------
# `long_form` is a CPU composition function. Runtime runs an ordinary serving child for each
# shot and owns its intermediate bytes. A segment is memoized on its full inputs, the previous
# segment's context included, so a resubmitted long_form skips the segments it already finished.
# Only encoded asset handles cross the loop. Assembly decodes one bounded event at a time.

class SegmentOutput(msgspec.Struct):
    """`H3VideoOutput` without a default factory, which an invocable result may not carry."""

    video: Annotated[
        VideoAsset,
        AssetBound(
            max_bytes=256 << 20,
            max_decoded_bytes=32 << 20,
            media_types=("video/mp4",),
        ),
    ]
    warnings: list[str]
    provenance: RenderProvenance


ContextFrames = Literal[0, 22, 39, 56]


class StorySegment(ClipInput, kw_only=True):
    """One story segment with an optional music override."""

    non_diegetic_music: str | None = None


class LongFormSegment(StorySegment, kw_only=True):
    """One segment: whether it continues the previous segment's picture and its sound."""

    context_frames: bool = True
    context_audio: bool = True


class LongFormInput(msgspec.Struct):
    """Fixed references; each segment may carry the previous one's picture and/or sound."""

    segments: Annotated[list[LongFormSegment], msgspec.Meta(min_length=1)]
    references: Annotated[list[StoryReference], msgspec.Meta(min_length=1)]
    style: str = ""
    overall_soundscape: str = ""
    non_diegetic_music: str = "N/A"
    context_frames: ContextFrames = 22
    mode: Literal["turbo", "standard"] = "turbo"
    steps: Steps | None = None
    clean_references: bool = True
    describe_references: bool = True


class LongFormOutput(msgspec.Struct):
    """The joined video and the reference images made for it. Each is published as it lands:
    every generated reference, and the video one segment at a time. A failed or canceled run
    keeps what it published."""

    video: Annotated[VideoAsset, AssetBound(max_bytes=256 << 20, media_types=("video/mp4",))]
    references: list[ImageAsset]
    delivered_frames: int
    fps: int
    warnings: list[str]


def segment_carries(payload: LongFormInput) -> list[Carry]:
    """What each segment takes from its predecessor: the global picture window unless the
    segment turns it off (global 0 carries no picture), and one second of sound."""
    return [
        Carry(
            payload.context_frames if index and shot.context_frames else 0,
            bool(index and shot.context_audio),
        )
        for index, shot in enumerate(payload.segments)
    ]


def shot_seed(shot: ClipInput, request_id: str, index: int) -> int:
    """Choose once per request/shot, without changing seeds when its attempt retries."""
    if shot.seed is not None:
        return shot.seed
    identity = f"{request_id}/h3/shot/{index}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def segment_clock(durations: Sequence[int]) -> tuple[list[int], int]:
    """Start frame of each shot and the delivered total, in exact integer frames.

    Exactly one replayed frame leaves the chain at every seam. This is integer arithmetic
    over frame counts and the caller turns it into seconds as `Fraction(frames, FPS)` —
    never a per-seam `round(sample_rate / fps)`, whose residue accumulated +8.33 ms per clip.
    """
    starts: list[int] = []
    total = 0
    for index, seconds in enumerate(durations):
        starts.append(total)
        total += frames_for(seconds) - (1 if index else 0)
    return starts, total


class CutInput(ReferenceClipInput, kw_only=True):
    steps: int | None
    turbo: bool
    expected_provenance: RenderProvenance | None = None
    previous_provenance: RenderProvenance | None = None


def _sampling(payload: CutInput | MotionInput, turbo_lora: H3TurboLoRA) -> H3TurboLoRA | None:
    """The segment's sampler over the one shared construction: the PDD adapter, or none."""
    if payload.turbo and payload.steps is not None:
        raise InvalidRequest("a turbo segment runs its LoRA's one schedule; it takes no steps")
    return turbo_lora if payload.turbo else None


def _render_cut(
    ctx: Context,
    payload: CutInput,
    assets: ReferenceAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    steps: int | None,
    turbo_lora: H3TurboLoRA | None = None,
) -> SegmentOutput:
    observed = provenance(
        model.checkpoint_ref, "" if turbo_lora is None else turbo_lora.checkpoint_ref
    )
    compatible(observed, payload.expected_provenance)
    compatible_base(observed, payload.previous_provenance)
    _reference_policy(assets)
    task: Task = "ref2va" if turbo_lora is None else "ref2va_turbo"
    shot = _references_to_video(
        ctx,
        task,
        payload,
        assets,
        model,
        out,
        tel,
        steps=steps,
        turbo_lora=turbo_lora,
    )
    tel.log("h3 cut rendering", native_path=task, seed=payload.seed, references=len(assets))
    return SegmentOutput(shot.video, list(shot.warnings), observed)


# Both sampling modes use one turbo-capable construction, reused by every segment.
@invocable(
    defaults={"base_model": _MODEL, "turbo_lora": _TURBO_LORA},
    components={"base_model": _REF2VA, "turbo_lora": ("ref2va_turbo",)},
)
async def cut_segment(
    ctx: Context,
    *,
    payload: CutInput,
    assets: ReferenceAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> SegmentOutput:
    lora = _sampling(payload, turbo_lora)
    return _render_cut(ctx, payload, assets, base_model, out, tel, steps=payload.steps, turbo_lora=lora)


app.entrypoint(internal=True)(cut_segment)


MotionContextAsset = Annotated[
    FileAsset, AssetBound(max_bytes=16 << 20, media_types=("application/octet-stream",))
]


def held_frames(plan: ContinuationPlan, videos: Sequence[VideoUse] = ()) -> int:
    """The frames a call's DiT holds: its sampled window, plus the carried picture again as
    condition rows (the double form; single-copy was tested and rejected, 2026-10-10), plus
    a guide's frames, at most the window's again."""
    guided = any(video.guide for video in videos)
    return plan.sample_frames * (1 + guided) + plan.carried.frames


class MotionInput(msgspec.Struct):
    prompt: str
    seed: int
    duration_s: SegmentSeconds
    steps: int | None
    turbo: bool
    #: `held_frames` of this segment: the shape Runtime banks its memory by. A continuation
    #: holds more than its seconds say, so a 12 s segment after a 6 s one ran on the 6 s
    #: peak and ran out on GPU 1 (run 1560). Any other value is refused.
    frames: int
    #: How the segment uses each of its videos, in packed order.
    videos: tuple[VideoUse, ...]
    #: What this segment carries from its predecessor, and what its successor carries.
    context_frames: ContextFrames = 0
    context_audio: bool = False
    next_context_frames: ContextFrames = 0
    next_context_audio: bool = False
    expected_provenance: RenderProvenance | None = None
    previous_provenance: RenderProvenance | None = None
    context: MotionContextAsset | None = None


class MotionOutput(SegmentOutput):
    # A fixed output slot: Runtime grants result outputs by field path, never inside a union.
    # A shot with no successor returns it empty instead of exporting any window.
    context: MotionContextAsset


def _render_motion(
    ctx: Context,
    payload: MotionInput,
    assets: ReferenceAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    turbo_lora: H3TurboLoRA | None = None,
) -> MotionOutput:
    ctx.raise_if_cancelled()
    observed = provenance(
        model.checkpoint_ref, "" if turbo_lora is None else turbo_lora.checkpoint_ref
    )
    compatible(observed, payload.expected_provenance)
    compatible_base(observed, payload.previous_provenance)
    native_provenance = context_provenance(observed)
    plan = ContinuationPlan(
        payload.duration_s,
        Carry(int(payload.context_frames), payload.context_audio),
        Carry(int(payload.next_context_frames), payload.next_context_audio),
    )
    context: AVContext | None = None
    if plan.carried:
        if payload.context is None:
            raise InvalidRequest("a carried picture or sound needs the previous context",
                                 fields=["payload.context"])
        context = decode_context(payload.context.read_bytes())
    _reference_policy(assets)
    saved_context: list[FileAsset] = []

    def save_context(state: Any) -> None:
        carried = model.pipe.export_context(state, provenance=native_provenance)
        assert carried is not None
        saved_context.append(
            out.save_bytes(encode_context(carried), media_type="application/octet-stream")
        )

    task: Task = "ref2va" if turbo_lora is None else "ref2va_turbo"
    shot = _references_to_video(
        ctx,
        task,
        ReferenceClipInput(prompt=payload.prompt, seed=payload.seed, duration_s=payload.duration_s,
                           frames=payload.frames, videos=payload.videos),
        assets,
        model,
        out,
        tel,
        steps=payload.steps,
        turbo_lora=turbo_lora,
        context=context,
        delivery=plan,
        expected_context_provenance=context_provenance(payload.previous_provenance or observed),
        completed=save_context if plan.successor else None,
    )
    return MotionOutput(
        shot.video,
        list(shot.warnings),
        observed,
        saved_context[0]
        if saved_context
        else out.save_bytes(b"", media_type="application/octet-stream"),
    )


@invocable(
    defaults={"base_model": _MODEL, "turbo_lora": _TURBO_LORA},
    components={"base_model": _REF2VA, "turbo_lora": ("ref2va_turbo",)},
    memoize=True,
)
async def motion_segment(
    ctx: Context,
    *,
    payload: MotionInput,
    assets: ReferenceAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> MotionOutput:
    lora = _sampling(payload, turbo_lora)
    return _render_motion(ctx, payload, assets, base_model, out, tel, turbo_lora=lora)


app.entrypoint(internal=True)(motion_segment)


class LongFormCutsInput(msgspec.Struct):
    segments: Annotated[list[StorySegment], msgspec.Meta(min_length=1)]
    references: Annotated[list[StoryReference], msgspec.Meta(min_length=1)]
    style: str = ""
    overall_soundscape: str = ""
    non_diegetic_music: str = "N/A"
    mode: Literal["turbo", "standard"] = "turbo"
    steps: Steps | None = None
    clean_references: bool = True
    describe_references: bool = True


def segment_sampling(
    payload: LongFormInput | LongFormCutsInput,
) -> tuple[bool, int | None]:
    """Whether segments run turbo, and their step count (None: the plan's own)."""
    turbo = payload.mode == "turbo"
    if turbo and payload.steps is not None:
        raise InvalidRequest(
            "turbo runs its LoRA's one schedule; omit steps or select mode=standard",
            fields=["mode", "steps"],
        )
    return turbo, payload.steps


#: Progress weight of one generated reference image, in segment frames.
_REFERENCE_FRAMES = 5


def log_prompt(tel: Telemetry, event: str, prompt: str, **fields: str | int) -> None:
    """The text's digest and ends: a log value holds 400 characters, not a whole prompt."""
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    tel.log(event, level="info", digest=digest, chars=len(prompt), head=prompt[:400], tail=prompt[-400:], **fields)


def segment_calls(payload: LongFormInput | LongFormCutsInput) -> list[SegmentCall]:
    """Each segment's child call: its text plus the global sections, and its references."""
    return compile_segments(
        [shot.prompt for shot in payload.segments],
        payload.references,
        style=payload.style,
        overall_soundscape=payload.overall_soundscape,
        non_diegetic_music=payload.non_diegetic_music,
        segment_music=[shot.non_diegetic_music for shot in payload.segments],
    )


async def _create_references(
    ctx: Context,
    references: list[StoryReference],
    tel: Telemetry,
    out: Outputs,
    shown: list[ImageAsset],
    *,
    style: str,
    prepare_characters: bool = True,
    describe_characters: bool = True,
) -> tuple[dict[str, Reference], list[StoryReference], list[str], dict[str, str]]:
    """Prepare, generate and describe references concurrently: the images, the described list,
    warnings, and each motion clip's description by name. Describing fills a supplied
    character's or place's missing description and describes every clip (`describe.clip`)."""
    reference_steps = 40
    # Managed dependencies install an asynchronous flat caller for the source
    # entrypoint, whose own annotations describe its injected implementation.
    reference_generator = cast(Callable[..., Awaitable[ImageResult]], generate_reference)
    background_remover = cast(Callable[..., Awaitable[RemoveBackgroundOutput]], remove_background)

    async def prepare_character(reference: StoryReference) -> ImageAsset:
        ctx.raise_if_cancelled()
        assert reference.image is not None
        with tel.scope(f"Removing background from {reference.name}"):
            result = await background_remover(image=reference.image)
        return result.image

    async def create_reference(reference: StoryReference) -> ImageAsset:
        ctx.raise_if_cancelled()
        # Concurrent child progress has no ordered overall range. Runtime owns
        # GPU admission; this CPU composer submits at most nine independent calls.
        # A character sheet's three panels share a wide 2 MP canvas: the face close-up and
        # each full-body view keep about as many pixels as H3's 1024 px short edge reads.
        sheet = reference.kind == "character"
        with tel.scope(f"Creating reference {reference.name}"):
            result = await reference_generator(
                prompt=image_prompt(reference, style),
                aspect_ratio="16:9" if sheet else "1:1",
                megapixels=2.0 if sheet else 1.0,
                steps=reference_steps,
                seed=reference_seed(reference, ctx.request_id),
            )
        return result.image

    def show(reference: StoryReference, image: ImageAsset) -> None:
        # In request order, each as soon as it and every earlier one exist.
        out.publish("references", image, label=f"Reference: {reference.name}")
        shown.append(image)

    references, warnings = _audible(references)
    async with asyncio.TaskGroup() as group:
        resolving = group.create_task(resolve_reference_images(
            references, create_reference, prepare_character if prepare_characters else None, show
        ))
        if describe_characters:
            people = group.create_task(_describe_images(references, "character", tel))
            places = group.create_task(_describe_images(references, "background", tel))
            watching = group.create_task(_describe_clips(references, tel))
    images = resolving.result()
    described, clips = references, {}
    if describe_characters:
        found = {**people.result()[0], **places.result()[0]}
        described = [replace(ref, description=found[ref.name]) if found.get(ref.name) else ref for ref in references]
        clips = watching.result()[0]
        warnings += people.result()[1] + places.result()[1] + watching.result()[1]
    for reference in references:
        ctx.raise_if_cancelled()
        tel.log(
            "h3 fixed reference",
            subject=reference.name,
            reference_kind=reference.kind,
            source=("background-removed" if reference.image is not None and prepare_characters
                    and reference.kind == "character"
                    else "generated" if FILE[reference.kind] == "image" and reference.image is None
                    else "attached"),
            digest=images[reference.name].digest,
        )
    return images, described, warnings, clips
def _audible(references: list[StoryReference]) -> tuple[list[StoryReference], list[str]]:
    """`use_audio` holds only for a clip with a soundtrack: on a silent one it is ignored, with a
    warning, so H3's `<Audio N>` labels match the prompt's."""
    silent = {
        ref.name for ref in references
        if ref.kind == "video" and ref.use_audio and ref.video is not None and ref.video.probe().soundtrack is None
    }
    return (
        [replace(ref, use_audio=False) if ref.name in silent else ref for ref in references],
        [f"{name} has no audio track; use_audio ignored" for name in silent],
    )


async def _describe_images(
    references: list[StoryReference], kind: str, tel: Telemetry
) -> tuple[dict[str, str], list[str]]:
    """One sentence for each supplied character (identity) or place (look, people left out) the
    author did not describe, all in one call. Descriptions only help, so a failed call, or an
    image that shows no character, warns and keeps the reference as written."""
    wanted = [ref for ref in references if ref.kind == kind and ref.image is not None and not ref.description.strip()]
    if not wanted:
        return {}, []
    names = ", ".join(ref.name for ref in wanted)
    images = [ref.image for ref in wanted if ref.image is not None]
    try:
        with tel.scope(f"Describing {names}"):
            texts = await (describe.characters(images, tel) if kind == "character" else describe.places(images))
    except Exception as exc:
        tel.log("h3 reference descriptions skipped", level="warning", subjects=names, error=str(exc)[:400])
        return {}, [f"the descriptions of {names} were skipped: {str(exc)[:200]}"]
    described = dict(zip((ref.name for ref in wanted), texts, strict=True))
    for name, description in described.items():
        if description:
            tel.log(f"h3 {kind} description", subject=name, description=description[:400])
    return described, [
        f"{name}'s description was skipped: the image shows no character's appearance"
        for name, description in described.items() if not description
    ]


async def _describe_clips(
    references: list[StoryReference], tel: Telemetry
) -> tuple[dict[str, str], list[str]]:
    """Each clip's people, beats, camera and cuts, watched as video (`describe.clip`); a clip
    that fails is left undescribed, with a warning."""
    clips = [ref for ref in references if ref.kind == "video" and ref.video is not None]

    async def one(ref: StoryReference) -> str:
        assert ref.video is not None
        with tel.scope(f"Describing {ref.name}"):
            return await describe.clip(ref.video, tel)

    results = await asyncio.gather(*(one(ref) for ref in clips), return_exceptions=True)
    described, warnings = {}, []
    for ref, result in zip(clips, results, strict=True):
        if isinstance(result, BaseException):
            tel.log("h3 clip description skipped", level="warning", subject=ref.name, error=str(result)[:400])
            warnings.append(f"the description of {ref.name} was skipped: {str(result)[:200]}")
        else:
            tel.log("h3 clip description", subject=ref.name, description=result[:2000])
            described[ref.name] = result
    return described, warnings


async def _write_prompts(
    prompts: list[str],
    references: list[StoryReference],
    clips: dict[str, str],
    style: str,
    soundscape: str,
    tel: Telemetry,
) -> tuple[list[str], list[str]]:
    """Each call with a described clip gains, from one writer pass over all of them, what its text
    lacks of MiniMax's guide format: the characters' roles in the clip, a 350-500-word
    description, a soundscape (`apply_written`). Writing only helps: a failed pass warns and
    sends the text as written."""
    if not clips:
        return prompts, []
    calls = compile_segments(prompts, references, style=style)
    asks = [
        (index, request) for index, (text, call) in enumerate(zip(prompts, calls, strict=True))
        if (request := writer_request(text, call.references, clips, style, soundscape))
    ]
    if not asks:
        return prompts, []
    write = cast(Callable[..., Awaitable[AskResult]], ask)
    try:
        with tel.scope("Writing the prompts"):
            result = await write(prompts=[AskPrompt(request) for _, request in asks], max_tokens=WRITER_TOKENS)
    except Exception as exc:
        tel.log("h3 prompt writing skipped", level="warning", error=str(exc)[:400])
        return prompts, [f"the prompts were sent as written: writing them failed: {str(exc)[:200]}"]
    written, warnings = list(prompts), []
    for (index, _), reply in zip(asks, result.texts, strict=True):
        tel.log("h3 written prompt", segment=index + 1, reply=reply[:6000])
        written[index], more = apply_written(
            prompts[index], reply, references, calls[index].references, clips, soundscape
        )
        warnings += [f"Segment {index + 1}: {warning}" for warning in more]
    return written, warnings


async def _written[T: (LongFormInput, LongFormCutsInput)](
    payload: T, references: list[StoryReference], clips: dict[str, str], tel: Telemetry
) -> tuple[T, list[str]]:
    """`payload` with the described references and its segments' prompts written (`_write_prompts`)."""
    prompts, warnings = await _write_prompts(
        [segment.prompt for segment in payload.segments], references, clips, payload.style,
        payload.overall_soundscape.strip(), tel,
    )
    segments = [replace(segment, prompt=prompt) for segment, prompt in zip(payload.segments, prompts, strict=True)]
    return replace(payload, references=references, segments=segments), warnings


async def long_form_cuts(
    ctx: Context,
    payload: LongFormCutsInput,
    out: Outputs,
    tel: Telemetry,
) -> LongFormOutput:
    """Generate reference-conditioned camera cuts. Each reference and the film so far are
    published as they land; a failed or canceled run keeps them.

    Each segment is free text naming references as `<Name>`; see LONG_FORM.md for what
    the workflow adds, such as `style` as each segment's `style:` section. H3 fills
    undescribed audio with invented speech: name the non-voice sounds, and write
    "silently, lips closed" where no one speaks.
    """
    ctx.raise_if_cancelled()
    turbo, steps = segment_sampling(payload)
    calls = segment_calls(payload)
    planned_frames = [frames_for(shot.duration_s) for shot in payload.segments]
    render_work = planned_frames
    reference_work = sum(FILE[ref.kind] == "image" and ref.image is None for ref in payload.references) * _REFERENCE_FRAMES
    total_work = sum(render_work) + reference_work
    prefetch(cut_segment)
    references: list[ImageAsset] = []
    images, described, warnings, clips = await _create_references(
        ctx, payload.references, tel, out, references, style=payload.style,
        prepare_characters=payload.clean_references,
        describe_characters=payload.describe_references,
    )
    payload, written = await _written(payload, described, clips, tel)
    warnings += written
    calls = segment_calls(payload)
    completed_work = reference_work
    expected = None
    warnings = truncation_warnings(named_videos(payload.references, calls)) + warnings
    warnings += [warning for call in calls for warning in call.warnings]
    count = len(payload.segments)
    cut = CutJoin(out, ctx.raise_if_cancelled)
    joining: asyncio.Future[None] | None = None
    with ScanAhead(ctx.raise_if_cancelled, pictures=False) as scanning:

        def show(index: int, last: bool) -> None:
            """Join shot `index` and publish the film so far, off the GPU path."""
            revision = cut.append(scanning.result(index), planned_frames[index], last=last)
            span = "shot 1" if index == 0 else f"shots 1-{index + 1}"
            out.publish("video", revision, label=f"Video ({span} of {count})")

        try:
            for index, shot in enumerate(payload.segments):
                ctx.raise_if_cancelled()
                selected = calls[index].references
                prompt = calls[index].prompt
                seed = shot_seed(shot, ctx.request_id, index)
                assets = _call_assets(images, selected)
                log_prompt(tel, "h3 segment prompt", prompt, segment=index + 1,
                           references=", ".join(ref.name for ref in selected))
                try:
                    with tel.scope(
                        f"Segment {index + 1} of {count}",
                        overall_range=(
                            completed_work / total_work,
                            (completed_work + render_work[index]) / total_work,
                        ),
                    ):
                        result = await cut_segment(  # type: ignore[call-arg]
                            payload=CutInput(
                                prompt=prompt,
                                seed=seed,
                                steps=steps,
                                turbo=turbo,
                                duration_s=shot.duration_s,
                                frames=held_frames(
                                    ContinuationPlan(shot.duration_s), video_uses(selected)
                                ),
                                videos=video_uses(selected),
                                expected_provenance=expected,
                                previous_provenance=expected,
                            ),
                            assets=assets,
                        )
                except ChildCallError as failure:
                    ctx.raise_if_cancelled()
                    raise OutputError(
                        f"shot {index + 1} of {count} failed ({failure.code}): {str(failure)[:512]}",
                        code="segment_failed",
                    ) from failure
                compatible(result.provenance, expected)
                expected = result.provenance
                scanning.add(result.video)
                warnings.extend(result.warnings)
                completed_work += render_work[index]
                if joining is not None:
                    await joining
                joining = asyncio.ensure_future(asyncio.to_thread(show, index, index + 1 == count))
            # Every GPU call has returned: the rest is CPU work, so the next request's shots
            # start now.
            ctx.release_gpus()
        finally:
            if joining is not None:
                # However the loop ends, the film through the last joined segment is published.
                await asyncio.gather(joining, return_exceptions=True)
        assert joining is not None
        joining.result()
        joined = cut.finish()
    delivered_frames = sum(planned_frames)
    if joined.frame_count != delivered_frames:
        raise OutputError("the joined cuts differ from the declared frame clock", code="shot_frames")
    return LongFormOutput(
        video=joined.video,
        references=references,
        delivered_frames=delivered_frames,
        fps=FPS,
        warnings=warnings,
    )


async def long_form(
    ctx: Context,
    payload: LongFormInput,
    out: Outputs,
    tel: Telemetry,
) -> LongFormOutput:
    """Render segments with their selected motion context and join them. Each reference and the video so far are
    published as they land; a failed or canceled run keeps them.

    Each segment is free text naming references as `<Name>`; see LONG_FORM.md for what
    the workflow adds, such as `style` as each segment's `style:` section. H3 fills
    undescribed audio with invented speech: name the non-voice sounds, and write
    "silently, lips closed" where no one speaks.
    """
    ctx.raise_if_cancelled()
    turbo, steps = segment_sampling(payload)
    calls = segment_calls(payload)
    carries = segment_carries(payload)
    # A carry's regenerated head shares the native window with new frames. Shorten requested
    # segments to its whole-second budget instead of refusing a valid 5-15s request.
    durations = [
        min(shot.duration_s, (MAX_FRAMES - carry.head_frames) // FPS)
        for shot, carry in zip(payload.segments, carries, strict=True)
    ]
    warnings = [warning for call in calls for warning in call.warnings]
    warnings.extend(opening_cut_warnings(
        [shot.prompt for shot in payload.segments], payload.references,
        [carry.frames > 0 for carry in carries],
    ))
    for index, (shot, duration) in enumerate(zip(payload.segments, durations, strict=True)):
        if duration != shot.duration_s:
            warning = (
                f"Segment {index + 1}: shortened from {shot.duration_s}s to {duration}s "
                f"to fit {carries[index].head_frames} carried frames."
            )
            warnings.append(warning)
            tel.log(
                "Adjusted segment duration", shot=index + 1,
                requested_seconds=shot.duration_s, delivered_seconds=duration,
                context_frames=carries[index].head_frames,
            )
    # Validate every adjusted generation window before creating reference images.
    plans = [
        ContinuationPlan(duration, carried, successor)
        for duration, carried, successor in zip(
            durations, carries, [*carries[1:], Carry()], strict=True
        )
    ]
    render_work = [plan.sample_frames for plan in plans]
    reference_work = sum(FILE[ref.kind] == "image" and ref.image is None for ref in payload.references) * _REFERENCE_FRAMES
    total_work = sum(render_work) + reference_work
    prefetch(motion_segment)
    references: list[ImageAsset] = []
    images, described, described_warnings, clips = await _create_references(
        ctx, payload.references, tel, out, references, style=payload.style,
        prepare_characters=payload.clean_references,
        describe_characters=payload.describe_references,
    )
    payload, written = await _written(payload, described, clips, tel)
    described_warnings += written
    calls = segment_calls(payload)
    warnings = (
        truncation_warnings(named_videos(payload.references, calls)) + described_warnings + warnings
    )
    completed_work = reference_work
    context = None
    expected: RenderProvenance | None = None
    count = len(payload.segments)
    cut = CutJoin(out, ctx.raise_if_cancelled, continuous=True)
    joining: asyncio.Future[None] | None = None
    with ScanAhead(ctx.raise_if_cancelled, pictures=False) as scanning:

        def show(index: int, last: bool) -> None:
            """Join segment `index` and publish the video so far, off the GPU path."""
            revision = cut.append(scanning.result(index), plans[index].delivered_frames, last=last)
            span = "segment 1" if index == 0 else f"segments 1-{index + 1}"
            out.publish("video", revision, label=f"Video ({span} of {count})")

        try:
            for index, shot in enumerate(payload.segments):
                ctx.raise_if_cancelled()
                seed = shot_seed(shot, ctx.request_id, index)
                carried, successor = plans[index].carried, plans[index].successor
                try:
                    with tel.scope(
                        f"Segment {index + 1} of {count}",
                        overall_range=(
                            completed_work / total_work,
                            (completed_work + render_work[index]) / total_work,
                        ),
                    ):
                        assets = _call_assets(images, calls[index].references)
                        log_prompt(tel, "h3 segment prompt", calls[index].prompt, segment=index + 1,
                                   references=", ".join(ref.name for ref in calls[index].references))
                        child_payload = MotionInput(
                            prompt=calls[index].prompt,
                            seed=seed,
                            duration_s=durations[index],
                            steps=steps,
                            videos=video_uses(calls[index].references),
                            turbo=turbo,
                            frames=held_frames(
                                plans[index], video_uses(calls[index].references)
                            ),
                            expected_provenance=expected,
                            previous_provenance=expected,
                            context=context if carried else None,
                            context_frames=cast(ContextFrames, carried.frames),
                            context_audio=carried.audio,
                            next_context_frames=cast(ContextFrames, successor.frames),
                            next_context_audio=successor.audio,
                        )
                        # The previous segment joins in a thread while this one renders.
                        shot_result = await motion_segment(  # type: ignore[call-arg]
                            payload=child_payload, assets=assets
                        )
                except ChildCallError as failure:
                    # Concurrent caller cancellation must not become a successful partial result.
                    ctx.raise_if_cancelled()
                    raise OutputError(
                        f"segment {index + 1} of {count} failed ({failure.code}): {str(failure)[:512]}",
                        code="segment_failed",
                    ) from failure
                compatible(shot_result.provenance, expected)
                expected = shot_result.provenance
                scanning.add(shot_result.video)
                warnings.extend(shot_result.warnings)
                context = shot_result.context if successor else None
                completed_work += render_work[index]
                if joining is not None:
                    await joining
                joining = asyncio.ensure_future(asyncio.to_thread(show, index, index + 1 == count))
            # Every GPU call has returned: the rest is CPU work, so the next request's shots
            # start now.
            ctx.release_gpus()
        finally:
            if joining is not None:
                # However the loop ends, the film through the last joined segment is published.
                await asyncio.gather(joining, return_exceptions=True)
        assert joining is not None
        joining.result()
        joined = cut.finish()
    delivered_frames = sum(plan.delivered_frames for plan in plans)
    if joined.frame_count != delivered_frames:
        raise OutputError("the joined video differs from the long-form clock", code="shot_frames")
    tel.log(
        "h3 long-form delivery",
        mode=payload.mode,
        steps=steps or "plan",
        segments=count,
        delivered_frames=delivered_frames,
    )
    return LongFormOutput(
        video=joined.video,
        references=references,
        delivered_frames=delivered_frames,
        fps=FPS,
        warnings=warnings,
    )


app.entrypoint(long_form, emits_media=True, accelerator=False)
app.entrypoint(long_form_cuts, emits_media=True, accelerator=False)
app.entrypoint(assemble_video, emits_media=True, accelerator=False)
app.entrypoint(internal=True, accelerator=False)(describe.find_cuts)
