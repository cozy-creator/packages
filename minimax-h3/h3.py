"""MiniMax H3 as three tools over one dual-DiT construction: `generate` renders a shot of any length in segments
(`minimax_h3/shot.py`), `edit` edits a plate under one of Alissonerdx's guide-edit LoRAs, and `upscale` re-renders a
take above 768p with NVIDIA's Regenerate-2K.

Runtime reads typed file handles and encodes outputs. Official Diffusers owns H3 presentation, conditioning, AdaLN,
solver, and decode math. This module validates the product contract, stages weighted roots, and joins those two
boundaries.
"""

from __future__ import annotations

import hashlib
import math
import queue
import re
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import partial
from itertools import accumulate
from types import ModuleType
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, cast

import msgspec
import numpy as np
import numpy.typing as npt
import torch
from cozy_runtime.accel import stepcache
from cozy_runtime.author import (
    call,
    ChildCallError,
    Context,
    InvalidRequest,
    Lora,
    ModelChoice,
    OutputError,
    Outputs,
    prefetch,
    SavedVideo,
    Seed,
    Telemetry,
    ThreadPoolExecutor,
    UnsupportedInput,
)
from cozy_runtime.fields import (
    Advanced,
    Count,
    FileBound,
    FileLimits,
    ImagePreparation,
    Label,
    Range,
    Shape,
    Spans,
)
from cozy_runtime.media import (
    AudioFile,
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaHeader,
    DecodedVideoFormat,
    DecodedVideoFrame,
    File,
    Files,
    ImageFile,
    Latents,
    VideoFile,
)
from msgspec.structs import replace
from PIL import Image as PILImage
from PIL import ImageOps
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors
from torch import Tensor

from gates import MediaFacts, refuse_before_encode
from minimax_h3.application import app
from minimax_h3 import regenerate as refine_math
from minimax_h3 import shot_latents
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
    FULL_BALANCED,
    FullRecipe,
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
    MAX_IMAGE_REFERENCES,
    MAX_REFERENCES,
    MAX_VIDEO_REFERENCES,
    MIN_FRAMES,
    REFERENCE_IMAGE_SHORT_EDGE,
    VAE_SPATIAL_RATIO,
    OfficialH3Pipeline,
    ReferencePolicyFacts,
    ScheduleFacts,
    Task,
    TimestepPlan,
    Trunk,
    assert_duration_envelope,
    clip_size,
    denoise_rows,
    frames_for,
    read_reference_video,
    reference_image_vision_tokens,
    reference_video_vision_tokens,
    shown_width,
    validate_reference_policy,
)
from minimax_h3.shot import (
    ANY_SIZE,
    AUDIO_LAYOUTS,
    FIRST_SHOT,
    JOIN_S,
    MAX_DURATION_S,
    MIN_DURATION_S,
    NATIVE,
    VIDEO_COLOR,
    Beat,
    ClipWindow,
    Quality,
    Shot,
    ShotImage,
    ShotResult,
    ShotVideo,
    blocks,
    canvas,
    cut_latents,
    fitted,
    fl2va_text,
    frame_at,
    here,
    join_kept,
    plan_segments,
    prompt_parts,
    ratio_frame,
    segment_seed,
    segment_texts,
    shot_latents_file,
    shot_windows,
    sound_chunks,
    SoFar,
    unusable,
    upscale_prompt,
    written,
)
from story import (
    AUDIO_READ_BYTES,
    CLIPS,
    FILE,
    IMAGE_READ_BYTES,
    REFERENCE_AUDIO,
    VIDEO_READ_BYTES,
    SegmentCall,
    StoryReference,
    encoder_prompt,
    own_retention,
    segment_call,
)

if TYPE_CHECKING:
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3Reference
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

# Register preparation functions on the same App before the media entrypoints below.
from minimax_h3.preparation import job as _preparation  # noqa: F401

_MIB = 1 << 20
#: Standalone audio references share one 15-second budget in a call; past it each is cut to its share.
_MAX_AUDIO_SECONDS = Fraction(15, 1)
#: A character's picture with less of it transparent than this is opaque: its background reads as part of it.
_TRANSPARENT = 0.05
#: A picture cozy adds at a join (the previous segment's last frame, a plate's first) is read at this short edge: the
#: full picture opens the segment on a copy of it (run 20661, first-frame SSIM 0.871), 256 px keeps the set but not who
#: stands where (run 20689).
_LOOSE_EDGE = 512

Reference = ImageFile | VideoFile | AudioFile
#: One reference of each kind, with its retained-read bound. A Ref2VA segment's `assets` declare
#: the preparations the shot's own fields do, so Runtime records a truncation on the call the
#: client made.
_REFERENCE = (
    Annotated[ImageFile, FileBound(max_decoded_bytes=IMAGE_READ_BYTES)]
    | Annotated[
        VideoFile, FileBound(max_decoded_bytes=VIDEO_READ_BYTES)
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
    REFERENCE_AUDIO,
    msgspec.Meta(min_length=1),
]


class VideoUse(msgspec.Struct, frozen=True):
    """How a call uses one of its videos: whether its frames are anchored in the output (`anchor_video`), and the
    frames of it, on the 24-fps clock, the call's window starts and stops at (`shot_windows`)."""

    anchor: bool = False
    start: Annotated[int, msgspec.Meta(ge=0)] = 0
    stop: Annotated[int, msgspec.Meta(ge=0)] = 0


def video_uses(references: Sequence[StoryReference], windows: dict[str, ClipWindow]) -> tuple[VideoUse, ...]:
    """Each of a call's videos, in packed order, from its `windows`."""
    return tuple(VideoUse(ref.kind == "anchor_video", *windows[ref.name]) for ref in references if ref.kind in CLIPS)


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


class Resolution(Enum):
    """A Regenerate render's size: native 768p (a refine only), or above it."""

    NATIVE = "768p"
    FULL_HD = "1080p"
    QHD = "1440p"
    UHD = "2160p"


#: The refine starts the Regenerate overlay serves (`efficient-large-model/open-h3-regenerate-2k@0.1.0`'s plan).
REGENERATE_STRENGTHS = (0.7, 0.8, 0.909)
#: The default start (Paul, 10-31): NVIDIA's 0.909 redraws a face as another person's, its hat and sunglasses, and a
#: painted look as a photograph; 0.7 keeps them (fidelity test, runs 24518-24554).
REGENERATE_STRENGTH = 0.7


class H3VideoOutput(msgspec.Struct):
    """The catalog wire shape. Checkpoint, plan, geometry, and digest facts remain
    attempt observations (se-012): they ride Telemetry, never the customer result."""

    video: Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, media_types=("video/mp4",))]
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
    """The images without a fidelity share the package default and step down the ladder together
    until the total fits the budget, never above the default; past its last rung the explicit ones
    step down with them."""

    def sized(edge: int, explicit: bool = False) -> ReferenceSizing:
        resolved = tuple(
            replace(image, resolved=min(image.requested, edge) if explicit and image.requested else image.requested or edge)
            for image in images
        )
        return ReferenceSizing(resolved, video_tokens, default)

    ladder = [default, *(rung for rung in _SHORT_EDGE_LADDER if rung < default)]
    tries = [*(sized(edge) for edge in ladder), *(sized(edge, True) for edge in _SHORT_EDGE_LADDER)]
    return next((sizing for sizing in tries if sizing.total <= MAX_CONDITIONER_VISION_TOKENS), tries[-1])


def assets_to_h3_refs(
    references: Files[Reference],
    *,
    pipe: OfficialH3Pipeline,
    frames: int,
    videos: Sequence[VideoUse] = (),
    edges: Sequence[int] = (),
    characters: Sequence[str] = (),
) -> tuple[list[MiniMaxH3Reference], ReferenceSizing, GuideClip | None, list[str]]:
    """Check every reference from its header and size the vision budget, then read each.

    A video is read over its window: from its `videos` entry's `start` to its `stop` (per video in
    packed order), the call's `frames` at most, which is all the official pipeline keeps of it, and
    budgeted at that length. It streams straight to the size it is used at on the 24-fps clock
    (`read_reference_video`), without its sound. An anchor video is not a reference (`guide_rows`):
    Qwen never sees it. A character's picture (`characters`: its name per picture, in packed order)
    that is opaque is said: its background goes into the shot.
    """
    audio_duration = Fraction(0)
    video_tokens = 0
    images: list[ImageSizing] = []
    use = iter(videos)
    uses = [
        next(use, VideoUse()) if isinstance(reference, VideoFile) else VideoUse()
        for reference in references
    ]
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
            edge = edges[len(images)] if len(images) < len(edges) else 0
            requested = edge or (
                None if info.fidelity == "auto" else _REFERENCE_FIDELITY_EDGES[info.fidelity]
            )
            images.append(ImageSizing(field, image.width, image.height, requested))
        elif isinstance(reference, VideoFile):
            video = reference.probe()
            if not uses[index].anchor:
                start, stop = uses[index].start, uses[index].stop
                window = min(Fraction(min(frames, stop - start), FPS), video.duration - Fraction(start, FPS))
                shown = shown_width(video.width, video.pixel_aspect_ratio)
                video_tokens += reference_video_vision_tokens(shown, video.height, window)
        else:
            audio_duration += reference.probe().duration
    sizing = resolve_reference_sizing(
        images, default=DEFAULT_REFERENCE_IMAGE_SHORT_EDGE, video_tokens=video_tokens
    )
    prepared: list[MiniMaxH3Reference] = []
    guide = None
    notes: list[str] = []
    share = min(Fraction(1), _MAX_AUDIO_SECONDS / audio_duration) if audio_duration else Fraction(1)
    if share < 1:
        notes.append(f"this segment's audio references run {float(audio_duration):.1f} s; each is cut to its share "
                     "of the 15 s H3 takes")  # fmt: skip
    names = iter(characters)
    for index, reference in enumerate(references):
        if isinstance(reference, ImageFile):
            picture = reference.read(mode="RGBA")
            alpha = np.asarray(picture.getchannel("A"))
            if (name := next(names, "")) and np.count_nonzero(alpha < 128) < _TRANSPARENT * alpha.size:
                notes.append(f"<{name}>'s picture is opaque, so its background goes into the shot: draw it on a blank "
                             "background, or take its background off with a tool")  # fmt: skip
            white = PILImage.new("RGBA", picture.size, (255, 255, 255, 255))
            prepared.append(pipe.image_reference(PILImage.alpha_composite(white, picture).convert("RGB")))
        elif isinstance(reference, VideoFile):
            start, stop = uses[index].start, uses[index].stop
            with reference.stream(start=Fraction(start, FPS), end=Fraction(min(start + frames, stop), FPS),
                                  audio=False) as events:  # fmt: skip
                clip = read_reference_video(events, frames=frames, audio=False, guide=uses[index].anchor)
            if uses[index].anchor:
                guide = pipe.guide_clip(clip)
            else:
                prepared.append(pipe.video_reference(clip))
        else:
            prepared.append(pipe.audio_reference(reference.read(end=reference.probe().duration * share)))
    return prepared, sizing, guide, notes


def _finish(
    model: H3Model,
    task: Task,
    state: PipelineState,
    schedule: ScheduleFacts,
    *,
    duration_s: int,
    out: Outputs,
    tel: Telemetry,
    progress: Callable[[int, int], None],
    cancel: Callable[[], None],
    completed: Callable[[PipelineState], None] | None = None,
    delivered_frames: int | None = None,
    sampled_frames: int | None = None,
    ended: Callable[[Tensor], None] | None = None,
    earlier: _Earlier | None = None,
) -> H3VideoOutput:
    """Decode and encode the render, with `earlier` segments before it as one video."""
    cancel()
    if completed is not None:
        # The successor's carry: a slice of the denoised latents, before either decode.
        completed(state)
    frames = frames_for(duration_s) if delivered_frames is None else delivered_frames
    waveform, sample_rate = _soundtrack(model, task, state, frames=frames, tel=tel, cancel=cancel)
    decode = partial(model.decode_video, task, state)
    if earlier is not None:
        # The whole shot as one latent: the VAE cross-fades its chunks across every join,
        # where decoding segment by segment steps the picture (run 20751: luma +3.7 at a join).
        latents = torch.cat([earlier.latents.to(state.latents.device), state.latents[:, :, earlier.head :]], dim=2)
        decode = partial(model.decode_latents, latents)
        frames += earlier.frames
        waveform = torch.cat([earlier.waveform, waveform], dim=1)

    # The encoder runs under the decode (h3a-017): each chunk is quantized on its device
    # and landed in the host buffer while a worker feeds the runtime's streaming MP4 sink.
    with tel.stage("decode_video"):
        stream = _VideoStream(
            torch,
            out,
            frames=frames,
            waveform=waveform,
            sample_rate=sample_rate,
            cancel=cancel,
            progress=progress,
        )
        try:
            decode(on_chunk=stream.push)
        except BaseException:
            stream.abandon()
            raise
    try:
        with tel.stage("encode_video"):
            saved = stream.finish(progress)
    except BaseException:
        stream.abandon()
        raise
    pixels = cast(Tensor, stream.pixels)  # `finish` saw every frame land
    _, height, width, _ = (int(value) for value in pixels.shape)
    if ended is not None:
        ended(pixels)

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
    model: H3Model, task: Task, state: PipelineState, *, frames: int, tel: Telemetry, cancel: Callable[[], None]
) -> tuple[Tensor, int]:
    """The render's decoded `(channels, samples)` soundtrack, checked against its clock."""
    with tel.stage("decode_audio"):
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


def _land(torch: ModuleType, chunk: Tensor, landed: Tensor) -> None:
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
        torch: ModuleType,
        out: Outputs,
        *,
        frames: int,
        waveform: Tensor | None,
        sample_rate: int,
        cancel: Callable[[], None],
        progress: Callable[[int, int], None],
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
        self.pixels: Tensor | None = None
        self.landed = 0

    def push(self, chunk: Tensor) -> None:
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
        self._progress(self.landed - 1, self._frames)

    def finish(self, progress: Callable[[int, int], None]) -> SavedVideo:
        """Close the stream once every requested frame has landed; the sink's probed asset,
        `progress(step, total)` told the frames it took meanwhile. A sink that wrote fewer frames
        fails the run."""
        if self.landed != self._frames or self._encode is None:
            raise OutputError(
                f"official H3 decode returned {self.landed} frames, expected {self._frames}",
                code="output_integrity",
            )
        self._slices.put(None)
        taken = self._taken  # the callback holds the queue, not the stream: no cycle keeps its frames
        self._encode.add_done_callback(lambda _: taken.put(None))
        try:
            while (index := self._taken.get()) is not None:
                if self._taken.empty():  # the newest only
                    progress(index, self._frames)
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
        pixels = cast(Tensor, self.pixels)  # the push that began this stream made them
        _, height, width, _ = (int(value) for value in pixels.shape)
        channels, samples = (0, 0) if self._waveform is None else (int(v) for v in self._waveform.shape)
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=width,
                height=height,
                time_base=Fraction(1, FPS),
                pixel_aspect_ratio=Fraction(1),
                nominal_frame_rate=Fraction(FPS),
                **VIDEO_COLOR,
            ),
            audio=None
            if self._waveform is None
            else DecodedAudioFormat(
                channels=channels,
                sample_rate=self._sample_rate,
                channel_layout=AUDIO_LAYOUTS[channels][0],
                channel_names=AUDIO_LAYOUTS[channels][1],
                time_base=Fraction(1, self._sample_rate),
            ),
        )
        submitted = 0
        while (item := self._slices.get()) is not None:
            start, stop = item
            landed = pixels[start:stop]
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
                    **VIDEO_COLOR,
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
        waveform = cast(Tensor, self._waveform)  # samples to send: there is a soundtrack
        channels = int(waveform.shape[0])
        yield DecodedAudioChunk(
            channels=channels,
            sample_count=stop - start,
            sample_rate=self._sample_rate,
            channel_layout=AUDIO_LAYOUTS[channels][0],
            channel_names=AUDIO_LAYOUTS[channels][1],
            pcm_f32le=tuple(
                waveform[channel, start:stop].numpy().tobytes() for channel in range(channels)
            ),
            pts=start,
            time_base=Fraction(1, self._sample_rate),
        )


def _nonfinite_fraction(torch: ModuleType, value: Tensor) -> float:
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
_MODEL = "minimaxai/minimax-h3@1.0.0-rc.8"
_TURBO_LORA = "alibaba-pai/minimax-h3-turbo-lora@1.0.1"
_REGENERATE_LORA = "efficient-large-model/open-h3-regenerate-2k@0.1.0"
_LATENT_UPSCALER = "lbh-123-ai/h3-latent-upscaler@0.1.1"

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


#: Turbo's dense steps of eight: all (NVIDIA HyperFlow's `quality`), four, and one (Sol-Engine's
#: `sol_bsa`), the rest sparse at tau 1.0.
TURBO_DENSE_STEPS: dict[Quality, int] = {"full": 8, "standard": 4, "draft": 1}
#: The full model's recipes, NVIDIA sol-engine's tiers (forward indices, as theirs): `full` H100's
#: 15 dense steps, tau 0.5 and EasyCache; `standard` its RTX `sol` arm, 10 dense at tau 1.0;
#: `draft` that plus H100 `aggressive` / GB200 `fullopt`'s FirstBlockCache. Turbo's eight distilled
#: steps never cache (NVIDIA keeps cache reuse off on distilled paths).
FULL_RECIPES: dict[Quality, FullRecipe] = {
    "full": FullRecipe(15, 0.5, stepcache.EasyCache(0.30, retain=10, cooldown=4, max_hits=1)),
    "standard": FULL_BALANCED,
    "draft": FullRecipe(10, 1.0, stepcache.FirstBlockCache(0.08)),
}


class RegenerateInput(msgspec.Struct, kw_only=True):
    """`upscale` (`regenerate_shot`): the shot's 768p latents (`latents`, its
    own, as NVIDIA's `--input-latents`), else its video's frames encoded, upscaled in latent space
    and refined by H3 at `resolution` from `prompt`, the video's soundtrack kept, at its own shape
    (`refine_math.target`): a 7:4 frame fills 16:9, stretched 1.6%, and 4:7 9:16; `768p` refines only."""

    video: Annotated[VideoFile, FileBound(max_decoded_bytes=2 << 30)]
    latents: Latents | None = None
    prompt: str
    resolution: RegenerateResolution = Resolution.QHD
    strength: float = REGENERATE_STRENGTH
    seed: int
    #: NVIDIA's refine forwards (`regenerate.PRESETS`): `full` dense (`standard`), `standard`
    #: step 0 dense then sparse at tau 1.0 (`sol_engine`), `draft` all sparse at tau 1.75.
    quality: Quality = "standard"


class RegenerateOutput(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, media_types=("video/mp4",))]
    warnings: list[str]


def _draft_size(video: VideoFile) -> tuple[int, int]:
    """The (height, width) the refine's latent is at: the H3 canvas of the video's shape, as its shot rendered."""
    probe = video.probe()
    tall, wide = clip_size(shown_width(probe.width, probe.pixel_aspect_ratio), probe.height, guide=True)
    return tall, wide


def _read_draft(video: VideoFile, draft: tuple[int, int]) -> tuple[Tensor, Tensor | None, int, int, list[str]]:
    """A video of any length as `(frames on the VAE grid, *draft, 3)` uint8, each frame
    cover-scaled and centre-cropped as it decodes into one buffer sized from the header, its
    last frame held up to the grid; its `(channels, samples)` soundtrack or None, the sample
    rate, and how many frames it delivers."""
    probe = video.probe()
    pixels = torch.empty((round(probe.duration * FPS) + 17, *draft, 3), dtype=torch.uint8)
    pcm: list[list[npt.NDArray[np.float32]]] = []
    count = width = height = rate = 0
    origin: Fraction | None = None
    with video.stream() as stream:
        for event in stream:
            if isinstance(event, DecodedMediaHeader):
                if event.video is None:
                    raise InvalidRequest("regenerate needs a video stream", fields=["video"])
                width, height = event.video.width, event.video.height
            elif isinstance(event, DecodedVideoFrame):
                # Onto the 24-fps clock: a faster source's extra frames go, a slower one's hold.
                origin = event.start_time if origin is None else origin
                slot = round((event.start_time - origin) * FPS)
                if slot < count:
                    continue
                if slot >= len(pixels):
                    pixels = torch.cat([pixels, torch.empty((slot + 17 - len(pixels), *draft, 3), dtype=torch.uint8)])
                frame = torch.frombuffer(bytearray(event.rgb), dtype=torch.uint8)
                pixels[count : slot + 1] = _cover(frame.reshape(height, width, 3), draft)
                count = slot + 1
            else:
                rate = event.sample_rate
                pcm.append([np.frombuffer(data, dtype=np.float32) for data in event.pcm_f32le])
    # Under 5 s the refine runs on its last frame held to 5 s, and the result is cut back.
    grid = max(count + (-(count - 5)) % 17, MIN_FRAMES)
    if grid > len(pixels):
        pixels = torch.cat([pixels, pixels[-1:].expand(grid - len(pixels), -1, -1, -1)])
    pixels = pixels[:grid]
    pixels[count:] = pixels[count - 1]
    warnings = [] if (height, width) == draft else [
        f"the {width}x{height} video was scaled to {draft[1]}x{draft[0]} for the refine"
    ]  # fmt: skip
    if probe.frame_rate != FPS:
        warnings.append(f"the {float(probe.frame_rate or 0):g} fps video was read at 24 fps for the refine")
    waveform = None
    if pcm:
        waveform = torch.from_numpy(np.stack([np.concatenate(c) for c in zip(*pcm, strict=True)]))
    return pixels, waveform, rate, count, warnings


def _read_soundtrack(video: VideoFile, frames: int) -> tuple[Tensor | None, int]:
    """The `(channels, samples)` soundtrack of a video its latents stand for (`frames` frames at
    24 fps), and its sample rate; its pictures are not decoded."""
    probe = video.probe()
    if abs(round(probe.duration * FPS) - frames) > 1:
        raise InvalidRequest(f"the latents are {frames} frames and the video {round(probe.duration * FPS)}: "
                             "they are not the same shot", fields=["latents"])  # fmt: skip
    pcm: list[list[npt.NDArray[np.float32]]] = []
    rate = 0
    with video.stream(pictures=False) as stream:
        for event in stream:
            if isinstance(event, DecodedAudioChunk):
                rate = event.sample_rate
                pcm.append([np.frombuffer(data, dtype=np.float32) for data in event.pcm_f32le])
    if not pcm:
        return None, rate
    waveform = torch.from_numpy(np.stack([np.concatenate(c) for c in zip(*pcm, strict=True)]))
    return waveform, rate


def _cover(frame: Tensor, draft: tuple[int, int]) -> Tensor:
    """One `(h, w, 3)` uint8 frame cover-scaled (bilinear) and centre-cropped to the `draft` (height, width)."""
    height, width = (int(value) for value in frame.shape[:2])
    if (height, width) == draft:
        return frame
    scale = max(draft[0] / height, draft[1] / width)
    size = (max(draft[0], round(height * scale)), max(draft[1], round(width * scale)))
    image = torch.nn.functional.interpolate(
        frame.permute(2, 0, 1)[None].float(), size=size, mode="bilinear", align_corners=False
    )[0]
    top, left = (size[0] - draft[0]) // 2, (size[1] - draft[1]) // 2
    crop = image[:, top : top + draft[0], left : left + draft[1]]
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
    first: PILImage.Image | None,
    last: PILImage.Image | None,
    canvas: tuple[int, ...],
    turbo_lora: H3TurboLoRA | None = None,
    quality: Quality = "standard",
    delivery: ContinuationPlan | None = None,
    completed: Callable[[PipelineState], None] | None = None,
    ended: Callable[[Tensor], None] | None = None,
) -> H3VideoOutput:
    """An FL2VA render: text, and a first and/or last keyframe, on the shot's (width, height)
    `canvas`. `delivery` and `completed` hand a successor its carry, and `ended` its last frame
    (`fl2va_segment`)."""
    plan = _plan(model, turbo_lora, "fl2va")
    steps = plan.schedule(steps).transformer_evaluations
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=seed)
    notes: list[str] = []
    log_prompt(tel, "h3 encoder prompt", prompt)
    with tel.stage("prepare"):
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
            canvas=(canvas[1], canvas[0]),
        )
    with tel.stage("condition_text"):
        model.condition_text(task, state)
    if first is not None or last is not None:
        with tel.stage("condition_media"):
            model.condition_fl2va_media(task, state)
    if turbo_lora is None:
        sample = partial(model.sample_fl2va, recipe=FULL_RECIPES[quality])
    else:
        assert isinstance(model, H3TurboBase)
        sample = partial(model.sample_fl2va_turbo, turbo_lora=turbo_lora, sol_dense_steps=TURBO_DENSE_STEPS[quality])
    with tel.stage("denoise"):
        schedule = sample(
            state,
            on_step=lambda step: ctx.progress(step, steps),
            cancel=ctx.raise_if_cancelled,
        )
    result = _finish(
        model, task, state, schedule, duration_s=duration_s, out=out, tel=tel, progress=ctx.progress,
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
    completed: Callable[[PipelineState], None] | None,
    anchors: Sequence[tuple[PILImage.Image, Literal["first", "last"]]],
    ended: Callable[[Tensor], None],
    earlier: _Earlier | None = None,
    sound: tuple[Tensor, Fraction] | None = None,
) -> H3VideoOutput:
    plan = _plan(model, turbo_lora, "ref2va")
    steps = plan.schedule(payload.steps).transformer_evaluations
    # A cut video's sound is encoded first: what the segment carries starts from it.
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=payload.seed)
    notes: list[str] = []
    if context is not None and context.provenance != expected_context_provenance:
        # The carry is latents, the same for every lane and release that share the VAE (`shot_latents.vae_space`).
        notes.append("this continues a video made with another model release, lane or turbo setting; the join may show")
    if payload.frames != held_frames(delivery, payload.videos):
        raise InvalidRequest(
            f"a {payload.duration_s} s segment with these references holds "
            f"{held_frames(delivery, payload.videos)} frames, not {payload.frames}",
            fields=["payload.frames"],
        )
    frames = delivery.sample_frames
    with tel.stage("prepare"):
        if sound is not None:
            assert context is not None
            context = context.with_sound(model.carried_sound(sound[0]), sound[1])
        references, sizing, guide, read_notes = assets_to_h3_refs(
            assets, pipe=model.pipe, frames=frames, videos=payload.videos, edges=payload.picture_edges,
            characters=payload.characters,
        )
        notes += read_notes
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
            reference_image_short_edges=sizing.edges,
            task=task,
            plan=plan,
            guide=guide,
            anchors=anchors,
            canvas=(payload.canvas[1], payload.canvas[0]),
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
    with tel.stage("condition"):
        condition_references(model, task, state, tel=tel)
    if turbo_lora is None:
        sample = partial(model.sample_ref2va, recipe=FULL_RECIPES[payload.quality])
    else:
        assert isinstance(model, H3TurboBase)
        sample = partial(model.sample_ref2va_turbo, turbo_lora=turbo_lora, sol_dense_steps=TURBO_DENSE_STEPS[payload.quality])
    with tel.stage("denoise"):
        schedule = sample(
            state,
            on_step=lambda step: ctx.progress(step, steps),
            cancel=ctx.raise_if_cancelled,
        )
    result = _finish(
        model, task, state, schedule, duration_s=payload.duration_s, out=out, tel=tel, progress=ctx.progress,
        cancel=ctx.raise_if_cancelled, completed=completed, ended=ended,
        delivered_frames=delivery.delivered_frames, sampled_frames=delivery.sample_frames, earlier=earlier,
    )  # fmt: skip
    return H3VideoOutput(result.video, notes + result.warnings)


def _keep_frames(out: Outputs, last: list[ImageFile], pixels: Tensor) -> None:
    """A call's delivered `(frames, H, W, 3)` RGB8: its last frame."""
    last.append(out.save_image(PILImage.fromarray(pixels[-1].numpy())))


def _anchors(
    first: ImageFile | None, last: ImageFile | None, shape: tuple[int, ...]
) -> list[tuple[PILImage.Image, Literal["first", "last"]]]:
    """A Ref2VA call's keyframes, read: anchored as ComfyUI's Add Guide anchors them."""
    pairs: list[tuple[ImageFile | None, Literal["first", "last"]]] = [(first, "first"), (last, "last")]
    return [(image, at) for asset, at in pairs if (image := _frame(asset, shape)) is not None]


# --- segments ----------------------------------------------------------------------------

ContextFrames = Literal[0, 22, 39, 56]


#: A segment's carry: a predecessor's exported context (~2.7 MB), or a resumed video's whole latents
#: file, whose carry the segment reads (`shot_latents.context`).
ContextFile = Annotated[
    File, FileBound(max_bytes=ANY_SIZE, media_types=("application/octet-stream",))
]
#: A segment's delivered video latents (`_Segments`), about 3 MB a second at 768p.
LatentFile = Annotated[File, FileBound(max_bytes=ANY_SIZE, media_types=("application/octet-stream",))]
SegmentVideo = Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, max_decoded_bytes=32 << 20, media_types=("video/mp4",))]


def held_frames(plan: ContinuationPlan, videos: Sequence[VideoUse] = ()) -> int:
    """The frames a call's DiT holds: its sampled window, plus the carried picture again as
    condition rows (the double form; single-copy was tested and rejected, 2026-10-10), plus
    an anchor video's frames, at most the window's again."""
    anchored = any(video.anchor for video in videos)
    return plan.sample_frames * (1 + anchored) + plan.carried.frames


class Ref2VAInput(msgspec.Struct):
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
    #: The (width, height) its keyframes fill: the shot's canvas (`shot.canvas`).
    canvas: tuple[int, ...]
    #: What this segment carries from its predecessor, and what its successor carries.
    context_frames: ContextFrames = 0
    context_audio: bool = False
    next_context_frames: ContextFrames = 0
    next_context_audio: bool = False
    previous_provenance: RenderProvenance | None = None
    context: ContextFile | None = None
    #: Keyframes, anchored as ComfyUI's Add Guide anchors them (`anchor_rows`).
    first_frame: FrameImage = None
    last_frame: FrameImage = None
    #: Each picture's short edge in packed order; 0 sizes it as usual.
    picture_edges: tuple[int, ...] = ()
    #: Each picture's name in packed order where it is a character, whose opaque background is said; else "".
    characters: tuple[str, ...] = ()
    #: Return the delivered latents (`SegmentOutput.latents`).
    latents: bool = False
    #: A shot's last segment: the earlier segments' latents and videos, decoded with it as one.
    earlier_latents: tuple[LatentFile, ...] = ()
    earlier_videos: tuple[SegmentVideo, ...] = ()
    quality: Quality = "standard"


class SegmentOutput(msgspec.Struct):
    #: The segment's video; a shot's last segment returns the whole shot.
    video: SegmentVideo
    warnings: list[str]
    provenance: RenderProvenance
    # A fixed output slot: Runtime grants result outputs by field path, never inside a union.
    # A segment with no successor returns it empty instead of exporting any window.
    context: ContextFile
    #: The segment's last delivered frame. A child's result is a granted input of the parent,
    #: so it may feed the next segment (a picture the parent saved itself may not, run 20651).
    last_frame: ImageFile
    #: Its delivered video latents when the call asks (`latents`), else empty: a shot's last
    #: segment decodes them with its own as one.
    latents: LatentFile


@dataclass(frozen=True)
class _Earlier:
    """A shot's segments before its last: their delivered latents end to end, their sound end to
    end (each starts where the last ended, as continuation cut it; a cut video's, kept in its
    latents, first), and their frames; and the last segment's head to leave out, whole VAE cycles,
    so its latents follow in phase."""

    latents: Tensor
    waveform: Tensor
    frames: int
    head: int

    @classmethod
    def read(cls, latents: Sequence[File], videos: Sequence[VideoFile], head: int) -> _Earlier:
        loaded = [load_tensors(file.read_bytes()) for file in latents]
        parts = [tensors["sound"].numpy() for tensors in loaded if "sound" in tensors]
        for video in videos:
            parts.extend(sound_chunks(video.stream(pictures=False)))
        joined = torch.cat([tensors["latents"] for tensors in loaded], dim=2)
        # 5k + 2 latents decode to 17k + 5 frames (`continuation.video_tokens`).
        frames = (int(joined.shape[2]) - 2) // 5 * 17 + 5
        return cls(joined, torch.from_numpy(np.concatenate(parts, axis=1)), frames, head)


class _Segments:
    """What a segment hands on once it has denoised, before its decode: its successor's carry
    and, when asked, its delivered latents."""

    def __init__(self, out: Outputs, model: H3Model, plan: ContinuationPlan, observed: RenderProvenance,
                 latents: bool) -> None:  # fmt: skip
        self.out, self.model, self.plan, self.observed, self.latents = out, model, plan, observed, latents
        self.saved: dict[str, File] = {}
        self.warnings: list[str] = []

    def __call__(self, state: PipelineState) -> None:
        if self.plan.successor:
            # What's denoised is worth more than what comes after it: a video stands without its carry, which only a later take reads.
            try:
                carried = self.model.pipe.export_context(state, provenance=context_provenance(self.observed))
                assert carried is not None
                self.saved["context"] = self.out.save_bytes(encode_context(carried), media_type="application/octet-stream")
            except InvalidRequest as refused:
                self.warnings.append(f"Pick up from here isn't available for this video: {refused.message}")
        if self.latents:
            # The head is whole VAE cycles (`continuation`), so what follows it joins the
            # previous segment's latents in phase.
            head = video_tokens(self.plan.carried.frames) if self.plan.carried.frames else 0
            delivered = state.latents[:, :, head:].detach().float().cpu().contiguous()
            space = self.model.latent_space()
            self.saved["latents"] = self.out.save_bytes(save_tensors({"latents": delivered}, {"vae_space": space}),
                                                       media_type="application/octet-stream")  # fmt: skip

    @property
    def wanted(self) -> bool:
        return bool(self.plan.successor) or self.latents

    def file(self, name: str) -> File:
        return self.saved.get(name) or self.out.save_bytes(b"", media_type="application/octet-stream")


def _sampling(payload: FL2VAInput | Ref2VAInput, turbo_lora: H3TurboLoRA) -> H3TurboLoRA | None:
    """The segment's sampler over the one shared construction: the PDD adapter, or none."""
    if payload.turbo and payload.steps is not None:
        raise InvalidRequest("a turbo segment runs its LoRA's one schedule; it takes no steps")
    return turbo_lora if payload.turbo else None


@app.entrypoint(
    private=True,
    defaults={"base_model": _MODEL, "turbo_lora": _TURBO_LORA},
    components={"base_model": _REF2VA, "turbo_lora": ("ref2va_turbo",)},
    memoize=True,
)
async def ref2va_segment(
    ctx: Context,
    *,
    payload: Ref2VAInput,
    assets: ReferenceFiles,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> SegmentOutput:
    """A segment on Ref2VA: its references, the context it carries in, and its successor's."""
    lora = _sampling(payload, turbo_lora)
    observed = provenance(base_model.checkpoint_ref, "" if lora is None else lora.checkpoint_ref)
    plan = ContinuationPlan(
        payload.duration_s,
        Carry(int(payload.context_frames), payload.context_audio),
        Carry(int(payload.next_context_frames), payload.next_context_audio),
    )
    context: AVContext | None = None
    sound = None
    if plan.carried:
        if payload.context is None:
            raise InvalidRequest("a carried picture or sound needs the previous context", fields=["payload.context"])
        data = payload.context.read_bytes()
        context, sound = decode_context(shot_latents.context(data)), shot_latents.sound(data)
    _reference_policy(assets)
    handed = _Segments(out, base_model, plan, observed, payload.latents)
    earlier = None
    if payload.earlier_latents:
        head = video_tokens(plan.carried.frames) if plan.carried.frames else 0
        earlier = _Earlier.read(payload.earlier_latents, payload.earlier_videos, head)
    ended: list[ImageFile] = []
    segment = _references_to_video(
        ctx, "ref2va" if lora is None else "ref2va_turbo", payload, assets, base_model, out, tel,
        turbo_lora=lora, context=context, delivery=plan,
        expected_context_provenance=context_provenance(payload.previous_provenance or observed),
        completed=handed if handed.wanted else None,
        anchors=_anchors(payload.first_frame, payload.last_frame, payload.canvas),
        ended=partial(_keep_frames, out, ended),
        earlier=earlier, sound=sound,
    )  # fmt: skip
    return SegmentOutput(segment.video, [*segment.warnings, *handed.warnings], observed, handed.file("context"), ended[0], handed.file("latents"))


def log_prompt(tel: Telemetry, event: str, prompt: str, **fields: str | int) -> None:
    """The text's digest and ends: a log value holds 400 characters, not a whole prompt."""
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    tel.log(event, level="info", digest=digest, chars=len(prompt), head=prompt[:400], tail=prompt[-400:], **fields)




#: A keyframe a segment fills its canvas with.
FrameImage = ShotImage | None


class FL2VAInput(msgspec.Struct, kw_only=True):
    """One FL2VA segment: its text, keyframes and length, and what its successor carries."""

    prompt: str
    seed: int
    duration_s: DurationSeconds
    steps: int | None
    turbo: bool
    quality: Quality = "standard"
    first_frame: FrameImage = None
    last_frame: FrameImage = None
    #: The (width, height) the keyframes fill: the shot's canvas (`shot.canvas`).
    canvas: tuple[int, ...]
    next_context_frames: ContextFrames = 0
    next_context_audio: bool = False
    #: Return the delivered latents (`SegmentOutput.latents`).
    latents: bool = False


def _frame(asset: ImageFile | None, shape: tuple[int, ...]) -> PILImage.Image | None:
    """A keyframe as a segment reads it: its middle, scaled to fill the shot's canvas (`shape`)."""
    if asset is None:
        return None
    return ImageOps.fit(asset.read(), shape, method=PILImage.Resampling.BICUBIC)  # type: ignore[arg-type]  # the canvas: (width, height)


@app.entrypoint(
    private=True,
    defaults={"base_model": _MODEL, "turbo_lora": _TURBO_LORA},
    components={"base_model": _FL2VA, "turbo_lora": ("fl2va_turbo",)},
    memoize=True,
)
async def fl2va_segment(
    ctx: Context,
    *,
    payload: FL2VAInput,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> SegmentOutput:
    """A segment on FL2VA: its text and its first and/or last keyframe, or text alone."""
    lora = _sampling(payload, turbo_lora)
    observed = provenance(base_model.checkpoint_ref, "" if lora is None else lora.checkpoint_ref)
    plan = ContinuationPlan(
        payload.duration_s, Carry(),
        Carry(int(payload.next_context_frames), payload.next_context_audio),
    )  # fmt: skip
    handed = _Segments(out, base_model, plan, observed, payload.latents)
    ended: list[ImageFile] = []
    shot = _render_keyframes(
        ctx, "fl2va" if lora is None else "fl2va_turbo", base_model, out, tel,
        prompt=encoder_prompt("fl2va", payload.prompt), seed=payload.seed,
        duration_s=payload.duration_s, steps=payload.steps,
        first=_frame(payload.first_frame, payload.canvas), last=_frame(payload.last_frame, payload.canvas),
        canvas=payload.canvas, turbo_lora=lora, quality=payload.quality, delivery=plan if plan.successor else None,
        completed=handed if handed.wanted else None,
        ended=partial(_keep_frames, out, ended),
    )  # fmt: skip
    return SegmentOutput(shot.video, [*shot.warnings, *handed.warnings], observed, handed.file("context"), ended[0], handed.file("latents"))


@app.entrypoint(
    private=True,
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
    """Regenerate-2K over a whole shot: its 768p latent upscaled, refined at
    `resolution` in overlapping windows where the card needs them, and decoded with its
    soundtrack."""
    # The shot's own latent: no encode, and no decode-and-encode loss (NVIDIA's `--input-latents`).
    # Latents it can't use, the video stands for them.
    warnings: list[str] = []
    latents = None
    draft = _draft_size(payload.video)
    if payload.latents is not None:
        try:
            grid = (draft[0] // VAE_SPATIAL_RATIO, draft[1] // VAE_SPATIAL_RATIO)
            latents, frames = shot_latents.load(payload.latents.read_bytes(), base_model.latent_space(), grid)
            waveform, sample_rate = _read_soundtrack(payload.video, frames)
        except InvalidRequest as refused:
            latents = None
            warnings.append(f"these latents can't be used, so it works from the video: {refused.message}")
    cancel = ctx.raise_if_cancelled
    if latents is None:
        with tel.stage("read_video"):
            pixels, waveform, sample_rate, frames, read = _read_draft(payload.video, draft)
            warnings += read
        with tel.stage("encode_video"):
            latents = base_model.encode_clip(pixels)
        del pixels
    resolution = payload.resolution.value
    with tel.stage("upscale_latent"):
        for height, width in refine_math.upscale_passes(resolution, draft):
            latents = latent_upscaler.upscale(latents, height=height, width=width)
    cancel()
    with tel.stage("condition_text"):
        text = base_model.encode_prompt(encoder_prompt("fl2va", payload.prompt), frames_for(MIN_DURATION_S))
    layout = refine_math.plan_windows(tuple(int(value) for value in latents.shape))
    (first, stop), *_ = layout.spans
    schedule = regenerate_lora.pipe.plan.refined(payload.strength)
    steps = schedule.transformer_evaluations
    seed = payload.seed
    tel.log("h3 regenerate", resolution=resolution, strength=payload.strength, quality=payload.quality, start_sigma=schedule.start,
            transformer_evaluations=steps, latent="x".join(str(int(value)) for value in latents.shape[2:]),
            windows=len(layout), window_latent_frames=stop - first)  # fmt: skip
    with tel.stage("refine"):
        refined = base_model.refine(
            latents, overlay=regenerate_lora, prompt_embeds=text[0], text_tags=text[1], layout=layout,
            audio=base_model.silence(refine_math.frames_of(stop - first)), schedule=schedule, seed=seed,
            quality=payload.quality,
            on_step=lambda step: ctx.progress(step, steps),
        )  # fmt: skip
    cancel()
    (height, width), (keep, keep_width) = refine_math.target(resolution, draft)
    top, left = (height - keep) // 2, (width - keep_width) // 2
    with tel.stage("decode_video"):
        stream = _VideoStream(
            torch, out, frames=frames, waveform=waveform, sample_rate=sample_rate, cancel=cancel,  # type: ignore[possibly-undefined]  # read with the frames
            progress=ctx.progress,
        )  # fmt: skip
        delivered = 0

        def push(chunk: Tensor) -> None:
            nonlocal delivered
            take = min(int(chunk.shape[0]), frames - delivered)
            if take > 0:
                stream.push(chunk[:take, :, top : top + keep, left : left + keep_width])
                delivered += take

        try:
            base_model.decode_latents(refined, on_chunk=push)
        except BaseException:
            stream.abandon()
            raise
    try:
        with tel.stage("encode_video"):
            saved = stream.finish(ctx.progress)
    except BaseException:
        stream.abandon()
        raise
    tel.log("h3 container facts", video_codec=saved.video_codec, frame_count=saved.frame_count,
            frame_rate=str(saved.frame_rate), video_bytes=saved.video.size_bytes)  # fmt: skip
    ctx.release_gpus()
    return RegenerateOutput(saved.video, warnings)


# --- the three tools ---------------------------------------------------------------------
# A tool is a CPU parent on the machine its render landed on. A shot renders in segments of at most 15 s, each going on
# from the last from its carried latents, never decoded; a segment is an ordinary serving child memoized on its full
# inputs (the previous segment's context included), so a resubmitted shot skips the segments it already finished.

#: A kept take reaches past the beats if less than this of them is left.
_REACHES_S = 1.0
#: The model's own tags in its text: `<d>` around dialogue, `<scenetrans>`, `<cutoff>`, and the language a line is spoken
#: in, `<d><English>Hello<d>`. MiniMax's guide lists no languages: these are the ones people write dialogue in.
_RESERVED = [
    "d", "scenetrans", "cutoff",
    "english", "chinese", "mandarin", "cantonese", "japanese", "korean", "spanish", "french", "german", "italian", "portuguese",
    "russian", "arabic", "hindi", "bengali", "urdu", "punjabi", "tamil", "telugu", "indonesian", "malay", "vietnamese", "thai",
    "filipino", "tagalog", "turkish", "persian", "hebrew", "greek", "dutch", "swedish", "norwegian", "danish", "finnish", "polish",
    "czech", "ukrainian", "romanian", "hungarian", "swahili",
]  # fmt: skip


def _anchor(name: str, people: bool) -> StoryReference:
    """A picture cozy adds to a segment: the frame it starts from, as a world state."""
    return StoryReference(name=name, kind="world_state",
                          description=f"This segment starts from it: {'who and what' if people else 'what'} is where now.")  # fmt: skip


async def _render(ctx: Context, shot: Shot, out: Outputs, tel: Telemetry, *, plate: VideoFile | None = None,
                  lora: ModelChoice | None = None, keep_latents: bool = True) -> ShotResult:  # fmt: skip
    """A shot's segments, rendered in order, each going on from the last from its carried latents, decoded as one video
    and published as it grows. A pickup goes on from its take at its frame: from the take's latents, else as a new video
    opening on that frame, joined after the take's kept start. A `plate` is followed frame by frame, and its length is
    the shot's. `lora` serves every segment. Without `keep_latents`, the video's latents aren't kept."""
    ctx.raise_if_cancelled()
    warnings: list[str] = []
    # Asking for steps asks for the full model, the only one that has a choice of them.
    turbo = shot.turbo and shot.steps is None
    if shot.turbo and not turbo:
        warnings.append(f"steps {shot.steps} runs the full model, not turbo's 8 steps")
    pickup, opening, ending = shot.pickup, shot.first_frame, shot.last_frame
    if pickup is not None and pickup.out_frame < MIN_FRAMES:
        warnings.append(f"Pick up from here at frame {pickup.out_frame} is before the shortest piece, {MIN_FRAMES} frames: "
                        "the whole shot renders")  # fmt: skip
        pickup = None
    if opening is not None and (plate is not None or pickup is not None):
        warnings.append(f"{'the plate' if plate is not None else 'the take picked up'} sets the opening; first_frame is ignored")
        opening = None
    take = pickup.take if pickup is not None else None
    latents: File | None = take.latents if take is not None else None
    if take is not None and latents is not None and (why := unusable(take)):
        warnings.append(f"this take's latents can't be used ({why}), so the new part opens on its frame and the join may show")
        latents = None
    carry, done, trimmed, offset = Carry(shot.carried_frames, True), 0, False, 0.0
    if pickup is not None and take is not None:
        if latents is not None:
            end = shot_latents.delivered(latents.read_bytes())
            done = min(shot_latents.nearest_join(pickup.out_frame), end)
        else:
            end = math.ceil(take.file.probe().duration * FPS)
            done = min(pickup.out_frame, end)
        if latents is not None and done < end:
            latents, trimmed = cut_latents(take, done, out, carry), True
        elif latents is not None and (own_frames := decode_context(shot_latents.context(latents.read_bytes())).carry.frames) not in (0, carry.frames):
            # Going on from a take's end takes the carry that take kept.
            warnings.append(f"the take carries {own_frames} frames, so this does too, not {carry.frames}")
            carry = Carry(own_frames, True)
        if latents is None:
            warnings.append("this take has no latents here, so the new part is a new video opening on its frame, joined "
                            "after its kept start; the join may show")  # fmt: skip
            opening = frame_at(take.file, done, out)
        beats_from = pickup.beats_from if pickup.beats_from is not None else done / FPS if take.own else 0.0
        offset = beats_from - done / FPS
    script = written(shot, offset=offset, length=None if plate is None else float(plate.probe().duration))
    warnings += script.warnings
    begun, duration_s, carried = done / FPS, script.duration_s, latents is not None
    if take is not None and duration_s - begun < _REACHES_S:
        raise InvalidRequest(f"the beats end at {duration_s:.2f} s and this picks up at {begun:.2f} s, so nothing is made: "
                             "add a beat, or pick up earlier", fields=["pickup"])  # fmt: skip
    timed, timing = blocks(prompt_parts(script.prompt)[1], duration_s)
    segments, opens, planned = plan_segments(duration_s - begun, [block.at - begun for block in timed if block.at > begun + JOIN_S],
                                             carried=carried, carry=carry, split=plate is not None)  # fmt: skip
    warnings += [*timing, *planned]
    count = len(segments)
    # The shot's canvas: a take it goes on from or a plate sets it; else the shape asked for; else Auto: its first or
    # last frame's, else 16:9. A reference picture never sets it.
    holder = take.file if carried and take is not None else plate
    asked = None if shot.aspect_ratio is None else canvas(ratio_frame(shot.aspect_ratio), count > 1)
    shape = NATIVE
    if holder is not None:
        probed = holder.probe()
        tall, wide = clip_size(shown_width(probed.width, probed.pixel_aspect_ratio), probed.height, guide=True)
        shape = (wide, tall)
    elif asked is not None and take is None:
        shape, moved = asked
        warnings += moved
    elif (keyframe := opening or ending) is not None:
        shape, moved = canvas((keyframe.probe().width, keyframe.probe().height), count > 1)
        warnings += moved
    if asked is not None and asked[0] != shape:
        warnings.append(f"{'the plate' if plate is not None else 'the take'} is {shape[0]}x{shape[1]} and sets the shape, "
                        f"not {shot.aspect_ratio}")  # fmt: skip
    for name, keyframe in (("first_frame", shot.first_frame if opening is shot.first_frame else None), ("last_frame", ending)):
        if keyframe is not None:
            warnings += fitted(name, (keyframe.probe().width, keyframe.probe().height), shape)
    retained = own_retention(script.prompt)
    if not FIRST_SHOT.search(script.prompt):
        warnings.append("this prompt has no `[Shot 1]`, which MiniMax's guide opens a shot with; it goes as written")
    steps = None if turbo else shot.steps
    # Every segment carries into a successor, its last too: the video's latents keep that carry, which a pickup resumes.
    plans = [ContinuationPlan(length, carry if index or carried else Carry(), carry) for index, length in enumerate(segments)]
    *starts, total = accumulate((plan.delivered_frames for plan in plans), initial=done)
    followed = StoryReference(name="Plate", kind="anchor_video", video=plate) if plate is not None else None
    clips = [(StoryReference(name=name, kind="video", video=video), start, end) for name, (video, start, end) in script.clips.items()]
    windows, notes = shot_windows([*clips, *([(followed, 0.0, math.inf)] if followed else [])],
                                  [(start - plan.prefix_frames, plan.sample_frames) for start, plan in zip(starts, plans, strict=True)])  # fmt: skip
    warnings += notes
    opened = [begun + at for at in opens]
    texts = segment_texts(script.prompt, opened, continued=take is not None, carried=carried, duration_s=duration_s, carry=carry)
    files: dict[str, ImageFile | VideoFile | AudioFile] = {}
    calls: list[SegmentCall] = []
    for index, (text, start, until) in enumerate(zip(texts, opened, [*opened[1:], math.inf], strict=True)):
        own = [ref for ref in here(script, start, until) if ref.kind not in CLIPS or ref.name in windows[index]]
        # A later segment starts from the previous segment's last frame as a world state, after the shot's own pictures
        # so their <Picture N> keep their numbers: who and what is where now (se-087: 9 of 9 seams joined cleanly).
        anchor = [_anchor(f"Segment {index} end", script.people)] if index else []
        refs, added = [*own, *anchor, *([followed] if followed else [])], frozenset(ref.name for ref in anchor)
        forced = added | ({"Plate"} if followed else set())
        role = {"keyframes": bool(index == 0 and opening or index + 1 == count and ending), "continues": index == 0 and carried}
        segment = segment_call(text, refs, soundscape=script.soundscape, music=script.music, index=index,
                               retained=retained, added=added, forced=forced, **role)  # fmt: skip
        if index == 0 and (plate is not None or carried) and not any(FILE[ref.kind] == "image" for ref in segment.references):
            # Ref2VA needs a picture: a plate's own first frame (se-101), or the frame the take goes on from (h3a-098,
            # run 25881), as the same world state.
            begin = _anchor("Plate opening" if plate is not None else "Segment 0 end", script.people)
            if plate is not None:
                warnings.append("a plate with no picture beside it uses its own first frame as the picture")
            files[begin.name] = frame_at(plate, 1, out) if plate is not None else frame_at(cast(ShotVideo, take).file, done, out)
            segment = segment_call(text, [*refs, begin], soundscape=script.soundscape, music=script.music, index=index,
                                   retained=retained, added=added | {begin.name}, forced=forced | {begin.name}, **role)  # fmt: skip
        files |= {ref.name: file for ref in segment.references if (file := getattr(ref, FILE[ref.kind])) is not None}
        calls.append(segment if segment.references else SegmentCall(fl2va_text(text, script.soundscape, script.music), [], segment.warnings))
        warnings += segment.warnings
    tel.log("h3 shot segments", segments=count, seconds=",".join(map(str, segments)), resumed_at=done,
            boundaries_s=",".join(f"{start / FPS:.2f}" for start in starts[1:]), delivered_frames=total,
            clip_windows=";".join(",".join(f"{name}@{a}-{b}" for name, (a, b) in w.items()) for w in windows))  # fmt: skip
    if not calls[0].references and any(later.references for later in calls[1:]):
        # Only a text-only first segment (FL2VA) runs ahead of a Ref2VA one. A hint the shot never uses loads on after
        # it ends and holds the GPU from the next run (0.2.0 gate 41159884913: h3-5s, then upscale waited 3 h 50 min).
        prefetch(ref2va_segment)
    seeds = [segment_seed(shot.seed, ctx.request_id, index) for index in range(count)]
    # A take's latents and its video join this one's at its last segment, decoded as one; a trimmed take's sound is in its latents.
    before = ((latents,), () if trimmed else (cast(ShotVideo, take).file,)) if latents is not None else ((), ())
    joined = count > 1 or carried
    context: File | None = latents
    expected = None
    models = {} if lora is None else {"base_model": lora}
    returned: list[SegmentOutput] = []
    so_far: SoFar | None = None
    for index, (segment, plan) in enumerate(zip(calls, plans, strict=True)):
        ctx.raise_if_cancelled()
        first = opening if index == 0 else None
        last = ending if index + 1 == count else None
        end_segment = joined and index + 1 == count
        try:
            with tel.scope(f"Segment {index + 1} of {count}"):
                log_prompt(tel, "h3 segment prompt", segment.prompt, segment=index + 1,
                           references=", ".join(ref.name for ref in segment.references))  # fmt: skip
                if segment.references:
                    uses = video_uses(segment.references, windows[index])
                    pictures = [ref for ref in segment.references if FILE[ref.kind] == "image"]
                    result = await call(
                        ref2va_segment,
                        payload=Ref2VAInput(
                            prompt=segment.prompt, seed=seeds[index], duration_s=segments[index], steps=steps,
                            turbo=turbo, quality=shot.quality, frames=held_frames(plan, uses), videos=uses,
                            previous_provenance=expected, context=context if plan.carried else None,
                            context_frames=cast(ContextFrames, plan.carried.frames),
                            context_audio=plan.carried.audio,
                            next_context_frames=cast(ContextFrames, plan.successor.frames),
                            next_context_audio=plan.successor.audio, first_frame=first, last_frame=last, canvas=shape,
                            latents=True,
                            earlier_latents=(*before[0], *(r.latents for r in returned)) if end_segment else (),
                            earlier_videos=(*before[1], *(r.video for r in returned)) if end_segment else (),
                            picture_edges=tuple(_LOOSE_EDGE if ref.kind == "world_state" else 0 for ref in pictures),
                            characters=tuple(ref.name if ref.kind == "character" else "" for ref in pictures),
                        ), assets=_call_assets(files, segment.references), **models,
                    )  # fmt: skip
                else:
                    result = await call(
                        fl2va_segment,
                        payload=FL2VAInput(
                            prompt=segment.prompt, seed=seeds[index], duration_s=segments[index], steps=steps,
                            turbo=turbo, quality=shot.quality, first_frame=first, last_frame=last, canvas=shape,
                            next_context_frames=cast(ContextFrames, plan.successor.frames),
                            next_context_audio=plan.successor.audio, latents=True,
                        ), **models,
                    )  # fmt: skip
        except ChildCallError:
            # Concurrent caller cancellation must not become a successful partial result.
            ctx.raise_if_cancelled()
            raise
        expected = result.provenance
        context = result.context
        returned.append(result)
        warnings += result.warnings
        files[f"Segment {index + 1} end"] = result.last_frame
        # Each segment shows as it lands, after those before it, so its box plays all made so far while the next renders.
        if index + 1 == count and (take is None or carried):
            out.publish("video", result.video, label="Video")
        else:
            so_far = so_far or SoFar(out, tel, result.video)
            out.publish("video", so_far.add(result.video), label=f"Segments 1-{index + 1} of {count}" if index else f"Segment 1 of {count}")
    video, kept = returned[-1].video, None
    if take is not None and not carried:
        video, said = join_kept(out, take.file, done, video)
        warnings += said
        out.publish("video", video, label="Video")
    elif keep_latents:
        kept = shot_latents_file(out, [*before[0], *(r.latents for r in returned)], total, returned[-1].context, upscale_prompt(script))
    ctx.release_gpus()
    tel.log("h3 shot", segments=count, delivered_frames=total, turbo=turbo, quality=shot.quality)
    return ShotResult(video=video, warnings=list(dict.fromkeys(warnings)), latents=cast("Latents | None", kept),  # a saved file fills a Latents field
                      kept_s=begun if take is not None else None)  # fmt: skip


@app.tool(label="Shot", named="shot", accelerator=False, reserved=_RESERVED)
async def generate(ctx: Context, payload: Shot, out: Outputs, tel: Telemetry) -> ShotResult:
    """One continuous shot of any length, with sound (README.md): a summary, then what happens, beat by beat, each text
    with the references it mentions; an optional first and last frame; and Pick up from here, which goes on from a take
    at a frame. Returns the video and its latents, which a pickup of it resumes and an upscale starts from."""
    return await _render(ctx, payload, out, tel)


class ShotLoRA(msgspec.Struct, frozen=True, kw_only=True):
    #: `[org/]name@release`.
    model: str
    strength: float = 1.0


class PackageShot(msgspec.Struct, kw_only=True):
    """A shot another package makes, with its LoRAs."""

    shot: Shot
    loras: list[ShotLoRA] = msgspec.field(default_factory=list)


class PackageShotResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, media_types=("video/mp4",))]
    warnings: list[str]


@app.entrypoint(accelerator=False)
async def render_shot(ctx: Context, payload: PackageShot, out: Outputs, tel: Telemetry) -> PackageShotResult:
    """`generate` for another package (`orbit`): its LoRAs on every segment after Turbo's, as `edit` puts its edit LoRA
    on, and only the video back."""
    choice = ModelChoice(lora=tuple(Lora(lora.model, lora.strength) for lora in payload.loras)) if payload.loras else None
    made = await _render(ctx, payload.shot, out, tel, lora=choice, keep_latents=False)
    return PackageShotResult(made.video, made.warnings)


class EditLoRA(NamedTuple):
    model: str
    trigger: str
    strength: float = 1.0


#: Alissonerdx's guide-edit LoRAs on Ref2VA, each taking the plate through `MiniMaxH3AddGuide` at frame 0
#: (hf://Alissonerdx/Minimax-H3-ComfyUI README.md: its LoRA table gives each trigger, and line 97 the guide), at the
#: Hub's mirror releases, whose cards carry the same triggers. Each one's guide starts at strength 1.0
#: (docs/character-swap.md:14, head-swap.md:28, style-transfer.md:13, vfx-edit.md:6, motion-transfer.md:55). Live
#: Wallpaper animates pictures without a plate, so it isn't an edit.
EDITS: dict[str, EditLoRA] = {
    "character_swap": EditLoRA("alissonerdx/minimax-h3-character-swap@0.1.0", "body_swap:"),
    "head_swap": EditLoRA("alissonerdx/minimax-h3-head-swap@0.1.0", "head_swap:"),
    "style_transfer": EditLoRA("alissonerdx/minimax-h3-style-transfer@0.1.0", "style_transfer:"),
    "vfx_edit": EditLoRA("alissonerdx/minimax-h3-vfx-edit@0.1.0", "vfx_edit:"),
    "motion_transfer": EditLoRA("alissonerdx/minimax-h3-motion-transfer@0.1.0", "motion_transfer:"),
}
ShotFootage = Annotated[VideoFile, FileBound(max_decoded_bytes=VIDEO_READ_BYTES)]


class Edit(msgspec.Struct, kw_only=True):
    """A plate edited by its beats under one of Alissonerdx's edit LoRAs: the plate is followed frame by frame, and its
    length is the edit's."""

    plate: Annotated[ShotFootage, Spans("beats")]
    # Each LoRA's words and what it does, from its row of the README's LoRA table (lines 28-34).
    edit: Annotated[
        Literal["character_swap", "head_swap", "style_transfer", "vfx_edit", "motion_transfer"],
        Label("Edit", options={
            "character_swap": ("Character swap", "Replace the main guide subject with the character in a reference image "
                                                 "while following the guide performance"),
            "head_swap": ("Head swap", "Experimental reference-image head replacement over an aligned source video"),
            "style_transfer": ("Style transfer", "Transfer a visual style from an image or text description onto a video"),
            "vfx_edit": ("VFX edit", "General prompt-directed VFX editing with an aligned source video"),
            "motion_transfer": ("Motion transfer", "The character in the reference performs the guide video's motion"),
        }),
    ]  # fmt: skip
    beats: Annotated[list[Beat], Count(min=1)]
    turbo: bool = True
    quality: Quality = "standard"
    seed: Seed | None = None


@app.tool(label="Edit", named="edit", accelerator=False, reserved=_RESERVED)
async def edit(ctx: Context, payload: Edit, out: Outputs, tel: Telemetry) -> ShotResult:
    """A plate edited by its beats, on Ref2VA under the chosen edit LoRA, its trigger opening each beat. With no picture
    in a beat, the plate's own first frame is the picture."""
    lora = EDITS[payload.edit]
    beats = [replace(beat, description=replace(beat.description, text=f"{lora.trigger} {beat.description.text}".rstrip()))
             for beat in payload.beats]  # fmt: skip
    shot = Shot(beats=beats, turbo=payload.turbo, quality=payload.quality, seed=payload.seed)
    choice = ModelChoice(lora=(Lora(lora.model, lora.strength),))
    return await _render(ctx, shot, out, tel, plate=payload.plate, lora=choice)


class Upscale(msgspec.Struct, kw_only=True):
    """A take re-rendered larger by NVIDIA's Regenerate-2K: from its latents when H3 made it, else from its frames and
    description."""

    video: ShotVideo
    to: Literal["1080p", "1440p", "2160p"] = "1440p"
    #: How much of the take the refine redraws, as its start noise: 0.7, 0.8 or 0.909 (NVIDIA's).
    strength: Annotated[float, Range(0.7, 0.909), Advanced()] = REGENERATE_STRENGTH


class UpscaleOutput(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, media_types=("video/mp4",))]
    warnings: list[str] = msgspec.field(default_factory=list)


@app.tool(label="Upscale", named="upscale", accelerator=False)
async def upscale(ctx: Context, payload: Upscale, out: Outputs, tel: Telemetry) -> UpscaleOutput:
    """A take at 1080p, 1440p or 2160p, its sound kept (UPSCALE.md): its 768p latents upscaled and refined by H3 under
    the Regenerate LoRA, from the text they carry, else its frames from its description. A video already that big comes
    back as it is."""
    video, warnings = payload.video, []
    probe = video.file.probe()
    if min(probe.width, probe.height) >= int(payload.to[:-1]):
        warnings.append(f"this video is {probe.width}x{probe.height}, already {payload.to}: it comes back as it is")
        with video.file.stream() as events:
            return UpscaleOutput(out.save_video_stream(iter(events)).video, warnings)
    # The text the video was made from: its latents carry it whole, else its description, names without brackets.
    carried = shot_latents.text(video.latents.read_bytes()) if video.latents is not None else ""
    prompt = carried or re.sub(r"<([^<>]+)>", r"\1", video.description).strip()
    if not prompt:
        raise InvalidRequest("this video has no latents from H3 and no description, and Regenerate re-renders it from the "
                             "text it was made from: give it a description", fields=["video"])  # fmt: skip
    strength = min(REGENERATE_STRENGTHS, key=lambda served: abs(served - payload.strength))
    if round(payload.strength, 3) != strength:
        warnings.append(f"strength {payload.strength} runs at {strength}, the nearest served")
    seed = int.from_bytes(hashlib.sha256(f"{video.file.digest or video.file.ref}/h3/upscale".encode()).digest()[:4], "big")
    request = RegenerateInput(video=video.file, latents=video.latents, prompt=prompt, resolution=Resolution(payload.to),
                              strength=strength, seed=seed)  # fmt: skip
    with tel.scope(f"Regenerate at {payload.to}"):
        result = await call(regenerate_shot, payload=request)
    warnings += result.warnings
    tel.log("h3 upscale", to=payload.to, strength=strength, warnings="; ".join(warnings))
    return UpscaleOutput(result.video, warnings)
