"""H3 segment continuation: the previous segment's sampled AV latents, carried across a join.

A segment carries a picture window and/or one second of sound, sliced from its predecessor's
denoised latents: no decode, no re-encode. A picture window is whole VAE cycles ending at the
predecessor's last sampled frame, so its tokens keep their 1/4/4/4/4 phase. They are condition
rows on the target head, which is regenerated and cut: the owner's double form (2026-10-10),
in the denoise window and as a reference, never one copy (tag
archive/h3-single-copy-context-20261010). The sound is the predecessor's last 40 audio ticks,
placed on this clip's timeline exactly where they ended on the predecessor's.

Reference: NikoDemon80/ComfyUI-H3-Motion-Context and AIMixer/ComfyUI_MiniMaxH3_Director
(latent tails, cycle-aligned windows, a 24-frame sound window ending at the join).
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, NoReturn, cast

import msgspec
import torch
from safetensors.torch import load, save

from cozy_runtime.author import InvalidRequest

FPS = 24
#: Audio latents run at 40 Hz: a video frame is 5/3 of an audio tick.
TICKS_PER_FRAME = Fraction(5, 3)
#: Picture windows of whole VAE cycles (17k + 5 frames).
CONTEXT_FRAMES = (22, 39, 56)
#: One second of sound, 24 frames: exactly 40 ticks, whatever the picture window.
AUDIO_FRAMES = 24
AUDIO_TICKS = 40
MAX_FRAMES = 362
MAX_CONTEXT_BYTES = 16 << 20
MAX_CONTEXT_PIXELS = 1344 * 768
CONTEXT_SCHEMA = "cozy.minimax-h3.av-context/3"


def _refuse(message: str) -> NoReturn:
    raise InvalidRequest(message, code="h3_continuation", fields=["context"])


def video_tokens(frames: int) -> int:
    """Video latents of a `17k + 5` frame run."""
    return 5 * ((frames - 5) // 17) + 2


def audio_ticks(frames: int) -> int:
    """H3's audio latents for `frames`: 5/3 a frame, rounded to the nearest tick."""
    return (5 * frames + 1) // 3


@dataclass(frozen=True)
class Carry:
    """What a segment takes from its predecessor: a picture window and/or a second of sound."""

    frames: int = 0
    audio: bool = False

    def __post_init__(self) -> None:
        if type(self.frames) is not int or self.frames not in (0, *CONTEXT_FRAMES):
            _refuse("a carried picture window is 0, 22, 39 or 56 frames")
        if type(self.audio) is not bool:
            _refuse("carried sound is a boolean")

    def __bool__(self) -> bool:
        return bool(self.frames or self.audio)

    @property
    def head_frames(self) -> int:
        """The regenerated, cut head this carry needs: the picture window, or one second for
        sound alone. Before the target start H3 reads carried sound as a reference to imitate
        (run 5485: 20 dB quieter, not continued); over the head it is this clip's own sound."""
        return self.frames or (AUDIO_FRAMES if self.audio else 0)


@dataclass(frozen=True)
class ContinuationPlan:
    """One segment's clocks. It samples its carry's head plus `seconds` on the `17n + 5` grid
    (snapped up, as `frames_for` does) and cuts the regenerated head. Its
    picture ends at the last sampled frame, where a successor's picture window must end;
    a successor carrying sound alone needs the sound to reach the join, so the picture
    then ends on the last frame the sound covers (one frame earlier when the grid rounds
    the sound down)."""

    seconds: int
    carried: Carry = Carry()
    successor: Carry = Carry()

    def __post_init__(self) -> None:
        if type(self.seconds) is not int or not 5 <= self.seconds <= 15:
            _refuse("a segment delivers 5-15 new seconds")
        if self.sample_frames > MAX_FRAMES:
            _refuse(
                f"{self.seconds} s plus {self.prefix_frames} carried frames samples "
                f"{self.sample_frames} frames, above the {MAX_FRAMES}-frame envelope; "
                "shorten the segment"
            )

    @property
    def prefix_frames(self) -> int:
        return self.carried.head_frames

    @property
    def sample_frames(self) -> int:
        frames = self.prefix_frames + FPS * self.seconds
        return frames + (5 - frames) % 17

    @property
    def delivery_end(self) -> int:
        if self.successor.audio and not self.successor.frames:
            return min(self.sample_frames, 3 * audio_ticks(self.sample_frames) // 5)
        return self.sample_frames

    @property
    def delivered_frames(self) -> int:
        return self.delivery_end - self.prefix_frames

    @property
    def sound_end(self) -> Fraction:
        """Where the delivered sound ends, in ticks: at the picture's end, or, when the
        successor carries it, on the first whole tick at or after it (the last sampled one
        at most), so the carried second is whole ticks within a third of the join."""
        picture = TICKS_PER_FRAME * self.delivery_end
        if not self.successor.audio:
            return picture
        return Fraction(min(audio_ticks(self.sample_frames), math.ceil(picture)))

    @property
    def exit_offset(self) -> Fraction:
        """Where the carried second of sound ends past the picture join, in ticks."""
        return self.sound_end - TICKS_PER_FRAME * self.delivery_end

    def audio_samples(self, entry: Fraction, sample_rate: int) -> tuple[int, int]:
        """The delivered sound: from where the predecessor's ended (`entry` ticks past the
        picture join, carried sound only) to `sound_end`."""
        start = TICKS_PER_FRAME * self.prefix_frames + (entry if self.carried.audio else 0)
        end = self.sound_end
        if start < 0:
            _refuse("carried sound ends before this segment begins")
        per_tick = Fraction(sample_rate, 40)
        return round(start * per_tick), round(end * per_tick)


@dataclass(frozen=True)
class AVContext:
    """A successor's carry in the model's normalized space: the last picture window
    `(1, 24, tokens, h, w)` and/or the last second of sound as packed channel-major audio
    rows `(80, 32)`, ending `audio_offset` ticks past the picture join."""

    video: torch.Tensor | None
    audio: torch.Tensor | None
    audio_offset: Fraction
    provenance: str

    def __post_init__(self) -> None:
        self.validate()

    @property
    def carry(self) -> Carry:
        tokens = None if self.video is None else int(self.video.shape[2])
        frames = next((n for n in CONTEXT_FRAMES if video_tokens(n) == tokens), 0)
        return Carry(frames, self.audio is not None)

    @property
    def canvas(self) -> tuple[int, int] | None:
        """The picture's (height, width) in pixels."""
        if self.video is None:
            return None
        return int(self.video.shape[3]) * 16, int(self.video.shape[4]) * 16

    def validate(self) -> None:
        if self.video is None and self.audio is None:
            _refuse("context carries neither picture nor sound")
        for tensor in (self.video, self.audio):
            if tensor is not None and (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or not bool(torch.isfinite(tensor).all())
            ):
                _refuse("context must hold finite float32 CPU latents")
        if self.video is not None:
            shape = tuple(self.video.shape)
            canvas = self.canvas
            if (
                len(shape) != 5
                or shape[:2] != (1, 24)
                or shape[2] not in {video_tokens(n) for n in CONTEXT_FRAMES}
                or canvas is None
                or min(canvas) < 32
                or canvas[0] % 32
                or canvas[1] % 32
                or canvas[0] * canvas[1] > MAX_CONTEXT_PIXELS
            ):
                _refuse("context picture is outside the qualified H3 window or canvas")
        if self.audio is not None and tuple(self.audio.shape) != (2 * AUDIO_TICKS, 32):
            _refuse("context sound must be one second of packed stereo audio latents")
        offset = self.audio_offset
        if (
            not isinstance(offset, Fraction)
            or 3 % offset.denominator
            or not -Fraction(1, 3) <= offset <= Fraction(1, 3)
            or (self.audio is None and offset)
        ):
            _refuse("context sound offset is off the frame/tick grid")
        if (
            not isinstance(self.provenance, str)
            or not self.provenance
            or len(self.provenance.encode("utf-8")) > 16384
        ):
            _refuse("context requires bounded renderer provenance")


@dataclass(frozen=True)
class ContextReferenceLayout:
    """Private packed-layout entry for the carried latents, after every public reference."""

    kind: str
    has_audio: bool

    @classmethod
    def of(cls, context: AVContext) -> ContextReferenceLayout:
        return cls("audio" if context.video is None else "video", context.audio is not None)


def export_context(
    video: torch.Tensor, audio: torch.Tensor, plan: ContinuationPlan, provenance: str
) -> AVContext | None:
    """The successor's carry, sliced from this segment's denoised latents: video
    `(1, 24, T, h, w)` and audio `(2, 32, N)`, both normalized."""
    carry = plan.successor
    if not carry:
        return None
    window = rows = None
    if carry.frames:
        tokens = video_tokens(carry.frames)
        total = int(video.shape[2])
        if total != video_tokens(plan.sample_frames) or (total - tokens) % 5:
            _refuse("picture window is not whole VAE cycles at the sampled end")
        window = video[:, :, total - tokens :].detach().float().cpu().contiguous()
    if carry.audio:
        if tuple(audio.shape) != (2, 32, audio_ticks(plan.sample_frames)):
            _refuse("sampled audio latents differ from the segment's clock")
        end = int(plan.sound_end)
        tail = audio[:, :, end - AUDIO_TICKS : end].permute(0, 2, 1).reshape(-1, 32)
        rows = tail.detach().float().cpu().contiguous()
    return AVContext(window, rows, plan.exit_offset, provenance)


def encode_context(context: AVContext) -> bytes:
    context.validate()
    tensors = {
        name: tensor
        for name, tensor in (("video", context.video), ("audio", context.audio))
        if tensor is not None
    }
    metadata = {
        "schema": CONTEXT_SCHEMA,
        "audio_offset": str(context.audio_offset),
        "provenance": context.provenance,
    }
    data = save(tensors, metadata=metadata)
    if len(data) > MAX_CONTEXT_BYTES:
        _refuse("encoded context exceeds its 16 MiB bound")
    return cast(bytes, data)


def decode_context(data: bytes) -> AVContext:
    if not isinstance(data, bytes) or not 8 <= len(data) <= MAX_CONTEXT_BYTES:
        _refuse("context carrier is absent or exceeds its 16 MiB bound")
    length = struct.unpack("<Q", data[:8])[0]
    if not 2 <= length <= min(65536, len(data) - 8):
        _refuse("context header is invalid or exceeds its bound")
    try:
        metadata = json.loads(data[8 : 8 + length]).get("__metadata__", {})
        if metadata.get("schema") != CONTEXT_SCHEMA or not {"audio_offset", "provenance"} <= set(
            metadata
        ):
            _refuse("context schema or fields do not match")
        # Bounded, uncompressed safetensors validates offsets before allocation.
        tensors = load(data)
        if not tensors.keys() <= {"video", "audio"}:
            _refuse("context holds unexpected tensors")
        return AVContext(
            tensors.get("video"),
            tensors.get("audio"),
            Fraction(metadata["audio_offset"]),
            metadata["provenance"],
        )
    except InvalidRequest:
        raise
    except Exception as exc:
        raise InvalidRequest(
            "invalid completed H3 AV context", code="h3_continuation", fields=["context"]
        ) from exc


def align_context_rows(state: Any, context: AVContext, patch_size: tuple[int, int, int]) -> None:
    """Put the carried rows, the last condition rows, on this clip's timeline.

    The picture tokens sit on the target head they condition, in phase. The sound ends at
    the picture join plus its offset, on fractional rotary positions where the grids
    disagree, so it continues exactly where it stopped. The native scheduler keeps both
    fixed on every step.
    """
    origin = state.position_ids[state.video_indices[state.num_condition_video_rows], 0].clone()
    if context.video is not None:
        _, patch_h, patch_w = patch_size
        _, _, frames, height, width = (int(n) for n in context.video.shape)
        count = frames * (height // patch_h) * (width // patch_w)
        if count > state.num_condition_video_rows:
            _refuse("context packed picture geometry is inconsistent")
        rows = state.video_indices[
            state.num_condition_video_rows - count : state.num_condition_video_rows
        ]
        state.position_ids[rows, 0] += origin - state.position_ids[rows[0], 0].clone()
    if context.audio is not None:
        count = int(context.audio.shape[0])
        if count > state.num_condition_audio_rows:
            _refuse("context packed sound geometry is inconsistent")
        rows = state.audio_indices[
            state.num_condition_audio_rows - count : state.num_condition_audio_rows
        ]
        ticks = TICKS_PER_FRAME * context.carry.head_frames + context.audio_offset - AUDIO_TICKS
        state.position_ids[rows, 0] += (
            origin + float(ticks) - state.position_ids[rows[0], 0].clone()
        )


class DeliveredFrames:
    """A streamed decode's delivered picture: frames `[prefix, delivery_end)` of the sample."""

    def __init__(self, plan: ContinuationPlan) -> None:
        self.plan = plan
        self.seen = 0
        self.delivered = 0

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 4 or chunk.shape[1] != 3:
            _refuse("continuation decode stream is invalid")
        start = max(0, self.plan.prefix_frames - self.seen)
        end = min(int(chunk.shape[0]), self.plan.delivery_end - self.seen)
        self.seen += int(chunk.shape[0])
        selected = chunk[start : max(start, end)]
        self.delivered += int(selected.shape[0])
        return selected

    def finish(self) -> int:
        if self.seen != self.plan.sample_frames or self.delivered != self.plan.delivered_frames:
            _refuse("completed continuation decode differs from its sampled/delivered frame plan")
        return self.delivered


class RenderProvenance(msgspec.Struct, frozen=True):
    """The model selections a piece rendered with: a carried context names them."""

    model_manifest: str
    turbo_lora_manifest: str = ""


def provenance(model_manifest: str, turbo_lora_manifest: str = "") -> RenderProvenance:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", model_manifest):
        raise InvalidRequest("H3 rendering needs its exact model manifest", code="render_model")
    if turbo_lora_manifest and not re.fullmatch(r"sha256:[0-9a-f]{64}", turbo_lora_manifest):
        raise InvalidRequest("H3 turbo needs its exact adapter manifest", code="render_model")
    return RenderProvenance(model_manifest, turbo_lora_manifest)


def context_provenance(renderer: RenderProvenance) -> str:
    """Record model selections for native context compatibility without code hashes."""
    return msgspec.json.encode(renderer).decode("utf-8")
