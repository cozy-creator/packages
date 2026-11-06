"""A shot's latents as a file: what `generate` returns, `upscale` starts from (Regenerate refines
the shot's own latent, as NVIDIA's `infer.py --input-latents` does, instead of re-encoding its
decoded video) and a pickup resumes.

The file is safetensors: `latents`, the normalized `(1, 24, T, h, w)` video latents in float16 on
the VAE's `5n + 2` grid; `context`, its last segment's carry (encoded `AVContext`), which a
continuation's first segment carries in; and metadata naming the format, the latent space
(`vae_space`), the frames the shot delivered (the grid's `17n + 5` frames, or fewer when its
duration was trimmed) and the text it was made from (`text`), which an upscale re-renders it from.

A shot cut at a join (a pickup's `out_frame`, `cut`) is the same file: its latents up to there, the carry there from those
latents, and `sound`, the sound that stays, whose last second the carry takes once encoded: the file holds no audio latents
before a shot's end."""

from __future__ import annotations

import hashlib
import json
import struct
from fractions import Fraction
from typing import Any

import torch
from cozy_runtime.author import InvalidRequest
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from .continuation import TICKS_PER_FRAME, AVContext, audio_ticks, decode_context, encode_context, video_tokens
from .official import FRAMES_PER_CHUNK, LATENTS_PER_CHUNK

FORMAT = "minimax-h3/latents/2"
#: H3's sound: 32 kHz, 40 audio latents a second.
SAMPLES_PER_TICK = 800
#: The config the latent space depends on: what the encoder writes and the decoder reads.
_SPACE = ("latent_channels", "latents_mean", "latents_std", "spatial_compression_ratio", "temporal_compression_ratio")


def vae_space(vae: Any) -> str:
    """The latent space's identity: the video VAE's channels, normalization and compression, the
    same for every lane and release that shares them."""
    config = vae.config
    space = {key: config[key] for key in _SPACE if key in config}
    return hashlib.sha256(json.dumps(space, sort_keys=True).encode()).hexdigest()[:16]


def save(latents: torch.Tensor, frames: int, space: str, context: bytes, text: str) -> bytes:
    """A shot's latents, with its last segment's carry (`context`, encoded `AVContext`), what a
    pickup of it carries into its first segment, and the text it was made from (`text`),
    which an upscale that re-renders it reads when given none."""
    carry = torch.frombuffer(bytearray(context), dtype=torch.uint8) if context else torch.zeros(0, dtype=torch.uint8)
    return save_tensors(
        {"latents": latents.detach().to("cpu", torch.float16).contiguous(), "context": carry},
        metadata={"format": FORMAT, "vae_space": space, "frames": str(frames), "text": text},
    )


def context(data: bytes) -> bytes:
    """The carry a segment continues from: a shot latents file's `context`, else `data` itself (a
    segment's own exported context)."""
    if metadata_or_none(data).get("format") != FORMAT:
        return data
    return bytes(load_tensors(data)["context"].numpy().tobytes())


def nearest_join(frames: float) -> int:
    """The grid's `17k + 5` frames nearest `frames`, past a carried second of sound."""
    return FRAMES_PER_CHUNK * max(2, round((frames - 5) / FRAMES_PER_CHUNK)) + 5


def cut(data: bytes, frames: int, carried: int, kept_sound: torch.Tensor) -> bytes:
    """The shot latents file `data` as a shot of `frames` (`nearest_join`) with `kept_sound`, its `(channels, samples)` sound
    up to `audio_ticks(frames)`: a successor's carry is its last `carried` frames there, its own tokens, in phase, whichever
    segment made them."""
    tensors, meta = load_tensors(data), metadata(data)
    tokens = video_tokens(frames)
    latents = tensors["latents"][:, :, :tokens]
    window = latents[:, :, tokens - video_tokens(carried) :].float().contiguous()
    made = decode_context(context(data)).provenance
    carry = encode_context(AVContext(window, None, Fraction(0), made))
    samples = audio_ticks(frames) * SAMPLES_PER_TICK
    pad = samples - int(kept_sound.shape[1])
    # A last frame's sound can fall a little short of the video's clock.
    sound = torch.nn.functional.pad(kept_sound, (0, max(pad, 0)))[:, :samples].contiguous()
    return save_tensors(
        {"latents": latents.contiguous(), "context": torch.frombuffer(bytearray(carry), dtype=torch.uint8), "sound": sound},
        metadata={**meta, "frames": str(frames)},
    )


def sound(data: bytes) -> tuple[torch.Tensor, Fraction] | None:
    """A cut shot's `sound` and how far past its picture it ends, in ticks; None for anything else."""
    kept = load_tensors(data).get("sound")
    if kept is None:
        return None
    frames = int(metadata(data)["frames"])
    return kept, audio_ticks(frames) - TICKS_PER_FRAME * frames


def delivered(data: bytes) -> int:
    """The frames a shot latents file stands for."""
    try:
        meta = metadata(data)
        frames = int(meta["frames"])
    except (struct.error, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise InvalidRequest("pickup.take.latents is not a minimax-h3 latents file", fields=["pickup.take.latents"]) from exc
    if meta.get("format") != FORMAT:
        raise InvalidRequest(f"pickup.take.latents is {meta.get('format')!r}, not {FORMAT}", fields=["pickup.take.latents"])
    return frames


def metadata_or_none(data: bytes) -> dict[str, str]:
    """`metadata`, or nothing for bytes that are not safetensors."""
    try:
        return metadata(data)
    except (struct.error, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def metadata(data: bytes) -> dict[str, str]:
    """A safetensors file's `__metadata__`, read from its header alone."""
    (length,) = struct.unpack("<Q", data[:8])
    return dict(json.loads(data[8 : 8 + length]).get("__metadata__") or {})


def text(data: bytes) -> str:
    """The text the shot was made from; empty when the file carries none."""
    try:
        return metadata(data).get("text", "")
    except (struct.error, ValueError, json.JSONDecodeError) as exc:
        raise InvalidRequest("latents is not a minimax-h3 latents file", fields=["latents"]) from exc


def load(data: bytes, space: str, size: tuple[int, int]) -> tuple[torch.Tensor, int]:
    """A shot's latents and its delivered frames, refused unless they are this format, this
    latent space and the `size` grid (latent height, width)."""
    try:
        meta = metadata(data)
        latents = load_tensors(data)["latents"]
        frames = int(meta["frames"])
    except (struct.error, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise InvalidRequest("latents is not a minimax-h3 latents file", fields=["latents"]) from exc
    if meta.get("format") != FORMAT:
        raise InvalidRequest(f"latents is {meta.get('format')!r}, not {FORMAT}", fields=["latents"])
    if meta.get("vae_space") != space:
        raise InvalidRequest("latents come from another video VAE's latent space", fields=["latents"])
    count = int(latents.shape[2]) if latents.ndim == 5 else 0
    grid = (count - 2) // LATENTS_PER_CHUNK * FRAMES_PER_CHUNK + 5
    if (
        latents.ndim != 5
        or tuple(latents.shape[:2]) != (1, 24)
        or tuple(latents.shape[3:]) != size
        or count < 2
        or (count - 2) % LATENTS_PER_CHUNK
        or not 0 < frames <= grid
    ):
        raise InvalidRequest(
            f"latents of shape {tuple(latents.shape)} for {frames} frames are not a {size[0]}x{size[1]} "
            "shot on the VAE's 5n + 2 grid",
            fields=["latents"],
        )
    return latents.float(), frames
