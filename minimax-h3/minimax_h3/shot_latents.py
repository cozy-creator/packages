"""A shot's latents as a file: what `generate` returns, `upscale` starts from (Regenerate refines
the shot's own latent, as NVIDIA's `infer.py --input-latents` does, instead of re-encoding its
decoded video) and `continue_from` resumes.

The file is safetensors: `latents`, the normalized `(1, 24, T, h, w)` video latents in float16 on
the VAE's `5n + 2` grid; `context`, its last piece's carry (encoded `AVContext`), which a
continuation's first piece carries in; and metadata naming the format, the latent space
(`vae_space`), the frames the shot delivered (the grid's `17n + 5` frames, or fewer when its
duration was trimmed) and the text it was made from (`text`), which an upscale re-renders it from."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import torch
from cozy_runtime.author import InvalidRequest
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from .official import FRAMES_PER_CHUNK, LATENTS_PER_CHUNK

FORMAT = "minimax-h3/latents/2"
#: The config the latent space depends on: what the encoder writes and the decoder reads.
_SPACE = ("latent_channels", "latents_mean", "latents_std", "spatial_compression_ratio", "temporal_compression_ratio")


def vae_space(vae: Any) -> str:
    """The latent space's identity: the video VAE's channels, normalization and compression, the
    same for every lane and release that shares them."""
    config = vae.config
    space = {key: config[key] for key in _SPACE if key in config}
    return hashlib.sha256(json.dumps(space, sort_keys=True).encode()).hexdigest()[:16]


def save(latents: torch.Tensor, frames: int, space: str, context: bytes, text: str) -> bytes:
    """A shot's latents, with its last piece's carry (`context`, encoded `AVContext`), what a
    `continue_from` of it carries into its first piece, and the text it was made from (`text`),
    which an upscale that re-renders it reads when given none."""
    carry = torch.frombuffer(bytearray(context), dtype=torch.uint8) if context else torch.zeros(0, dtype=torch.uint8)
    return save_tensors(
        {"latents": latents.detach().to("cpu", torch.float16).contiguous(), "context": carry},
        metadata={"format": FORMAT, "vae_space": space, "frames": str(frames), "text": text},
    )


def context(data: bytes) -> bytes:
    """The carry a piece continues from: a shot latents file's `context`, else `data` itself (a
    piece's own exported context)."""
    if metadata_or_none(data).get("format") != FORMAT:
        return data
    return bytes(load_tensors(data)["context"].numpy().tobytes())


def delivered(data: bytes) -> int:
    """The frames a shot latents file stands for."""
    try:
        meta = metadata(data)
        frames = int(meta["frames"])
    except (struct.error, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise InvalidRequest("continue_from.latents is not a minimax-h3 latents file", fields=["continue_from.latents"]) from exc
    if meta.get("format") != FORMAT:
        raise InvalidRequest(f"continue_from.latents is {meta.get('format')!r}, not {FORMAT}", fields=["continue_from.latents"])
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
