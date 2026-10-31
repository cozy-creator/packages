"""A shot's latents as a file: what `shot` returns and `upscale` starts from (`ShotResult.latents`,
`UpscaleRequest.latents`), so Regenerate refines the shot's own latent, as NVIDIA's
`infer.py --input-latents` does, instead of re-encoding its decoded video.

The file is safetensors: `latents`, the normalized `(1, 24, T, h, w)` video latents in float16 on
the VAE's `5n + 2` grid, and metadata naming the format, the latent space (`vae_space`) and the
frames the shot delivered (the grid's `17n + 5` frames, or fewer when its duration was trimmed)."""

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

FORMAT = "minimax-h3/latents/1"
#: The config the latent space depends on: what the encoder writes and the decoder reads.
_SPACE = ("latent_channels", "latents_mean", "latents_std", "spatial_compression_ratio", "temporal_compression_ratio")


def vae_space(vae: Any) -> str:
    """The latent space's identity: the video VAE's channels, normalization and compression, the
    same for every lane and release that shares them."""
    config = vae.config
    space = {key: config[key] for key in _SPACE if key in config}
    return hashlib.sha256(json.dumps(space, sort_keys=True).encode()).hexdigest()[:16]


def save(latents: torch.Tensor, frames: int, space: str) -> bytes:
    return save_tensors(
        {"latents": latents.detach().to("cpu", torch.float16).contiguous()},
        metadata={"format": FORMAT, "vae_space": space, "frames": str(frames)},
    )


def metadata(data: bytes) -> dict[str, str]:
    """A safetensors file's `__metadata__`, read from its header alone."""
    (length,) = struct.unpack("<Q", data[:8])
    return dict(json.loads(data[8 : 8 + length]).get("__metadata__") or {})


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
