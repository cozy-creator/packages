"""se-150's measurement, not merged: the VAE encoder alone on reference pictures as MiniMax-H3-Image reads them, timed
per pass, with each normalized latent and, on request, the largest activation each encoder module writes."""

from __future__ import annotations

import time
from typing import Annotated, cast

import msgspec
import torch
from cozy_runtime.author import Context, Outputs, Telemetry
from cozy_runtime.fields import FileBound
from cozy_runtime.media import File, ImageFile
from PIL import Image
from safetensors.torch import save as save_tensors
from torch import nn

from .application import app
from .model import LATENT_CHANNELS, H3ImageEncoder, latent_grid, upright
from .vae import ImageEncoder

MODEL = "minimaxai/minimax-h3-image@0.1.0"
LatentFile = Annotated[File, FileBound(max_bytes=1 << 40, media_types=("application/octet-stream",))]


class EncodeProbe(msgspec.Struct, kw_only=True):
    images: list[ImageFile]
    passes: int = 3
    activations: bool = False


class EncodeProbeOutput(msgspec.Struct):
    latents: LatentFile
    #: "<picture>": [seconds per pass], pass 0 the cold one
    times_s: dict[str, list[float]]
    #: "<picture>": latent (h, w)
    grids: dict[str, list[int]]
    #: "<picture>": {module: largest |output|}
    peaks: dict[str, dict[str, float]]


class ProbeEncoder(H3ImageEncoder):
    def peaks(self, picture: Image.Image, noise: torch.Tensor) -> dict[str, float]:
        """`encode` once with every encoder module's largest |output| recorded."""
        encoder = cast(ImageEncoder, self.graph.components["vae_encoder"])
        peaks: dict[str, float] = {}
        handles = []
        for name, module in [*encoder.encoder.named_modules(prefix="encoder"), ("quant_conv", encoder.quant_conv)]:

            def hook(_m: nn.Module, _a: object, out: object, name: str = name) -> None:
                if isinstance(out, torch.Tensor):
                    peaks[name] = max(peaks.get(name, 0.0), float(out.detach().abs().float().max()))

            handles.append(module.register_forward_hook(hook))
        try:
            self.encode([picture], [noise])
        finally:
            for handle in handles:
                handle.remove()
        return peaks


@app.entrypoint(defaults={"vae_encoder": MODEL}, components={"vae_encoder": ("vae_encoder",)})
async def encode_probe(
    ctx: Context, *, payload: EncodeProbe, vae_encoder: ProbeEncoder, out: Outputs, tel: Telemetry
) -> EncodeProbeOutput:
    """se-150: each reference encoded `passes` times as `generate` encodes it, the posterior's noise drawn from seed 0."""
    saved: dict[str, torch.Tensor] = {}
    times: dict[str, list[float]] = {}
    grids: dict[str, list[int]] = {}
    peaks: dict[str, dict[str, float]] = {}
    for index, image in enumerate(payload.images):
        picture = upright(image.read())
        h, w = latent_grid(picture)
        noise = torch.randn((1, LATENT_CHANNELS, h, w), generator=torch.Generator().manual_seed(0))
        key = str(index)
        grids[key], times[key] = [h, w], []
        latents = torch.empty(0)
        for _ in range(payload.passes):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            with tel.stage("vae_encode"):
                (latents,) = vae_encoder.encode([picture], [noise])
            times[key].append(round(time.perf_counter() - started, 4))
        saved[key] = latents.float().contiguous()
        if payload.activations:
            peaks[key] = vae_encoder.peaks(picture, noise)
        tel.log("encode probe", key=key, grid=[h, w], times=times[key], peak=max(peaks.get(key, {}).values(), default=0.0))
    blob = out.save_bytes(save_tensors(saved), media_type="application/octet-stream")
    return EncodeProbeOutput(blob, times, grids, peaks)
