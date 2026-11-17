"""se-150's measurement, not merged: the VAE encoder alone on pictures at the sizes P-H3-Image encodes them, timed
per pass, with each normalized latent and, on request, the largest activation each encoder module writes."""

from __future__ import annotations

import time
from typing import Annotated, cast

import msgspec
import numpy as np
import torch
from cozy_runtime.accel import fusion
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import Config, Context, Loader, Model, Outputs, Telemetry, uses_components
from cozy_runtime.fields import FileBound
from cozy_runtime.media import File, ImageFile
from safetensors.torch import save as save_tensors
from torch import nn

from .application import app
from .model import ENCODE_SEED, MEGAPIXEL, resized, size_at_area, stored
from .vae import VAE_ENCODER, ImageVAE, encode_picture

MODEL = "prunaai/p-h3-image@0.2.0"
LatentFile = Annotated[File, FileBound(max_bytes=1 << 40, media_types=("application/octet-stream",))]


class VAEGraph:
    def __init__(self, config: Config) -> None:
        vae = ImageVAE.build(config.section("vae").mapping())
        self.components: dict[str, nn.Module] = {"vae": stored(cast(nn.Module, vae), "vae", config).eval()}


class PH3ImageVAE(Model[VAEGraph]):
    graph: VAEGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(VAEGraph)

    def unload(self, loader: Loader) -> None:
        return None

    def execution_plan(self, plan: Plan) -> str:
        return fusion.install_encoder(plan, VAE_ENCODER, self.graph.components["vae"])

    @uses_components("vae")
    def encode(self, pixels: torch.Tensor, activations: bool) -> tuple[torch.Tensor, dict[str, float], float]:
        """`PH3Image.encode_pictures` for one picture: its normalized latents, the largest |output| of each encoder
        module (when asked), and the encoder's dtype's name."""
        vae = cast(ImageVAE, self.graph.components["vae"])
        mean = torch.tensor(vae.config["latents_mean"]).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config["latents_std"]).view(1, -1, 1, 1, 1)
        peaks: dict[str, float] = {}
        handles = []
        if activations:
            for name, module in [*vae.encoder.named_modules(prefix="encoder"), ("quant_conv", vae.quant_conv)]:

                def hook(_m: nn.Module, _a: object, out: object, name: str = name) -> None:
                    if isinstance(out, torch.Tensor):
                        peaks[name] = max(peaks.get(name, 0.0), float(out.detach().abs().float().max()))

                handles.append(module.register_forward_hook(hook))
        try:
            with torch.no_grad():
                latents = (encode_picture(vae, pixels, ENCODE_SEED).cpu() - mean) / std
        finally:
            for handle in handles:
                handle.remove()
        return latents, peaks, float(next(vae.encoder.parameters()).element_size())


class EncodeProbe(msgspec.Struct, kw_only=True):
    images: list[ImageFile]
    megapixels: list[float] = msgspec.field(default_factory=lambda: [8.0, 16.0])
    passes: int = 3
    activations: bool = False


class EncodeProbeOutput(msgspec.Struct):
    latents: LatentFile
    #: "<picture>@<mp>": [seconds per pass], pass 0 the cold one
    times_s: dict[str, list[float]]
    sizes: dict[str, list[int]]
    #: "<picture>@<mp>": {module: largest |output|}
    peaks: dict[str, dict[str, float]]
    encoder_bytes_per_element: float


@app.entrypoint(defaults={"model": MODEL}, components={"model": ("vae",)})
async def encode_probe(
    ctx: Context, *, payload: EncodeProbe, model: PH3ImageVAE, out: Outputs, tel: Telemetry
) -> EncodeProbeOutput:
    """se-150: each picture at each size, encoded `passes` times as an edit's pictures are."""
    saved: dict[str, torch.Tensor] = {}
    times: dict[str, list[float]] = {}
    sizes: dict[str, list[int]] = {}
    peaks: dict[str, dict[str, float]] = {}
    width = 0.0
    for index, image in enumerate(payload.images):
        picture = image.read().convert("RGB")
        for mp in payload.megapixels:
            size = size_at_area(*picture.size, round(mp * MEGAPIXEL))
            pixels = torch.from_numpy(np.array(resized(picture, size)))
            key = f"{index}@{mp:g}"
            sizes[key] = list(size)
            times[key] = []
            for _ in range(payload.passes):
                ctx.raise_if_cancelled()
                started = time.perf_counter()
                with tel.stage("vae_encode"):
                    latents, _, width = model.encode(pixels, False)
                times[key].append(round(time.perf_counter() - started, 4))
            saved[key] = latents.float().contiguous()
            if payload.activations:
                _, peaks[key], _ = model.encode(pixels, True)
            tel.log("encode probe", key=key, size=list(size), times=times[key], peak=max(peaks.get(key, {}).values(), default=0.0))
    blob = out.save_bytes(save_tensors(saved), media_type="application/octet-stream")
    return EncodeProbeOutput(blob, times, sizes, peaks, width)
