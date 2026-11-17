"""se-150's measurement, not merged. `encode_probe`: the VAE encoder of two checkpoints (`reference`, float32 rows,
eager; `vae_encoder`, float16 rows, fused) on the same reference pictures as `generate` encodes them, passes
alternating, with the latents' PSNR and each module's largest activation."""

from __future__ import annotations

import math
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
Picture = Annotated[ImageFile, FileBound(max_bytes=1 << 30, max_decoded_bytes=1 << 30)]
SAME = 999.0
Image.MAX_IMAGE_PIXELS = None


class ProbeEncoder(H3ImageEncoder):
    def latents(self, picture: Image.Image, noise: torch.Tensor, peaks: dict[str, float] | None = None) -> tuple[torch.Tensor, str]:
        """`encode` of one picture, its encoder's dtype, and with `peaks` each encoder module's largest |output| (the
        modules a fused block reads instead of calling report nothing)."""
        encoder = cast(ImageEncoder, self.graph.components["vae_encoder"])
        handles = []
        if peaks is not None:
            for name, module in [*encoder.encoder.named_modules(prefix="encoder"), ("quant_conv", encoder.quant_conv)]:

                def hook(_m: nn.Module, _a: object, out: object, name: str = name) -> None:
                    if isinstance(out, torch.Tensor):
                        peaks[name] = max(peaks.get(name, 0.0), float(out.detach().abs().float().max()))

                handles.append(module.register_forward_hook(hook))
        try:
            (latents,) = self.encode([picture], [noise])
        finally:
            for handle in handles:
                handle.remove()
        return latents, str(next(encoder.encoder.parameters()).dtype)


class EncodeProbe(msgspec.Struct, kw_only=True):
    images: list[Picture]
    #: timed passes per arm, alternating, after one untimed pass each
    passes: int = 3


class Measured(msgspec.Struct):
    grid: list[int]
    reference_s: list[float]
    model_s: list[float]
    psnr_db: float
    max_abs: float
    finite: bool
    reference_peak: float
    reference_peak_at: str
    model_peak: float
    model_peak_at: str


class EncodeProbeOutput(msgspec.Struct):
    latents: LatentFile
    dtypes: list[str]
    measured: dict[str, Measured]
    reference_peaks: dict[str, dict[str, float]]


@app.entrypoint(defaults={"reference": MODEL, "vae_encoder": MODEL},
                components={"reference": ("vae_encoder",), "vae_encoder": ("vae_encoder",)})  # fmt: skip
async def encode_probe(
    ctx: Context, *, payload: EncodeProbe, reference: ProbeEncoder, vae_encoder: ProbeEncoder, out: Outputs, tel: Telemetry
) -> EncodeProbeOutput:
    """se-150: each reference through both encoders as `generate` encodes it (noise from seed 0), `passes` timed passes
    each, alternating."""
    saved: dict[str, torch.Tensor] = {}
    measured: dict[str, Measured] = {}
    all_peaks: dict[str, dict[str, float]] = {}
    dtypes: list[str] = []
    for index, image in enumerate(payload.images):
        picture = upright(image.read())
        h, w = latent_grid(picture)
        noise = torch.randn((1, LATENT_CHANNELS, h, w), generator=torch.Generator().manual_seed(0))
        key = str(index)
        ref_peaks: dict[str, float] = {}
        got_peaks: dict[str, float] = {}
        want, ref_dtype = reference.latents(picture, noise, ref_peaks)
        got, dtype = vae_encoder.latents(picture, noise, got_peaks)
        dtypes = [ref_dtype, dtype]
        times: dict[str, list[float]] = {"reference": [], "model": []}
        for _ in range(payload.passes):
            for arm, encoder in (("reference", reference), ("model", vae_encoder)):
                ctx.raise_if_cancelled()
                started = time.perf_counter()
                with tel.stage(f"vae_encode_{arm}"):
                    encoder.latents(picture, noise)
                times[arm].append(round(time.perf_counter() - started, 4))
        saved[f"{key}/reference"], saved[f"{key}/model"] = want.float().contiguous(), got.float().contiguous()
        a, b = want.double(), got.double()
        mse = float((a - b).square().mean())
        value = round(10 * math.log10(float(a.max() - a.min()) ** 2 / mse), 2) if mse > 0 else SAME
        top = max(ref_peaks, key=ref_peaks.__getitem__)
        top_model = max(got_peaks, key=got_peaks.__getitem__) if got_peaks else ""
        measured[key] = Measured([h, w], times["reference"], times["model"], value, float((a - b).abs().max()),
                                 bool(torch.isfinite(b).all()), ref_peaks[top], top, got_peaks.get(top_model, 0.0), top_model)  # fmt: skip
        all_peaks[key] = ref_peaks
        tel.log("encode probe", key=key, grid=[h, w], dtypes=dtypes, psnr_db=value, reference_s=times["reference"],
                model_s=times["model"], reference_peak=ref_peaks[top], at=top)  # fmt: skip
    blob = out.save_bytes(save_tensors(saved), media_type="application/octet-stream")
    return EncodeProbeOutput(blob, dtypes, measured, all_peaks)
