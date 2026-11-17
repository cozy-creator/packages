"""se-150's measurement, not merged. `encode_probe`: the VAE encoder of two checkpoints (`reference`, float32 rows,
eager; `vae_encoder`, float16 rows, fused) on the same reference pictures as `generate` encodes them, passes
alternating, with the latents' PSNR and each module's largest activation."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Annotated, cast

import msgspec
import torch
from cozy_runtime.accel import fusion
from cozy_runtime.author import Context, Outputs, Telemetry
from cozy_runtime.fields import FileBound
from cozy_runtime.media import File, ImageFile
from diffusers.models.modeling_utils import get_parameter_dtype
from PIL import Image
from safetensors.torch import save as save_tensors
from torch import nn

from .application import app
from .model import LATENT_CHANNELS, H3ImageEncoder, latent_grid, upright
from .vae import ImageEncoder, split_tiles, stitch

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


# --- cr-323's measurement, not merged ------------------------------------------------------------------------------

#: sequential tiles (the loop before cr-323) or one batch, each through all three temporal taps or the last alone
ARMS = ("seq_three", "seq_last", "batch_three", "batch_last")


def _sequential(encoder: ImageEncoder, pixels: torch.Tensor) -> torch.Tensor:
    """`ImageEncoder.forward` before cr-323: tile by tile."""
    pixels = pixels.to(get_parameter_dtype(encoder.encoder))
    ys, y_lengths, y_overlaps = split_tiles(pixels.shape[-2], encoder.ratio)
    xs, x_lengths, x_overlaps = split_tiles(pixels.shape[-1], encoder.ratio)
    rows = [[encoder._clip(pixels[..., y : y + h, x : x + w]) for x, w in zip(xs, x_lengths, strict=True)]
            for y, h in zip(ys, y_lengths, strict=True)]  # fmt: skip
    return stitch(rows, [o // encoder.ratio for o in y_overlaps], [o // encoder.ratio for o in x_overlaps])


@contextmanager
def as_arm(name: str, encoder: ImageEncoder) -> Iterator[None]:
    """The encoder as arm `name` runs it: `seq_*` tile by tile, `*_three` with the Runtime's `fusion._lone_frame`
    answering no (before cr-323). A Runtime before cr-323 runs three taps either way."""
    lone = getattr(fusion, "_lone_frame", None)
    if name.endswith("three") and lone is not None:
        setattr(fusion, "_lone_frame", lambda *_: False)
    if name.startswith("seq"):
        setattr(encoder, "forward", partial(_sequential, encoder))
    try:
        yield
    finally:
        if name.startswith("seq"):
            delattr(encoder, "forward")
        if lone is not None:
            setattr(fusion, "_lone_frame", lone)


class LoneProbe(msgspec.Struct, kw_only=True):
    images: list[Picture]
    #: timed passes per arm, alternating, after one untimed pass each
    passes: int = 3


class LoneMeasured(msgspec.Struct):
    grid: list[int]
    seconds: dict[str, list[float]]
    psnr_db: dict[str, float]
    max_abs: dict[str, float]
    peak_gib: dict[str, float]


class LoneProbeOutput(msgspec.Struct):
    runtime: str
    lone_path: bool
    measured: dict[str, LoneMeasured]
    latents: LatentFile


@app.entrypoint(defaults={"vae_encoder": MODEL}, components={"vae_encoder": ("vae_encoder",)})
async def lone_probe(
    ctx: Context, *, payload: LoneProbe, vae_encoder: ProbeEncoder, out: Outputs, tel: Telemetry
) -> LoneProbeOutput:
    """cr-323: each reference as `generate` encodes it (noise from seed 0) in each arm (`ARMS`), `passes` timed passes
    each, alternating, and each arm's latents' PSNR against the current fused path's (`seq_three`)."""
    from importlib.metadata import version

    encoder = cast(ImageEncoder, vae_encoder.graph.components["vae_encoder"])
    saved: dict[str, torch.Tensor] = {}
    measured: dict[str, LoneMeasured] = {}
    for index, image in enumerate(payload.images):
        picture = upright(image.read())
        h, w = latent_grid(picture)
        noise = torch.randn((1, LATENT_CHANNELS, h, w), generator=torch.Generator().manual_seed(0))
        key = str(index)
        latents: dict[str, torch.Tensor] = {}
        for name in ARMS:
            with as_arm(name, encoder):
                latents[name], _ = vae_encoder.latents(picture, noise)
        seconds: dict[str, list[float]] = {name: [] for name in ARMS}
        peaks: dict[str, float] = dict.fromkeys(ARMS, 0.0)
        for _ in range(payload.passes):
            for name in ARMS:
                ctx.raise_if_cancelled()
                with as_arm(name, encoder):
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with tel.stage(f"vae_encode_{name}"):
                        vae_encoder.latents(picture, noise)
                    seconds[name].append(round(time.perf_counter() - started, 4))
                    peaks[name] = max(peaks[name], round(torch.cuda.max_memory_allocated() / 2**30, 2))
        a = latents["seq_three"].double()
        values: dict[str, float] = {}
        biggest: dict[str, float] = {}
        for name in ARMS:
            b = latents[name].double()
            mse = float((a - b).square().mean())
            values[name] = round(10 * math.log10(float(a.max() - a.min()) ** 2 / mse), 2) if mse > 0 else SAME
            biggest[name] = float((a - b).abs().max())
            saved[f"{key}/{name}"] = latents[name].float().contiguous()
        measured[key] = LoneMeasured([h, w], seconds, values, biggest, peaks)
        tel.log("lone probe", key=key, grid=[h, w], seconds=seconds, psnr_db=values, peak_gib=peaks)
    blob = out.save_bytes(save_tensors(saved), media_type="application/octet-stream")
    return LoneProbeOutput(version("cozy-runtime"), hasattr(fusion, "_lone_frame"), measured, blob)
