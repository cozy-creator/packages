"""se-150's measurement, not merged. `encode_probe`: the VAE encoder of two checkpoints (`reference`, float32 rows,
eager; `model`, float16 rows, fused) on the same pictures at the sizes P-H3-Image encodes them, passes alternating,
with the latents' PSNR and each module's largest activation. `fetch_picture` and `compare_images` serve the session."""

from __future__ import annotations

import io
import math
import time
from collections.abc import Iterator
from functools import partial
from contextlib import contextmanager
import urllib.request
from typing import Annotated, cast

import msgspec
import numpy as np
import torch
from cozy_runtime.accel import fusion
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import Config, Context, Loader, Model, Outputs, Telemetry, uses_components
from cozy_runtime.fields import FileBound
from cozy_runtime.media import File, ImageFile
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from PIL import Image
from safetensors.torch import save as save_tensors
from torch import nn

from .application import app
from .model import ENCODE_SEED, MEGAPIXEL, resized, size_at_area, stored
from .vae import VAE_ENCODER, ImageVAE, encode_picture

MODEL = "prunaai/p-h3-image@0.2.0"
LatentFile = Annotated[File, FileBound(max_bytes=1 << 40, media_types=("application/octet-stream",))]
Picture = Annotated[ImageFile, FileBound(max_bytes=1 << 30, max_decoded_bytes=1 << 30)]
#: PSNR of identical latents or pictures
SAME = 999.0
Image.MAX_IMAGE_PIXELS = None


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
    def encode(self, pixels: torch.Tensor, peaks: dict[str, float] | None = None) -> tuple[torch.Tensor, str]:
        """`PH3Image.encode_pictures` for one picture: its normalized latents and the encoder's dtype. With `peaks`,
        each encoder module's largest |output| (the modules a fused block reads instead of calling report nothing)."""
        vae = cast(ImageVAE, self.graph.components["vae"])
        mean = torch.tensor(vae.config["latents_mean"]).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config["latents_std"]).view(1, -1, 1, 1, 1)
        handles = []
        if peaks is not None:
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
        return latents, str(next(vae.encoder.parameters()).dtype)


def psnr(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    """PSNR over the reference's peak-to-peak range, and the largest absolute difference."""
    a, b = reference.double(), other.double()
    mse = float((a - b).square().mean())
    peak = float(a.max() - a.min())
    return {"psnr_db": round(10 * math.log10(peak**2 / mse), 2) if mse > 0 else SAME,
            "max_abs": float((a - b).abs().max()), "finite": bool(torch.isfinite(b).all())}  # fmt: skip


class EncodeProbe(msgspec.Struct, kw_only=True):
    images: list[Picture]
    megapixels: list[float]
    #: timed passes per arm, alternating, after one untimed pass each
    passes: int = 3


class Measured(msgspec.Struct):
    size: list[int]
    reference_s: list[float]
    model_s: list[float]
    psnr_db: float
    max_abs: float
    finite: bool
    #: the reference's largest |output| over every encoder module, and where
    reference_peak: float
    reference_peak_at: str
    #: the model's, over the modules it calls
    model_peak: float
    model_peak_at: str


class EncodeProbeOutput(msgspec.Struct):
    latents: LatentFile
    dtypes: list[str]
    #: "<picture>@<mp>"
    measured: dict[str, Measured]
    reference_peaks: dict[str, dict[str, float]]


@app.entrypoint(defaults={"reference": MODEL, "model": MODEL}, components={"reference": ("vae",), "model": ("vae",)})
async def encode_probe(
    ctx: Context, *, payload: EncodeProbe, reference: PH3ImageVAE, model: PH3ImageVAE, out: Outputs, tel: Telemetry
) -> EncodeProbeOutput:
    """se-150: each picture at each size through both encoders, `passes` timed passes each, alternating."""
    saved: dict[str, torch.Tensor] = {}
    measured: dict[str, Measured] = {}
    all_peaks: dict[str, dict[str, float]] = {}
    dtypes: list[str] = []
    for index, image in enumerate(payload.images):
        picture = image.read().convert("RGB")
        for mp in payload.megapixels:
            size = size_at_area(*picture.size, round(mp * MEGAPIXEL))
            pixels = torch.from_numpy(np.array(resized(picture, size)))
            key = f"{index}@{mp:g}"
            ref_peaks: dict[str, float] = {}
            got_peaks: dict[str, float] = {}
            want, ref_dtype = reference.encode(pixels, ref_peaks)
            got, dtype = model.encode(pixels, got_peaks)
            dtypes = [ref_dtype, dtype]
            times: dict[str, list[float]] = {"reference": [], "model": []}
            for _ in range(payload.passes):
                for arm, encoder in (("reference", reference), ("model", model)):
                    ctx.raise_if_cancelled()
                    started = time.perf_counter()
                    with tel.stage(f"vae_encode_{arm}"):
                        encoder.encode(pixels)
                    times[arm].append(round(time.perf_counter() - started, 4))
            saved[f"{key}/reference"], saved[f"{key}/model"] = want.float().contiguous(), got.float().contiguous()
            compared = psnr(want, got)
            top = max(ref_peaks, key=ref_peaks.__getitem__)
            top_model = max(got_peaks, key=got_peaks.__getitem__) if got_peaks else ""
            measured[key] = Measured(list(size), times["reference"], times["model"], compared["psnr_db"], compared["max_abs"],
                                     bool(compared["finite"]), ref_peaks[top], top, got_peaks.get(top_model, 0.0), top_model)  # fmt: skip
            all_peaks[key] = ref_peaks
            tel.log("encode probe", key=key, size=list(size), dtypes=dtypes, **compared, reference_s=times["reference"],
                    model_s=times["model"], reference_peak=ref_peaks[top], at=top)  # fmt: skip
    blob = out.save_bytes(save_tensors(saved), media_type="application/octet-stream")
    return EncodeProbeOutput(blob, dtypes, measured, all_peaks)


class Fetch(msgspec.Struct, kw_only=True):
    url: str


class Fetched(msgspec.Struct):
    image: Picture
    size: list[int]


@app.entrypoint(accelerator=False)
async def fetch_picture(ctx: Context, *, payload: Fetch) -> Fetched:
    """se-150: a public photo by URL, as a PNG."""
    request = urllib.request.Request(payload.url, headers={"User-Agent": "cozy-measurement/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    picture = Image.open(io.BytesIO(data)).convert("RGB")
    return Fetched(ImageFile.from_image(picture, format="png"), list(picture.size))


class Pair(msgspec.Struct, kw_only=True):
    a: Picture
    b: Picture


class Compared(msgspec.Struct):
    psnr_db: float
    max_abs: float
    mean_abs: float
    size: list[int]


@app.entrypoint(accelerator=False)
async def compare_images(ctx: Context, *, payload: Pair) -> Compared:
    """se-150: two pictures' PSNR (8-bit), largest and mean absolute difference."""
    a = np.asarray(payload.a.read().convert("RGB"), dtype=np.float64)
    b = np.asarray(payload.b.read().convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        return Compared(-1.0, -1.0, -1.0, [*a.shape[:2], *b.shape[:2]])
    mse = float(np.mean((a - b) ** 2))
    value = 10 * math.log10(255.0**2 / mse) if mse > 0 else SAME
    return Compared(round(value, 2), float(np.abs(a - b).max()), round(float(np.abs(a - b).mean()), 4), list(a.shape[:2]))


# --- cr-323's measurement, not merged ------------------------------------------------------------------------------

#: sequential tiles (upstream's loop) or one batch, each through all three temporal taps or the last alone
ARMS = ("seq_three", "seq_last", "batch_three", "batch_last")


@contextmanager
def as_arm(name: str, vae: nn.Module) -> Iterator[None]:
    """The encoder as arm `name` runs it: `seq_*` encodes tile by tile (Diffusers' `_encode_clip`), `*_three` with the
    Runtime's `fusion._lone_frame` answering no (before cr-323). A Runtime before cr-323 runs three taps either way."""
    lone = getattr(fusion, "_lone_frame", None)
    if name.endswith("three") and lone is not None:
        setattr(fusion, "_lone_frame", lambda *_: False)
    if name.startswith("seq"):
        setattr(vae, "_encode_clip", partial(AutoencoderKLMiniMaxH3._encode_clip, vae))
    try:
        yield
    finally:
        if name.startswith("seq"):
            delattr(vae, "_encode_clip")
        if lone is not None:
            setattr(fusion, "_lone_frame", lone)


class LoneProbe(msgspec.Struct, kw_only=True):
    images: list[Picture]
    megapixels: list[float]
    #: timed passes per arm, alternating, after one untimed pass each
    passes: int = 3


class LoneMeasured(msgspec.Struct):
    size: list[int]
    #: seconds per pass, per arm
    seconds: dict[str, list[float]]
    #: each arm's latents against `seq_three`'s (the current fused path)
    psnr_db: dict[str, float]
    max_abs: dict[str, float]


class LoneProbeOutput(msgspec.Struct):
    runtime: str
    lone_path: bool
    measured: dict[str, LoneMeasured]
    latents: LatentFile


@app.entrypoint(defaults={"model": MODEL}, components={"model": ("vae",)})
async def lone_probe(ctx: Context, *, payload: LoneProbe, model: PH3ImageVAE, out: Outputs, tel: Telemetry) -> LoneProbeOutput:
    """cr-323: each picture at each size through the fused encoder in each arm (`ARMS`), `passes` timed passes each,
    alternating, and each arm's latents' PSNR against the current fused path's."""
    from importlib.metadata import version

    vae = model.graph.components["vae"]
    saved: dict[str, torch.Tensor] = {}
    measured: dict[str, LoneMeasured] = {}
    for index, image in enumerate(payload.images):
        picture = image.read().convert("RGB")
        for mp in payload.megapixels:
            size = size_at_area(*picture.size, round(mp * MEGAPIXEL))
            pixels = torch.from_numpy(np.array(resized(picture, size)))
            key = f"{index}@{mp:g}"
            latents: dict[str, torch.Tensor] = {}
            for name in ARMS:
                with as_arm(name, vae):
                    latents[name], _ = model.encode(pixels)
            seconds: dict[str, list[float]] = {name: [] for name in ARMS}
            for _ in range(payload.passes):
                for name in ARMS:
                    ctx.raise_if_cancelled()
                    with as_arm(name, vae):
                        started = time.perf_counter()
                        with tel.stage(f"vae_encode_{name}"):
                            model.encode(pixels)
                        seconds[name].append(round(time.perf_counter() - started, 4))
            compared = {name: psnr(latents["seq_three"], latents[name]) for name in ARMS}
            for name in ARMS:
                saved[f"{key}/{name}"] = latents[name].float().contiguous()
            measured[key] = LoneMeasured(list(size), seconds, {n: c["psnr_db"] for n, c in compared.items()},
                                         {n: c["max_abs"] for n, c in compared.items()})  # fmt: skip
            tel.log("lone probe", key=key, size=list(size), seconds=seconds, psnr_db=measured[key].psnr_db)
    blob = out.save_bytes(save_tensors(saved), media_type="application/octet-stream")
    return LoneProbeOutput(version("cozy-runtime"), hasattr(fusion, "_lone_frame"), measured, blob)
