"""LTX-2.5 Distilled: synchronized video and audio from text or a starting picture."""

from __future__ import annotations

import math
import secrets
from typing import Annotated, Literal

import msgspec
from cozy_runtime import rng
from cozy_runtime.author import Adjustments, App, Context, Outputs, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, ImagePreparation, Range, Ratios
from cozy_runtime.media import ImageFile, VideoFile
from PIL import Image

from . import quantize as quantize  # its recipe
from .model import COMPONENTS, FPS, REFINE_SIGMAS, SIGMAS, Ltx25

app = App()
MODEL = "lightricks/ltx-2.5@0.2.0"


class Generate(msgspec.Struct, kw_only=True):
    #: Describe the picture, motion, camera, sounds and dialogue to generate.
    prompt: str
    first_frame: Annotated[ImageFile, ImagePreparation(max_pixels=16 << 20)] | None = None
    duration_s: Annotated[float, Range(1, 20, unit="s")] = 5.0
    resolution: Literal["720p", "1080p"] = "1080p"
    #: Unset: the first frame's shape, else 16:9. Output sides are on LTX's 64-pixel two-stage grid.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=0.5, max=2)] | None = None
    seed: Seed | None = None
    #: Render the soundtrack with the synchronized audio branch; false returns a silent MP4.
    audio: Annotated[bool, Advanced()] = True
    #: proof/se-161 only: Diffusers' Euler, or native LTX-2.5's Euler ancestral.
    sampler: Annotated[Literal["euler", "ancestral"], Advanced()] = "euler"
    #: proof/se-161 only: stage 2's first frame, the upsampled stage-1 encode or the picture encoded at full size.
    stage2_image: Annotated[Literal["upsampled", "encoded"], Advanced()] = "upsampled"
    #: proof/se-161 only: the picture stretched to the canvas (Diffusers) or centre-cropped to its shape (native).
    image_fit: Annotated[Literal["stretch", "crop"], Advanced()] = "stretch"


class VideoResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 40, media_types=("video/mp4",))]
    width: int
    height: int
    fps: float
    frame_count: int
    seed: int
    prompt: str


def fit(picture: Image.Image, width: int, height: int) -> Image.Image:
    """The picture's centre at the canvas's shape, as native LTX and ComfyUI fit a conditioning image."""
    w, h = picture.size
    if w * height > h * width:
        crop = round(h * width / height)
        return picture.crop(((w - crop) // 2, 0, (w - crop) // 2 + crop, h))
    crop = round(w * height / width)
    return picture.crop((0, (h - crop) // 2, w, (h - crop) // 2 + crop))


def canvas(resolution: str, ratio: AspectRatio | None, first: tuple[int, int] | None) -> tuple[int, int]:
    native = (1920, 1088) if resolution == "1080p" else (1280, 704)
    if ratio is None and first is None:
        return native
    if ratio is not None:
        rw, rh = (float(value) for value in str(ratio).split(":"))
        r = rw / rh
    else:
        w, h = first or native
        r = w / h
    r = min(2.0, max(0.5, r))
    area = native[0] * native[1]
    return max(64, round(math.sqrt(area * r) / 64) * 64), max(64, round(math.sqrt(area / r) / 64) * 64)


@app.tool(label="LTX-2.5", named="video", defaults={"model": MODEL}, components={"model": COMPONENTS}, memoize=True)
def generate(ctx: Context, payload: Generate, model: Ltx25, out: Outputs, tel: Telemetry, adj: Adjustments) -> VideoResult:
    """LTX-2.5 Distilled, in two stages: a scene and its synchronized sound, then x2 spatial refinement."""
    if not payload.prompt.strip():
        raise ValueError("LTX-2.5 needs a prompt describing the video and its sound")
    seed = int(payload.seed) if payload.seed is not None else secrets.randbits(32)
    seconds = min(20.0, max(1.0, payload.duration_s))
    frames = min(481, 8 * math.ceil(seconds * FPS / 8) + 1)
    if seconds != payload.duration_s:
        adj.warn("duration_s", f"LTX-2.5 renders 1-20 seconds, so this clip runs {(frames - 1) / FPS:g} seconds")
    picture = payload.first_frame.read().convert("RGB") if payload.first_frame else None
    width, height = canvas(payload.resolution, payload.aspect_ratio, picture.size if picture else None)
    if picture is not None and payload.image_fit == "crop":
        picture = fit(picture, width, height)
    generator = rng.generator(seed)
    ancestral = payload.sampler == "ancestral"
    with tel.stage("encoding prompt"):
        embeds = model.encode(payload.prompt)
    with tel.stage("generating video and audio"):
        latents = model.denoise(embeds, picture, width=width // 2, height=height // 2, frames=frames,
                                generator=generator, noise=seed + 10000 if ancestral else None, check=ctx.raise_if_cancelled,
                                progress=lambda step: ctx.progress(step, len(SIGMAS)))
    ctx.raise_if_cancelled()
    with tel.stage("upscaling latents"):
        high = model.upscale(latents[0])
    with tel.stage("refining video and audio"):
        latents = model.denoise(embeds, picture, width=width, height=height, frames=frames, latents=(high, latents[1]),
                                generator=generator, noise=seed + 20000 if ancestral else None,
                                encoded=payload.stage2_image == "encoded", check=ctx.raise_if_cancelled,
                                progress=lambda step: ctx.progress(step, len(REFINE_SIGMAS)))
    ctx.raise_if_cancelled()
    with tel.stage("decoding video"):
        pixels = model.decode_video(latents[0])
    audio, rate = None, None
    if payload.audio:
        with tel.stage("decoding audio"):
            audio, rate = model.decode_audio(latents[1])
    with tel.stage("saving video"):
        video = out.save_video(pixels, fps=FPS, audio=audio, sample_rate=rate)
    tel.log("ltx-2.5", width=width, height=height, frames=len(pixels), audio=payload.audio,
            sampler=payload.sampler, stage2_image=payload.stage2_image, image_fit=payload.image_fit, stages=2, steps=len(SIGMAS) + len(REFINE_SIGMAS))
    return VideoResult(video, width, height, float(FPS), len(pixels), seed, payload.prompt)
