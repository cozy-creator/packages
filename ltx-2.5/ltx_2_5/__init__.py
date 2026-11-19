"""LTX-2.5 Distilled: synchronized video and audio from text or a starting picture."""

from __future__ import annotations

import math
import secrets
from collections.abc import Iterator
from fractions import Fraction
from typing import Annotated, Literal

import msgspec
import numpy as np
import torch
from cozy_runtime import rng
from cozy_runtime.author import Adjustments, App, Context, Outputs, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, ImagePreparation, Range, Ratios
from cozy_runtime.media import (
    DecodedAudioChunk, DecodedAudioFormat, DecodedMediaHeader, DecodedVideoFormat, DecodedVideoFrame, ImageFile, VideoFile,
)
from PIL import Image

from . import quantize as quantize  # its recipe
from .model import FPS, REFINE_SIGMAS, SIGMAS, Ltx25, Ltx25TextEncoder
from .vae import VideoPixels

app = App()
MODEL = "lightricks/ltx-2.5@0.2.0"
#: The Gemma 4 12B encoder is its own slot, abliterated by default (Paul, 11-17: "use abliterated models wherever
#: possible"): igorls' Heretic edit of Gemma 4 12B moved onto Lightricks' fine-tune of it, the encoder and its tokenizer
#: alone (se-160: a release holds only its author's work). `Ltx25TextEncoder` builds only the encoder, so Lightricks' own
#: is any LTX-2.5 checkpoint: `text_encoder=lightricks/ltx-2.5@0.2.0`. `model` reads everything else.
TEXT_ENCODER = "igorls/gemma-4-12b-it-heretic-v1-ltx-2.5@0.1.0"
_COMPONENTS = {"model": ("transformer", "connectors", "vae", "audio_vae", "vocoder", "latent_upsampler"),
               "text_encoder": ("text_encoder",)}

from . import encoder as encoder  # noqa: E402  # `transplant`, which writes TEXT_ENCODER; it registers on `app`


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


class VideoResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 40, media_types=("video/mp4",))]
    width: int
    height: int
    fps: float
    frame_count: int
    seed: int
    prompt: str


def fit(picture: Image.Image, width: int, height: int) -> Image.Image:
    """The picture's centre at the canvas's shape: native LTX and its ComfyUI template crop a conditioning picture to
    the canvas rather than stretch it, as Diffusers' resize would."""
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


@app.tool(label="LTX-2.5", named="video", defaults={"model": MODEL, "text_encoder": TEXT_ENCODER}, components=_COMPONENTS,
          memoize=True)
def generate(ctx: Context, payload: Generate, model: Ltx25, text_encoder: Ltx25TextEncoder, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> VideoResult:
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
    if picture is not None:
        picture = fit(picture, width, height)
    generator = rng.generator(seed)
    with tel.stage("encoding prompt"):
        embeds = text_encoder.encode(payload.prompt)
    with tel.stage("generating video and audio"):
        latents = model.denoise(embeds, picture, width=width // 2, height=height // 2, frames=frames,
                                generator=generator, check=ctx.raise_if_cancelled,
                                progress=lambda step: ctx.progress(step, len(SIGMAS)))
    ctx.raise_if_cancelled()
    with tel.stage("upscaling latents"):
        high = model.upscale(latents[0])
    with tel.stage("refining video and audio"):
        latents = model.denoise(embeds, picture, width=width, height=height, frames=frames, latents=(high, latents[1]),
                                generator=generator, check=ctx.raise_if_cancelled,
                                progress=lambda step: ctx.progress(step, len(REFINE_SIGMAS)))
    ctx.raise_if_cancelled()
    with tel.stage("decoding video"):
        pixels = model.decode_video(latents[0])
    audio, rate = None, None
    if payload.audio:
        with tel.stage("decoding audio"):
            audio, rate = model.decode_audio(latents[1])
    with tel.stage("saving video"):
        video = out.save_video_stream(_events(pixels, audio, rate)).video
    tel.log("ltx-2.5", width=width, height=height, frames=len(pixels), audio=payload.audio,
            sampler="diffusers-euler", stages=2, steps=len(SIGMAS) + len(REFINE_SIGMAS))
    return VideoResult(video, width, height, float(FPS), len(pixels), seed, payload.prompt)


type Waveform = np.ndarray[tuple[int, int], np.dtype[np.float32]]
_LAYOUTS = {1: ("mono", ("FC",)), 2: ("stereo", ("FL", "FR"))}


def _events(pixels: VideoPixels, audio: torch.Tensor | None,
            rate: int | None) -> Iterator[DecodedMediaHeader | DecodedVideoFrame | DecodedAudioChunk]:
    """The clip for `save_video_stream`, BT.709 as H3's tools write theirs: each frame as it is read, each second of
    frames followed by its sound, so the encoder never takes the whole video at once."""
    frames, height, width, _ = pixels.shape
    sound: Waveform | None = None if audio is None or rate is None else audio.numpy()
    channels, samples, sample_rate = (0, 0, 1) if sound is None or rate is None else (*sound.shape, rate)
    yield DecodedMediaHeader(
        video=DecodedVideoFormat(width=width, height=height, time_base=Fraction(1, FPS), pixel_aspect_ratio=Fraction(1),
                                 nominal_frame_rate=Fraction(FPS), color_primaries=1, color_transfer=1, color_matrix=1,
                                 color_range=1),
        audio=None if sound is None else DecodedAudioFormat(
            channels=channels, sample_rate=sample_rate, channel_layout=_LAYOUTS[channels][0],
            channel_names=_LAYOUTS[channels][1], time_base=Fraction(1, sample_rate)),
    )
    sent = 0
    for index in range(frames):
        yield DecodedVideoFrame(width=width, height=height, rgb=pixels[index].tobytes(), pts=index, duration=1,
                                time_base=Fraction(1, FPS), pixel_aspect_ratio=Fraction(1), color_primaries=1,
                                color_transfer=1, color_matrix=1, color_range=1)
        last = index + 1 == frames
        if sound is None or not (last or (index + 1) % FPS == 0):
            continue
        upto = samples if last else min(samples, (index + 1) * sample_rate // FPS)
        if upto > sent:
            yield DecodedAudioChunk(channels=channels, sample_count=upto - sent, sample_rate=sample_rate,
                                    channel_layout=_LAYOUTS[channels][0], channel_names=_LAYOUTS[channels][1],
                                    pcm_f32le=tuple(sound[c, sent:upto].tobytes() for c in range(channels)), pts=sent,
                                    time_base=Fraction(1, sample_rate))
            sent = upto
