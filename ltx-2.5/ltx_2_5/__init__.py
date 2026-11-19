"""LTX-2.5 Distilled: synchronized video and audio from text or a starting picture, a take continued from its last
frames and their sound, and a range of a take made again between the frames on both sides."""

from __future__ import annotations

import itertools
import math
import secrets
from collections.abc import Iterable, Iterator
from fractions import Fraction
from typing import Annotated, Literal

import msgspec
import numpy as np
import torch
from cozy_runtime import rng
from cozy_runtime.author import Adjustments, App, Context, OutputError, Outputs, SavedVideo, Seed, Telemetry
from cozy_runtime.fields import (
    Advanced, AspectRatio, FileBound, ImagePreparation, Label, Pickup, Placeholder, Range, Ratios,
)
from cozy_runtime.media import (
    DecodedAudioChunk, DecodedAudioFormat, DecodedMediaHeader, DecodedVideoFormat, DecodedVideoFrame, ImageFile,
    VideoFile,
)
from cozy_runtime.media import DecodedMediaEvent as Event
from PIL import Image

from . import edits, mel
from . import quantize as quantize  # its recipe
from .edits import Edit, Piece, Side, Waveform
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
#: Frames read from a take at once: two seconds, never the whole take.
_READ = 2 * FPS
#: Steps a canvas takes: the ancestral stage, then the refine.
_STEPS = len(SIGMAS) + len(REFINE_SIGMAS)


class Generate(msgspec.Struct, kw_only=True):
    #: Describe the picture, motion, camera, sounds and dialogue to generate.
    prompt: str
    first_frame: Annotated[ImageFile, ImagePreparation(max_pixels=16 << 20)] | None = None
    #: With a pickup, the length of what follows it.
    duration_s: Annotated[float, Range(1, 20, unit="s")] = 5.0
    resolution: Literal["720p", "1080p"] = "1080p"
    #: Unset: the first frame's shape, else 16:9. Output sides are on LTX's 64-pixel two-stage grid.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=0.5, max=2)] | None = None
    seed: Seed | None = None
    #: Render the soundtrack with the synchronized audio branch; false returns a silent MP4.
    audio: Annotated[bool, Advanced()] = True
    #: Pick up from here: the take goes on from its frame, its last 25 frames' motion and their sound carried into
    #: what follows. The take sets the size.
    pickup: Pickup | None = None


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


def grid(width: int, height: int) -> tuple[int, int]:
    """The size a take is continued or redone at: its own, the sides on the VAE's 32-pixel grid, at most 1080p's
    area."""
    scale = min(1.0, math.sqrt(1920 * 1088 / (width * height)))
    return max(64, round(width * scale / 32) * 32), max(64, round(height * scale / 32) * 32)


@app.tool(label="LTX-2.5", named="video", defaults={"model": MODEL, "text_encoder": TEXT_ENCODER}, components=_COMPONENTS,
          memoize=True)
def generate(ctx: Context, payload: Generate, model: Ltx25, text_encoder: Ltx25TextEncoder, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> VideoResult:
    """LTX-2.5 Distilled, in two stages: a scene and its synchronized sound, then x2 spatial refinement. A pickup
    instead continues its take on one canvas: the take's last 25 frames and their sound held, what follows made."""
    if not payload.prompt.strip():
        raise ValueError("LTX-2.5 needs a prompt describing the video and its sound")
    seed = int(payload.seed) if payload.seed is not None else secrets.randbits(32)
    if payload.pickup is not None:
        picked = _pickup(ctx, payload, payload.pickup, seed, model, text_encoder, out, tel, adj)
        if picked is not None:
            return picked
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
        sound = None if audio is None or rate is None else (np.ascontiguousarray(audio.numpy(), dtype=np.float32), rate)
        video = out.save_video_stream(_stream(pixels.shape[2], pixels.shape[1], (frame.tobytes() for frame in pixels),
                                              sound)).video
    tel.log("ltx-2.5", width=width, height=height, frames=len(pixels), audio=payload.audio,
            sampler="diffusers-euler", stages=2, steps=len(SIGMAS) + len(REFINE_SIGMAS))
    return VideoResult(video, width, height, float(FPS), len(pixels), seed, payload.prompt)


def _pickup(ctx: Context, payload: Generate, pickup: Pickup, seed: int, model: Ltx25, text_encoder: Ltx25TextEncoder,
            out: Outputs, tel: Telemetry, adj: Adjustments) -> VideoResult | None:
    """The take to its frame, then what follows, made on one canvas holding the take's last 25 frames and their
    sound (upstream's long-video carry; LTX Desktop's extend is the same canvas). The take's frames come back as they
    were, blending into the made ones over the last 16; the sound crossfades over 40 ms. None: nothing to carry."""
    take = pickup.take.file
    probe = take.probe()
    count = _frame_count(probe.duration)
    done = min(pickup.out_frame, count)
    if done < 1:
        adj.warn("pickup", f"Pick up from here at frame {pickup.out_frame} carries nothing, so the whole video renders")
        return None
    for name, given in (("first_frame", payload.first_frame), ("aspect_ratio", payload.aspect_ratio)):
        if given is not None:
            adj.warn(name, f"the take picked up sets the opening and the shape; {name} is ignored")
    asked = math.ceil(min(20.0, max(1.0, payload.duration_s)) * FPS)
    piece = edits.pickup(done, asked)
    new = piece.end - piece.start
    if new < asked:
        adj.warn("duration_s", f"a canvas holds 20 s with the carry, so {new / FPS:g} s follow the pickup")
    size = grid(probe.width, probe.height)
    sound = _soundtrack(take, Fraction(done, FPS))
    with tel.stage("encoding prompt"):
        embeds = text_encoder.encode(payload.prompt)
    edit = Edit()
    latents = _render(ctx, model, take, piece, edit, sound, embeds, size, seed=seed, redo_video=True,
                      redo_sound=True, steps=_STEPS, tel=tel)
    with tel.stage("decoding video"):
        edit.put(piece, model.decode_video(latents[0]))
    track: tuple[Waveform, int] | None = None
    if payload.audio:
        with tel.stage("decoding audio"):
            made, made_rate = model.decode_audio(latents[1])
        own, rate = sound if sound is not None else (np.zeros((made.shape[0], 0), dtype=np.float32), made_rate)
        wave = np.zeros((own.shape[0], round((done + new) / FPS * rate)), dtype=np.float32)
        kept = min(own.shape[1], round(done / FPS * rate))
        wave[:, :kept] = own[:, :kept]
        edits.splice(wave, _fitted(made, made_rate, rate, own.shape[0]), round(piece.first / FPS * rate),
                     round(done / FPS * rate), wave.shape[1], lead="context", tail="none",
                     fade=round(edits.CROSSFADE_S * rate))
        track = wave, rate
    with tel.stage("saving video"):
        saved = _save(out, adj, take, done - min(edits.FADE, piece.before), done, done, edit, track)
    tel.log("ltx-2.5 pickup", width=size[0], height=size[1], carry=piece.before, frames=done + new,
            sampler="ancestral+euler", steps=_STEPS)
    return VideoResult(saved.video, saved.width, saved.height, float(saved.frame_rate), saved.frame_count, seed,
                       payload.prompt)


class Retake(msgspec.Struct, kw_only=True):
    #: The take to redo a range of; everything outside the range stays as it is.
    video: VideoFile
    #: Where the range starts, in seconds from the take's start.
    start_s: Annotated[float, Range(0, 3600, unit="s")]
    #: Where it ends. A shot change within 0.8 s of either edge becomes that edge, with nothing held across it.
    end_s: Annotated[float, Range(0, 3600, unit="s")]
    #: What happens in the range, with its sound and dialogue.
    prompt: Annotated[str, Placeholder("What happens instead, and what is heard")] = ""
    redo: Annotated[Literal["both", "video", "sound"], Label("Redo", {
        "both": ("Picture and sound", "Both are made again in the range"),
        "video": ("Picture only", "The take's sound stays as it is"),
        "sound": ("Sound only", "The take's picture stays as it is")})] = "both"
    seed: Seed | None = None


@app.tool(label="LTX-2.5 Retake", named="retake", defaults={"model": MODEL, "text_encoder": TEXT_ENCODER},
          components={"model": ("transformer", "connectors", "vae", "audio_vae", "vocoder"),
                      "text_encoder": ("text_encoder",)}, memoize=True)
def retake(ctx: Context, payload: Retake, model: Ltx25, text_encoder: Ltx25TextEncoder, out: Outputs, tel: Telemetry,
           adj: Adjustments) -> VideoResult:
    """A range of a take made again from text, the frames and sound on both sides held as context: upstream's
    retake (`TemporalRegionMask`) on LTX Desktop's window and sampler, at most 20 s a canvas, a longer range in pieces
    left to right. Only the range's span is read for the model. The take's own frames and sound come back outside the
    range, blending into the made ones over 16 frames and 40 ms; at a shot change the range ends there instead."""
    text = payload.prompt.strip()
    if not text:
        raise ValueError("LTX-2.5 Retake needs a prompt describing the range made again")
    seed = int(payload.seed) if payload.seed is not None else secrets.randbits(32)
    take = payload.video
    probe = take.probe()
    count = _frame_count(probe.duration)
    if count < 9:
        raise ValueError(f"the take has {count} frames: a retake needs at least 9 ({9 / FPS:g} s)")
    start, end = max(0, round(payload.start_s * FPS)), min(count, round(payload.end_s * FPS))
    if end <= start:
        raise ValueError(f"end_s ({payload.end_s:g}) must come after start_s ({payload.start_s:g}), within the take's "
                         f"{count / FPS:g} s")
    if end < round(payload.end_s * FPS):
        adj.warn("end_s", f"the take lasts {count / FPS:g} s, so the range ends there")
    with tel.stage("finding shot changes"):
        cut_end, cut_start = _cut_near(take, count, end), _cut_near(take, count, start)
    if cut_end is not None and cut_end > start:
        end = cut_end
        adj.warn("end_s", f"the shot changes at {end / FPS:.2f} s, so the range ends on that cut")
    else:
        cut_end = None
    if cut_start is not None and cut_start < end:
        start = cut_start
        adj.warn("start_s", f"the shot changes at {start / FPS:.2f} s, so the range starts on that cut")
    else:
        cut_start = None
    pieces = edits.retake(start, end, before=0 if cut_start is not None else start,
                          after=0 if cut_end is not None else count - end)
    size = grid(probe.width, probe.height)
    redo_video, redo_sound = payload.redo != "sound", payload.redo != "video"
    sound = _soundtrack(take, None)
    lead: Side = "context" if start and cut_start is None else "inside" if cut_start is not None else "none"
    tail: Side = "context" if end < count and cut_end is None else "inside" if cut_end is not None else "none"
    with tel.stage("encoding prompt"):
        embeds = text_encoder.encode(text)
    edit = Edit()
    for index, piece in enumerate(pieces):
        latents = _render(ctx, model, take, piece, edit, sound, embeds, size, seed=seed + index, redo_video=redo_video,
                          redo_sound=redo_sound, steps=_STEPS * len(pieces), done=_STEPS * index, tel=tel)
        if redo_video:
            with tel.stage("decoding video"):
                edit.put(piece, model.decode_video(latents[0]))
        if redo_sound:
            with tel.stage("decoding audio"):
                made, made_rate = model.decode_audio(latents[1])
            if sound is None:
                sound = (np.zeros((made.shape[0], round(count / FPS * made_rate)), dtype=np.float32), made_rate)
            wave, rate = sound
            edits.splice(wave, _fitted(made, made_rate, rate, wave.shape[0]), round(piece.first / FPS * rate),
                         round(piece.start / FPS * rate), round(piece.end / FPS * rate),
                         lead=lead if index == 0 else "context", tail=tail if index == len(pieces) - 1 else "none",
                         fade=round(edits.CROSSFADE_S * rate))
    with tel.stage("saving video"):
        lo = pieces[0].start - min(edits.FADE, pieces[0].before)
        hi = min(count, pieces[-1].end + min(edits.FADE, pieces[-1].after))
        saved = _save(out, adj, take, lo, hi, count, edit, sound)
    tel.log("ltx-2.5 retake", width=size[0], height=size[1], frames=count, start=start, end=end, pieces=len(pieces),
            redo=payload.redo, cut_start=cut_start is not None, cut_end=cut_end is not None,
            sampler="ancestral+euler", steps=_STEPS * len(pieces))
    return VideoResult(saved.video, saved.width, saved.height, float(saved.frame_rate), saved.frame_count, seed, text)


def _render(ctx: Context, model: Ltx25, take: VideoFile, piece: Piece, edit: Edit,
            sound: tuple[Waveform, int] | None, embeds: tuple[torch.Tensor, torch.Tensor], size: tuple[int, int], *,
            seed: int, redo_video: bool, redo_sound: bool, steps: int, tel: Telemetry,
            done: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """One canvas: its frames read from the take at `size` (as `edit` leaves them so far), the frames past what it
    holds padded with the last one it holds, its sound held where there is any; then made."""
    with tel.stage("reading the take"):
        pixels = np.empty((piece.frames, size[1], size[0], 3), dtype=np.uint8)
        held = _sized(take, piece.first, min(piece.limit, piece.first + piece.frames), size, pixels)
        if not held:
            raise ValueError(f"the take has no frame at {piece.first / FPS:g} s to hold")
        for index in range(held):
            pixels[index] = edit.at(piece.first + index, pixels[index])
        pixels[held:] = pixels[held - 1]
    with tel.stage("encoding the take"):
        video = model.encode_video(pixels)
        del pixels
        audio = None
        if sound is not None:
            track, rate = sound
            lo, hi = round(piece.first / FPS * rate), round((piece.first + piece.frames) / FPS * rate)
            clip = np.pad(track[:, lo:hi], ((0, 0), (0, max(0, hi - lo - track[:, lo:hi].shape[1]))))
            audio = model.encode_audio(torch.from_numpy(np.ascontiguousarray(clip)), rate)
    span = piece.span()
    with tel.stage("generating video and audio"):
        return model.canvas(embeds, video, audio, video_span=span if redo_video else None,
                            audio_span=span if redo_sound else None, seed=seed, check=ctx.raise_if_cancelled,
                            progress=lambda step: ctx.progress(done + step, steps))


def _frame_count(duration: Fraction) -> int:
    """Frames a take has on LTX's 24 fps clock."""
    return round(duration * FPS)


def _slot(frame: int, rate: Fraction) -> int:
    """A frame on the 24 fps clock as a frame on clock `rate`."""
    return math.floor(Fraction(frame) * rate / FPS + Fraction(1, 2))


def _shown(slot: int, rate: Fraction) -> int:
    """The frame on the 24 fps clock showing at frame `slot` of clock `rate`."""
    return math.floor(Fraction(slot) * FPS / rate + Fraction(1, 2))


def _chunks(take: VideoFile, first: int, stop: int,
            rate: Fraction = Fraction(FPS)) -> Iterator[tuple[tuple[int, int], tuple[bytes, ...]]]:
    """The take's frames `[first, stop)` on clock `rate` as decoded RGB, two seconds at a time, with their size."""
    step = max(1, round(_READ * rate / FPS))
    for at in range(first, stop, step):
        read = take.read(start=Fraction(at) / rate, end=Fraction(min(stop, at + step)) / rate, fps=rate, audio=False)
        yield (read.width, read.height), read.frames_rgb


def _sized(take: VideoFile, first: int, stop: int, size: tuple[int, int], into: VideoPixels) -> int:
    """The take's frames `[first, stop)` at `size` into `into`'s first rows; how many there were."""
    index = 0
    for shape, chunk in _chunks(take, first, stop):
        for rgb in chunk:
            frame = Image.frombytes("RGB", shape, rgb)
            if frame.size != size:
                frame = frame.resize(size, Image.Resampling.LANCZOS)
            into[index] = np.asarray(frame)
            index += 1
    return index


def _cut_near(take: VideoFile, count: int, edge: int) -> int | None:
    """The shot change within `edits.NEAR` frames of take frame `edge` (24 fps clock), the closest; None when there is
    none. The take is read on its own clock, `edits.FLASH` frames further each way, so a change at either end is seen
    to hold."""
    if not 0 < edge < count:
        return None
    first, last = max(1, edge - edits.NEAR), min(count, edge + edits.NEAR + 1)
    begin = max(0, first - 1 - edits.FLASH)
    slots: list[int] = []

    def pictures() -> Iterator[edits.Frame]:
        origin: Fraction | None = None
        with take.stream(start=Fraction(begin, FPS), end=Fraction(min(count, last + edits.FLASH), FPS),
                         audio=False) as events:
            for event in events:
                if isinstance(event, DecodedVideoFrame):
                    origin = event.start_time if origin is None else origin
                    slots.append(begin + round((event.start_time - origin) * FPS))
                    yield np.frombuffer(event.rgb, dtype=np.uint8).reshape(event.height, event.width, 3)

    return edits.nearest([slots[index] for index in edits.shot_changes(pictures())], edge)


def _soundtrack(take: VideoFile, end: Fraction | None) -> tuple[Waveform, int] | None:
    """The take's sound to `end` (all of it: it is small beside the pictures), at most two channels; None if silent."""
    if take.probe().soundtrack is None:
        return None
    parts: list[Waveform] = []
    rate = 0
    with take.stream(end=end, pictures=False) as events:
        for event in events:
            if isinstance(event, DecodedAudioChunk):
                rate = event.sample_rate
                parts.append(np.stack([np.frombuffer(pcm, dtype=np.float32) for pcm in event.pcm_f32le[:2]]))
    if not parts:
        return None
    return np.ascontiguousarray(np.concatenate(parts, axis=1)), rate


def _fitted(made: torch.Tensor, rate: int, target: int, channels: int) -> Waveform:
    """Made sound [channels, samples] at `rate` as `target`'s rate and the take's channel count."""
    sound = mel.resample(made.float(), rate, target)
    sound = sound.mean(dim=0, keepdim=True) if channels == 1 else sound[:channels]
    return np.ascontiguousarray(sound.numpy(), dtype=np.float32)


def _save(out: Outputs, adj: Adjustments, take: VideoFile, lo: int, hi: int, stop: int, edit: Edit,
          sound: tuple[Waveform, int] | None) -> SavedVideo:
    """The take's first `stop` frames (24 fps clock) with the edit over them, then the frames it made past them, under
    `sound`. Only the window `[lo, hi)` (the edit, its blends and the sound's crossfades) is written, on the take's own
    clock: `Outputs.splice_video` copies every other packet of the take, byte for byte. On a take not at 24 fps the
    made frames show on its clock, some twice (MiniMax H3's retake does the same). A take it can't splice
    (`video_copy_incompatible`), or a silent take given sound, is written again whole at 24 fps, with one warning."""
    probe = take.probe()
    rate = probe.frame_rate or Fraction(FPS)
    if probe.soundtrack is None and sound is not None:
        said = "the take has no sound track for the new sound"
    else:
        try:
            saved = out.splice_video(take, _slot(lo, rate), None if hi >= stop else _slot(hi, rate),
                                     _written(take, rate, lo, hi, edit, sound))
        except OutputError as error:
            if error.code != "video_copy_incompatible":
                raise
            said = f"the take can't be spliced by packet copy ({error.message})"
        else:
            if rate != FPS:
                adj.warn("video", f"the take runs at {float(rate):g} fps, so the new part's {FPS} fps frames show on "
                                  "its clock, some twice")
            return saved
    adj.warn("video", f"{said}, so the whole take is written again at {FPS} fps")
    return out.save_video_stream(_written(take, Fraction(FPS), 0, stop, edit, sound))


def _written(take: VideoFile, rate: Fraction, first: int, stop: int, edit: Edit,
             sound: tuple[Waveform, int] | None) -> Iterator[Event]:
    """The take's frames from `first` to before `stop` (24 fps clock) on clock `rate` at the take's size, each the
    take's own with the edit's frame showing then mixed over it, then the made frames that follow, under `sound` from
    the first frame's time."""
    a, b = _slot(first, rate), _slot(stop, rate)
    chunks = _chunks(take, a, b, rate)
    size, head = next(chunks)

    def frames() -> Iterator[bytes]:
        at = a
        for chunk in itertools.chain((head,), (rest for _, rest in chunks)):
            for rgb in chunk:
                index = _shown(at, rate)
                if index in edit.frames:
                    own = np.frombuffer(rgb, dtype=np.uint8).reshape(size[1], size[0], 3)
                    yield edits.mix(own, _resized(edit.frames[index], size), edit.weights[index]).tobytes()
                else:
                    yield rgb
                at += 1
        while (index := _shown(at, rate)) in edit.frames:
            yield _resized(edit.frames[index], size).tobytes()
            at += 1

    track = None
    if sound is not None:
        wave, sample_rate = sound
        # Made frames past `stop` (a pickup's new part) take the sound to its end; else it ends with `stop`.
        end = wave.shape[1] if max(edit.frames, default=-1) >= stop else round(Fraction(b) / rate * sample_rate)
        track = np.ascontiguousarray(wave[:, round(Fraction(a) / rate * sample_rate):end]), sample_rate
    yield from _stream(size[0], size[1], frames(), track, rate)


def _resized(frame: edits.Frame, size: tuple[int, int]) -> edits.Frame:
    if (frame.shape[1], frame.shape[0]) == size:
        return frame
    return np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS))


_LAYOUTS = {1: ("mono", ("FC",)), 2: ("stereo", ("FL", "FR"))}


def _stream(width: int, height: int, frames: Iterable[bytes], track: tuple[Waveform, int] | None,
            rate: Fraction = Fraction(FPS)) -> Iterator[Event]:
    """A clip on clock `rate` for `save_video_stream` or a splice's window, BT.709 as H3's tools write theirs: each
    frame as it comes, each second of frames followed by its sound, so the encoder never takes the whole video at
    once."""
    sound = None if track is None else track[0]
    channels, samples, sample_rate = (0, 0, 1) if track is None else (*track[0].shape, track[1])
    second = max(1, round(rate))
    yield DecodedMediaHeader(
        video=DecodedVideoFormat(width=width, height=height, time_base=1 / rate, pixel_aspect_ratio=Fraction(1),
                                 nominal_frame_rate=rate, color_primaries=1, color_transfer=1, color_matrix=1,
                                 color_range=1),
        audio=None if sound is None else DecodedAudioFormat(
            channels=channels, sample_rate=sample_rate, channel_layout=_LAYOUTS[channels][0],
            channel_names=_LAYOUTS[channels][1], time_base=Fraction(1, sample_rate)),
    )
    sent = 0

    def chunk(upto: int) -> DecodedAudioChunk:
        assert sound is not None
        return DecodedAudioChunk(channels=channels, sample_count=upto - sent, sample_rate=sample_rate,
                                 channel_layout=_LAYOUTS[channels][0], channel_names=_LAYOUTS[channels][1],
                                 pcm_f32le=tuple(sound[c, sent:upto].tobytes() for c in range(channels)), pts=sent,
                                 time_base=Fraction(1, sample_rate))

    for index, rgb in enumerate(frames, 1):
        yield DecodedVideoFrame(width=width, height=height, rgb=rgb, pts=index - 1, duration=1, time_base=1 / rate,
                                pixel_aspect_ratio=Fraction(1), color_primaries=1, color_transfer=1, color_matrix=1,
                                color_range=1)
        upto = min(samples, math.floor(index * sample_rate / rate))
        if sound is not None and index % second == 0 and upto > sent:
            yield chunk(upto)
            sent = upto
    if sound is not None and samples > sent:
        yield chunk(samples)
