"""LTX-2.5 Distilled: synchronized video and audio from text, the pictures it starts and ends on, the pictures and the
sound its prompt mentions, and a take it picks up from."""

from __future__ import annotations

import math
import secrets
from fractions import Fraction
from typing import Annotated, Literal

import msgspec
import numpy as np
import torch
from cozy_runtime import rng
from cozy_runtime.author import Adjustments, App, Context, Outputs, Seed, Telemetry
from cozy_runtime.fields import (
    Advanced, AspectRatio, Audio, FileBound, Image, ImagePreparation, Label, Pickup, Prompt, Range, Ratios, Video,
)
from cozy_runtime.media import ImageFile, VideoFile
from PIL import Image as PILImage

from . import annotators, controls, sound
from . import quantize as quantize  # its recipe
from .adapter import ICLora
from .annotators import PersonDetector, VideoDepth, WholeBody
from .enhancer import Enhancer
from .model import COMPONENTS, FPS, REFINE_SIGMAS, SIGMAS, Frame, Ltx25
from .prompts import names, rendered, sheet

app = App()
MODEL = "lightricks/ltx-2.5@0.2.0"
#: Lightricks' Ingredients IC-LoRA for LTX-2.5, mirrored (`Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients@5f83f6f7`).
INGREDIENTS = "lightricks/ingredients@0.1.0"
#: Lightricks' Union Control IC-LoRA, the LTX-2.3 adapter their LTX-2.5 workflow runs (`Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control@20a580c8`).
UNION_CONTROL = "lightricks/union-control@0.1.0"
#: Union Control's annotators, as Lightricks' workflow runs them: Video-Depth-Anything Small for depth, DWPose (Megvii's
#: YOLOX-L finding people, DWPose-l reading their bodies, hands and faces) for pose.
DEPTH = "depth-anything/video-depth-anything-small@0.1.0"
PEOPLE = "megvii-basedetection/yolox-l@0.1.0"
POSE = "yzd-v/dw-ll-ucoco-384@0.1.0"
#: Upstream's prompt enhancer for LTX-2.5 (`google/gemma-4-E2B-it@4adf1574`, ltx-pipelines `--prompt-enhancer-gemma-root`).
ENHANCER = "google/gemma-4-e2b-it@0.2.0"
#: Frames a pickup carries from its take into the new part: upstream's long-video carry
#: (ltx_pipelines/utils/args.py `--chunk-carry-frames`, 25), a causal latent start (`1 + 8k`).
CARRY = 25

Picture = Annotated[ImageFile, ImagePreparation(max_pixels=16 << 20)]


class Keyframe(msgspec.Struct, kw_only=True):
    #: A picture the video passes through.
    picture: Picture
    #: When, in seconds from the start; it lands on that exact frame.
    at_s: Annotated[float, Range(0, 20, unit="s")]


class Generate(msgspec.Struct, kw_only=True):
    #: What happens, in your words, with its sound and dialogue. A picture it mentions is a character, prop or place
    #: the video keeps (Lightricks' Ingredients reference sheet); a video it mentions sets the structure (Union
    #: Control); a sound it mentions is the soundtrack it is timed to.
    prompt: Prompt[Image | Video | Audio] = Prompt()
    #: The picture the video starts from.
    first_frame: Picture | None = None
    #: The picture the video ends on.
    last_frame: Picture | None = None
    #: Pictures the video passes through at given times (upstream's `--image PATH FRAME_IDX 1.0 guide`).
    keyframes: list[Keyframe] = msgspec.field(default_factory=list)
    duration_s: Annotated[float, Range(1, 20, unit="s")] = 5.0
    resolution: Literal["720p", "1080p"] = "1080p"
    #: Unset: the first frame's shape, else the last's, else 16:9. Output sides are on LTX's 64-pixel two-stage grid.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=0.5, max=2)] | None = None
    seed: Seed | None = None
    #: What a mentioned video gives Union Control, by Lightricks' annotators: its depth (their default), its people's
    #: pose (motion transfer), its edges, or the clip as it is, already a control map.
    control: Annotated[Literal["depth", "pose", "edges", "map"], Advanced(), Label("Reference video", {
        "depth": ("Depth", "Follows the depth and layout of the video you mention"),
        "pose": ("Motion transfer", "Moves like the people in the video you mention: body, hands and face"),
        "edges": ("Structure", "Follows the shapes and layout of the video you mention"),
        "map": ("As given", "The video is already a control map: edges, depth or pose")})] = "depth"
    #: Lightricks' enhancer (Gemma 4 E2B) first rewrites the prompt as LTX-2.5's captions read, as their templates do.
    prompt_upsampling: Annotated[bool, Advanced()] = True
    #: Render the soundtrack with the synchronized audio branch; false returns a silent MP4.
    audio: Annotated[bool, Advanced()] = True
    #: Pick up from here: the take goes on from its frame, its last second carried into the new part.
    pickup: Pickup | None = None


class VideoResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 40, media_types=("video/mp4",))]
    width: int
    height: int
    fps: float
    frame_count: int
    seed: int
    #: The text the model read.
    prompt: str


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


def waveform(planes: tuple[bytes, ...]) -> torch.Tensor:
    return torch.from_numpy(np.stack([np.frombuffer(plane, dtype=np.float32) for plane in planes]).copy())


def at_rate(audio: torch.Tensor, rate: int, target: int, channels: int) -> torch.Tensor:
    audio = sound.resample(audio, rate, target)
    return audio.expand(channels, -1).clone() if audio.shape[0] < channels else audio[:channels]


@app.tool(label="LTX-2.5", named="video",
          defaults={"model": MODEL, "ingredients": INGREDIENTS, "union_control": UNION_CONTROL, "enhancer": ENHANCER,
                    "depth": DEPTH, "people": PEOPLE, "pose": POSE},
          components={"model": COMPONENTS},
          on_demand=("ingredients", "union_control", "enhancer", "depth", "people", "pose"), memoize=True)
def generate(ctx: Context, payload: Generate, model: Ltx25, ingredients: ICLora, union_control: ICLora, enhancer: Enhancer,
             depth: VideoDepth, people: PersonDetector, pose: WholeBody, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> VideoResult:
    """LTX-2.5 Distilled, in two stages: a scene and its synchronized sound, then x2 spatial refinement. A first frame
    replaces the first latent and a last frame guides the last frame (upstream's `--image PATH FRAME_IDX 1.0`); mentioned
    pictures are the Ingredients IC-LoRA's reference sheet in stage 1 (upstream's IC-LoRA stages), a mentioned video the
    Union Control IC-LoRA's reference there; a mentioned sound is
    held as the soundtrack through both (`A2VidPipelineTwoStage`); a pickup carries its take's last 25 frames and their
    sound as the new part's opening (upstream's long-video carry)."""
    text = names(payload.prompt.text)
    if not text:
        raise ValueError("LTX-2.5 needs a prompt describing the video and its sound")
    seed = int(payload.seed) if payload.seed is not None else secrets.randbits(32)
    seconds = min(20.0, max(1.0, payload.duration_s))
    frames = min(481, 8 * math.ceil(seconds * FPS / 8) + 1)
    if seconds != payload.duration_s:
        adj.warn("duration_s", f"LTX-2.5 renders 1-20 seconds, so this clip runs {(frames - 1) / FPS:g} seconds")
    mentioned = payload.prompt.references
    panels = [key for key, value in mentioned.items() if isinstance(value, Image)]
    sounds = [key for key, value in mentioned.items() if isinstance(value, Audio)]
    videos = [key for key, value in mentioned.items() if isinstance(value, Video)]
    if videos and panels:
        adj.warn("prompt", f"Ingredients and Union Control take references at different sizes, so LTX-2.5 runs one: the "
                           f"pictures; {', '.join(f'<{key}>' for key in videos)} goes unused")
        videos = []
    if len(videos) > 1:
        adj.warn("prompt", f"Union Control takes one reference video, so it goes without {', '.join(f'<{key}>' for key in videos[1:])}")
    first = payload.first_frame.read().convert("RGB") if payload.first_frame else None
    last = payload.last_frame.read().convert("RGB") if payload.last_frame else None
    pickup, kept, kept_sound, carry = payload.pickup, None, None, 0
    if pickup is not None:
        done = min(pickup.out_frame, math.floor(pickup.take.file.probe().duration * FPS))
        carry = 8 * ((min(CARRY, done) - 1) // 8) + 1 if done >= 9 else 0
        if carry == 0:
            adj.warn("pickup", f"Pick up from here at frame {done} carries nothing, so the whole video renders")
            pickup = None
        else:
            read = pickup.take.file.read(end=Fraction(done, FPS), fps=FPS)
            kept = [PILImage.frombytes("RGB", (read.width, read.height), frame) for frame in read.frames_rgb[:done]]
            kept_sound = read.soundtrack
            frames = max(frames, 8 * math.ceil((carry + 1) / 8) + 1)
            for name, given in (("first_frame", first), ("aspect_ratio", payload.aspect_ratio)):
                if given is not None:
                    adj.warn(name, f"the take picked up sets the opening and the shape; {name} is ignored")
            first = None
    if len(sounds) > 1:
        adj.warn("prompt", f"LTX-2.5 takes one soundtrack, so it goes without {', '.join(f'<{key}>' for key in sounds[1:])}")
    soundtrack = None
    if sounds and pickup is not None:
        adj.warn("prompt", f"the take picked up carries its own sound, so <{sounds[0]}> is not the soundtrack")
    elif sounds:
        adj.warn("prompt", f"<{sounds[0]}> is the soundtrack the video is timed to: LTX-2.5 takes no voice or music style "
                           "from a sound")
        clip = mentioned[sounds[0]]
        assert isinstance(clip, Audio)
        soundtrack = clip.file.read(end=Fraction(frames, FPS))
        fits = 8 * ((math.floor(float(soundtrack.duration) * FPS) - 1) // 8) + 1
        if fits < frames:
            if fits < 9:
                raise ValueError(f"<{sounds[0]}> lasts {float(soundtrack.duration):.2f} s: a soundtrack needs at least 9 frames' worth")
            adj.warn("duration_s", f"<{sounds[0]}> lasts {float(soundtrack.duration):.2f} s, so the clip runs {(fits - 1) / FPS:g} s")
            frames = fits
    shape = kept[-1].size if kept else first.size if first is not None else last.size if last is not None else None
    width, height = canvas(payload.resolution, None if kept else payload.aspect_ratio, shape)
    pictures: list[Frame] = [(picture, frame) for picture, frame in ((first, 0), (last, frames - 1)) if picture is not None]
    for index, keyframe in enumerate(payload.keyframes):
        frame = round(keyframe.at_s * FPS)
        if frame > frames - 1:
            adj.warn(f"keyframes.{index}.at_s", f"{keyframe.at_s:g} s is past the clip's end, so it lands on its last frame")
        frame = min(frame, frames - 1)
        if frame == 0 and any(at == 0 for _, at in pictures):
            adj.warn(f"keyframes.{index}", "the video already opens on a picture, so this keyframe moves to frame 1")
            frame = 1
        pictures.append((keyframe.picture.read().convert("RGB"), frame))
    if kept is not None:
        pictures.insert(0, (kept[-carry:], 0))
    video_text = text
    if payload.prompt_upsampling:
        labels = [key.replace(".", " ") for key in [*panels, *videos]]
        with tel.stage("enhancing prompt"):
            rewritten = enhancer.rewrite(text, kept[-1] if kept else first, labels)
        if rewritten is None:
            adj.warn("prompt_upsampling", f"the enhancer's rewrite lost a reference ({', '.join(labels)}), so the prompt "
                                          "goes as written")
        video_text = rewritten or text
    prompt = rendered(video_text, panels)
    held = None
    if soundtrack is not None or kept_sound is not None:
        with tel.stage("encoding sound"):
            if soundtrack is not None:
                held = model.encode_audio(waveform(soundtrack.pcm_f32le), soundtrack.sample_rate)
            elif kept_sound is not None:
                start = round((len(kept or ()) - carry) / FPS * kept_sound.sample_rate)
                held = model.encode_audio(waveform(kept_sound.pcm_f32le)[:, start:], kept_sound.sample_rate)
    generator = rng.generator(seed)
    with tel.stage("encoding prompt"):
        embeds = model.encode(prompt)

    adapter, reference = None, None
    if panels:
        with tel.stage("making the reference sheet"):
            board = sheet([value.file.read() for value in (mentioned[key] for key in panels) if isinstance(value, Image)],
                          width // 2, height // 2)
        adapter, reference = ingredients, [board] * frames
    elif videos:
        clip = mentioned[videos[0]]
        assert isinstance(clip, Video)
        with tel.stage("reading the reference video"):
            read = clip.file.read(end=Fraction(frames, FPS), fps=FPS, audio=False)
            shown = [PILImage.frombytes("RGB", (read.width, read.height), frame) for frame in read.frames_rgb[:frames]]
        with tel.stage("annotating the reference video"):
            reference = (depth.gray(shown) if payload.control == "depth"
                         else annotators.drawn(shown, people, pose) if payload.control == "pose"
                         else controls.edges(annotators.scaled(shown)) if payload.control == "edges" else shown)
            # A clip shorter than the video holds its last frame, so every frame has its reference.
            reference += reference[-1:] * (frames - len(reference))
        adapter = union_control

    def stage_one() -> tuple[torch.Tensor, torch.Tensor]:
        return model.denoise(embeds, pictures, width=width // 2, height=height // 2, frames=frames, reference=reference,
                             downscale=adapter.graph.downscale if adapter is not None else 1, held=held,
                             generator=generator, check=ctx.raise_if_cancelled,
                             progress=lambda step: ctx.progress(step, len(SIGMAS)))

    with tel.stage("generating video and audio"):
        latents = adapter.applied(model.graph.sites, stage_one) if adapter is not None else stage_one()
    ctx.raise_if_cancelled()
    with tel.stage("upscaling latents"):
        high = model.upscale(latents[0])
    with tel.stage("refining video and audio"):
        latents = model.denoise(embeds, pictures, width=width, height=height, frames=frames, latents=(high, latents[1]),
                                held=held, generator=generator, check=ctx.raise_if_cancelled,
                                progress=lambda step: ctx.progress(step, len(REFINE_SIGMAS)))
    ctx.raise_if_cancelled()
    with tel.stage("decoding video"):
        pixels = model.decode_video(latents[0])
    audio: torch.Tensor | None = None
    rate: int | None = None
    if payload.audio:
        with tel.stage("decoding audio"):
            if soundtrack is not None:
                # The soundtrack goes out as given, as upstream's A2V returns the source waveform.
                rate = soundtrack.sample_rate
                audio = waveform(soundtrack.pcm_f32le)[:, : round(len(pixels) / FPS * rate)]
            else:
                audio, rate = model.decode_audio(latents[1])
    if kept is not None:
        with tel.stage("joining the take"):
            # The take up to its frame, as it was, then what follows the carried frames.
            joined = [np.asarray(frame.resize((width, height), PILImage.Resampling.LANCZOS) if frame.size != (width, height)
                                 else frame) for frame in kept]
            pixels = np.concatenate([np.stack(joined), pixels[carry:]])
            if audio is not None and rate is not None:
                before = (at_rate(waveform(kept_sound.pcm_f32le), kept_sound.sample_rate, rate, audio.shape[0])
                          if kept_sound is not None else torch.zeros(audio.shape[0], round(len(kept) / FPS * rate)))
                audio = torch.cat([before, audio[:, round(carry / FPS * rate):]], dim=1)
    with tel.stage("saving video"):
        video = out.save_video(pixels, fps=FPS, audio=audio, sample_rate=rate)
    tel.log("ltx-2.5", width=width, height=height, frames=len(pixels), audio=payload.audio,
            sampler="diffusers-euler", stages=2, steps=len(SIGMAS) + len(REFINE_SIGMAS))
    return VideoResult(video, width, height, float(FPS), len(pixels), seed, prompt)
