"""PAN2: a scene unfolding under timed actions, from text or its first frame."""

from __future__ import annotations

import json
import math
import secrets
from pathlib import Path
from typing import Annotated

import msgspec
from cozy_runtime import vlm
from cozy_runtime.author import Adjustments, App, Context, Outputs, OutputError, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, ImagePreparation, Range, Ratios
from cozy_runtime.media import ImageFile, VideoFile
from cozy_runtime.vlm import VisionLM

from .model import COMPONENTS, DEFAULT_STEPS, FPS, Pan2

app = App()
MODEL = "ifm/pan2@0.1.0"

REWRITE = Path(__file__).with_name("prompt-policy.md").read_text()


class Generate(msgspec.Struct, kw_only=True):
    prompt: str
    first_frame: Annotated[ImageFile, ImagePreparation(max_pixels=16 << 20)] | None = None
    duration_s: Annotated[float, Range(1, 20, unit="s")] = 10.0
    #: Unset: the first frame's shape, else the model's native 16:9.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=0.5, max=2)] | None = None
    seed: Seed | None = None
    prompt_upsampling: Annotated[bool, Advanced()] = True
    steps: Annotated[int, Range(1, 100), Advanced()] = DEFAULT_STEPS
    guidance: Annotated[float, Range(1, 10), Advanced()] = 2.5
    negative_prompt: Annotated[str | None, Advanced()] = None


class VideoResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 40, media_types=("video/mp4",))]
    width: int
    height: int
    fps: float
    frame_count: int
    seed: int
    prompt: str


def canvas(ratio: AspectRatio | None, first: tuple[int, int] | None) -> tuple[int, int]:
    if ratio is None:
        w, h = first or (1248, 704)
        r = w / h
    else:
        rw, rh = (float(value) for value in str(ratio).split(":"))
        r = rw / rh
    r = min(2.0, max(0.5, r))
    area = 1248 * 704
    return max(32, round(math.sqrt(area * r) / 32) * 32), max(32, round(math.sqrt(area / r) / 32) * 32)


def caption(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.partition("\n")[2].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("refined_prompt"), str):
            value = json.loads(value["refined_prompt"])
    except json.JSONDecodeError as exc:
        raise OutputError(f"PAN2's prompt rewriter did not return a JSON caption: {exc.msg}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("events"), list):
        raise OutputError("PAN2's prompt rewriter returned no structured event timeline")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@app.tool(label="PAN2", named="video", defaults={"model": MODEL, "reasoner": "qwen/qwen3.5-9b@0.2.0"},
          components={"model": COMPONENTS}, on_demand=("reasoner",), memoize=True)
def generate(ctx: Context, payload: Generate, model: Pan2, reasoner: VisionLM, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> VideoResult:
    """PAN2's silent video, at 24 fps for up to 20 seconds. A first frame animates that scene."""
    seed = int(payload.seed) if payload.seed is not None else secrets.randbits(32)
    seconds = min(20.0, max(1.0, payload.duration_s))
    frames = min(481, 4 * math.ceil(seconds * FPS / 4) + 1)
    picture = payload.first_frame.read().convert("RGB") if payload.first_frame else None
    width, height = canvas(payload.aspect_ratio, picture.size if picture else None)
    prompt = payload.prompt
    if not prompt.strip():
        raise ValueError("PAN2 needs a prompt describing the scene and its actions")
    if payload.prompt_upsampling:
        with tel.stage("writing action timeline"):
            replies = reasoner.ask(
                [vlm.Question(f"Clip duration: {(frames - 1) / FPS:g} seconds.\nUser request: {prompt}", [picture] if picture else [], [])],
                system=REWRITE, thinking=False, stop=[], max_new_tokens=3072, presence_penalty=0.0,
                temperature=0.0, top_p=1.0, seed=seed,
                progress=lambda _: ctx.raise_if_cancelled(),
            )
        prompt = caption(replies[0])
    else:
        prompt = caption(prompt)
    if seconds != payload.duration_s:
        adj.warn("duration_s", f"PAN2 renders 1-20 seconds, so this clip runs {(frames - 1) / FPS:g} seconds")
    with tel.stage("encoding caption"):
        embeds = model.encode(prompt, frames, payload.guidance, payload.negative_prompt)
    image = None
    if picture is not None:
        with tel.stage("encoding first frame"):
            image = model.encode_image(picture, width, height)
    with tel.stage("generating video"):
        latents = model.denoise(embeds, image, width=width, height=height, frames=frames, steps=payload.steps,
                                guidance=payload.guidance, seed=seed, check=ctx.raise_if_cancelled,
                                progress=lambda step: ctx.progress(step, payload.steps))
    ctx.raise_if_cancelled()
    with tel.stage("decoding video"):
        pixels = model.decode(latents)
    with tel.stage("saving video"):
        video = out.save_video(pixels, fps=FPS)
    tel.log("pan2", width=width, height=height, frames=frames, steps=payload.steps, audio=False)
    return VideoResult(video, width, height, float(FPS), len(pixels), seed, prompt)
