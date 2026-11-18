"""PAN2: a scene unfolding under timed actions, from instructions and the first frame they mention."""

from __future__ import annotations

import json
import math
import secrets
from pathlib import Path
from typing import Annotated

import msgspec
from cozy_runtime import vlm
from cozy_runtime.author import Adjustments, App, Context, InvalidRequest, Outputs, OutputError, Seed, Telemetry
from cozy_runtime.fields import MENTION, Advanced, AspectRatio, FileBound, Image, Prompt, Range, Ratios
from cozy_runtime.media import VideoFile

from . import quantize as quantize  # its recipe
from .model import DEFAULT_STEPS, FPS, Pan2, Reasoner

app = App()
MODEL = "ifm/pan2@0.3.0"
REASONER = "huihui-ai/huihui-qwen3.5-9b-abliterated@0.1.0"
#: PAN2's text encoder is stock Qwen3.5-9B's language model, its own slot: pottokao's Heretic abliteration of it by default
#: (Paul, 11-17: "use abliterated models wherever possible"), the weights moss-image encodes with. Stock is any PAN2
#: checkpoint: `model.text_encoder=ifm/pan2@0.3.0`. `model` reads the DiT and the VAE.
TEXT_ENCODER = "pottokao/moss-image-text-encoder-heretic-pan2@0.1.0"

from . import lanes as lanes  # noqa: E402  # the `curve` lane function; it registers on `app`

REWRITE = Path(__file__).with_name("prompt-policy.md").read_text()


class Generate(msgspec.Struct, kw_only=True):
    #: The scene and its actions, in your words. The first picture it mentions is the first frame, which the scene opens
    #: on, as qwen-image-2's Image and Orbit take pictures: by mention.
    instructions: Prompt[Image] = Prompt()
    #: The clip's seconds, as H3's beats take them, on PAN2's `4k+1` frame grid.
    duration: Annotated[int, Range(1, 20, unit="s")] = 10
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


def names(text: str) -> str:
    """`text` with each mention as its name: `<amy.2>` is amy 2."""
    return MENTION.sub(lambda found: found[1].replace(".", " "), text).strip()


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


# The reasoner only for a request that upsamples, which it rewrites into the event timeline (cl-1103). It is
# huihui-ai's abliterated Qwen3.5-9B by default (Paul, 11-17: "use abliterated models wherever possible"); Qwen's own
# is `model.reasoner=qwen/qwen3.5-9b@0.2.0`.
@app.tool(label="PAN2", named="video", defaults={"model": MODEL, "text_encoder": TEXT_ENCODER, "reasoner": REASONER},
          components={"model": ("transformer", "vae"), "text_encoder": ("text_encoder",)},
          needs={"reasoner": "request.prompt_upsampling"}, memoize=True)
def generate(ctx: Context, payload: Generate, model: Pan2, text_encoder: Pan2, reasoner: Reasoner, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> VideoResult:
    """PAN2's silent video, at 24 fps for up to 20 seconds. The first picture the instructions mention is the first
    frame, and the scene opens on it."""
    seed = int(payload.seed) if payload.seed is not None else secrets.randbits(32)
    seconds = min(20, max(1, payload.duration))
    frames = min(481, 4 * math.ceil(seconds * FPS / 4) + 1)
    pictures = [key for key, value in payload.instructions.references.items() if isinstance(value, Image)]
    if len(pictures) > 1:
        adj.warn("instructions", f"PAN2 opens on one picture, <{pictures[0]}>, so it goes without "
                                 f"{', '.join(f'<{key}>' for key in pictures[1:])}")
    picture = payload.instructions.references[pictures[0]].file.read().convert("RGB") if pictures else None
    width, height = canvas(payload.aspect_ratio, picture.size if picture else None)
    prompt = names(payload.instructions.text)
    if not prompt:
        raise InvalidRequest("PAN2 needs instructions: the scene and its actions", fields=["instructions"])
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
    if seconds != payload.duration:
        adj.warn("duration", f"PAN2 renders 1-20 seconds, so this clip runs {(frames - 1) / FPS:g} seconds")
    with tel.stage("encoding caption"):
        embeds = text_encoder.encode(prompt, frames, payload.guidance, payload.negative_prompt)
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
