"""Tencent's HunyuanImage 3.0: text to image, or an edit or fusion of up to three ordered pictures, with the
model's own think-and-rewrite stage. `generate_turbo` is Instruct-Distil (8 steps), `generate` Instruct (50)."""

from __future__ import annotations

import secrets
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    AspectRatio,
    Context,
    Guidance,
    Image,
    ImageResult,
    Memoize,
    Outputs,
    Prompt,
    Ratios,
    Seed,
    Steps,
    Telemetry,
    ratio_of,
)

from .model import GUIDANCE, MAX_NEW_TOKENS, HunyuanImage3, caption
from . import quantize as quantize  # its recipes
from .tencent.tokenization_hunyuan_image_3 import Resolution, ResolutionGroup

app = App()

#: The model's sizes, as Tencent's image processor builds them (`image_base_size` 1024): 33 ratios from
#: 1:4 to 4:1 on a 64 px step with width + height = 2048, plus four common ones; all about 1 MP.
SIZES = ResolutionGroup(base_size=1024, align=16, extra_resolutions=[
    Resolution("1024x768"), Resolution("1280x720"), Resolution("768x1024"), Resolution("720x1280"),
])
_WIDEST = max(r.w / r.h for r in SIZES.data)
_TALLEST = min(r.w / r.h for r in SIZES.data)
#: Each checkpoint's steps when none is given: Instruct-Distil's and Instruct's.
DISTIL_STEPS = 8
INSTRUCT_STEPS = 50
#: The pictures HunyuanImage 3.0 takes: past them the last mentioned go, with a warning.
MAX_PICTURES = 3


class HunyuanRequest(msgspec.Struct):
    """No `megapixels`: the model makes about 1 MP. With no `aspect_ratio` it picks its size, as Tencent's
    `image_size="auto"`."""

    #: Its text, and the pictures it edits or fuses, in order of first mention.
    prompt: Prompt[Image] = Prompt()
    aspect_ratio: AspectRatio | None = None
    steps: Steps = None
    guidance: Guidance = None
    seed: Seed | None = None
    prompt_upsampling: bool | None = None


def _said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[Image]]:
    """Its text with each mention as HunyuanImage 3.0's `图N`: every prompt Tencent documents names them 图1,
    图2, 图3 (`run_demo_instruct.sh`), and the images precede the prompt with no label of their own. And the
    pictures."""
    keys = list(prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"HunyuanImage 3.0 takes {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
        keys = keys[:MAX_PICTURES]
    said = prompt.text
    for n, key in enumerate(keys, 1):
        said = said.replace(f"<{key}>", f"图{n}")
    return said, [prompt.references[key] for key in keys]


def requested_size(adj: Adjustments, ratio: str) -> tuple[int, int]:
    """The model size nearest `W:H` (Tencent's `get_target_size`); a ratio past 1:4 or 4:1 is a `clamp`."""
    w, h = ratio_of(ratio)
    width, height = SIZES.get_target_size(w, h)
    if not _TALLEST <= w / h <= _WIDEST:
        adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "the model's sizes run from 1:4 to 4:1")
    return width, height


def _generate(ctx: Context, payload: HunyuanRequest, steps: int, model: HunyuanImage3, out: Outputs,
              tel: Telemetry, adj: Adjustments) -> ImageResult:
    prompt, pictures = _said(payload.prompt, adj)
    seed = payload.seed if payload.seed is not None else secrets.randbits(53)
    references = [picture.file.read() for picture in pictures]
    guidance = payload.guidance if payload.guidance is not None else GUIDANCE
    # No aspect ratio: the model picks its size, as Tencent's `image_size="auto"`.
    size = requested_size(adj, payload.aspect_ratio) if payload.aspect_ratio else None
    cot = None
    if payload.prompt_upsampling is not False:
        with tel.stage("thinking", overall_range=(0.0, 0.3)):
            cot, picked, routing = model.think(
                prompt, references, seed=seed, pick_size=size is None, cancel=ctx.raise_if_cancelled,
                on_token=tel.step_callback(MAX_NEW_TOKENS, stage="thinking", overall_range=(0.0, 0.3)))
        tel.log("thinking routing", **routing)
        tel.log("rewritten prompt", caption=caption(cot, prompt)[:2000])
        size = size or picked
    if size is None:
        with tel.stage("picking size", overall_range=(0.0, 0.05)):
            size = model.pick_size(prompt, references, seed=seed)
    with tel.stage("generating image", overall_range=(0.3, 0.97)):
        image, routing = model.paint(
            prompt, references, cot, size=size, steps=steps, guidance=guidance, seed=seed,
            cancel=ctx.raise_if_cancelled,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.3, 0.95)))
    tel.log("denoising routing", **routing)
    with tel.stage("saving image", overall_range=(0.97, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, image.width, image.height, seed, caption(cot, prompt))


_COMPONENTS = {"model": ("model",)}
_DISTIL = "tencent/hunyuan-image-3-instruct-distil@0.1.1"
_INSTRUCT = "tencent/hunyuan-image-3-instruct@0.1.1"


@app.entrypoint(defaults={"model": _DISTIL}, components=_COMPONENTS, memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: HunyuanRequest, model: HunyuanImage3, out: Outputs, tel: Telemetry,
                   adj: Adjustments) -> ImageResult:
    """HunyuanImage 3.0 Instruct-Distil, 8 steps at guidance 2.5: text to image, or an edit of up to three pictures.

    The model thinks and rewrites the prompt first unless `prompt_upsampling` is false."""
    return _generate(ctx, payload, payload.steps or DISTIL_STEPS, model, out, tel, adj)


@app.entrypoint(defaults={"model": _INSTRUCT}, components=_COMPONENTS, memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: HunyuanRequest, model: HunyuanImage3, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> ImageResult:
    """HunyuanImage 3.0 Instruct, 50 steps with CFG 2.5: text to image, or an edit of up to three pictures.

    The model thinks and rewrites the prompt first unless `prompt_upsampling` is false."""
    return _generate(ctx, payload, payload.steps or INSTRUCT_STEPS, model, out, tel, adj)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it edits or fuses (`_said`).
    prompt: Prompt[Image] = Prompt()
    #: Instruct-Distil, 8 steps, else Instruct, 50 steps with CFG (`_generate`).
    turbo: bool = True
    #: `requested_size`: Tencent's sizes run 1:4 to 4:1. Unset, the model picks its size.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 4, max=4)] | None = None
    #: `_generate`'s seed, which also repeats the thought.
    seed: Seed | None = None
    #: The model's own think-and-rewrite stage (`_generate`).
    prompt_upsampling: bool = True


# Each checkpoint is built, and its weights fetched, only when a render asks for it.
@app.tool(label="HunyuanImage 3.0", named="image", defaults={"model": _DISTIL, "full": _INSTRUCT},
          components={"model": ("model",), "full": ("model",)}, on_demand=("full", "model"),
          memoize=Memoize(requires=["seed"]))
def image(ctx: Context, payload: ImageBox, full: HunyuanImage3, model: HunyuanImage3, out: Outputs, tel: Telemetry,
          adj: Adjustments) -> ImageResult:
    """HunyuanImage 3.0 on a canvas: Instruct-Distil, or Instruct."""
    request = HunyuanRequest(prompt=payload.prompt, aspect_ratio=payload.aspect_ratio, seed=payload.seed,
                             prompt_upsampling=payload.prompt_upsampling)
    if payload.turbo:
        return _generate(ctx, request, DISTIL_STEPS, model, out, tel, adj)
    return _generate(ctx, request, INSTRUCT_STEPS, full, out, tel, adj)
