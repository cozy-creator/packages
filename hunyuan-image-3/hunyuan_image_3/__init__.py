"""Tencent's HunyuanImage 3.0: text to image, or an edit or fusion of up to three ordered references, with the
model's own think-and-rewrite stage. `generate_turbo` is Instruct-Distil (8 steps), `generate` Instruct (50)."""

from __future__ import annotations

from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    ImageReference,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    Telemetry,
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


class HunyuanRequest(ImageRequest, forbid_unknown_fields=True):
    """The shared image request. With no `aspect_ratio` the model picks its size, as Tencent's
    `image_size="auto"`; it makes about 1 MP whatever `megapixels` asks."""

    references: Annotated[list[ImageReference], msgspec.Meta(max_length=3)] = msgspec.field(default_factory=list)


def label(n: int, reference: ImageReference) -> str:
    """HunyuanImage 3.0's word for the Nth reference: every prompt Tencent documents names them 图1, 图2,
    图3 (`run_demo_instruct.sh`); the images precede the prompt with no label of their own."""
    return f"图{n}"


def requested_size(adj: Adjustments, ratio: str) -> tuple[int, int]:
    """The model size nearest `W:H` (Tencent's `get_target_size`); a ratio past 1:4 or 4:1 is a `clamp`."""
    w, h = map(int, ratio.split(":"))
    width, height = SIZES.get_target_size(w, h)
    if not _TALLEST <= w / h <= _WIDEST:
        adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "the model's sizes run from 1:4 to 4:1")
    return width, height


def check_pixels(adj: Adjustments, megapixels: float | None, size: tuple[int, int]) -> None:
    """The model makes about 1 MP: `megapixels` more than 10% off the size made is a `clamp`."""
    made = size[0] * size[1]
    if megapixels is not None and abs(megapixels * (1 << 20) - made) > 0.1 * made:
        adj.clamp("megapixels", megapixels, round(made / (1 << 20), 2), "HunyuanImage 3.0 makes about 1 MP")


def _generate(ctx: Context, payload: HunyuanRequest, steps: int, model: HunyuanImage3, out: Outputs,
              tel: Telemetry, adj: Adjustments) -> ImageResult:
    payload.ignored(adj, "negative_prompt", reason="HunyuanImage 3.0's unconditional branch is the prompt dropped")
    payload.unplaced(adj, "HunyuanImage 3.0 has no layout control")
    prompt = payload.rendered(adj, label)
    seed = payload.drawn_seed()
    references = [asset.read() for asset in payload.images()]
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
    check_pixels(adj, payload.megapixels, size)
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


@app.entrypoint(defaults={"model": "tencent/hunyuan-image-3-instruct-distil@0.1.0"}, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: HunyuanRequest, model: HunyuanImage3, out: Outputs, tel: Telemetry,
                   adj: Adjustments) -> ImageResult:
    """HunyuanImage 3.0 Instruct-Distil, 8 steps, guidance 2.5 (distilled): text to image, or an edit of up to
    three references; the model thinks and rewrites the prompt first unless `prompt_upsampling` is false."""
    return _generate(ctx, payload, payload.steps or 8, model, out, tel, adj)


@app.entrypoint(defaults={"model": "tencent/hunyuan-image-3-instruct@0.1.0"}, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: HunyuanRequest, model: HunyuanImage3, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> ImageResult:
    """HunyuanImage 3.0 Instruct, 50 steps with CFG 2.5: text to image, or an edit of up to three references;
    the model thinks and rewrites the prompt first unless `prompt_upsampling` is false."""
    return _generate(ctx, payload, payload.steps or 50, model, out, tel, adj)
