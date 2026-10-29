"""MOSS-Image (OpenMOSS) text-to-image, with layout boxes for objects and rendered text."""

from __future__ import annotations

import json
import math
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    Region,
    Sizes,
    Telemetry,
    corners,
    held,
)

from . import quantize as quantize  # its recipes
from .model import MossImage

app = App()

#: The user guide: best at 1-2 MP, extrapolates from 0.25 MP (512²) to 4 MP; the pipeline's default 1024².
SIZES = Sizes((0.25, 4.0), (256, 4096), 16, "MOSS-Image's 0.25-4 MP")
STEPS, MOST_STEPS = 50, 200
GUIDANCE = 3.0

class GenerateInput(ImageRequest):
    #: Elements and rendered text placed by box, described in words (`\n` in `text` breaks a line); the
    #: prompt describes the whole image.
    regions: Annotated[list[Region], msgspec.Meta(max_length=64)] | None = None


def pixel_box(box: list[float], width: int, height: int, field: str) -> list[int]:
    """A normalized `[x0, y0, x1, y1]` as MOSS's `yxyx` pixel box from the canvas centre: left and top
    floored, right and bottom ceiled, as upstream's layout compiler encloses a box."""
    x0, y0, x1, y1 = corners(box, field)
    return [math.floor(round(y0 * height, 9)) - height // 2, math.floor(round(x0 * width, 9)) - width // 2,
            math.ceil(round(y1 * height, 9)) - height // 2, math.ceil(round(x1 * width, 9)) - width // 2]


def layout_prompt(prompt: str, regions: list[Region] | None, width: int, height: int) -> str:
    """The prompt as given, or with regions MOSS's layout JSON: the prompt is the scene's caption, each
    region an item (`text` when it renders text, else `object`), compact as upstream compiles it."""
    if not regions:
        return prompt
    items: list[dict[str, object]] = []
    for index, region in enumerate(regions):
        item: dict[str, object] = {"label": "object" if region.text is None else "text", "caption": region.prompt}
        if region.text is not None:
            item["text"] = region.text
        item["bbox"] = pixel_box(region.box, width, height, f"regions.{index}.box")
        items.append(item)
    layout = {"summary": {"caption": prompt.strip()}, "items": items, "bbox_coord_space": "abs_centralized",
              "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


_MEMO = ("moss_image", "moss_image.model")


@app.entrypoint(defaults={"model": "moss-image@0.1.0"}, components={"model": ("transformer", "text_encoder", "vae")},
                memoize=Memoize(requires=["seed"]), memo_dependencies=_MEMO)
def generate(ctx: Context, payload: GenerateInput, model: MossImage, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> ImageResult:
    """MOSS-Image SFT: 50 Euler steps at CFG 3 (the negative branch skips block 22), with `regions` laid
    out as upstream's layout JSON."""
    model.hold_vae(tel)
    if payload.reference_images:
        adj.warn("reference_images", "ignored: MOSS-Image generates from text alone")
    payload.ignored(adj, "reference_boxes", reason="MOSS-Image takes no references; place elements with regions")
    payload.ignored(adj, "prompt_upsampling", reason="MOSS-Image has no prompt upsampler")
    steps = held(adj, "steps", payload.steps or STEPS, 1, MOST_STEPS, "MOSS-Image's 1-200 steps")
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0,
                    "MOSS-Image's CFG 1-20")
    seed = payload.drawn_seed()
    width, height = payload.size(adj, None, SIZES, megapixels=1.0)
    prompt = layout_prompt(payload.prompt, payload.regions, width, height)
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt", overall_range=(0.0, 0.05)):
        positive = model.encode(prompt)
        negative = None if guidance == 1.0 else model.encode(payload.negative_prompt or "")
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.05, 0.97)):
        latents = model.denoise(
            positive, negative, width=width, height=height, steps=steps,
            guidance=guidance, seed=seed, tel=tel,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.05, 0.97)),
        )
    with tel.stage("decoding image", overall_range=(0.97, 0.99)):
        image = model.decode(latents)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)
