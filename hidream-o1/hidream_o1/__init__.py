"""HiDream-O1-Image text-to-image and reference editing, full and Dev."""

from __future__ import annotations

import math
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    PictureAsset,
    ImageRequest,
    InvalidRequest,
    ImageResult,
    Memoize,
    Outputs,
    Sizes,
    Telemetry,
    held,
)
from PIL import Image

from .model import MAX_BOXES, HiDreamO1, Recipe, area_resize, canvas_box, placed, reference_size
from . import quantize as quantize  # its recipes

app = App()

#: 1 MP is 1024² pixels. The model was refined at 2048² (tech report §4.1) and upstream serves only
#: about 4 MP: 2560x1440 (3.52 MP) to 2048² (4). At 1024² and below it falls apart (upstream #9).
MP = 1 << 20
NATIVE_PIXELS = 4 * MP
SIZES = Sizes((2560 * 1440 / MP, 4.0), (256, 4096), 32, "HiDream-O1's 3.52-4 MP")
FULL_STEPS, DEV_STEPS = 50, 28
GUIDANCE = 5.0
#: The full model's shift; upstream's README runs two or more references at 1.
FULL_SHIFT, SUBJECT_SHIFT = 3.0, 1.0
#: The Dev model's flash sampler: noise scale 7.5, clipped at 2.5 standard deviations.
DEV_FLASH = Recipe("flash", noise=7.5, clip=2.5)
#: Upstream's unconditional caption.
UNCONDITIONAL = " "


class GenerateInput(ImageRequest):
    #: Given, the image edits them, or composes them in order. HiDream-O1 has no words of its own
    #: for them, so a `<name>` mention reads as the name.
    assets: Annotated[list[PictureAsset], msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )
    #: CFG over the first 80% of the steps only (limited-interval guidance): with CFG to the end,
    #: flat areas show a 32 px grid of slightly miscoloured patches. False: CFG on every step, as upstream.
    guidance_interval: bool = True


def layout(
    adj: Adjustments, payload: GenerateInput, images: list[Image.Image]
) -> tuple[tuple[int, int], list[Image.Image]]:
    """The output size and the prepared references. What is not given is the first reference's
    (ratio, pixel count), else 1:1 at 4 MP; pixels are held to 3.52-4 MP. A lone reference given
    neither is upstream's `keep_original_aspect`: resized to the output's area and the output takes
    its size. Otherwise each is resized to the area of a square on the output's longer side, smaller
    with more references (upstream's `generate_image`). References placed by box are elements, not
    the canvas: the output is 1:1 at 4 MP unless given, and upstream's layout canvas follows them."""
    boxes = payload.placements()
    place = any(box is not None for box in boxes)
    first = images[0].size if images and not place else None
    target = payload.target(first, megapixels=NATIVE_PIXELS / MP)
    pixels = SIZES.pixels(adj, target)
    if len(images) == 1 and payload.aspect_ratio is None and payload.megapixels is None and not place:
        lone = area_resize(images[0], math.sqrt(pixels))
        if SIZES.sides[0] <= min(lone.size) and max(lone.size) <= SIZES.sides[1]:
            return lone.size, [lone]
    width, height = SIZES.fit(adj, target.ratio, pixels)
    if not place:
        side = reference_size(max(width, height), len(images))
        return (width, height), [area_resize(image, side) for image in images]
    pixel_boxes = [None if box is None else canvas_box(box, width, height) for box in boxes]
    for index, (box, drawn) in enumerate(zip(boxes, pixel_boxes, strict=True)):
        if box is not None and drawn is None:
            raise InvalidRequest(f"assets.{index}.box is under a pixel on the {width}x{height} output",
                                 code="invalid_box", fields=(f"assets.{index}.box",))
    if sum(box is not None for box in pixel_boxes) > MAX_BOXES:
        adj.warn("assets", f"HiDream-O1 draws only the {MAX_BOXES} largest boxes, as upstream")
    side = reference_size(max(width, height), len(images) + 1)
    return (width, height), placed(images, pixel_boxes, width, height, side)


def _generate(
    ctx: Context, model: HiDreamO1, out: Outputs, tel: Telemetry, adj: Adjustments, *,
    payload: GenerateInput, recipe: Recipe, steps: int, guidance: float, negative: str | None,
    interval: bool = False,
) -> ImageResult:
    payload.ignored(adj, "prompt_upsampling", reason="HiDream-O1 has no prompt upsampler")
    seed = payload.drawn_seed()
    prompt = payload.rendered(adj)
    images = [asset.read() for asset in payload.images()]
    (width, height), references = layout(adj, payload, images)
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.0, 0.98)):
        image = model.generate(
            prompt, negative, references, width=width, height=height, recipe=recipe, steps=steps,
            guidance=guidance, seed=seed, cancel=ctx.raise_if_cancelled, interval=interval,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.0, 0.98)),
        )
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


_FULL = {"model": "hidream-ai/hidream-o1@0.1.0"}
_DEV = {"model": "hidream-ai/hidream-o1-dev@0.1.0"}
_COMPONENTS = {"model": ("model",)}
_MEMO = ("hidream_o1", "hidream_o1.model", "hidream_o1.sampling")


@app.entrypoint(defaults=_FULL, components=_COMPONENTS, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate(ctx: Context, payload: GenerateInput, model: HiDreamO1, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image: 50 UniPC steps with CFG 5.0 over the first 40. The model HiDream recommends
    for editing."""
    boxed = any(asset.box is not None for asset in payload.assets)
    shift = SUBJECT_SHIFT if len(payload.assets) > 1 or boxed else FULL_SHIFT
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
    cfg = guidance > 1.0
    negative = (payload.negative_prompt if payload.negative_prompt is not None else UNCONDITIONAL) if cfg else None
    steps = held(adj, "steps", payload.steps or FULL_STEPS, 1, 200)
    return _generate(ctx, model, out, tel, adj, payload=payload, recipe=Recipe("unipc", shift),
                     steps=steps, guidance=guidance, negative=negative, interval=payload.guidance_interval)


@app.entrypoint(defaults=_DEV, components=_COMPONENTS, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate_turbo(ctx: Context, payload: GenerateInput, model: HiDreamO1, out: Outputs,
                   tel: Telemetry, adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image-Dev: 28 distilled steps, no CFG. A one-asset edit samples flow-match Euler."""
    if payload.steps not in (None, DEV_STEPS):
        adj.clamp("steps", payload.steps, DEV_STEPS, "the Dev model's distilled schedule")
    payload.ignored(adj, "guidance", "negative_prompt", reason="HiDream-O1-Dev samples without CFG")
    if not payload.guidance_interval:
        adj.warn("guidance_interval", "ignored: HiDream-O1-Dev samples without CFG")
    recipe = Recipe("euler") if len(payload.assets) == 1 else DEV_FLASH
    return _generate(ctx, model, out, tel, adj, payload=payload, recipe=recipe, steps=DEV_STEPS,
                     guidance=0.0, negative=None)
