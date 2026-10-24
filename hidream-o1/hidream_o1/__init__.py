"""HiDream-O1-Image text-to-image and reference editing, full and Dev, with HiDream's prompt agent."""

from __future__ import annotations

import math
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    ImageRequest,
    InvalidRequest,
    ImageResult,
    Memoize,
    Outputs,
    ReferenceImages,
    Telemetry,
    held,
)
from PIL import Image

from .model import MAX_BOXES, HiDreamO1, Recipe, area_resize, canvas_box, grid_size, placed, reference_size
from . import quantize as quantize  # its recipes
from .agent import PromptAgent

app = App()

#: 1 MP is 1024² pixels. The model was refined at 2048² (tech report §4.1) and upstream serves only
#: about 4 MP: 2560x1440 (3.52 MP) to 2048² (4). At 1024² and below it falls apart (upstream #9).
MP = 1 << 20
MIN_PIXELS, NATIVE_PIXELS = 2560 * 1440, 4 * MP
MIN_SIDE, MAX_SIDE = 256, 4096
FULL_STEPS, DEV_STEPS = 50, 28
GUIDANCE = 5.0
#: The full model's shift; upstream's README runs two or more references at 1.
FULL_SHIFT, SUBJECT_SHIFT = 3.0, 1.0
#: The Dev model's flash sampler: noise scale 7.5, clipped at 2.5 standard deviations.
DEV_FLASH = Recipe("flash", noise=7.5, clip=2.5)
#: Upstream's unconditional caption.
UNCONDITIONAL = " "
_UNIT = Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]
#: `[x0, y0, x1, y1]`: corners, 0-1 of the output canvas, origin top-left, x0 < x1, y0 < y1.
Box = Annotated[list[_UNIT], msgspec.Meta(min_length=4, max_length=4)]


class GenerateInput(ImageRequest):
    #: Given, the image edits them, or composes them in order.
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )
    #: One per reference, in order: where it goes in the output, or null for none. Soft: see the README.
    reference_boxes: list[Box | None] | None = None
    #: CFG over the first 80% of the steps only (limited-interval guidance): with CFG to the end,
    #: flat areas show a 32 px grid of slightly miscoloured patches. False: CFG on every step, as upstream.
    guidance_interval: bool = True


def fitted(adj: Adjustments, ratio: tuple[int, int], pixels: int) -> tuple[int, int]:
    """`pixels` at `ratio` on the 32 px grid (at most `pixels`), the ratio held to 256-4096 px sides."""
    w, h = ratio
    widest = min(MAX_SIDE**2 / pixels, pixels / MIN_SIDE**2)
    aspect = min(max(w / h, 1 / widest), widest)
    width, height = grid_size(aspect, 1.0, pixels)
    width, height = min(max(width, MIN_SIDE), MAX_SIDE), min(max(height, MIN_SIDE), MAX_SIDE)
    if aspect != w / h:
        adj.clamp("aspect_ratio", f"{w}:{h}", f"{width}:{height}", "HiDream-O1's 256-4096 px sides")
    return width, height


def layout(
    adj: Adjustments, payload: GenerateInput, images: list[Image.Image]
) -> tuple[tuple[int, int], list[Image.Image]]:
    """The output size and the prepared references. What is not given is the first reference's
    (ratio, pixel count), else 1:1 at 4 MP; pixels are held to 3.52-4 MP. A lone reference given
    neither is upstream's `keep_original_aspect`: resized to the output's area and the output takes
    its size. Otherwise each is resized to the area of a square on the output's longer side, smaller
    with more references (upstream's `generate_image`). References placed by box are elements, not
    the canvas: the output is 1:1 at 4 MP unless given, and upstream's layout canvas follows them."""
    boxes = _boxes(payload)
    place = any(box is not None for box in boxes)
    first = images[0].size if images and not place else None
    target = payload.target(first, megapixels=NATIVE_PIXELS / MP)
    pixels = held(adj, "megapixels", target.pixels / MP, MIN_PIXELS / MP, NATIVE_PIXELS / MP,
                  "HiDream-O1's 3.52-4 MP") * MP
    if len(images) == 1 and payload.aspect_ratio is None and payload.megapixels is None and not place:
        lone = area_resize(images[0], math.sqrt(pixels))
        if MIN_SIDE <= min(lone.size) and max(lone.size) <= MAX_SIDE:
            return lone.size, [lone]
    width, height = fitted(adj, target.ratio, int(pixels))
    if not place:
        side = reference_size(max(width, height), len(images))
        return (width, height), [area_resize(image, side) for image in images]
    pixel_boxes = [None if box is None else canvas_box(box, width, height) for box in boxes]
    for index, (box, drawn) in enumerate(zip(boxes, pixel_boxes, strict=True)):
        if box is not None and drawn is None:
            raise InvalidRequest(f"reference_boxes.{index} is under a pixel on the {width}x{height} output",
                                 code="invalid_box", fields=(f"reference_boxes.{index}",))
    if sum(box is not None for box in pixel_boxes) > MAX_BOXES:
        adj.warn("reference_boxes", f"HiDream-O1 draws only the {MAX_BOXES} largest boxes, as upstream")
    side = reference_size(max(width, height), len(images) + 1)
    return (width, height), placed(images, pixel_boxes, width, height, side)


def _boxes(payload: GenerateInput) -> list[list[float] | None]:
    """`reference_boxes` checked against the references: one each, x0 < x1 and y0 < y1."""
    boxes, count = payload.reference_boxes, len(payload.reference_images)
    if boxes is None:
        return [None] * count
    if len(boxes) != count:
        raise InvalidRequest(f"reference_boxes has {len(boxes)} entries for {count} reference_images",
                             code="reference_boxes_length", fields=("reference_boxes",))
    for index, box in enumerate(boxes):
        if box is not None and not (box[0] < box[2] and box[1] < box[3]):
            raise InvalidRequest(f"a box is [x0, y0, x1, y1] with x0 < x1 and y0 < y1, got {box}",
                                 code="invalid_box", fields=(f"reference_boxes.{index}",))
    return list(boxes)


def _caption(prompt: str, upsample: bool, images: list[Image.Image], agent: PromptAgent, seed: int,
             adj: Adjustments, tel: Telemetry) -> str:
    """The prompt the image is made from: HiDream's agent's rewrite of a text-to-image prompt when asked."""
    if not upsample:
        return prompt
    if images:
        adj.warn("prompt_upsampling", "HiDream's prompt agent rewrites text-to-image prompts: an edit's goes as typed")
        return prompt
    with tel.stage("rewriting prompt", overall_range=(0.0, 0.15)):
        caption = agent.rewrite(prompt, seed)
    tel.log("rewritten prompt", caption=caption[:2000])
    return caption


def _generate(
    ctx: Context, model: HiDreamO1, agent: PromptAgent, out: Outputs, tel: Telemetry, adj: Adjustments, *,
    payload: GenerateInput, recipe: Recipe, steps: int, guidance: float, negative: str | None,
    interval: bool = False,
) -> ImageResult:
    seed = payload.drawn_seed()
    images = [asset.read() for asset in payload.reference_images]
    (width, height), references = layout(adj, payload, images)
    caption = _caption(payload.prompt, payload.prompt_upsampling is True, images, agent, seed, adj, tel)
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.15, 0.98)):
        image = model.generate(
            caption, negative, references, width=width, height=height, recipe=recipe, steps=steps,
            guidance=guidance, seed=seed, cancel=ctx.raise_if_cancelled, interval=interval,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.15, 0.98)),
        )
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, caption)


_FULL = {"model": "hidream-o1@0.1.0", "agent": "gemma-4-31b-it@0.1.0"}
_DEV = {"model": "hidream-o1-dev@0.1.0", "agent": "gemma-4-31b-it@0.1.0"}
_COMPONENTS = {"model": ("model",), "agent": ("model",)}
#: Built and fetched on first use: without `prompt_upsampling` the agent is never pulled.
_ON_DEMAND = ("agent",)
_MEMO = ("hidream_o1", "hidream_o1.model", "hidream_o1.sampling", "hidream_o1.agent")


@app.entrypoint(defaults=_FULL, components=_COMPONENTS, on_demand=_ON_DEMAND, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate(ctx: Context, payload: GenerateInput, model: HiDreamO1, agent: PromptAgent, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image: 50 UniPC steps with CFG 5.0 over the first 40. The model HiDream recommends
    for editing."""
    boxed = any(box is not None for box in payload.reference_boxes or ())
    shift = SUBJECT_SHIFT if len(payload.reference_images) > 1 or boxed else FULL_SHIFT
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
    cfg = guidance > 1.0
    negative = (payload.negative_prompt if payload.negative_prompt is not None else UNCONDITIONAL) if cfg else None
    steps = held(adj, "steps", payload.steps or FULL_STEPS, 1, 200)
    return _generate(ctx, model, agent, out, tel, adj, payload=payload, recipe=Recipe("unipc", shift),
                     steps=steps, guidance=guidance, negative=negative, interval=payload.guidance_interval)


@app.entrypoint(defaults=_DEV, components=_COMPONENTS, on_demand=_ON_DEMAND, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate_turbo(ctx: Context, payload: GenerateInput, model: HiDreamO1, agent: PromptAgent, out: Outputs,
                   tel: Telemetry, adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image-Dev: 28 distilled steps, no CFG. A one-reference edit samples flow-match Euler."""
    if payload.steps not in (None, DEV_STEPS):
        adj.clamp("steps", payload.steps, DEV_STEPS, "the Dev model's distilled schedule")
    payload.ignored(adj, "guidance", "negative_prompt", reason="HiDream-O1-Dev samples without CFG")
    if not payload.guidance_interval:
        adj.warn("guidance_interval", "ignored: HiDream-O1-Dev samples without CFG")
    recipe = Recipe("euler") if len(payload.reference_images) == 1 else DEV_FLASH
    return _generate(ctx, model, agent, out, tel, adj, payload=payload, recipe=recipe, steps=DEV_STEPS,
                     guidance=0.0, negative=None)
