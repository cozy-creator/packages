"""HiDream-O1-Image text-to-image and reference editing, full and Dev."""

from __future__ import annotations

import math
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    AspectRatio,
    Box,
    Context,
    Guidance,
    Image,
    ImageRequest,
    ImageResult,
    Memoize,
    NegativePrompt,
    Outputs,
    Prompt,
    Ratios,
    Seed,
    Sizes,
    Steps,
    Telemetry,
    corners,
    held,
    name_key,
)
from PIL import Image as PILImage

from .model import MAX_BOXES, HiDreamO1, Recipe, area_resize, canvas_box, placed, reference_size
from . import quantize as quantize  # its recipes

app = App()

#: 1 MP is 1024² pixels. The model was refined at 2048² (tech report §4.1) and upstream serves only
#: about 4 MP: 2560x1440 (3.52 MP) to 2048² (4). At 1024² and below it falls apart (upstream #9).
MP = 1 << 20
NATIVE_PIXELS = 4 * MP
MIN_MEGAPIXELS = 2560 * 1440 / MP
MAX_MEGAPIXELS = 4.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (256, 4096), 32, "HiDream-O1's 3.52-4 MP")
#: The widest those sizes hold: 4096 px sides at 3.52 MP, about 4.55:1.
WIDEST = 4096**2 / (2560 * 1440)
FULL_STEPS, DEV_STEPS = 50, 28
GUIDANCE = 5.0
#: The full model's shift; upstream's README runs two or more references at 1.
FULL_SHIFT, SUBJECT_SHIFT = 3.0, 1.0
#: The Dev model's flash sampler: noise scale 7.5, clipped at 2.5 standard deviations.
DEV_FLASH = Recipe("flash", noise=7.5, clip=2.5)
#: Upstream's unconditional caption.
UNCONDITIONAL = " "
#: The pictures HiDream-O1 takes: past them the last mentioned go, with a warning.
MAX_PICTURES = 10
#: A picture's corners on the output, 0-1: `[x0, y0, x1, y1]`.
Corners = tuple[float, float, float, float]


class TurboInput(ImageRequest):
    #: Its text, and the pictures it mentions, which it edits or composes in the order it first mentions them.
    #: HiDream-O1 has no words of its own for them, so a `<name>` mention reads as the name.
    prompt: Prompt[Image] = Prompt()
    #: Where a mentioned picture goes in the output, by its key (upstream's `--layout_bboxes`).
    boxes: dict[str, Box] = msgspec.field(default_factory=dict)


class GenerateInput(TurboInput):
    steps: Steps = None
    guidance: Guidance = None
    negative_prompt: NegativePrompt = None
    #: CFG over the first 80% of the steps only (limited-interval guidance): with CFG to the end,
    #: flat areas show a 32 px grid of slightly miscoloured patches. False: CFG on every step, as upstream.
    guidance_interval: bool = True


def pictures(adj: Adjustments, payload: TurboInput) -> dict[str, Corners | None]:
    """The pictures HiDream-O1 takes, by key in order of first mention, each with its box or None. A box's key
    matches as a mention does: any case, `_`, `-` and space alike."""
    keys = list(payload.prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"HiDream-O1 takes {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
        keys = keys[:MAX_PICTURES]
    given = {name_key(key): key for key in payload.boxes}
    boxed: dict[str, Corners | None] = {}
    for key in keys:
        named = given.pop(name_key(key), None)
        boxed[key] = None if named is None else corners(payload.boxes[named], f"boxes.{named}")
    if given:
        adj.warn("boxes", f"ignored: {', '.join(f'<{key}>' for key in given.values())} names no picture HiDream-O1 takes")
    return boxed


def layout(
    adj: Adjustments, payload: TurboInput, images: list[PILImage.Image], boxed: dict[str, Corners | None]
) -> tuple[tuple[int, int], list[PILImage.Image]]:
    """The output size and the prepared references. What is not given is the first reference's
    (ratio, pixel count), else 1:1 at 4 MP; pixels are held to 3.52-4 MP. A lone reference given
    neither is upstream's `keep_original_aspect`: resized to the output's area and the output takes
    its size. Otherwise each is resized to the area of a square on the output's longer side, smaller
    with more references (upstream's `generate_image`). References placed by box are elements, not
    the canvas: the output is 1:1 at 4 MP unless given, and upstream's layout canvas follows them."""
    place = any(box is not None for box in boxed.values())
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
    pixel_boxes = [None if box is None else canvas_box(box, width, height) for box in boxed.values()]
    for (key, box), drawn in zip(boxed.items(), pixel_boxes, strict=True):
        if box is not None and drawn is None:
            adj.warn(f"boxes.{key}", f"ignored: it is under a pixel on the {width}x{height} output")
    if sum(box is not None for box in pixel_boxes) > MAX_BOXES:
        adj.warn("boxes", f"HiDream-O1 draws only the {MAX_BOXES} largest boxes, as upstream")
    side = reference_size(max(width, height), len(images) + 1)
    return (width, height), placed(images, pixel_boxes, width, height, side)


def _generate(ctx: Context, payload: TurboInput, model: HiDreamO1, out: Outputs, tel: Telemetry,
              adj: Adjustments) -> ImageResult:
    """Both models' body: a `GenerateInput` samples the full model, UniPC with CFG; else Dev's 28 distilled steps,
    flow-match Euler for a one-picture edit."""
    boxed = pictures(adj, payload)
    interval = False
    if isinstance(payload, GenerateInput):
        shift = SUBJECT_SHIFT if len(boxed) > 1 or any(box is not None for box in boxed.values()) else FULL_SHIFT
        guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
        cfg = guidance > 1.0
        negative = (payload.negative_prompt if payload.negative_prompt is not None else UNCONDITIONAL) if cfg else None
        steps = held(adj, "steps", payload.steps or FULL_STEPS, 1, 200)
        recipe, interval = Recipe("unipc", shift), payload.guidance_interval
    else:
        recipe = Recipe("euler") if len(boxed) == 1 else DEV_FLASH
        steps, guidance, negative = DEV_STEPS, 0.0, None
    seed = payload.drawn_seed()
    prompt = payload.prompt.text
    for key in payload.prompt.references:
        prompt = prompt.replace(f"<{key}>", key.replace(".", " "))
    images = [payload.prompt.references[key].file.read() for key in boxed]
    (width, height), references = layout(adj, payload, images, boxed)
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.0, 0.98)):
        image = model.generate(
            prompt, negative, references, width=width, height=height, recipe=recipe, steps=steps,
            guidance=guidance, seed=seed, interval=interval,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.0, 0.98)),
        )
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


_FULL = "hidream-ai/hidream-o1@0.1.0"
_DEV = "hidream-ai/hidream-o1-dev@0.1.0"
_COMPONENTS = {"model": ("model",)}
_MEMO = ("hidream_o1", "hidream_o1.model", "hidream_o1.sampling")


@app.entrypoint(defaults={"model": _FULL}, components=_COMPONENTS, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate(ctx: Context, payload: GenerateInput, model: HiDreamO1, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image: 50 UniPC steps with CFG 5.0 over the first 40. The model HiDream recommends
    for editing."""
    return _generate(ctx, payload, model, out, tel, adj)


@app.entrypoint(defaults={"model": _DEV}, components=_COMPONENTS, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate_turbo(ctx: Context, payload: TurboInput, model: HiDreamO1, out: Outputs,
                   tel: Telemetry, adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image-Dev: 28 distilled steps, no CFG. A one-picture edit samples flow-match Euler."""
    return _generate(ctx, payload, model, out, tel, adj)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it mentions (`pictures`, `_generate`).
    instructions: Prompt[Image] = Prompt()
    #: HiDream-O1-Image-Dev's 28 distilled steps, else the full model's 50 with CFG (`_generate`).
    turbo: bool = True
    #: `layout`, through `ImageRequest.target`; `SIZES.fit` holds it to `WIDEST` either way.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / WIDEST, max=WIDEST)] | None = None
    #: `_generate`'s `drawn_seed`.
    seed: Seed | None = None


# Each checkpoint is built, and its weights fetched, only when a box asks for it.
@app.tool(label="Image", named="image", defaults={"full": _FULL, "model": _DEV},
          components={"full": ("model",), "model": ("model",)}, on_demand=("full", "model"),
          memoize=Memoize(requires=["seed"]), memo_dependencies=_MEMO)  # fmt: skip
def image(ctx: Context, payload: ImageBox, full: HiDreamO1, model: HiDreamO1, out: Outputs, tel: Telemetry,
          adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image on a canvas: Dev, or the full model, which loads only when asked for."""
    kind = TurboInput if payload.turbo else GenerateInput
    request = kind(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio,
                   seed=payload.seed)  # fmt: skip
    return _generate(ctx, request, model if payload.turbo else full, out, tel, adj)
