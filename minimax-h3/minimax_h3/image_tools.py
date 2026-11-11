"""`image`: P-H3-Image text-to-image, or an edit of the pictures its instructions mention (`image.py`)."""

from __future__ import annotations

import dataclasses
from typing import Annotated

import msgspec
from cozy_runtime.author import Adjustments, Context, Outputs, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, Image, Prompt, Range, Ratios
from cozy_runtime.image import MEGAPIXEL, ImageRequest, ImageResult, Sizes
from PIL import Image as PILImage

from .application import app
from .image import (
    CANVAS_MIN_PIXELS,
    GRID,
    IMAGE_COMPONENTS,
    MAX_PICTURES,
    PICTURE_PIXELS,
    STEPS,
    UPSAMPLER_COMPONENTS,
    PH3Image,
    PH3Upsampler,
    aspect_label,
    resized,
    size_in_range,
)

MODEL = "prunaai/p-h3-image@0.1.0"
#: The card's 1-16 MP ladder on a 32 px grid, 8 MP by default (its presets: 1:1 is 2912x2912, 16:9 3872x2176).
SIZES = Sizes((1.0, 16.0), (256, 8192), GRID, "P-H3-Image's sizes", widest=4)
DEFAULT_MEGAPIXELS = 8.0


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and for an edit the pictures it mentions in order of first mention: the first is edited, the
    #: rest are references (`<Picture 2>` ...), up to 8.
    instructions: Prompt[Image] = Prompt()
    #: `ImageRequest.target`, held to 1:4 through 4:1 (`SIZES.fit`). Unset: an edit's Picture 1's, else 1:1.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 4, max=4)] | None = None
    #: `SIZES.pixels`, 1-16 MP. Unset: 8, or an edit's Picture 1's held to 4-8 MP.
    megapixels: Annotated[float, Range(1, 16)] | None = None
    seed: Seed | None = None
    #: Pruna's upsampler writes the full prompt from it first, as P-H3-Image renders best.
    prompt_upsampling: Annotated[bool, Advanced()] = True


class _Request(ImageRequest):
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text


@app.tool(
    label="Image",
    named="image",
    defaults={"model": MODEL, "upsampler": MODEL},
    components={"model": IMAGE_COMPONENTS, "upsampler": UPSAMPLER_COMPONENTS},
    on_demand=("upsampler",),
    memoize=True,
)
def image(ctx: Context, payload: ImageBox, model: PH3Image, upsampler: PH3Upsampler, out: Outputs,
          tel: Telemetry, adj: Adjustments) -> ImageResult:  # fmt: skip
    """P-H3-Image, 8 steps without CFG at 8 MP: text to image, or an edit of the pictures the instructions mention."""
    request = _Request(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio,
                       megapixels=payload.megapixels, seed=payload.seed)  # fmt: skip
    keys = list(request.prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("instructions", f"P-H3-Image edits with {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")  # fmt: skip
        keys = keys[:MAX_PICTURES]
    pictures: list[PILImage.Image] = [request.prompt.references[key].file.read().convert("RGB") for key in keys]
    text = request.prompt.text
    for n, key in enumerate(keys, 1):
        text = text.replace(f"<{key}>", f"<Picture {n}>")
    first = pictures[0].size if pictures else None
    target = request.target(first, megapixels=DEFAULT_MEGAPIXELS)
    if first is not None and payload.megapixels is None:
        target = dataclasses.replace(target, pixels=float(min(max(target.pixels, CANVAS_MIN_PIXELS), PICTURE_PIXELS[1])))
    width, height = SIZES.fit(adj, target.ratio, SIZES.pixels(adj, target))
    seed = request.drawn_seed()
    prompt = text
    if payload.prompt_upsampling:
        with tel.stage("upsampling prompt"):
            prompt, why = upsampler.upsample(text, pictures, aspect_label(width, height))
        if why:
            adj.warn("prompt_upsampling", f"the upsampler's answer was {why}, so the prompt renders as written")
        tel.log("upsampled prompt", prompt=prompt[:4000])
    ctx.raise_if_cancelled()
    conditions = []
    if pictures:
        # The VAE reads each picture at its own size held to 1-8 MP; Picture 1 at the canvas unless one was asked for.
        asked = payload.aspect_ratio is not None or payload.megapixels is not None
        sizes = [size_in_range(*picture.size, *PICTURE_PIXELS) for picture in pictures]
        if not asked:
            sizes[0] = (width, height)
        with tel.stage("encoding pictures"):
            conditions = model.encode_pictures([resized(p, s) for p, s in zip(pictures, sizes, strict=True)])
    with tel.stage("encoding prompt"):
        embeds, tags = model.encode_text(prompt, pictures)
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(embeds, tags, conditions, width=width, height=height, seed=seed,
                                on_step=lambda step: ctx.progress(step, STEPS))  # fmt: skip
    with tel.stage("decoding image"):
        picture = model.decode(latents)
    with tel.stage("saving image"):
        asset = out.save_image(picture, format="png")
    tel.log("p-h3-image", width=width, height=height, pictures=len(pictures), megapixels=round(width * height / MEGAPIXEL, 2))
    return ImageResult(asset, width, height, seed, prompt)
