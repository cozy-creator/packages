"""MOSS-Image (OpenMOSS) text-to-image, with layout boxes for objects and rendered text; with pictures mentioned,
Moss-Image-Edit (chinoll's edit LoRA) edits or composes them; image-to-image from a picture."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import Adjustments, App, Context, Lora, ModelChoice, Outputs, Seed, Telemetry, call
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, Image, Prompt, Ratios
from cozy_runtime.media import ImageFile
from PIL import Image as PILImage
from cozy_runtime.image import (
    MEGAPIXEL,
    Guidance,
    ImageRequest,
    ImageResult,
    NegativePrompt,
    Region,
    Sizes,
    Steps,
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
#: Moss-Image-Edit (chinoll/Moss-Image-Edit, rank 64 on every block linear), the LoRA a prompt mentioning pictures
#: runs under: its caller stacks it on `model` (`edit`).
EDIT_LORA = "chinoll/moss-image-edit@0.1.0"
#: Moss-Image-Edit's workflows: 30 Euler steps at CFG 4 against an empty negative.
EDIT_STEPS, EDIT_GUIDANCE = 30, 4.0
#: Each reference's long edge at most, on the 16-pixel grid (its ComfyUI node's `reference_max_edge`).
REFERENCE_EDGE = 512
#: The pictures Moss-Image-Edit takes: past them the last mentioned go, with a warning.
MAX_PICTURES = 4


class GenerateInput(ImageRequest):
    #: Its text, and the pictures it mentions in order of first mention: given, Moss-Image-Edit edits or composes
    #: them (`edit`); else MOSS-Image generates from the text.
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text
    steps: Steps = None
    guidance: Guidance = None
    #: `generate`'s negative branch, encoded only with guidance above 1.
    negative_prompt: NegativePrompt = None
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


_DEFAULTS = {"model": "openmoss-team/moss-image@0.1.0"}
_COMPONENTS = {"model": ("transformer", "text_encoder", "vae")}


def said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[Image]]:
    """Its text with each mention as Moss-Image-Edit reads it, and the pictures: one picture as its name, several as
    `image N` in order (no convention upstream: its instructions speak of "the image")."""
    keys = list(prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"Moss-Image-Edit takes {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
        keys = keys[:MAX_PICTURES]
    text = prompt.text
    for n, key in enumerate(keys, 1):
        text = text.replace(f"<{key}>", key.replace(".", " ") if len(keys) == 1 else f"image {n}")
    return text, [prompt.references[key] for key in keys]


def reference(picture: PILImage.Image) -> PILImage.Image:
    """A picture as Moss-Image-Edit's ComfyUI node hands it to the VAE: its long edge at most `REFERENCE_EDGE`, each
    side on the 16-pixel grid (Lanczos)."""
    width, height = picture.size
    scale = min(1.0, REFERENCE_EDGE / max(width, height))
    size = (max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16))
    return picture if size == picture.size else picture.resize(size, PILImage.Resampling.LANCZOS)


@app.entrypoint(accelerator=False)
async def generate(ctx: Context, payload: GenerateInput) -> ImageResult:
    """MOSS-Image from the text (`text_to_image`); with pictures mentioned, `edit` under Moss-Image-Edit
    (`EDIT_LORA`), stacked on the model after any caller LoRA. Each is this request's GPU call, so the request's
    model and LoRAs reach it."""
    if payload.prompt.references:
        return await call(edit, payload=payload, model=ModelChoice(lora=(Lora(EDIT_LORA),)))
    return await call(text_to_image, payload=payload)


@app.entrypoint(defaults=_DEFAULTS, components=_COMPONENTS, memoize=True, private=True)
def text_to_image(ctx: Context, payload: GenerateInput, model: MossImage, out: Outputs, tel: Telemetry,
                  adj: Adjustments) -> ImageResult:
    """MOSS-Image SFT, 50 Euler steps at CFG 3, with `regions` laid out as upstream's layout JSON. The negative
    branch skips block 22."""
    model.hold_vae(tel)
    steps = held(adj, "steps", payload.steps or STEPS, 1, MOST_STEPS, "MOSS-Image's 1-200 steps")
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0,
                    "MOSS-Image's CFG 1-20")
    seed = payload.drawn_seed()
    width, height = payload.size(adj, None, SIZES, megapixels=1.0)
    prompt = layout_prompt(payload.prompt.text, payload.regions, width, height)
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt"):
        positive = model.encode(prompt)
        negative = None if guidance == 1.0 else model.encode(payload.negative_prompt or "")
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(
            positive, negative, width=width, height=height, steps=steps,
            guidance=guidance, seed=seed, tel=tel, progress=ctx.progress,
        )
    with tel.stage("decoding image"):
        image = model.decode(latents)
    with tel.stage("saving image"):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


@app.entrypoint(defaults=_DEFAULTS, components=_COMPONENTS, memoize=True, private=True)
def edit(ctx: Context, payload: GenerateInput, model: MossImage, out: Outputs, tel: Telemetry,
         adj: Adjustments) -> ImageResult:
    """`generate` with pictures, Moss-Image-Edit stacked on `model` by its caller: each picture a noise-free
    reference at t = 0 (`MossImage.denoise`'s `references`), 30 Euler steps at CFG 4 against an empty negative that
    attends to them too, at the first picture's ratio (ComfyUI's Moss-Image-Edit workflows)."""
    model.hold_vae(tel)
    text, pictures = said(payload.prompt, adj)
    if payload.regions:
        adj.warn("regions", "Moss-Image-Edit reads an instruction, not a layout, so it goes without the regions")
    steps = held(adj, "steps", payload.steps or EDIT_STEPS, 1, MOST_STEPS, "MOSS-Image's 1-200 steps")
    guidance = held(adj, "guidance", EDIT_GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0,
                    "MOSS-Image's CFG 1-20")
    seed = payload.drawn_seed()
    with tel.stage("reading pictures"):
        images = [picture.file.read() for picture in pictures]
    width, height = payload.size(adj, images[0].size, SIZES, megapixels=1.0)
    ctx.raise_if_cancelled()
    with tel.stage("encoding reference images"):
        references = [model.encode_image(reference(image)) for image in images]
    with tel.stage("encoding prompt"):
        positive = model.encode(text)
        negative = None if guidance == 1.0 else model.encode(payload.negative_prompt or "")
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(positive, negative, width=width, height=height, steps=steps, guidance=guidance,
                                seed=seed, tel=tel, progress=ctx.progress, references=references)  # fmt: skip
    with tel.stage("decoding image"):
        image = model.decode(latents)
    with tel.stage("saving image"):
        asset = out.save_image(image, format="png")
    tel.log("moss edit", pictures=len(images), references=" ".join(f"{r.shape[3] * 16}x{r.shape[2] * 16}" for r in references))
    return ImageResult(asset, width, height, seed, text)


class ImageBox(msgspec.Struct, kw_only=True):
    #: `generate`'s prompt: its words, and the pictures it mentions, which Moss-Image-Edit edits or composes.
    instructions: Prompt[Image] = Prompt()
    #: `generate`, through `ImageRequest.size`; held to 1:16 through 16:1.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `generate`, through `SIZES.pixels`: the 1-2 MP the user guide recommends (README, Sizes).
    megapixels: Literal[1, 2] = 1
    #: `generate`'s negative branch, encoded only with guidance above 1.
    negative_prompt: str | None = None
    seed: Seed | None = None


@app.tool(label="Image", named="image", accelerator=False)
async def image(ctx: Context, payload: ImageBox) -> ImageResult:
    """MOSS-Image on a canvas: `generate` from the box, an edit of its pictures when it mentions any."""
    request = GenerateInput(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=payload.megapixels,
                            seed=payload.seed, negative_prompt=payload.negative_prompt)  # fmt: skip
    return await generate(ctx, request)


#: What MOSS-Image is told of a picture it upscales.
UPSCALE_PROMPT = "a high quality, highly detailed, sharp picture"


class UpscaleInput(msgspec.Struct, kw_only=True):
    #: `upscale`: the picture, the box's own version.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
    #: `upscale`: the picture's ratio at this size (`SIZES.fit`), up to MOSS-Image's 4 MP.
    megapixels: Literal[2, 4] = 4
    #: `MossImage.denoise`'s `strength`: the noise the picture starts under (se-123).
    denoise: Annotated[float, msgspec.Meta(gt=0, le=1), Advanced()] = 0.4
    seed: Seed | None = None


@app.tool(label="Upscale", named="image", defaults=_DEFAULTS, components=_COMPONENTS, memoize=True)
def upscale(ctx: Context, payload: UpscaleInput, model: MossImage, out: Outputs, tel: Telemetry,
            adj: Adjustments) -> ImageResult:
    """The picture larger: `remake` at the size asked, from `UPSCALE_PROMPT` (50-step schedule, CFG 3)."""
    with tel.stage("reading picture"):
        picture = payload.image.read()
    width, height = SIZES.fit(adj, picture.size, payload.megapixels * MEGAPIXEL)
    seed = payload.seed if payload.seed is not None else ImageRequest(prompt="").drawn_seed()
    result = remake(ctx, model, out, tel, picture, UPSCALE_PROMPT, "", width=width, height=height, steps=STEPS,
                    guidance=GUIDANCE, seed=seed, strength=payload.denoise)  # fmt: skip
    tel.log("upscale", source=f"{picture.width}x{picture.height}", width=width, height=height, denoise=payload.denoise)
    return result


class ImageToImageInput(msgspec.Struct, kw_only=True):
    #: The picture it starts from.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
    #: What the picture becomes.
    prompt: str = ""
    #: How much changes: the last fraction of the steps that run from the picture, as ComfyUI's denoise
    #: (`MossImage.denoise`), a light touch near 0, a new picture at 1.
    strength: Annotated[float, msgspec.Meta(gt=0, le=1)] = 0.75
    #: The picture's ratio at this many megapixels (`SIZES.fit`); unset, the picture's own size.
    megapixels: Annotated[float, msgspec.Meta(gt=0)] | None = None
    steps: Steps = None
    guidance: Guidance = None
    negative_prompt: NegativePrompt = None
    seed: Seed | None = None


@app.entrypoint(defaults=_DEFAULTS, components=_COMPONENTS, memoize=True)
def image_to_image(ctx: Context, payload: ImageToImageInput, model: MossImage, out: Outputs, tel: Telemetry,
                   adj: Adjustments) -> ImageResult:
    """The picture remade from the prompt (`remake`): 50 Euler steps at CFG 3, those above `strength` skipped."""
    with tel.stage("reading picture"):
        picture = payload.image.read()
    width, height = SIZES.fit(adj, picture.size, (payload.megapixels or picture.width * picture.height / MEGAPIXEL) * MEGAPIXEL)
    steps = held(adj, "steps", payload.steps or STEPS, 1, MOST_STEPS, "MOSS-Image's 1-200 steps")
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0,
                    "MOSS-Image's CFG 1-20")
    seed = payload.seed if payload.seed is not None else ImageRequest(prompt="").drawn_seed()
    return remake(ctx, model, out, tel, picture, payload.prompt, payload.negative_prompt or "", width=width,
                  height=height, steps=steps, guidance=guidance, seed=seed, denoise=payload.strength)  # fmt: skip


def remake(ctx: Context, model: MossImage, out: Outputs, tel: Telemetry, picture: PILImage.Image, prompt: str,
           negative_prompt: str, *, width: int, height: int, steps: int, guidance: float, seed: int,
           strength: float = 1.0, denoise: float | None = None) -> ImageResult:
    """Image-to-image: the picture Lanczos to `width` x `height`, encoded, noised to the sigma `strength` (or the
    schedule's last `denoise` fraction) and denoised from there (`MossImage.denoise`'s `start`), CFG against
    `negative_prompt`."""
    model.hold_vae(tel)
    with tel.stage("encoding picture"):
        start = model.encode_image(picture.convert("RGB").resize((width, height), PILImage.Resampling.LANCZOS))
    with tel.stage("encoding prompt"):
        positive = model.encode(prompt)
        negative = None if guidance == 1.0 else model.encode(negative_prompt)
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(positive, negative, width=width, height=height, steps=steps, guidance=guidance, seed=seed,
                                tel=tel, progress=ctx.progress, start=start, strength=strength, denoise=denoise)  # fmt: skip
    with tel.stage("decoding image"):
        image = model.decode(latents)
    with tel.stage("saving image"):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)
