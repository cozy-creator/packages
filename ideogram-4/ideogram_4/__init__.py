"""Ideogram 4: text to image from a structured JSON caption, with layout boxes and rendered text."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Literal

import msgspec
import numpy as np
from cozy_runtime.author import Adjustments, App, Context, Seed, Telemetry, UnsupportedInput
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, Ratios
from cozy_runtime.image import (
    Guidance,
    ImageRequest,
    ImageResult,
    MEGAPIXEL,
    Region,
    Sizes,
    Steps,
    corners,
    held,
)
from cozy_runtime.media import ImageFile

from . import quantize as quantize  # its recipes
from .model import PATCH, Ideogram4, guidance_schedule
from .text_encoder import QwenEncoder, QwenUpsampler

if TYPE_CHECKING:
    from PIL import Image as PILImage

app = App()

#: Ideogram 4 serves 256-2048 px sides on its 16 px grid, at most 6:1 (README, `docs/inference.md`); the
#: reference CLI's 1024² by default, up to 2048² (4 MP, Ideogram's highest quality).
SIZES = Sizes((256 * 256 / MEGAPIXEL, 4.0), (256, 2048), PATCH, "Ideogram 4 serves 0.06-4 MP", widest=6.0)
#: Diffusers' default: V4_QUALITY_48, 48 steps at guidance 7 with a 3.0 polish tail; V4_TURBO_12 is the box's Turbo.
STEPS, TURBO_STEPS, GUIDANCE = 48, 12, 7.0
#: How much of a start picture is redrawn by default (se-175's runs).
STRENGTH = 0.8

class GenerateInput(ImageRequest):
    steps: Steps = None
    guidance: Guidance = None
    #: `caption`: None or on has the upsampler write a plain prompt's caption.
    prompt_upsampling: bool | None = None
    #: Elements placed by box, in order: an object (`prompt`), or rendered text (`text`, styled as `prompt` says).
    regions: list[Region] = msgspec.field(default_factory=list)
    #: Image to image: start from this picture's latents, `strength` of the schedule redrawing it (`Ideogram4.denoise`).
    #: The model takes no picture otherwise (README).
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    #: How much of the picture is redrawn: 1 is from noise.
    strength: Annotated[float, msgspec.Meta(gt=0.0, le=1.0), Advanced()] = STRENGTH


def element(region: Region, index: int) -> dict[str, object]:
    """A region as Ideogram's caption element: `bbox` is `[y_min, x_min, y_max, x_max]` on its
    0-1000 grid, keys in the order it was trained on."""
    x0, y0, x1, y1 = corners(region.box, f"regions.{index}.box")
    bbox = [round(y0 * 1000), round(x0 * 1000), round(y1 * 1000), round(x1 * 1000)]
    if region.text is None:
        return {"type": "obj", "bbox": bbox, "desc": region.prompt}
    return {"type": "text", "bbox": bbox, "text": region.text, "desc": region.prompt}


def _json_object(prompt: str) -> dict[str, object] | None:
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _with_regions(caption: dict[str, object], regions: list[Region]) -> str:
    composition = caption.setdefault("compositional_deconstruction", {"background": "", "elements": []})
    if not isinstance(composition, dict):
        raise UnsupportedInput("the caption's compositional_deconstruction is not an object: give it a background and elements",
                               code="prompt")
    if not isinstance(composition.setdefault("elements", []), list):
        composition["elements"] = [composition["elements"]]
    composition["elements"].extend(element(region, i) for i, region in enumerate(regions))
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def caption(payload: GenerateInput, prompt: str, upsampler: QwenUpsampler, *, width: int, height: int,
            seed: int, tel: Telemetry) -> str:
    """The JSON caption Ideogram 4 reads. A JSON prompt is the caption as given; otherwise the
    upsampler writes one. `regions` are appended to its elements. A plain prompt without the
    upsampler is refused: Ideogram 4 reads only captions, and on a bare description it draws its own
    "Image blocked by safety filter" card (runs 19807, 19809), so this package writes no caption
    content the request did not give."""
    given = _json_object(prompt)
    if given is not None:
        return _with_regions(given, payload.regions) if payload.regions else prompt
    if payload.prompt_upsampling is False:
        raise UnsupportedInput(
            "Ideogram 4 reads a JSON caption, and a plain prompt with prompt_upsampling=false has none: "
            "leave prompt_upsampling on (the default) to have the upsampler write it, or pass the caption "
            "itself as the prompt (high_level_description, and compositional_deconstruction with its "
            "background and elements)", code="prompt", fields=("prompt", "prompt_upsampling"))
    with tel.stage("upsampling prompt"):
        written = json.loads(upsampler.caption(prompt, width=width, height=height, seed=seed))
    return _with_regions(written, payload.regions)


def refusal_card(image: PILImage.Image) -> bool:
    """Whether `image` looks like Ideogram 4's own refusal, the flat grey "Image blocked by safety
    filter" card its weights draw for some captions and seeds (RGB about 114 with one line of light
    text): every pixel grey, and nearly all of them the field's one shade. A flat grey design can
    match too, so this only warns."""
    rgb = np.asarray(image.convert("RGB").reduce(4), dtype=np.int16)
    luma = rgb.mean(axis=-1)
    field = float(np.median(luma))
    return (float(np.percentile(rgb.max(axis=-1) - rgb.min(axis=-1), 99)) <= 12 and 95 <= field <= 135
            and float((np.abs(luma - field) <= 8).mean()) >= 0.85)


#: The text encoder and the upsampler are stock Qwen3-VL-8B-Instruct, so both default to pottokao's Heretic
#: abliteration of it (Paul, 11-17: "use abliterated models wherever possible"), one checkpoint for both; Qwen's own
#: is `model.text_encoder=qwen/qwen3-vl-8b-instruct@0.1.1` (and `model.upsampler=`).
_QWEN = "pottokao/qwen-image-2.1-text-encoder-heretic@0.1.0"
_DEFAULTS = {"model": "ideogram-ai/ideogram-4@0.2.0", "text_encoder": _QWEN, "upsampler": _QWEN}
_COMPONENTS = {"model": ("transformer", "unconditional_transformer", "vae"), "text_encoder": ("model",),
               "upsampler": ("model",)}


# The upsampler is fetched only for a request with `prompt_upsampling` on (unset is on), and read only for a plain
# prompt (`caption`): a JSON caption, which only its text shows, over-fetches it (cl-1103). Model parameters are in
# name order: Runtime refuses a binding set whose paths are not sorted.
@app.entrypoint(defaults=_DEFAULTS, components=_COMPONENTS, needs={"upsampler": "request.prompt_upsampling != false"},
                memoize=True)
def generate(ctx: Context, payload: GenerateInput, model: Ideogram4, text_encoder: QwenEncoder,
             upsampler: QwenUpsampler, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Ideogram 4, 48 steps at guidance 7: a JSON caption or a prompt, with `regions` placed by box.

    Guidance is 3 on the last steps; the upsampler expands a prompt into the caption."""
    model.hold_vae(tel)
    prompt = payload.prompt
    steps = held(adj, "steps", payload.steps or STEPS, 1, 200)
    weight = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
    seed = payload.drawn_seed()
    picture = payload.image.read() if payload.image is not None else None
    width, height = payload.size(adj, None if picture is None else picture.size, SIZES, megapixels=1.0)
    written = caption(payload, prompt, upsampler, width=width, height=height, seed=seed, tel=tel)
    if written != prompt:
        tel.log("caption", caption=written[:4000])
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt"):
        features = text_encoder.encode(written)
    start = None
    if picture is not None:
        with tel.stage("encoding picture"):
            start = model.encode(picture, width=width, height=height)
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(
            features, width=width, height=height, guidance=guidance_schedule(steps, weight), seed=seed,
            progress=ctx.progress, start=start, strength=payload.strength,
        )
    with tel.stage("decoding image"):
        image = model.decode(latents, width=width, height=height)
    if refusal_card(image):
        tel.log("the image looks like Ideogram 4's own refusal card (\"Image blocked by safety filter\"): "
                "its weights draw it for some captions and seeds, harmless ones included; a more detailed "
                "prompt or another seed usually renders", level="warning", seed=seed)
    with tel.stage("saving image"):
        asset = ImageFile.from_image(image, format="png")
    return ImageResult(asset, width, height, seed, written)


class ImageBox(msgspec.Struct, kw_only=True):
    #: `generate`'s prompt, as words, or a JSON caption.
    instructions: str = ""
    #: `generate`, through `ImageRequest.size`; held to 1:6 through 6:1.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 6, max=6)] | None = None
    #: `generate`, through `SIZES.pixels`: the reference CLI's 1 MP default up to its 4 MP, 2048 px sides (README).
    megapixels: Literal[1, 2, 4] = 1
    #: Ideogram's V4_TURBO_12, 12 steps, else V4_QUALITY_48 (`generate`'s steps).
    turbo: bool = True
    #: `generate`: start from this picture (image to image).
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    #: `generate`: how much of the start picture is redrawn.
    strength: Annotated[float, msgspec.Meta(gt=0.0, le=1.0), Advanced()] = STRENGTH
    seed: Seed | None = None


# A plain prompt is always upsampled into Ideogram's caption (`caption`); a JSON caption goes as given, but only its
# text says so, so every box fetches the upsampler (cl-1103).
@app.tool(label="Image", named="image", defaults=_DEFAULTS, components=_COMPONENTS, needs={"upsampler": "true"},
          memoize=True)  # fmt: skip
def image(ctx: Context, payload: ImageBox, model: Ideogram4, text_encoder: QwenEncoder, upsampler: QwenUpsampler,
          tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Ideogram 4 on a canvas: Turbo's 12 steps, or 48."""
    request = GenerateInput(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=payload.megapixels,
                            steps=TURBO_STEPS if payload.turbo else STEPS, seed=payload.seed, image=payload.image,
                            strength=payload.strength)  # fmt: skip
    return generate(ctx, request, model, text_encoder, upsampler, tel, adj)
