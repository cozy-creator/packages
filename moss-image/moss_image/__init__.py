"""MOSS-Image (OpenMOSS) text-to-image, with layout boxes for objects and rendered text; with a control map (pose,
depth, canny, lineart, sketch), thedeoxen's control LoRA of its kind steers it."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import Adjustments, App, Context, Lora, ModelChoice, Outputs, Seed, Telemetry, call
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, Ratios
from cozy_runtime.media import ImageFile
from PIL import Image as PILImage
from cozy_runtime.image import (
    MEGAPIXEL,
    Control,
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
#: The control maps MOSS-Image reads (`control`).
type Kinds = Literal["pose", "depth", "canny", "lineart", "sketch"]
#: thedeoxen's control LoRAs for MOSS-Image SFT, one a kind, stacked on `model` at the control's strength: AI-Toolkit
#: edit LoRAs, each trained alone with its map as the one reference.
CONTROL_LORAS: dict[str, str] = {
    "pose": "thedeoxen/moss-image-pose-controlnet@0.1.0",  # a DWPose skeleton: colored bones on black
    "depth": "thedeoxen/moss-image-depth-controlnet@0.1.0",  # near is white
    "canny": "thedeoxen/moss-image-canny-controlnet@0.1.0",  # thin white edges on black
    "lineart": "thedeoxen/moss-image-lineart-controlnet@0.1.0",  # fine white lines on black
    "sketch": "thedeoxen/moss-image-scribble-controlnet@0.1.0",  # loose white strokes on black
}
#: thedeoxen's control workflows: 30 Euler steps at CFG 4.
CONTROL_STEPS, CONTROL_GUIDANCE = 30, 4.0
#: Their workflows' negative without its "nsfw": this platform's users stack NSFW LoRAs, which it would fight.
CONTROL_NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"


class GenerateInput(ImageRequest):
    steps: Steps = None
    guidance: Guidance = None
    #: `generate`'s negative branch, encoded only with guidance above 1.
    negative_prompt: NegativePrompt = None
    #: Elements and rendered text placed by box, described in words (`\n` in `text` breaks a line); the
    #: prompt describes the whole image.
    regions: Annotated[list[Region], msgspec.Meta(max_length=64)] | None = None
    #: A map the picture follows (`control`): the first with a strength above 0 steers it through its kind's adapter
    #: (`CONTROL_LORAS`), at its strength.
    control: list[Control[Literal["pose", "depth", "canny", "lineart", "sketch"]]] = msgspec.field(default_factory=list)


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


_DEFAULTS = {"model": "openmoss-team/moss-image@0.1.1"}
_COMPONENTS = {"model": ("transformer", "text_encoder", "vae")}


def steering(payload: GenerateInput) -> list[Control[Kinds]]:
    """The request's control maps that steer: each with a strength above 0, in order."""
    return [map_ for map_ in payload.control if map_.strength > 0]


@app.entrypoint(accelerator=False)
async def generate(ctx: Context, payload: GenerateInput) -> ImageResult:
    """MOSS-Image from the text (`text_to_image`); with a control map, `control` under its kind's adapter
    (`CONTROL_LORAS`, at its strength), stacked on the model after any caller LoRA. Each is this request's GPU call,
    so the request's model and LoRAs reach it."""
    if maps := steering(payload):
        lora = Lora(CONTROL_LORAS[maps[0].kind], maps[0].strength)
        return await call(control, payload=payload, model=ModelChoice(lora=(lora,)))
    return await call(text_to_image, payload=payload)


@app.entrypoint(defaults=_DEFAULTS, components=_COMPONENTS, memoize=True, private=True)
def text_to_image(ctx: Context, payload: GenerateInput, model: MossImage, out: Outputs, tel: Telemetry,
                  adj: Adjustments) -> ImageResult:
    """MOSS-Image SFT, 50 Euler steps at CFG 3, with `regions` laid out as upstream's layout JSON.

    The negative branch skips block 22."""
    model.hold_vae(tel)
    steps = held(adj, "steps", payload.steps or STEPS, 1, MOST_STEPS, "MOSS-Image's 1-200 steps")
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0,
                    "MOSS-Image's CFG 1-20")
    seed = payload.drawn_seed()
    width, height = payload.size(adj, None, SIZES, megapixels=1.0)
    prompt = layout_prompt(payload.prompt, payload.regions, width, height)
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
def control(ctx: Context, payload: GenerateInput, model: MossImage, out: Outputs, tel: Telemetry,
            adj: Adjustments) -> ImageResult:
    """`generate` under its first control map, the kind's adapter stacked on `model` by its caller (thedeoxen's
    workflows, ComfyUI's `ai-toolkit` edit format): the map at the picture's size is the one noise-free reference at
    t = 0, the image beside it at f = 0 (`MossImage.denoise`'s `edit_format`), and the text encoder sees it before the
    text in both branches; 30 Euler steps at CFG 4 against `CONTROL_NEGATIVE`, at the map's size unless the request
    names `aspect_ratio` or `megapixels` (their workflows' latent is the map's size). Each adapter trained alone on
    one map, so one map steers."""
    model.hold_vae(tel)
    maps = steering(payload)
    if len(maps) > 1:
        adj.warn("control", f"each MOSS-Image control adapter reads one map, so it goes without the {len(maps) - 1} "
                            "after the first")  # fmt: skip
    steer = maps[0]
    if payload.regions:
        adj.warn("regions", "the control adapters read a caption, not a layout, so it goes without the regions")
    steps = held(adj, "steps", payload.steps or CONTROL_STEPS, 1, MOST_STEPS, "MOSS-Image's 1-200 steps")
    guidance = held(adj, "guidance", CONTROL_GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0,
                    "MOSS-Image's CFG 1-20")
    seed = payload.drawn_seed()
    with tel.stage("reading control map"):
        picture = steer.image.read()
    width, height = payload.size(adj, picture.size, SIZES, megapixels=1.0)
    shown = picture.convert("RGB").resize((width, height), PILImage.Resampling.LANCZOS)
    ctx.raise_if_cancelled()
    with tel.stage("encoding control map"):
        reference = model.encode_image(shown)
    with tel.stage("encoding prompt"):
        positive = model.encode(payload.prompt, (shown,))
        negative = None if guidance == 1.0 else model.encode(payload.negative_prompt or CONTROL_NEGATIVE, (shown,))
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(positive, negative, width=width, height=height, steps=steps, guidance=guidance,
                                seed=seed, tel=tel, progress=ctx.progress, references=[reference],
                                edit_format="ai-toolkit")  # fmt: skip
    with tel.stage("decoding image"):
        image = model.decode(latents)
    with tel.stage("saving image"):
        asset = out.save_image(image, format="png")
    tel.log("moss control", kind=steer.kind, strength=steer.strength, map=f"{picture.width}x{picture.height}")
    return ImageResult(asset, width, height, seed, payload.prompt)


class ImageBox(msgspec.Struct, kw_only=True):
    #: `generate`'s prompt, as words.
    instructions: str = ""
    #: `generate`, through `ImageRequest.size`; held to 1:16 through 16:1.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `generate`, through `SIZES.pixels`: the 1-2 MP the user guide recommends (README, Sizes).
    megapixels: Literal[1, 2] = 1
    #: `generate`'s negative branch, encoded only with guidance above 1.
    negative_prompt: str | None = None
    seed: Seed | None = None
    #: `generate`'s control maps: the first steers the picture through its kind's adapter.
    control: list[Control[Literal["pose", "depth", "canny", "lineart", "sketch"]]] = msgspec.field(default_factory=list)


@app.tool(label="Image", named="image", accelerator=False)
async def image(ctx: Context, payload: ImageBox) -> ImageResult:
    """MOSS-Image on a canvas: `generate` from the box, steered by its first control map when it has one."""
    request = GenerateInput(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=payload.megapixels,
                            seed=payload.seed, negative_prompt=payload.negative_prompt, control=payload.control)  # fmt: skip
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
    """The picture larger: Lanczos to the size asked, then MOSS-Image img2img over it (50-step schedule, CFG 3)."""
    model.hold_vae(tel)
    with tel.stage("reading picture"):
        picture = payload.image.read()
    width, height = SIZES.fit(adj, picture.size, payload.megapixels * MEGAPIXEL)
    seed = payload.seed if payload.seed is not None else ImageRequest(prompt="").drawn_seed()
    with tel.stage("encoding picture"):
        start = model.encode_image(picture.resize((width, height), PILImage.Resampling.LANCZOS))
    with tel.stage("encoding prompt"):
        positive, negative = model.encode(UPSCALE_PROMPT), model.encode("")
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(positive, negative, width=width, height=height, steps=STEPS, guidance=GUIDANCE, seed=seed,
                                tel=tel, progress=ctx.progress, start=start, strength=payload.denoise)  # fmt: skip
    with tel.stage("decoding image"):
        image = model.decode(latents)
    with tel.stage("saving image"):
        asset = out.save_image(image, format="png")
    tel.log("upscale", source=f"{picture.width}x{picture.height}", width=width, height=height, denoise=payload.denoise)
    return ImageResult(asset, width, height, seed, UPSCALE_PROMPT)
