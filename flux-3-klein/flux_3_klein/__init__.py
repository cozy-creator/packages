"""FLUX.3 Image Klein text-to-image and reference editing, distilled and base, with BFL's prompt upsampler."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import Adjustments, App, Context, Outputs, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, Image, Prompt, Ratios, name_key
from cozy_runtime.image import (
    Box,
    Guidance,
    ImageRequest,
    ImageResult,
    MEGAPIXEL,
    NegativePrompt,
    Region,
    Sizes,
    Steps,
    corners,
    held,
)
from PIL import Image as PILImage

from .klein import (
    MAX_SIDE,
    MIN_SIDE,
    Klein,
    prepare_multi_reference,
    prepare_reference,
    reference_canvas,
)
from . import quantize as quantize  # its recipes
from .upsampler import MAX_REFERENCES, Drafter, Row, Upsampler, canvas_box, is_finished, with_table

app = App()

#: Each function's steps when none is given, and the most it admits; Base's CFG scale (BFL's 4.0).
_STEPS = {"generate": (50, 200), "generate_turbo": (16, 100)}
_GUIDANCE, _MAX_GUIDANCE = 4.0, 20.0


#: BFL's 64-4096 px sides and 4 MiP: from one 64 px square to 4 MP.
MIN_MEGAPIXELS = 64**2 / (1 << 20)
MAX_MEGAPIXELS = 4.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (MIN_SIDE, MAX_SIDE), 32, "the sizes Klein admits")


#: A picture's corners on the output, 0-1: `[x0, y0, x1, y1]`.
Corners = tuple[float, float, float, float]


class TurboInput(ImageRequest):
    #: Its text, and the pictures it mentions, in the order it first mentions them: up to ten.
    prompt: Prompt[Image] = Prompt()
    #: Where a mentioned picture goes in the output, by its key. Placed, the pictures are elements, not the canvas.
    boxes: dict[str, Box] = msgspec.field(default_factory=dict)
    steps: Steps = None
    prompt_upsampling: bool | None = None
    #: Elements and rendered text placed by box, described in words.
    regions: list[Region] | None = None
    #: Upsample with Qwen3.5-9B's MTP head drafting tokens (1.1-1.2× faster a caption on a 4090);
    #: its first use downloads the 0.5 GB head.
    prompt_drafting: bool = True


class GenerateInput(TurboInput):
    guidance: Guidance = None
    negative_prompt: NegativePrompt = None


def _render(
    ctx: Context,
    model: Klein,
    upsampler: Upsampler,
    drafter: Drafter,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
    *,
    prompt: str,
    upsample: bool,
    draft: bool,
    images: list[PILImage.Image],
    prepared: list[PILImage.Image],
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    canvas: str | None,
    rows: list[Row],
    steps: int,
    guidance: float | None,
    negative: str,
    seed: int,
) -> ImageResult:
    model.hold_vae(tel)
    width, height = size
    caption = with_table(prompt, rows) if rows else prompt
    if upsample and (rows or not is_finished(prompt)):
        table = json.dumps(rows, ensure_ascii=False) if rows else ""
        with tel.stage("upsampling prompt"):
            def run() -> str:
                return upsampler.upsample(prompt, images, canvas, table)
            written = drafter.drafting(run) if draft else run()
        if written:
            caption = written
        else:
            adj.warn("prompt_upsampling", "the upsampler returned nothing; the prompt goes as written")
        tel.log("upsampled prompt", caption=caption[:2000])
    ctx.raise_if_cancelled()
    references = None
    if prepared:
        with tel.stage("encoding reference images"):
            references = model.encode_references(prepared)
    with tel.stage("encoding prompt"):
        embeds = model.encode(caption)
        negatives = model.encode(negative) if guidance is not None else None
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        tokens = model.denoise(
            embeds, negatives, references,
            width=width, height=height, steps=steps, guidance=guidance, seed=seed,
            progress=ctx.progress,
        )
    with tel.stage("decoding image"):
        image = model.decode(tokens, width=width, height=height)
        if crop is not None:
            image = image.crop(crop)
    with tel.stage("saving image"):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, image.width, image.height, seed, caption)


def _layout(
    adj: Adjustments, payload: TurboInput, images: list[PILImage.Image], placed: bool
) -> tuple[list[PILImage.Image], tuple[int, int], tuple[int, int, int, int] | None, str | None]:
    """The prepared references, the output size, the crop after decoding and the upsampler's canvas.
    No references, or references placed by box (elements, not the canvas): the ratio (1:1) at the
    size (1 MP). Otherwise what is not given is the first reference's: given neither, its own size,
    fitted to the 4 MiP Klein admits and generated on the 16 px grid, then cropped back; given one,
    its ratio or its pixel count."""
    if len(images) == 1:
        prepared = [prepare_reference(images[0])[0]]
    else:
        prepared = [prepare_multi_reference(image) for image in images]
    first = images[0].size if images and not placed else None
    if first is not None and payload.aspect_ratio is None and payload.megapixels is None:
        fit, canvas = reference_canvas(*first)
        if fit != first:
            adj.clamp("megapixels", round(first[0] * first[1] / MEGAPIXEL, 3), round(fit[0] * fit[1] / MEGAPIXEL, 3),
                      "the first reference's size, fitted to the 4 MiP and 4096 px sides Klein admits")
        left, top = (canvas[0] - fit[0]) // 2, (canvas[1] - fit[1]) // 2
        return prepared, canvas, None if fit == canvas else (left, top, left + fit[0], top + fit[1]), None
    target = payload.target(first, megapixels=1.0)
    size = SIZES.fit(adj, target.ratio, SIZES.pixels(adj, target))
    return prepared, size, None, None if images else f"{target.ratio[0]}:{target.ratio[1]}"


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def ordinal(n: int) -> str:
    """The Nth picture as BFL's upsampler is told of them: "the first image", "the second image", ..."""
    return f"the {_ORDINALS[n - 1]} image"


def ref_id(n: int) -> str:
    """The Nth picture as Klein's layout record names it, from 0: `ref_image_0`, ..."""
    return f"ref_image_{n - 1}"


def tagged(n: int) -> str:
    """The Nth picture in a prompt Klein reads directly, from 0: `<ref_image_0>`, ... (Diffusers' Flux3 docs)."""
    return f"<ref_image_{n - 1}>"


def pictures(adj: Adjustments, payload: TurboInput) -> dict[str, Corners | None]:
    """The pictures Klein takes, by key in order of first mention, each with its box or None. Past ten the last
    mentioned go, with a warning. A box's key matches as a mention does: any case, `_`, `-` and space alike."""
    keys = list(payload.prompt.references)
    if len(keys) > MAX_REFERENCES:
        adj.warn("prompt", f"Klein takes {MAX_REFERENCES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_REFERENCES:])}")
        keys = keys[:MAX_REFERENCES]
    given = {name_key(key): key for key in payload.boxes}
    placed: dict[str, Corners | None] = {}
    for key in keys:
        named = given.pop(name_key(key), None)
        placed[key] = None if named is None else corners(payload.boxes[named], f"boxes.{named}")
    if given:
        adj.warn("boxes", f"ignored: {', '.join(f'<{key}>' for key in given.values())} names no picture Klein takes")
    return placed


def worded(prompt: Prompt[Image], placed: dict[str, Corners | None], label: Callable[[int], str]) -> str:
    """The prompt as Klein's reader takes it: each mention of the Nth picture in its words (`label`)."""
    text = prompt.text
    for n, key in enumerate(placed, 1):
        text = text.replace(f"<{key}>", label(n))
    return text


def layout_rows(payload: TurboInput, placed: dict[str, Corners | None]) -> list[Row]:
    """Klein's table rows for the request's boxes: a placed picture is `ref_image_<i>` at its box (the
    upsampler describes it from the image); a region is a new element, its text quoted in the desc."""
    rows: list[Row] = []
    for i, box in enumerate(placed.values()):
        if box is not None:
            rows.append({"id": f"obj_{len(rows) + 1}", "from": f"ref_image_{i}", "src_bbox": None,
                         "tgt_bbox": canvas_box(box), "desc": ""})
    for i, region in enumerate(payload.regions or ()):
        grid = canvas_box(corners(region.box, f"regions.{i}.box"))
        desc = f'{region.prompt.strip()} "{region.text}"' if region.text else region.prompt.strip()
        row_id = f"obj_{len(rows) + 1}"
        rows.append({"id": row_id, "from": "target", "src_bbox": None, "tgt_bbox": grid, "desc": desc}
                    if placed else {"id": row_id, "bbox": grid, "desc": desc})
    return rows


def _generate(ctx: Context, function: str, payload: TurboInput, drafter: Drafter, model: Klein,
              upsampler: Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    default, most = _STEPS[function]
    steps = held(adj, "steps", payload.steps or default, 1, most)
    guidance: float | None = None
    negative = ""
    if isinstance(payload, GenerateInput):
        given = _GUIDANCE if payload.guidance is None else payload.guidance
        guidance = held(adj, "guidance", given, 1.0, _MAX_GUIDANCE)
        negative = payload.negative_prompt or ""
    placed = pictures(adj, payload)
    rows = layout_rows(payload, placed)
    images = [payload.prompt.references[key].file.read() for key in placed]
    prepared, size, crop, canvas = _layout(adj, payload, images, any(box is not None for box in placed.values()))
    upsample = True if payload.prompt_upsampling is None else payload.prompt_upsampling
    # The upsampler is told the references' order in words, or as the layout record's ids; Klein
    # reading the prompt itself was trained on `<ref_image_N>`.
    prompt = worded(payload.prompt, placed, (ref_id if rows else ordinal) if upsample else tagged)
    return _render(ctx, model, upsampler, drafter, out, tel, adj, prompt=prompt, upsample=upsample,
                   draft=payload.prompt_drafting, images=images, prepared=prepared, size=size, crop=crop,
                   canvas=canvas, rows=rows, steps=steps, guidance=guidance, negative=negative,
                   seed=payload.drawn_seed())


#: The upsampler is AtomicChat's abliterated one by default (Paul, 11-06): BFL's with its refusal direction projected
#: out. BFL's own is `model.upsampler=black-forest-labs/flux-3-klein-upsampler@0.2.0`.
_UPSAMPLER = "atomicchat/flux-3-klein-upsampler-abliterated@0.1.0"
# Qwen3.5-9B's MTP head alone (`drafter_head` makes it): neither upsampler checkpoint has one.
_DRAFTER = "qwen/qwen3.5-9b-mtp@0.2.0"
_DISTILLED = "black-forest-labs/flux-3-klein@0.1.0"
_BASE = "black-forest-labs/flux-3-klein-base@0.1.0"
_KLEIN = ("transformer", "text_encoder", "vae")
_COMPONENTS = {"model": _KLEIN, "upsampler": ("model",), "drafter": ("model",)}
#: Built and fetched on first use: `prompt_upsampling=False` never pulls the upsampler, and
#: `prompt_drafting=False` never pulls the drafter.
_ON_DEMAND = ("upsampler", "drafter")


@app.entrypoint(defaults={"model": _BASE, "upsampler": _UPSAMPLER, "drafter": _DRAFTER}, components=_COMPONENTS,
                on_demand=_ON_DEMAND, memoize=True)
def generate(ctx: Context, payload: GenerateInput, drafter: Drafter, model: Klein, upsampler: Upsampler,
             out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Klein Base, 50 steps with CFG 4: text to image, or an edit of up to ten ordered pictures."""
    return _generate(ctx, "generate", payload, drafter, model, upsampler, out, tel, adj)


@app.entrypoint(defaults={"model": _DISTILLED, "upsampler": _UPSAMPLER, "drafter": _DRAFTER}, components=_COMPONENTS,
                on_demand=_ON_DEMAND, memoize=True)
def generate_turbo(ctx: Context, payload: TurboInput, drafter: Drafter, model: Klein, upsampler: Upsampler,
                   out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The distilled Klein, 16 steps, no guidance: text to image, or an edit of up to ten ordered pictures."""
    return _generate(ctx, "generate_turbo", payload, drafter, model, upsampler, out, tel, adj)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it mentions (`pictures`, `worded`).
    instructions: Prompt[Image] = Prompt()
    #: The distilled Klein's 16 steps, else Base's 50 with CFG 4 (`_generate`).
    turbo: bool = True
    #: `_layout`, through `ImageRequest.target`; `SIZES.fit` holds it to 1:16 through 16:1.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `_layout`, through `SIZES.pixels`: BFL's 1 MP `default_size` up to Klein's 4 MP (README).
    megapixels: Literal[1, 2, 4] = 1
    #: `_generate`'s `drawn_seed`.
    seed: Seed | None = None
    #: BFL's upsampler rewrites the prompt first (`_render`).
    prompt_upsampling: Annotated[bool, Advanced()] = True


# Each checkpoint is built, and its weights fetched, only when a box asks for it.
@app.tool(label="Image", named="image",
          defaults={"drafter": _DRAFTER, "full": _BASE, "model": _DISTILLED, "upsampler": _UPSAMPLER},
          components={"drafter": ("model",), "full": _KLEIN, "model": _KLEIN, "upsampler": ("model",)},
          on_demand=("drafter", "full", "model", "upsampler"), memoize=True)  # fmt: skip
def image(ctx: Context, payload: ImageBox, drafter: Drafter, full: Klein, model: Klein, upsampler: Upsampler,
          out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """FLUX.3 Image Klein on a canvas, upsampled: the distilled model, or Base, which loads only when asked for."""
    kind, function, klein = (TurboInput, "generate_turbo", model) if payload.turbo else (GenerateInput, "generate", full)
    request = kind(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=payload.megapixels,
                   seed=payload.seed, prompt_upsampling=payload.prompt_upsampling)  # fmt: skip
    return _generate(ctx, function, request, drafter, klein, upsampler, out, tel, adj)
