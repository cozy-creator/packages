"""FLUX.3 Klein text-to-image and reference editing, distilled and base, with BFL's prompt upsampler."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import Adjustments, App, Context, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, Image, Prompt, Ratios, name_key
from cozy_runtime.media import ImageFile
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
    KleinTextEncoder,
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
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text
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
    text_encoder: KleinTextEncoder,
    upsampler: Upsampler,
    drafter: Drafter,
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
        embeds = text_encoder.encode(caption)
        negatives = text_encoder.encode(negative) if guidance is not None else None
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
        asset = ImageFile.from_image(image, format="png")
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


def _generate(ctx: Context, function: str, payload: TurboInput, drafter: Drafter, model: Klein, text_encoder: KleinTextEncoder,
              upsampler: Upsampler, tel: Telemetry, adj: Adjustments) -> ImageResult:
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
    return _render(ctx, model, text_encoder, upsampler, drafter, tel, adj, prompt=prompt, upsample=upsample,
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
#: Klein's text encoder is stock Qwen3-VL-4B-Instruct (BFL's shards are Qwen's), its own slot: DreamFast's Heretic
#: abliteration by default (Paul, 11-17: "use abliterated models wherever possible"), the same weights as krea-2's.
#: 0.2.0 holds the encoder alone (se-160: a release holds only its author's work); `KleinTextEncoder` builds only the
#: encoder, so BFL's own is any Klein checkpoint: `model.text_encoder=black-forest-labs/flux-3-klein@0.1.0`. `model`
#: (and `full`) read only the DiT and the VAE.
_TEXT_ENCODER = "dreamfast/qwen3-vl-4b-heretic-klein@0.2.0"
_KLEIN = ("transformer", "vae")
_TEXT = ("text_encoder",)
_COMPONENTS = {"model": _KLEIN, "text_encoder": _TEXT, "upsampler": ("model",), "drafter": ("model",)}
#: Which requests need each (cl-1103, `_render`): the upsampler with upsampling on (unset is on), the drafter with
#: drafting on too. Over-fetched: a finished layout caption (`is_finished`), which goes as written, but only its text says so.
_NEEDS = {"upsampler": "request.prompt_upsampling != false",
          "drafter": "request.prompt_upsampling != false && request.prompt_drafting"}


@app.entrypoint(defaults={"model": _BASE, "text_encoder": _TEXT_ENCODER, "upsampler": _UPSAMPLER, "drafter": _DRAFTER},
                components=_COMPONENTS, needs=_NEEDS, memoize=True)
def generate(ctx: Context, payload: GenerateInput, drafter: Drafter, model: Klein, text_encoder: KleinTextEncoder,
             upsampler: Upsampler, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Klein Base, 50 steps with CFG 4: text to image, or an edit of up to ten ordered pictures."""
    return _generate(ctx, "generate", payload, drafter, model, text_encoder, upsampler, tel, adj)


@app.entrypoint(defaults={"model": _DISTILLED, "text_encoder": _TEXT_ENCODER, "upsampler": _UPSAMPLER,
                          "drafter": _DRAFTER}, components=_COMPONENTS, needs=_NEEDS, memoize=True)
def generate_turbo(ctx: Context, payload: TurboInput, drafter: Drafter, model: Klein, text_encoder: KleinTextEncoder,
                   upsampler: Upsampler, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The distilled Klein, 16 steps, no guidance: text to image, or an edit of up to ten ordered pictures."""
    return _generate(ctx, "generate_turbo", payload, drafter, model, text_encoder, upsampler, tel, adj)


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


# Each checkpoint is fetched only for a box that uses it: the distilled Klein (`model`) with `turbo`, else Base
# (`full`); the upsampler and drafter as `_NEEDS` says, a box always drafting.
_BOX_NEEDS = {"drafter": "request.prompt_upsampling", "full": "!request.turbo", "model": "request.turbo",
              "upsampler": "request.prompt_upsampling"}


@app.tool(label="FLUX.3 Klein", named="image",
          defaults={"drafter": _DRAFTER, "full": _BASE, "model": _DISTILLED, "text_encoder": _TEXT_ENCODER,
                    "upsampler": _UPSAMPLER},
          components={"drafter": ("model",), "full": _KLEIN, "model": _KLEIN, "text_encoder": _TEXT,
                      "upsampler": ("model",)},
          needs=_BOX_NEEDS, memoize=True)  # fmt: skip
def image(ctx: Context, payload: ImageBox, drafter: Drafter, full: Klein, model: Klein, text_encoder: KleinTextEncoder,
          upsampler: Upsampler, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """FLUX.3 Klein on a canvas, upsampled: the distilled model, or Base, which loads only when asked for."""
    kind, function, klein = (TurboInput, "generate_turbo", model) if payload.turbo else (GenerateInput, "generate", full)
    request = kind(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=payload.megapixels,
                   seed=payload.seed, prompt_upsampling=payload.prompt_upsampling)  # fmt: skip
    return _generate(ctx, function, request, drafter, klein, text_encoder, upsampler, tel, adj)


#: What Klein is asked of a picture it upscales, read directly (`tagged`).
UPSCALE_PROMPT = (f"Upscale {tagged(1)} to a higher resolution. Keep the picture exactly as it is: the same composition, "
                  "framing, colours, lighting and every detail; only make it sharper and more finely detailed.")


class UpscaleInput(msgspec.Struct, kw_only=True):
    #: `upscale`: the picture, the box's own version.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
    #: `upscale`: the picture's ratio at this size (`SIZES.fit`), up to Klein's 4 MiP.
    megapixels: Literal[2, 4] = 4
    #: `Klein.denoise`'s `strength`: the noise the picture starts under; 1 redraws it from noise (se-123).
    denoise: Annotated[float, msgspec.Meta(gt=0, le=1), Advanced()] = 0.35
    seed: Seed | None = None


@app.tool(label="Upscale", named="image", defaults={"model": _DISTILLED, "text_encoder": _TEXT_ENCODER},
          components={"model": _KLEIN, "text_encoder": _TEXT}, memoize=True)  # fmt: skip
def upscale(ctx: Context, payload: UpscaleInput, model: Klein, text_encoder: KleinTextEncoder, tel: Telemetry,
            adj: Adjustments) -> ImageResult:
    """The picture larger: Klein edits it at the size asked, the picture Lanczos-upscaled to the canvas both as the
    reference and, below `denoise`, as the start."""
    with tel.stage("reading picture"):
        picture = payload.image.read()
    width, height = SIZES.fit(adj, picture.size, payload.megapixels * MEGAPIXEL)
    canvas = picture.resize((width, height), PILImage.Resampling.LANCZOS)
    seed = payload.seed if payload.seed is not None else ImageRequest(prompt="").drawn_seed()
    model.hold_vae(tel)
    with tel.stage("encoding reference images"):
        references = model.encode_references([canvas])
    with tel.stage("encoding prompt"):
        embeds = text_encoder.encode(UPSCALE_PROMPT)
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        tokens = model.denoise(
            embeds, None, references, width=width, height=height, steps=_STEPS["generate_turbo"][0], guidance=None,
            seed=seed, progress=ctx.progress, start=references[0] if payload.denoise < 1 else None,
            strength=payload.denoise,
        )
    with tel.stage("decoding image"):
        image = model.decode(tokens, width=width, height=height)
    with tel.stage("saving image"):
        asset = ImageFile.from_image(image, format="png")
    tel.log("upscale", source=f"{picture.width}x{picture.height}", width=width, height=height, denoise=payload.denoise)
    return ImageResult(asset, width, height, seed, UPSCALE_PROMPT)
