"""FLUX.3 Image Klein text-to-image and reference editing, distilled and base, with BFL's prompt upsampler."""

from __future__ import annotations

import json
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    MEGAPIXEL,
    Adjustments,
    App,
    Context,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    ReferenceImages,
    Region,
    Sizes,
    Telemetry,
    corners,
    held,
)
from PIL import Image

from .klein import (
    MATCH_MAX_PIXELS,
    MAX_SIDE,
    MIN_SIDE,
    Klein,
    prepare_multi_reference,
    prepare_reference,
    reference_canvas,
)
from . import quantize as quantize  # its recipes
from .upsampler import Drafter, Row, Upsampler, canvas_box, is_finished, with_table

app = App()

#: Each function's steps when none is given, and the most it admits; Base's CFG scale (BFL's 4.0).
_STEPS = {"generate": (50, 200), "generate_turbo": (16, 100)}
_GUIDANCE, _MAX_GUIDANCE = 4.0, 20.0


#: BFL's 64-4096 px sides and 4 MiP.
SIZES = Sizes((MIN_SIDE**2 / MEGAPIXEL, MATCH_MAX_PIXELS / MEGAPIXEL), (MIN_SIDE, MAX_SIDE), 32,
              "the sizes Klein admits")


class GenerateInput(ImageRequest):
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )
    #: Elements and rendered text placed by box, described in words.
    regions: list[Region] | None = None
    #: Upsample with Qwen3.5-9B's MTP head drafting tokens (1.1-1.2× faster a caption on a 4090);
    #: its first use downloads the 0.5 GB head.
    prompt_drafting: bool = True


def _render(
    ctx: Context,
    model: Klein,
    upsampler: Upsampler,
    drafter: Drafter,
    out: Outputs,
    tel: Telemetry,
    *,
    prompt: str,
    upsample: bool,
    draft: bool,
    images: list[Image.Image],
    prepared: list[Image.Image],
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    canvas: str | None,
    rows: list[Row],
    steps: int,
    guidance: float | None,
    negative: str,
    seed: int,
) -> ImageResult:
    width, height = size
    caption = with_table(prompt, rows) if rows else prompt
    if upsample and (rows or not is_finished(prompt)):
        table = json.dumps(rows, ensure_ascii=False) if rows else ""
        with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
            def run() -> str:
                return upsampler.upsample(prompt, images, canvas, table)
            caption = drafter.drafting(run) if draft else run()
        tel.log("upsampled prompt", caption=caption[:2000])
    ctx.raise_if_cancelled()
    references = None
    if prepared:
        with tel.stage("encoding reference images", overall_range=(0.15, 0.2)):
            references = model.encode_references(prepared)
    with tel.stage("encoding prompt", overall_range=(0.2, 0.25)):
        embeds = model.encode(caption)
        negatives = model.encode(negative) if guidance is not None else None
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.25, 0.95)):
        tokens = model.denoise(
            embeds, negatives, references,
            width=width, height=height, steps=steps, guidance=guidance, seed=seed,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.25, 0.95)),
            cancel=ctx.raise_if_cancelled,
        )
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(tokens, width=width, height=height)
        if crop is not None:
            image = image.crop(crop)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, image.width, image.height, seed, caption)


def _layout(
    adj: Adjustments, payload: GenerateInput, images: list[Image.Image], placed: bool
) -> tuple[list[Image.Image], tuple[int, int], tuple[int, int, int, int] | None, str | None]:
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


def layout_rows(payload: GenerateInput) -> list[Row]:
    """Klein's table rows for the request's boxes: a placed reference is `ref_image_<i>` at its box (the
    upsampler describes it from the image); a region is a new element, its text quoted in the desc."""
    rows: list[Row] = []
    for i, box in enumerate(payload.placements()):
        if box is not None:
            rows.append({"id": f"obj_{len(rows) + 1}", "from": f"ref_image_{i}", "src_bbox": None,
                         "tgt_bbox": canvas_box(box), "desc": ""})
    for i, region in enumerate(payload.regions or ()):
        grid = canvas_box(corners(region.box, f"regions.{i}.box"))
        desc = f'{region.prompt.strip()} "{region.text}"' if region.text else region.prompt.strip()
        row_id = f"obj_{len(rows) + 1}"
        rows.append({"id": row_id, "from": "target", "src_bbox": None, "tgt_bbox": grid, "desc": desc}
                    if payload.reference_images else {"id": row_id, "bbox": grid, "desc": desc})
    return rows


def _generate(ctx: Context, function: str, payload: GenerateInput, drafter: Drafter, model: Klein,
              upsampler: Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    default, most = _STEPS[function]
    steps = held(adj, "steps", payload.steps or default, 1, most)
    if function == "generate":
        given = _GUIDANCE if payload.guidance is None else payload.guidance
        guidance: float | None = held(adj, "guidance", given, 1.0, _MAX_GUIDANCE)
    else:
        guidance = None
        payload.ignored(adj, "guidance", "negative_prompt", reason="the distilled Klein samples without CFG")
    rows = layout_rows(payload)
    images = [asset.read() for asset in payload.reference_images]
    placed = any(box is not None for box in payload.reference_boxes or ())
    prepared, size, crop, canvas = _layout(adj, payload, images, placed)
    upsample = True if payload.prompt_upsampling is None else payload.prompt_upsampling
    return _render(ctx, model, upsampler, drafter, out, tel, prompt=payload.prompt, upsample=upsample,
                   draft=payload.prompt_drafting, images=images, prepared=prepared, size=size, crop=crop,
                   canvas=canvas, rows=rows, steps=steps, guidance=guidance, negative=payload.negative_prompt or "",
                   seed=payload.drawn_seed())


# Qwen3.5-9B's MTP head alone (`drafter_head` makes it): BFL's upsampler checkpoint has none.
_DRAFTER = "qwen/qwen3.5-9b-mtp@0.2.0"
_DISTILLED = {"model": "black-forest-labs/flux-3-klein@0.1.0", "upsampler": "black-forest-labs/flux-3-klein-upsampler@0.2.0", "drafter": _DRAFTER}
_BASE = {"model": "black-forest-labs/flux-3-klein-base@0.1.0", "upsampler": "black-forest-labs/flux-3-klein-upsampler@0.2.0", "drafter": _DRAFTER}
_COMPONENTS = {
    "model": ("transformer", "text_encoder", "vae"),
    "upsampler": ("model",),
    "drafter": ("model",),
}
#: Built and fetched on first use: `prompt_upsampling=False` never pulls the upsampler, and
#: `prompt_drafting=False` never pulls the drafter.
_ON_DEMAND = ("upsampler", "drafter")


@app.entrypoint(defaults=_BASE, components=_COMPONENTS, on_demand=_ON_DEMAND,
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, drafter: Drafter, model: Klein, upsampler: Upsampler,
             out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Klein Base, 50 steps with CFG 4: text to image, or an edit of up to ten ordered references."""
    return _generate(ctx, "generate", payload, drafter, model, upsampler, out, tel, adj)


@app.entrypoint(defaults=_DISTILLED, components=_COMPONENTS, on_demand=_ON_DEMAND,
                memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: GenerateInput, drafter: Drafter, model: Klein, upsampler: Upsampler,
                   out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The distilled Klein, 16 steps, no guidance: text to image, or an edit of up to ten ordered references."""
    return _generate(ctx, "generate_turbo", payload, drafter, model, upsampler, out, tel, adj)
