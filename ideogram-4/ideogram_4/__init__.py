"""Ideogram 4: text to image from a structured JSON caption, with layout boxes and rendered text."""

from __future__ import annotations

import json
from typing import Annotated, Any

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
    UnsupportedInput,
    corners,
    held,
)

from . import quantize as quantize  # its recipes
from .model import PATCH, Ideogram4, guidance_schedule
from .text_encoder import QwenEncoder, QwenUpsampler

app = App()

#: Ideogram 4 serves 256-2048 px sides on its 16 px grid, at most 6:1 (README, `docs/inference.md`); the
#: reference CLI's 1024² by default, up to 2048² (4 MP, Ideogram's highest quality).
SIZES = Sizes((256 * 256 / MEGAPIXEL, 4.0), (256, 2048), PATCH, "Ideogram 4 serves 0.06-4 MP", widest=6.0)
#: Diffusers' default: V4_QUALITY_48, 48 steps at guidance 7 with a 3.0 polish tail.
STEPS, GUIDANCE = 48, 7.0

class GenerateInput(ImageRequest):
    #: Ideogram 4 is text to image only.
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=0)] = msgspec.field(default_factory=list)
    #: Elements placed by box, in order: an object (`prompt`), or rendered text (`text`, styled as `prompt` says).
    regions: list[Region] = msgspec.field(default_factory=list)


def element(region: Region, index: int) -> dict[str, Any]:
    """A region as Ideogram's caption element: `bbox` is `[y_min, x_min, y_max, x_max]` on its
    0-1000 grid, keys in the order it was trained on."""
    x0, y0, x1, y1 = corners(region.box, f"regions.{index}.box")
    bbox = [round(y0 * 1000), round(x0 * 1000), round(y1 * 1000), round(x1 * 1000)]
    if region.text is None:
        return {"type": "obj", "bbox": bbox, "desc": region.prompt}
    return {"type": "text", "bbox": bbox, "text": region.text, "desc": region.prompt}


def _json_object(prompt: str) -> dict[str, Any] | None:
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _with_regions(caption: dict[str, Any], regions: list[Region]) -> str:
    composition = caption.setdefault("compositional_deconstruction", {"background": "", "elements": []})
    if not isinstance(composition, dict) or not isinstance(composition.setdefault("elements", []), list):
        raise UnsupportedInput("the caption's compositional_deconstruction.elements is not a list", code="prompt")
    composition["elements"].extend(element(region, i) for i, region in enumerate(regions))
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def caption(payload: GenerateInput, upsampler: QwenUpsampler, *, width: int, height: int, seed: int,
            tel: Telemetry) -> str:
    """The JSON caption Ideogram 4 reads. A JSON prompt is the caption as given; otherwise the
    upsampler writes one, or with `prompt_upsampling=false` the prompt is its one-line description.
    `regions` are appended to its elements."""
    given = _json_object(payload.prompt)
    if given is not None:
        return _with_regions(given, payload.regions) if payload.regions else payload.prompt
    if payload.prompt_upsampling is False:
        written: dict[str, Any] = {"high_level_description": payload.prompt,
                                   "compositional_deconstruction": {"background": "", "elements": []}}
    else:
        with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
            written = json.loads(upsampler.caption(payload.prompt, width=width, height=height, seed=seed))
    return _with_regions(written, payload.regions)


# The upsampler is built and fetched only for a plain prompt with `prompt_upsampling` on. Model
# parameters are in name order: Runtime refuses a binding set whose paths are not sorted.
@app.entrypoint(
    defaults={"model": "ideogram-4@0.1.0", "text_encoder": "qwen3-vl-8b-instruct@0.1.0",
              "upsampler": "qwen3-vl-8b-instruct@0.1.0"},
    components={"model": ("transformer", "unconditional_transformer", "vae"), "text_encoder": ("model",),
                "upsampler": ("model",)},
    on_demand=("upsampler",),
    memoize=Memoize(requires=["seed"]),
)
def generate(ctx: Context, payload: GenerateInput, model: Ideogram4, text_encoder: QwenEncoder,
             upsampler: QwenUpsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Ideogram 4, 48 steps at guidance 7 (3 on the last steps): a JSON caption, or a prompt the
    upsampler expands into one, with `regions` placed by box."""
    payload.ignored(adj, "negative_prompt", reason="Ideogram 4's unconditional DiT reads no text")
    payload.placements()  # no references: any reference_boxes entry is a length mismatch
    steps = held(adj, "steps", payload.steps or STEPS, 1, 200)
    weight = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
    seed = payload.drawn_seed()
    width, height = payload.size(adj, None, SIZES, megapixels=1.0)
    written = caption(payload, upsampler, width=width, height=height, seed=seed, tel=tel)
    if written != payload.prompt:
        tel.log("caption", caption=written[:4000])
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt", overall_range=(0.15, 0.2)):
        features = text_encoder.encode(written)
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.2, 0.96)):
        latents = model.denoise(
            features, width=width, height=height, guidance=guidance_schedule(steps, weight), seed=seed,
            cancel=ctx.raise_if_cancelled,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.2, 0.96)),
        )
    with tel.stage("decoding image", overall_range=(0.96, 0.99)):
        image = model.decode(latents, width=width, height=height)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, written)
