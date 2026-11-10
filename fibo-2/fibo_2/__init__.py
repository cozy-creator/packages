"""Bria Fibo 2.0 text-to-image and image editing, Dev and Edit Turbo, with Bria's prompt-to-JSON VLMs."""

from __future__ import annotations

import json
import re
from typing import Annotated

import msgspec
from cozy_runtime.author import Adjustments, App, Context, Outputs, Seed, Telemetry, UnsupportedInput
from cozy_runtime.fields import Advanced, AspectRatio, Image, Prompt, Ratios
from cozy_runtime.image import Guidance, ImageRequest, ImageResult, NegativePrompt, Sizes, Steps, held
from PIL import Image as PILImage

from .model import Fibo2
from . import quantize as quantize  # its recipes
from .upsampler import Captioner

app = App()

#: Fibo 2.0 was trained at about 1 MP (model card: "multiples of 16 at about one megapixel").
#: 0.25-1 MP on the 16 px grid; Diffusers' smallest side (4 latent patches) to the RoPE table's 512
#: positions an axis.
MIN_MEGAPIXELS = 0.25
MAX_MEGAPIXELS = 1.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (4 * 16, 512 * 16), 16, "Fibo 2.0's 0.25-1 MP")
#: The pictures Fibo 2.0 edits: past them the last mentioned go, with a warning.
MAX_PICTURES = 4
#: Each function's steps when none is given, and the most it admits; Dev's CFG scale (the card's 5.0).
_STEPS = {"generate": (50, 200), "generate_turbo": (4, 50)}
GUIDANCE = 5.0
#: The model card's text-to-image negative: away from soft, featureless, artifact-laden images.
#: Edits take Diffusers' default, the empty caption the model was trained with.
NEGATIVE_T2I = (
    '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
    'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified '
    'shapes, no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],'
    '"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and '
    'featureless, no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours",'
    '"neon colour cast","overexposed highlights","lens flare","slight blur","motion blur","haze"]},'
    '"photographic_characteristics":{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame",'
    '"focus":"soft, slightly out of focus, lacking fine texture"},"style_medium":"","context":"",'
    '"artistic_style":""}'
)

class TurboInput(ImageRequest):
    #: Its text, and the pictures it edits, in order of first mention.
    prompt: Prompt[Image] = Prompt()
    steps: Steps = None
    prompt_upsampling: bool | None = None


class GenerateInput(TurboInput):
    guidance: Guidance = None
    negative_prompt: NegativePrompt = None


def _said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[Image]]:
    """Its text in Fibo's words, and the pictures. Several read as `<image_N>`: the edit wraps them as
    `image_1`, `image_2`, ... (Diffusers `BriaFibo2Pipeline`), and Bria's instructions name them so. A lone
    picture is the edit's `image`, so it reads as its name, `kyogre.2` as `kyogre 2`."""
    keys = list(prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"Fibo 2.0 edits {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
        keys = keys[:MAX_PICTURES]
    words = {f"<{key}>": f"<image_{n}>" if len(keys) > 1 else key.replace(".", " ") for n, key in enumerate(keys, 1)}
    # In one pass, so an `<image_1>` put in is never read again as a box named `image_1`.
    said = re.sub("|".join(map(re.escape, words)), lambda found: words[found[0]], prompt.text) if words else prompt.text
    return said, [prompt.references[key] for key in keys]


def _json_object(prompt: str) -> dict[str, object] | None:
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _caption(
    prompt: str, images: list[PILImage.Image], upsample: bool, upsampler: Captioner, edit_upsampler: Captioner,
    adj: Adjustments,
) -> str:
    """The prompt Fibo 2.0 reads. A JSON object is already structured; otherwise Bria's VLMs write
    one: `FIBO-vlm` from a prompt, `FIBO-edit-vlm` from a request and its one image. Several images
    take the instruction as typed, a form Fibo 2.0 was also trained on."""
    structured = _json_object(prompt)
    if structured is not None:
        if images and "edit_instruction" not in structured:
            raise UnsupportedInput(
                'an edit\'s JSON prompt needs an "edit_instruction": add one saying what to change, or send plain text',
                code="prompt",
            )
        return prompt
    if not upsample:
        return prompt
    if len(images) > 1:
        adj.warn("prompt_upsampling", "Bria's edit VLM reads one image: with several, the instruction goes as typed")
        return prompt
    try:
        if not images:
            return upsampler.caption(prompt, None)
        caption = json.loads(edit_upsampler.caption(prompt, images[0]))
    except UnsupportedInput as failed:
        adj.warn("prompt_upsampling", f"{failed.message}; the prompt goes as typed")
        return prompt
    # FIBO-edit-vlm restates the request as `reworded_prompt`, which Bria's cleaning drops; Fibo 2.0
    # reads the request as `edit_instruction`.
    caption["edit_instruction"] = caption.pop("reworded_prompt", None) or caption.get("edit_instruction") or prompt
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def _generate(
    ctx: Context,
    function: str,
    payload: TurboInput,
    model: Fibo2,
    upsampler: Captioner,
    edit_upsampler: Captioner,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    model.hold_vae(tel)
    default, most = _STEPS[function]
    steps = held(adj, "steps", payload.steps or default, 1, most)
    prompt, pictures = _said(payload.prompt, adj)
    images = [picture.file.read() for picture in pictures]
    if isinstance(payload, GenerateInput):
        guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
        negative = payload.negative_prompt or (None if images else NEGATIVE_T2I)
    else:
        guidance, negative = 1.0, None
    seed = payload.drawn_seed()
    width, height = payload.size(adj, images[0].size if images else None, SIZES, megapixels=1.0)
    upsample = True if payload.prompt_upsampling is None else payload.prompt_upsampling
    with tel.stage("upsampling prompt"):
        caption = _caption(prompt, images, upsample, upsampler, edit_upsampler, adj)
    if caption != prompt:
        tel.log("upsampled prompt", caption=caption[:2000])
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt"):
        encoded = model.encode(caption, images, width=width, height=height, guidance=guidance, negative=negative)
    latents = []
    if images:
        with tel.stage("encoding reference images"):
            latents = model.encode_images(images, width=width, height=height)
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        tokens = model.denoise(
            encoded, latents, width=width, height=height, steps=steps, guidance=guidance, seed=seed,
            progress=ctx.progress,
        )
    with tel.stage("decoding image"):
        image = model.decode(tokens)
    with tel.stage("saving image"):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, image.width, image.height, seed, caption)


_MODEL = ("transformer", "text_encoder", "vae")
_DEV = "briaai/fibo-2-dev@0.1.0"
_TURBO = "briaai/fibo-2-edit-turbo@0.1.0"
_VLM = "briaai/fibo-vlm@0.1.0"
_EDIT_VLM = "briaai/fibo-edit-vlm@0.1.0"
_COMPONENTS = {"model": _MODEL, "upsampler": ("model",), "edit_upsampler": ("model",)}


# A VLM is built and fetched on first use: a request that skips upsampling never pulls one. Model
# parameters are in name order: Runtime refuses a binding set whose paths are not sorted.
@app.entrypoint(defaults={"model": _DEV, "upsampler": _VLM, "edit_upsampler": _EDIT_VLM}, components=_COMPONENTS,
                on_demand=("upsampler", "edit_upsampler"), memoize=True)
def generate(ctx: Context, payload: GenerateInput, edit_upsampler: Captioner, model: Fibo2, upsampler: Captioner,
             out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Fibo 2.0 Dev, 50 steps with CFG 5: text to image, or an edit of up to four ordered pictures."""
    return _generate(ctx, "generate", payload, model, upsampler, edit_upsampler, out, tel, adj)


@app.entrypoint(defaults={"model": _TURBO, "upsampler": _VLM, "edit_upsampler": _EDIT_VLM}, components=_COMPONENTS,
                on_demand=("upsampler", "edit_upsampler"), memoize=True)
def generate_turbo(ctx: Context, payload: TurboInput, edit_upsampler: Captioner, model: Fibo2,
                   upsampler: Captioner, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Fibo 2.0 Edit Turbo, 4 steps without guidance: text to image, or an edit of up to four ordered pictures."""
    return _generate(ctx, "generate_turbo", payload, model, upsampler, edit_upsampler, out, tel, adj)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it edits (`_said`).
    instructions: Prompt[Image] = Prompt()
    #: Edit Turbo, 4 steps without guidance, else Dev, 50 steps with CFG 5 (`_generate`).
    turbo: bool = True
    #: `ImageRequest.target`, held to the model's sizes (`SIZES.fit`).
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `ImageRequest.drawn_seed`.
    seed: Seed | None = None
    #: Bria's VLMs write the JSON caption from the prompt (`_caption`).
    prompt_upsampling: Annotated[bool, Advanced()] = True


# Each checkpoint and VLM is built, and its weights fetched, only when a render asks for it.
@app.tool(label="Image", named="image",
          defaults={"model": _TURBO, "full": _DEV, "upsampler": _VLM, "edit_upsampler": _EDIT_VLM},
          components={"model": _MODEL, "full": _MODEL, "upsampler": ("model",), "edit_upsampler": ("model",)},
          on_demand=("edit_upsampler", "full", "model", "upsampler"), memoize=True)
def image(ctx: Context, payload: ImageBox, edit_upsampler: Captioner, full: Fibo2, model: Fibo2, upsampler: Captioner,
          out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Fibo 2.0 on a canvas: Edit Turbo, or Dev."""
    kind, function, dit = (TurboInput, "generate_turbo", model) if payload.turbo else (GenerateInput, "generate", full)
    request = kind(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio,
                   seed=payload.seed, prompt_upsampling=payload.prompt_upsampling)
    return _generate(ctx, function, request, dit, upsampler, edit_upsampler, out, tel, adj)
