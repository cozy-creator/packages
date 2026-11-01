"""Bria Fibo 2.0 text-to-image and image editing, Dev and Edit Turbo, with Bria's prompt-to-JSON VLMs."""

from __future__ import annotations

import json
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    Picture,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    Sizes,
    Telemetry,
    UnsupportedInput,
    held,
    name_label,
)
from PIL import Image

from .model import Fibo2
from . import quantize as quantize  # its recipes
from .upsampler import Captioner

app = App()

#: Fibo 2.0 was trained at about 1 MP (model card: "multiples of 16 at about one megapixel").
#: 0.25-1 MP on the 16 px grid; Diffusers' smallest side (4 latent patches) to the RoPE table's 512
#: positions an axis.
SIZES = Sizes((0.25, 1.0), (4 * 16, 512 * 16), 16, "Fibo 2.0's 0.25-1 MP")
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

class GenerateInput(ImageRequest):
    pictures: Annotated[list[Picture], msgspec.Meta(max_length=4)] = msgspec.field(
        default_factory=list
    )


def label(n: int, picture: Picture) -> str:
    """Fibo's word for the Nth of several pictures: the edit wraps them as `image_1`, `image_2`,
    ... (Diffusers `BriaFibo2Pipeline`), and Bria's instructions name them `<image_N>`. A lone
    picture is the edit's `image`, so it keeps its name."""
    return f"<image_{n}>"


def _json_object(prompt: str) -> dict[str, object] | None:
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _caption(
    prompt: str, images: list[Image.Image], upsample: bool, upsampler: Captioner, edit_upsampler: Captioner,
    adj: Adjustments,
) -> str:
    """The prompt Fibo 2.0 reads. A JSON object is already structured; otherwise Bria's VLMs write
    one: `FIBO-vlm` from a prompt, `FIBO-edit-vlm` from a request and its one image. Several images
    take the instruction as typed, a form Fibo 2.0 was also trained on."""
    structured = _json_object(prompt)
    if structured is not None:
        if images and "edit_instruction" not in structured:
            raise UnsupportedInput('an edit\'s JSON prompt needs an "edit_instruction"', code="prompt")
        return prompt
    if not upsample:
        return prompt
    if not images:
        return upsampler.caption(prompt, None)
    if len(images) > 1:
        adj.warn("prompt_upsampling", "Bria's edit VLM reads one image: with several, the instruction goes as typed")
        return prompt
    caption = json.loads(edit_upsampler.caption(prompt, images[0]))
    # FIBO-edit-vlm restates the request as `reworded_prompt`, which Bria's cleaning drops; Fibo 2.0
    # reads the request as `edit_instruction`.
    caption["edit_instruction"] = caption.pop("reworded_prompt", None) or caption.get("edit_instruction") or prompt
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def _generate(
    ctx: Context,
    function: str,
    payload: GenerateInput,
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
    payload.unplaced(adj, "Fibo 2.0 has no layout control")
    prompt = payload.rendered(adj, label if len(payload.pictures) > 1 else name_label)
    images = [asset.read() for asset in payload.images()]
    if function == "generate":
        guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
        negative = payload.negative_prompt or (None if images else NEGATIVE_T2I)
    else:
        guidance, negative = 1.0, None
        payload.ignored(adj, "guidance", "negative_prompt", reason="Edit Turbo is distilled to sample without guidance")
    seed = payload.drawn_seed()
    width, height = payload.size(adj, images[0].size if images else None, SIZES, megapixels=1.0)
    upsample = True if payload.prompt_upsampling is None else payload.prompt_upsampling
    with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
        caption = _caption(prompt, images, upsample, upsampler, edit_upsampler, adj)
    if caption != prompt:
        tel.log("upsampled prompt", caption=caption[:2000])
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt", overall_range=(0.15, 0.2)):
        encoded = model.encode(caption, images, width=width, height=height, guidance=guidance, negative=negative)
    latents = []
    if images:
        with tel.stage("encoding reference images", overall_range=(0.2, 0.25)):
            latents = model.encode_images(images, width=width, height=height)
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.25, 0.95)):
        tokens = model.denoise(
            caption, images, encoded, latents, width=width, height=height, steps=steps, guidance=guidance,
            negative=negative, seed=seed, cancel=ctx.raise_if_cancelled,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.25, 0.95)),
        )
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(tokens)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, image.width, image.height, seed, caption)


_MODEL = ("transformer", "text_encoder", "vae")


# A VLM is built and fetched on first use: a request that skips upsampling never pulls one. Model
# parameters are in name order: Runtime refuses a binding set whose paths are not sorted.
@app.entrypoint(defaults={"model": "briaai/fibo-2-dev@0.1.0", "upsampler": "briaai/fibo-vlm@0.1.0", "edit_upsampler": "briaai/fibo-edit-vlm@0.1.0"},
                components={"model": _MODEL, "upsampler": ("model",), "edit_upsampler": ("model",)},
                on_demand=("upsampler", "edit_upsampler"), memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, edit_upsampler: Captioner, model: Fibo2, upsampler: Captioner,
             out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Fibo 2.0 Dev, 50 steps with CFG 5: text to image, or an edit of up to four ordered pictures."""
    return _generate(ctx, "generate", payload, model, upsampler, edit_upsampler, out, tel, adj)


@app.entrypoint(defaults={"model": "briaai/fibo-2-edit-turbo@0.1.0", "upsampler": "briaai/fibo-vlm@0.1.0",
                          "edit_upsampler": "briaai/fibo-edit-vlm@0.1.0"},
                components={"model": _MODEL, "upsampler": ("model",), "edit_upsampler": ("model",)},
                on_demand=("upsampler", "edit_upsampler"), memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: GenerateInput, edit_upsampler: Captioner, model: Fibo2,
                   upsampler: Captioner, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Fibo 2.0 Edit Turbo, 4 steps without guidance: text to image, or an edit of up to four ordered pictures."""
    return _generate(ctx, "generate_turbo", payload, model, upsampler, edit_upsampler, out, tel, adj)
