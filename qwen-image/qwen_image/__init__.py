"""Qwen-Image: Qwen-Image-2512 text-to-image and Qwen-Image-Edit-2511 editing, full and Lightning,
Runtime-executed."""

from __future__ import annotations

from functools import partial
from typing import Annotated

import msgspec
from cozy_runtime.author import Adjustments, App, Context, Memoize, Outputs, Seed, Telemetry
from cozy_runtime.fields import AspectRatio, Image, Prompt, Ratios
from cozy_runtime.image import (
    Guidance,
    ImageRequest,
    ImageResult,
    MEGAPIXEL,
    NegativePrompt,
    Sizes,
    Steps,
    held,
)

from . import quantize as quantize  # its recipes
from .model import PATCH, QwenImage, QwenImageLoRA, QwenImageTransformer

app = App()

#: The sizes the model serves: 256² to 2 MiP (Qwen-Image trains from 256² up to 1328², 1.7 MiP),
#: sides of 128-4096 px on its 16 px grid.
MIN_MEGAPIXELS = 0.0625
MAX_MEGAPIXELS = 2.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (128, 4096), PATCH, "the model serves 0.06-2 MiP")
#: Text to image defaults to Qwen-Image-2512's 1328² at 1:1.
TEXT_MEGAPIXELS = 1328 * 1328 / MEGAPIXEL
#: `generate`'s recommended steps and negative prompt per DiT (each model card's), at true CFG 4.0.
TEXT_STEPS, EDIT_STEPS, TURBO_STEPS = 50, 40, 8
TEXT_NEGATIVE = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。"  # noqa: RUF001
    "构图混乱。文字模糊，扭曲。"  # noqa: RUF001
)
EDIT_NEGATIVE, GUIDANCE = " ", 4.0
#: The pictures Qwen-Image-Edit-2511 takes: past them the last mentioned go, with a warning.
MAX_PICTURES = 10


class TurboInput(ImageRequest):
    #: Its text, and the pictures it mentions in order of first mention: given, Qwen-Image-Edit-2511
    #: edits or composes them; else Qwen-Image-2512 generates from the text.
    prompt: Prompt[Image] = Prompt()
    steps: Steps = None


class GenerateInput(TurboInput):
    guidance: Guidance = None
    negative_prompt: NegativePrompt = None


def _said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[Image]]:
    """Its text with each mention as Qwen-Image-Edit's `Picture N`, the label its template writes before
    each image (Diffusers 0.41 `QwenImageEditPlusPipeline`), and the pictures."""
    keys = list(prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"Qwen-Image-Edit takes {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
        keys = keys[:MAX_PICTURES]
    said = prompt.text
    for n, key in enumerate(keys, 1):
        said = said.replace(f"<{key}>", f"Picture {n}")
    return said, [prompt.references[key] for key in keys]


def _render(
    ctx: Context,
    model: QwenImage,
    dit: QwenImage | QwenImageTransformer,
    lora: QwenImageLoRA | None,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
    payload: TurboInput,
    *,
    steps: int,
    guidance: float,
    negative: str,
) -> ImageResult:
    seed = payload.drawn_seed()
    model.hold_vae(tel)
    prompt, pictures = _said(payload.prompt, adj)
    # Upright, with any transparency over white, as upstream's RGB references.
    images = [picture.file.read() for picture in pictures]
    first = images[0].size if images else None
    width, height = payload.size(adj, first, SIZES, megapixels=TEXT_MEGAPIXELS)
    ctx.raise_if_cancelled()
    references = None
    if images:
        with tel.stage("encoding reference images", overall_range=(0.0, 0.05)):
            references = model.encode_references(images)
    with tel.stage("encoding prompt", overall_range=(0.05, 0.1)):
        embeds, mask = model.encode(prompt, images)
        unconditional = model.encode(negative, images) if guidance > 1 else None
    ctx.raise_if_cancelled()
    if images:
        # The edit DiT and its LoRA build on their first read: here, so step 1 holds only a step.
        with tel.stage("loading the edit model"):
            _ = dit.graph, lora and lora.graph
    with tel.stage("generating image", overall_range=(0.1, 0.95)):
        denoise = dit.denoise if lora is None else partial(lora.denoise, dit)
        latents = denoise(
            embeds, mask, unconditional, references, width=width, height=height, steps=steps,
            guidance=guidance, seed=seed,
        )
    ctx.raise_if_cancelled()
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(latents, width=width, height=height)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


def _full(ctx: Context, payload: GenerateInput, edit_model: QwenImageTransformer, model: QwenImage, out: Outputs,
          tel: Telemetry, adj: Adjustments) -> ImageResult:
    edit = bool(payload.prompt.references)
    steps = held(adj, "steps", payload.steps or (EDIT_STEPS if edit else TEXT_STEPS), 2, 100,
                 "at least 2 (the schedule stretches its last step to the terminal sigma), at most 100")
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
    negative = payload.negative_prompt
    if negative is None:
        negative = EDIT_NEGATIVE if edit else TEXT_NEGATIVE
    return _render(ctx, model, edit_model if edit else model, None, out, tel, adj, payload,
                   steps=steps, guidance=guidance, negative=negative)


def _turbo(ctx: Context, payload: TurboInput, edit_model: QwenImageTransformer, edit_turbo_lora: QwenImageLoRA,
           model: QwenImage, turbo_lora: QwenImageLoRA, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    steps = held(adj, "steps", payload.steps or TURBO_STEPS, 1, 100)
    edit = bool(payload.prompt.references)
    dit, lora = (edit_model, edit_turbo_lora) if edit else (model, turbo_lora)
    return _render(ctx, model, dit, lora, out, tel, adj, payload, steps=steps, guidance=1.0, negative="")


_BASE = "qwen/qwen-image-2512@0.1.1"
_EDIT = "qwen/qwen-image-edit-2511@0.1.0"
_TURBO_DEFAULTS = {"model": _BASE, "edit_model": _EDIT,
                   "turbo_lora": "lightx2v/qwen-image-2512-lightning-lora@0.1.0",
                   "edit_turbo_lora": "lightx2v/qwen-image-edit-2511-lightning-lora@0.1.0"}
_COMPONENTS = {"edit_model": ("transformer",)}


# The edit DiT is all `edit_model` binds: the text encoder, VAE and tokenizer are `model`'s. It, and
# its LoRA, are fetched and built by the first request with references.
@app.entrypoint(defaults={"model": _BASE, "edit_model": _EDIT}, components=_COMPONENTS, on_demand=("edit_model",),
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, edit_model: QwenImageTransformer, model: QwenImage, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Qwen-Image-2512 from text (50 steps), or Qwen-Image-Edit-2511 with pictures (40 steps).

    Both with true CFG 4.0."""
    return _full(ctx, payload, edit_model, model, out, tel, adj)


@app.entrypoint(defaults=_TURBO_DEFAULTS, components=_COMPONENTS, on_demand=("edit_model", "edit_turbo_lora"),
                memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: TurboInput, edit_model: QwenImageTransformer, edit_turbo_lora: QwenImageLoRA,
                   model: QwenImage, turbo_lora: QwenImageLoRA, out: Outputs, tel: Telemetry,
                   adj: Adjustments) -> ImageResult:
    """Each DiT with lightx2v's 8-step Lightning LoRA for it: 8 steps, no CFG."""
    return _turbo(ctx, payload, edit_model, edit_turbo_lora, model, turbo_lora, out, tel, adj)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it edits or composes (`_said`).
    instructions: Prompt[Image] = Prompt()
    #: Each DiT with its 8-step Lightning LoRA (`_turbo`), else true CFG at 50 or 40 steps (`_full`).
    turbo: bool = True
    #: `ImageRequest.target`, held to the model's sizes (`SIZES.fit`).
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `ImageRequest.drawn_seed`.
    seed: Seed | None = None


# `model` holds the text encoder and VAE every render reads; each other DiT and LoRA is built, and its
# weights fetched, only when a render asks for it.
@app.tool(label="Image", named="image", defaults=_TURBO_DEFAULTS, components=_COMPONENTS,
          on_demand=("edit_model", "edit_turbo_lora", "turbo_lora"), memoize=Memoize(requires=["seed"]))  # fmt: skip
def image(ctx: Context, payload: ImageBox, edit_model: QwenImageTransformer, edit_turbo_lora: QwenImageLoRA,
          model: QwenImage, turbo_lora: QwenImageLoRA, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Qwen Image on a canvas: Lightning, or the full models."""
    request = GenerateInput(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio,
                            seed=payload.seed)  # fmt: skip
    if payload.turbo:
        return _turbo(ctx, request, edit_model, edit_turbo_lora, model, turbo_lora, out, tel, adj)
    return _full(ctx, request, edit_model, model, out, tel, adj)
