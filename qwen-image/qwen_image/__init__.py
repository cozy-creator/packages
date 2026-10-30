"""Qwen-Image: Qwen-Image-2512 text-to-image and Qwen-Image-Edit-2511 editing, full and Lightning,
Runtime-executed."""

from __future__ import annotations

from functools import partial
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    MEGAPIXEL,
    Adjustments,
    App,
    Context,
    PictureAsset,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    Sizes,
    Telemetry,
    held,
)

from . import quantize as quantize  # its recipes
from .model import PATCH, QwenImage, QwenImageLoRA, QwenImageTransformer

app = App()

#: The sizes the model serves: 256² to 2 MiP (Qwen-Image trains from 256² up to 1328², 1.7 MiP),
#: sides of 128-4096 px on its 16 px grid.
SIZES = Sizes((256 * 256 / MEGAPIXEL, 2.0), (128, 4096), PATCH, "the model serves 0.06-2 MiP")
#: Text to image defaults to Qwen-Image-2512's 1328² at 1:1.
TEXT_MEGAPIXELS = 1328 * 1328 / MEGAPIXEL
#: `generate`'s recommended steps and negative prompt per DiT (each model card's), at true CFG 4.0.
TEXT_STEPS, EDIT_STEPS, TURBO_STEPS = 50, 40, 8
TEXT_NEGATIVE = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。"  # noqa: RUF001
    "构图混乱。文字模糊，扭曲。"  # noqa: RUF001
)
EDIT_NEGATIVE, GUIDANCE = " ", 4.0


class GenerateInput(ImageRequest):
    #: Given, Qwen-Image-Edit-2511 edits or composes them, in order; else Qwen-Image-2512 generates
    #: from the prompt.
    assets: Annotated[list[PictureAsset], msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )


def label(n: int, asset: PictureAsset) -> str:
    """Qwen-Image-Edit's word for the Nth asset: its template writes `Picture N:` before each
    image (Diffusers 0.41 `QwenImageEditPlusPipeline`)."""
    return f"Picture {n}"


def _render(
    ctx: Context,
    model: QwenImage,
    dit: QwenImage | QwenImageTransformer,
    lora: QwenImageLoRA | None,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
    payload: GenerateInput,
    *,
    steps: int,
    guidance: float,
    negative: str,
) -> ImageResult:
    seed = payload.drawn_seed()
    model.hold_vae(tel)
    payload.ignored(adj, "prompt_upsampling", reason="Qwen-Image has no prompt upsampler")
    payload.unplaced(adj, "Qwen-Image has no layout control")
    prompt = payload.rendered(adj, label)
    # Upright, with any transparency over white, as upstream's RGB references.
    images = [asset.read() for asset in payload.images()]
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
    with tel.stage("generating image", overall_range=(0.1, 0.95)):
        denoise = dit.denoise if lora is None else partial(lora.denoise, dit)
        latents = denoise(
            embeds, mask, unconditional, references, width=width, height=height, steps=steps,
            guidance=guidance, seed=seed, cancel=ctx.raise_if_cancelled,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.1, 0.95)),
        )
    ctx.raise_if_cancelled()
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(latents, width=width, height=height)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


_MEMO = ("qwen_image", "qwen_image.model")


# The edit DiT is all `edit_model` binds: the text encoder, VAE and tokenizer are `model`'s. It, and
# its LoRA, are fetched and built by the first request with references.
@app.entrypoint(
    defaults={"model": "qwen/qwen-image-2512@0.1.1", "edit_model": "qwen/qwen-image-edit-2511@0.1.0"},
    components={"edit_model": ("transformer",)},
    on_demand=("edit_model",),
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
)
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage,
    edit_model: QwenImageTransformer,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2512 from text (50 steps), Qwen-Image-Edit-2511 with picture assets (40 steps), both
    with true CFG 4.0."""
    edit = bool(payload.assets)
    steps = held(adj, "steps", payload.steps or (EDIT_STEPS if edit else TEXT_STEPS), 2, 100,
                 "at least 2 (the schedule stretches its last step to the terminal sigma), at most 100")
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
    negative = payload.negative_prompt
    if negative is None:
        negative = EDIT_NEGATIVE if edit else TEXT_NEGATIVE
    return _render(ctx, model, edit_model if edit else model, None, out, tel, adj, payload,
                   steps=steps, guidance=guidance, negative=negative)


@app.entrypoint(
    defaults={
        "model": "qwen/qwen-image-2512@0.1.1",
        "edit_model": "qwen/qwen-image-edit-2511@0.1.0",
        "turbo_lora": "lightx2v/qwen-image-2512-lightning-lora@0.1.0",
        "edit_turbo_lora": "lightx2v/qwen-image-edit-2511-lightning-lora@0.1.0",
    },
    components={"edit_model": ("transformer",)},
    on_demand=("edit_model", "edit_turbo_lora"),
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
)
def generate_turbo(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage,
    edit_model: QwenImageTransformer,
    turbo_lora: QwenImageLoRA,
    edit_turbo_lora: QwenImageLoRA,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Each DiT with lightx2v's 8-step Lightning LoRA for it: 8 steps, no CFG."""
    payload.ignored(adj, "guidance", "negative_prompt", reason="the Lightning LoRA samples without CFG")
    steps = held(adj, "steps", payload.steps or TURBO_STEPS, 1, 100)
    edit = bool(payload.assets)
    dit, lora = (edit_model, edit_turbo_lora) if edit else (model, turbo_lora)
    return _render(ctx, model, dit, lora, out, tel, adj, payload, steps=steps, guidance=1.0, negative="")
