"""Qwen-Image-2.1 generation and reference editing with Runtime-owned execution."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import msgspec
import torch
from cozy_runtime.author import (
    MENTION,
    Adjustments,
    App,
    AspectRatio,
    Context,
    Image,
    ImageRequest,
    ImageResult,
    Memoize,
    OutputError,
    Outputs,
    Prompt,
    Ratios,
    Seed,
    Sizes,
    Telemetry,
)
from PIL import Image as PILImage

from . import quantize as quantize  # its recipes
from .model import QwenImage21Model

app = App()

#: The pixel counts Qwen serves: upstream's 1 MP default to its native 2K (4 MP);
#: upstream's rotary table centres at most 2048 tokens of 16 px on an axis.
MIN_MEGAPIXELS = 1.0
MAX_MEGAPIXELS = 4.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (32, 2048 * 16), 32, "the 1-4 MP Qwen-Image serves")
#: The size without references (Paul, 11-06: "2 megapixels is a good default").
DEFAULT_MEGAPIXELS = 2
#: The full model's steps; Turbo's are its schedule's.
_STEPS = 40
#: Qwen-Image-2.1-Turbo's distilled schedule, its `model_index.json` `sample_sigmas`
#: (hf://Qwen/Qwen-Image-2.1-Turbo@fc13ba19). The Runtime's model config holds components only,
#: so the function, not the checkpoint, names the regime.
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


#: The pictures Qwen-Image-2.1 takes: past them the last mentioned go, with a warning.
MAX_PICTURES = 10


class GenerateInput(ImageRequest):
    #: Its text, and the pictures it mentions, in the order it first mentions them.
    prompt: Prompt[Image] = Prompt()


def label(n: int, key: str, count: int) -> str:
    """Qwen-Image-2.1's word for the Nth of `count` pictures: its template writes `<imageN>` before each image
    (Diffusers 0.41 `QwenImage21Pipeline`), and Qwen's rewriter tags only two or more; a lone picture is its name,
    `kyogre.2` as `kyogre 2`."""
    return f"<image{n}>" if count > 1 else key.replace(".", " ")


#: The decoder's alpha never quite reaches 0 or 255: a transparent background sits at 1–4 and an
#: opaque subject at 251–255 (runs 12867, 12878, 12891). Those bands snap to exact.
_ALPHA_NOISE = 4


def decoded_image(decoded: Any) -> PILImage.Image:
    """The decoder's pixels: RGBA when its alpha has any transparency, else RGB. Quantized on the
    decoder's device, NumPy's arithmetic in Torch: a 4 MP canvas sends 16 MB of bytes to the host,
    not 67 MB of floats for six host passes."""
    # The model returns B,C,H,W in [-1,1], just as upstream VaeImageProcessor consumes.
    pixels = decoded[0].detach().float()
    if not bool(torch.isfinite(pixels).all()):
        raise OutputError(
            "Qwen Image decoder produced non-finite pixels, so there is no picture: try another seed, "
            "and if it repeats another lane with `--lane bf16`",
            code="output_integrity",
        )
    pixels = ((pixels / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
    if pixels.shape[-1] == 4:
        alpha = pixels[..., 3]
        alpha[alpha <= _ALPHA_NOISE] = 0
        alpha[alpha >= 255 - _ALPHA_NOISE] = 255
        if int(alpha.min()) == 255:
            pixels = pixels[..., :3]
    return PILImage.fromarray(pixels.contiguous().cpu().numpy())


def _said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[Image]]:
    """Its text, and what each picture it mentions is, each mention in the model's words (`label`), and the pictures:
    a mention reads as `<image3>` alone, and with no words for it a third picture was dropped in 3 of 3 seeds (runs
    27130, 27161, 27174), kept with them (27190, 27200)."""
    keys = list(prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"Qwen Image 2 takes {MAX_PICTURES} pictures, so it goes without {', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
        keys = keys[:MAX_PICTURES]
    words = "\n".join(f"<{key}>: {prompt.references[key].description.strip()}" for key in keys if prompt.references[key].description.strip())
    said = "\n\n".join(filter(None, (prompt.text.strip(), words)))
    # One pass: a label never reads as another key (a box named `image1`).
    labels = {key: label(n, key, len(keys)) for n, key in enumerate(keys, 1)}
    return MENTION.sub(lambda found: labels.get(found[1], found[0]), said), [prompt.references[key] for key in keys]


def _generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
    steps: int,
    sigmas: tuple[float, ...] | None,
) -> ImageResult:
    """Every function's body: `sigmas` is the distilled schedule (Turbo), else the base model's uniform spacing."""
    prompt, pictures = _said(payload.prompt, adj)
    tel.log("sampling", regime="turbo" if sigmas else "base", steps=steps,
            schedule="distilled sigmas" if sigmas else "uniform, dynamic shift")
    seed = payload.drawn_seed()
    model.hold_vae(tel)
    ctx.raise_if_cancelled()
    # Phase order for residency: references encode on the VAE first, then the text encoder, the
    # transformer denoise (the `denoise_loop`), and the VAE decode.
    if pictures:
        ctx.plan("vae", "text_encoder", "transformer", "vae")
    else:
        ctx.plan("text_encoder", "transformer", "vae")
    reference_images, reference_latents, image_pad_mask = None, None, None
    if pictures:
        with tel.stage("encoding reference images", overall_range=(0.0, 0.08)):
            # Upstream feeds the VAE RGBA; its vision encoder sees alpha composited over white.
            reference_images = [picture.file.read(mode="RGBA") for picture in pictures]
            conditioning_images, reference_latents = model.encode_reference_images(reference_images)
        ctx.raise_if_cancelled()
    # What is not given is the first reference's (upstream takes the LAST reference's; every Cozy
    # image package takes the first, the image being edited).
    first = reference_images[0].size if reference_images else None
    width, height = payload.size(adj, first, SIZES, megapixels=DEFAULT_MEGAPIXELS)
    with tel.stage("encoding prompt", overall_range=(0.08 if reference_images else 0.0, 0.1)):
        if reference_images is None:
            embeds, mask = model.encode(prompt)
        else:
            embeds, mask, image_pad_mask = model.encode_image_prompt(prompt, conditioning_images)
    with tel.stage("generating image", overall_range=(0.1, 0.9)):
        latents = model.denoise(
            embeds,
            mask,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.1, 0.9)),
            cancel=ctx.raise_if_cancelled,
            sigmas=sigmas,
            reference_images=reference_images,
            reference_latents=reference_latents,
            image_pad_mask=image_pad_mask,
        )
    ctx.raise_if_cancelled()
    with tel.stage("decoding image", overall_range=(0.9, 0.98)):
        decoded = model.decode(latents, width=width, height=height)
        image = decoded_image(decoded)
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        # WebP cannot represent a side of 16384 pixels; retain every supported canvas.
        output_format = "webp" if max(width, height) < 16384 else "png"
        asset = out.save_image(image, format=output_format)
    return ImageResult(asset, width, height, seed, prompt)


_MEMO = ("qwen_image_2", "qwen_image_2.model")
_FULL = "qwen/qwen-image-2.1@0.1.0"
_TURBO = "qwen/qwen-image-2.1-turbo@0.1.1"


@app.entrypoint(defaults={"model": _FULL}, memoize=Memoize(requires=["seed"]), memo_dependencies=_MEMO)
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2.1, the 40-step base model."""
    return _generate(ctx, payload, model, out, tel, adj, _STEPS, None)


@app.entrypoint(defaults={"model": _TURBO}, memoize=Memoize(requires=["seed"]), memo_dependencies=_MEMO)
def generate_turbo(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2.1-Turbo: the same inputs, on its 8-step distilled schedule."""
    return _generate(ctx, payload, model, out, tel, adj, len(TURBO_SIGMAS), TURBO_SIGMAS)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it mentions (`_said`).
    instructions: Prompt[Image] = Prompt()
    #: Qwen-Image-2.1-Turbo's 8 steps, else the 40-step base model.
    turbo: bool = True
    #: `ImageRequest.target`; Qwen's rotary range holds it to 1:16 through 16:1 (`SIZES.fit`).
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `SIZES.pixels`: Qwen Image 2.1 renders at 1, 2 or 4 MP, by default 2 (Paul, 11-06).
    megapixels: Literal[1, 2, 4] = DEFAULT_MEGAPIXELS
    seed: Seed | None = None


# Each checkpoint is built, and its weights fetched, only when a box asks for it.
@app.tool(label="Qwen Image 2", named="image", defaults={"model": _TURBO, "full": _FULL}, on_demand=("full", "model"),
          memoize=Memoize(requires=["seed"]), memo_dependencies=_MEMO)  # fmt: skip
def image(
    ctx: Context,
    payload: ImageBox,
    full: QwenImage21Model,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen Image 2 on a canvas: Turbo, or the full model, which loads only when asked for."""
    request = GenerateInput(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio,
                            megapixels=payload.megapixels, seed=payload.seed)  # fmt: skip
    if payload.turbo:
        return _generate(ctx, request, model, out, tel, adj, len(TURBO_SIGMAS), TURBO_SIGMAS)
    return _generate(ctx, request, full, out, tel, adj, _STEPS, None)
