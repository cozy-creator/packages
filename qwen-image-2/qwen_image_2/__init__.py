"""Qwen-Image-2.1 generation and reference editing with Runtime-owned execution."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import msgspec
import torch
from cozy_runtime import stepcache
from cozy_runtime.author import (
    OutputError,
    Adjustments,
    App,
    AspectRatio,
    Context,
    Picture,
    ImageFile,
    ImagePreparation,
    ImageRequest,
    ImageResult,
    Memoize,
    ModelDefault,
    Node,
    Outputs,
    Sizes,
    Telemetry,
    held,
    name_label,
    render,
)
from PIL import Image

from .model import QwenImage21Model
from . import quantize as quantize  # its recipes

app = App()

#: The pixel counts Qwen serves: upstream's 1 MP default to its native 2K (4 MP), the default without
#: references; upstream's rotary table centres at most 2048 tokens of 16 px on an axis.
MIN_MEGAPIXELS = 1.0
MAX_MEGAPIXELS = 4.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (32, 2048 * 16), 32, "the 1-4 MP Qwen-Image serves")
#: Steps without a checkpoint schedule (Turbo names its own 8), and the most admitted.
_STEPS = 40
_MAX_STEPS = 100
#: Qwen-Image-2.1-Turbo's distilled schedule, its `model_index.json` `sample_sigmas`
#: (hf://Qwen/Qwen-Image-2.1-Turbo@fc13ba19). The Runtime's model config holds components only,
#: so the function, not the checkpoint, names the regime.
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


class Mentioned(Picture, kw_only=True):
    """A picture, or with no file an asset that's only words, which its `description` gives the model."""

    image: Annotated[ImageFile, ImagePreparation(max_pixels=16 << 20)] | None = None


class GenerateInput(ImageRequest):
    pictures: Annotated[list[Mentioned], msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )
    #: Bounds for a box, not refusals: past them `_generate` holds it and says so.
    aspect_ratio: Annotated[AspectRatio | None, msgspec.Meta(extra={"min": 0.0625, "max": 16})] = None
    megapixels: Annotated[float | None, msgspec.Meta(extra={"min": MIN_MEGAPIXELS, "max": MAX_MEGAPIXELS})] = None
    steps: Annotated[ModelDefault[Annotated[int, msgspec.Meta(ge=1)] | None], msgspec.Meta(extra={"max": _MAX_STEPS})] = None
    #: Speed against fidelity: `fast` reuses the transformer's cached residual on steps that
    #: change it little (`CACHE`); `quality` and `balanced` run every step.
    preset: Literal["quality", "balanced", "fast"] = "balanced"


#: The text-to-image denoise's step cache per preset (`stepcache`): FirstBlockCache 0.08 for
#: `fast`, NVIDIA sol-engine's aggressive threshold; none otherwise.
CACHE: dict[str, stepcache.Policy | None] = {
    "quality": None,
    "balanced": None,
    "fast": stepcache.FirstBlockCache(0.08),
}


def label(n: int, picture: Picture) -> str:
    """Qwen-Image-2.1's word for the Nth picture: its template writes `<imageN>` before each
    image (Diffusers 0.41 `QwenImage21Pipeline`), and Qwen's rewriter tags only two or more."""
    return f"<image{n}>"


#: The decoder's alpha never quite reaches 0 or 255: a transparent background sits at 1–4 and an
#: opaque subject at 251–255 (runs 12867, 12878, 12891). Those bands snap to exact.
_ALPHA_NOISE = 4


def decoded_image(decoded: Any) -> Image.Image:
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
    return Image.fromarray(pixels.contiguous().cpu().numpy())


def _said(payload: GenerateInput) -> str:
    """Its text, and what each picture it mentions is: a mention reads as `<image3>` alone, and with no words for it a third picture was
    dropped in 3 of 3 seeds (runs 27130, 27161, 27174), kept with them (27190, 27200)."""
    words = "\n".join(f"<{picture.name}>: {picture.description.strip()}" for picture in payload.pictures if picture.description.strip())
    return "\n\n".join(filter(None, (payload.prompt.strip(), words)))


def _generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
    sigmas: tuple[float, ...] | None,
) -> ImageResult:
    """Both functions' body: `sigmas` is the distilled schedule (`generate_turbo`), else the base
    model's uniform spacing."""
    payload.ignored(adj, "guidance", "negative_prompt", "prompt_upsampling",
                    reason="Qwen-Image-2.1 samples without CFG and has no prompt upsampler")
    said = _said(payload)
    payload = msgspec.structs.replace(payload, pictures=[picture for picture in payload.pictures if picture.image is not None])
    payload.unplaced(adj, "Qwen-Image-2.1 has no layout control")
    prompt = render(said, payload.pictures, label if len(payload.pictures) > 1 else name_label, adj=adj, listed="pictures")
    if sigmas:
        steps = len(sigmas)
        if payload.steps not in (None, steps):
            adj.clamp("steps", payload.steps, steps, "Turbo's own distilled schedule")
    else:
        steps = held(adj, "steps", payload.steps or _STEPS, 1, _MAX_STEPS)
    tel.log("sampling", regime="turbo" if sigmas else "base", steps=steps,
            schedule="distilled sigmas" if sigmas else "uniform, dynamic shift")
    seed = payload.drawn_seed()
    model.hold_vae(tel)
    ctx.raise_if_cancelled()
    # Phase order for residency: references encode on the VAE first, then the text encoder, the
    # transformer denoise (the `denoise_loop`), and the VAE decode.
    if payload.pictures:
        ctx.plan("vae", "text_encoder", "transformer", "vae")
    else:
        ctx.plan("text_encoder", "transformer", "vae")
    reference_images, reference_latents, image_pad_mask = None, None, None
    if payload.pictures:
        with tel.stage("encoding reference images", overall_range=(0.0, 0.08)):
            # Upstream feeds the VAE RGBA; its vision encoder sees alpha composited over white.
            reference_images = [asset.read(mode="RGBA") for asset in payload.images()]
            conditioning_images, reference_latents = model.encode_reference_images(reference_images)
        ctx.raise_if_cancelled()
    # What is not given is the first reference's (upstream takes the LAST reference's; every Cozy
    # image package takes the first, the image being edited).
    first = reference_images[0].size if reference_images else None
    width, height = payload.size(adj, first, SIZES, megapixels=SIZES.megapixels[1])
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
            cache=CACHE[payload.preset],
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


@app.entrypoint(
    defaults={"model": "qwen/qwen-image-2.1@0.1.0"},
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
    node=Node(label="Qwen Image 2", hide=["guidance", "negative_prompt", "prompt_upsampling"]),
)
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2.1, the 40-step base model."""
    return _generate(ctx, payload, model, out, tel, adj, None)


@app.entrypoint(
    defaults={"model": "qwen/qwen-image-2.1-turbo@0.1.1"},
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
    node=Node(label="Qwen Image 2 Turbo", hide=["steps", "guidance", "negative_prompt", "prompt_upsampling"]),
)
def generate_turbo(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2.1-Turbo: the same inputs, on its 8-step distilled schedule."""
    return _generate(ctx, payload, model, out, tel, adj, TURBO_SIGMAS)
