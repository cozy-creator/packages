"""Qwen-Image-2.1 generation and reference editing with Runtime-owned execution."""

from __future__ import annotations

from typing import Literal, Annotated, Any

import msgspec
import torch
from cozy_runtime.author import (
    Adjustments,
    App,
    AspectRatio,
    Context,
    ImageFile,
    ImagePreparation,
    ImageResult,
    Memoize,
    Node,
    OutputError,
    Outputs,
    Picture,
    PictureRequest,
    Sizes,
    Telemetry,
    name_label,
    ratio_of,
    render,
)
from PIL import Image

from .model import QwenImage21Model
from .rewriter import Creativity, Enhancer, rewrite
from . import quantize as quantize  # its recipes

app = App()

#: The pixel counts Qwen serves: upstream's 1 MP default to its native 2K (4 MP), the default without
#: references; upstream's rotary table centres at most 2048 tokens of 16 px on an axis.
MIN_MEGAPIXELS = 1.0
MAX_MEGAPIXELS = 4.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (32, 2048 * 16), 32, "the 1-4 MP Qwen-Image serves")
#: The full model's steps; Turbo's are its schedule's.
_STEPS = 40
#: Qwen-Image-2.1-Turbo's distilled schedule, its `model_index.json` `sample_sigmas`
#: (hf://Qwen/Qwen-Image-2.1-Turbo@fc13ba19). The Runtime's model config holds components only,
#: so the function, not the checkpoint, names the regime.
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


class Mentioned(Picture, kw_only=True):
    """A picture, or with no file an asset that's only words, which its `description` gives the model."""

    image: Annotated[ImageFile, ImagePreparation(max_pixels=16 << 20)] | None = None
    description: str = ""


class GenerateInput(PictureRequest):
    pictures: Annotated[list[Mentioned], msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )
    #: Bounds for a box, not refusals: past them `_generate` holds it and says so.
    aspect_ratio: Annotated[AspectRatio | None, msgspec.Meta(extra={"min": 0.0625, "max": 16})] = None
    #: Qwen Image 2.1 renders at 1, 2 or 4 MP (its native 2K); 2 by default (Paul, 11-06).
    megapixels: Literal[1, 2, 4] = 2
    #: Qwen's prompt enhancer rewrites the request first (`rewriter`): an edit from its pictures, else the words.
    prompt_upsampling: bool = True
    #: How far the rewrite goes beyond the words; `balanced` is Qwen's own.
    creativity: Creativity = "balanced"


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


def _upsampled(
    payload: GenerateInput,
    prompt: str,
    images: list[Image.Image],
    seed: int,
    enhancer: Enhancer,
    tel: Telemetry,
    adj: Adjustments,
) -> tuple[GenerateInput, str]:
    """The request as Qwen's enhancer rewrites it, sampled from the render's `seed`, and the canvas it chose
    when the request names none (upstream passes `wh_ratio` and `ratio_follow` to the renderer)."""
    with tel.stage("upsampling prompt", overall_range=(0.0, 0.3)):
        found, answer, thought, said = rewrite(enhancer, prompt, images, payload.creativity, seed,
                                               lambda done: tel.progress(done, stage="upsampling prompt"))
    if found is None:
        adj.warn("prompt_upsampling", "Qwen's prompt enhancer gave no rewrite, so the prompt goes as written")
        tel.log("upsampled prompt", parsed=False, answer=answer[:2000], thought_tokens=thought, answer_tokens=said)
        return payload, prompt
    ratio = found.ratio
    if found.follow is not None:
        ratio = "{}:{}".format(*images[found.follow - 1].size)
    tel.log("upsampled prompt", prompt=found.prompt, seed=seed, creativity=payload.creativity, thought_tokens=thought,
            answer_tokens=said,
            wh_ratio=found.ratio or "", ratio_follow=f"<image{found.follow}>" if found.follow else "")
    if payload.aspect_ratio is None and ratio is not None:
        payload = msgspec.structs.replace(payload, aspect_ratio=":".join(map(str, ratio_of(ratio))))
    return payload, found.prompt


def _generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    rewriter: Enhancer,
    expander: Enhancer,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
    steps: int,
    sigmas: tuple[float, ...] | None,
) -> ImageResult:
    """Both functions' body: `sigmas` is the distilled schedule (`generate_turbo`), else the base
    model's uniform spacing."""
    said = _said(payload)
    payload = msgspec.structs.replace(payload, pictures=[picture for picture in payload.pictures if picture.image is not None])
    prompt = render(said, payload.pictures, label if len(payload.pictures) > 1 else name_label, adj=adj, listed="pictures")
    tel.log("sampling", regime="turbo" if sigmas else "base", steps=steps,
            schedule="distilled sigmas" if sigmas else "uniform, dynamic shift")
    seed = payload.drawn_seed()
    # Upstream feeds the VAE RGBA; its vision encoder sees alpha composited over white.
    reference_images = [asset.read(mode="RGBA") for asset in payload.images()] or None
    if payload.prompt_upsampling:
        enhancer = rewriter if reference_images else expander
        payload, prompt = _upsampled(payload, prompt, reference_images or [], seed, enhancer, tel, adj)
    model.hold_vae(tel)
    ctx.raise_if_cancelled()
    begin = 0.3 if payload.prompt_upsampling else 0.0
    # Phase order for residency: references encode on the VAE first, then the text encoder, the
    # transformer denoise (the `denoise_loop`), and the VAE decode.
    if payload.pictures:
        ctx.plan("vae", "text_encoder", "transformer", "vae")
    else:
        ctx.plan("text_encoder", "transformer", "vae")
    reference_latents, image_pad_mask = None, None
    if reference_images:
        with tel.stage("encoding reference images", overall_range=(begin, begin + 0.05)):
            conditioning_images, reference_latents = model.encode_reference_images(reference_images)
        ctx.raise_if_cancelled()
    # What is not given is the first reference's (upstream takes the LAST reference's; every Cozy
    # image package takes the first, the image being edited).
    first = reference_images[0].size if reference_images else None
    width, height = payload.size(adj, first, SIZES, megapixels=SIZES.megapixels[1])
    with tel.stage("encoding prompt", overall_range=(begin + 0.05, begin + 0.1)):
        if reference_images is None:
            embeds, mask = model.encode(prompt)
        else:
            embeds, mask, image_pad_mask = model.encode_image_prompt(prompt, conditioning_images)
    with tel.stage("generating image", overall_range=(begin + 0.1, 0.9)):
        latents = model.denoise(
            embeds,
            mask,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(begin + 0.1, 0.9)),
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


_MEMO = ("qwen_image_2", "qwen_image_2.model", "qwen_image_2.rewriter")
#: Each model, then Qwen's prompt enhancers (QwenLM/Qwen-Image-2.1 `prompt_rewrite/`): an edit's, a picture from words'.
_FULL = {"model": "qwen/qwen-image-2.1@0.1.0", "rewriter": "qwen/qwen-image-2.1-pe-i2i@0.1.0",
         "expander": "qwen/qwen-image-2.1-pe-t2i@0.1.0"}
_TURBO = {"model": "qwen/qwen-image-2.1-turbo@0.1.1", "rewriter": "qwen/qwen-image-2.1-pe-i2i@0.1.0",
          "expander": "qwen/qwen-image-2.1-pe-t2i@0.1.0"}
_COMPONENTS = {"model": ("text_encoder", "transformer", "vae"), "rewriter": ("model",), "expander": ("model",)}
#: Built and fetched on first use: an edit never pulls the expander, `prompt_upsampling=False` neither enhancer.
_ON_DEMAND = ("rewriter", "expander")


@app.entrypoint(
    defaults=_FULL,
    components=_COMPONENTS,
    on_demand=_ON_DEMAND,
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
    node=Node(name="image", label="Qwen Image 2", show=["megapixels"]),
)
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    rewriter: Enhancer,
    expander: Enhancer,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2.1, the 40-step base model."""
    return _generate(ctx, payload, model, rewriter, expander, out, tel, adj, _STEPS, None)


@app.entrypoint(
    defaults=_TURBO,
    components=_COMPONENTS,
    on_demand=_ON_DEMAND,
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
    node=Node(name="image", label="Qwen Image 2", variant="Turbo", show=["megapixels"]),
)
def generate_turbo(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    rewriter: Enhancer,
    expander: Enhancer,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2.1-Turbo: the same inputs, on its 8-step distilled schedule."""
    return _generate(ctx, payload, model, rewriter, expander, out, tel, adj, len(TURBO_SIGMAS), TURBO_SIGMAS)
