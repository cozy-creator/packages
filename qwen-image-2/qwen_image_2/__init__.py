"""Qwen-Image-2.1 generation and reference editing with Runtime-owned execution."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Annotated, Literal

import msgspec
import torch
from cozy_runtime.author import Adjustments, App, Context, OutputError, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, FileBound, Image, MENTION, Prompt, Ratios, When
from cozy_runtime.image import MEGAPIXEL, ImageRequest, ImageResult, Sizes, ratio_of
from cozy_runtime.media import ImageFile
from PIL import Image as PILImage

from . import quantize as quantize  # its recipes
from .keep import keep_unchanged
from .model import QwenImage21DiT, QwenImage21TextEncoder, QwenImage21VAE, reference_grids
from .rewriter import Creativity, Enhancer, keeps_transparency, rewrite, transparent
from .turbo import FunAccLoRA

app = App()

#: The pixel counts Qwen serves: upstream's 1 MP default to its native 2K (4 MP);
#: upstream's rotary table centres at most 2048 tokens of 16 px on an axis.
MIN_MEGAPIXELS = 1.0
MAX_MEGAPIXELS = 4.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (32, 2048 * 16), 32, "the 1-4 MP Qwen-Image serves")
#: The size without references (Paul, 11-06: "2 megapixels is a good default").
DEFAULT_MEGAPIXELS = 2
#: The full model's steps; Turbo's are its overlay checkpoint's (`turbo.read_plan`).
_STEPS = 40
#: Upstream's conditioning area for a reference (Diffusers' `output_resolution` 1024): the pictures past the first.
REFERENCE_AREA = 1024 * 1024


#: The pictures Qwen-Image-2.1 takes: past them the last mentioned go, with a warning.
MAX_PICTURES = 10


class GenerateInput(ImageRequest):
    #: Its text, and the pictures it mentions, in the order it first mentions them.
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text
    #: Qwen Image 2.1 renders at 1, 2 or 4 MP (its native 2K); 2 by default (Paul, 11-06).
    megapixels: Literal[1, 2, 4] = DEFAULT_MEGAPIXELS  # type: ignore[assignment]  # 2, a Literal member
    #: Qwen's prompt enhancer rewrites the request first (`rewriter.py`): an edit from its pictures, else the words.
    prompt_upsampling: bool = True
    #: How far the rewrite goes beyond the words, so only with it (`_upsampled`); `balanced` is Qwen's own.
    creativity: Annotated[Creativity, When("prompt_upsampling")] = "balanced"
    #: An edit keeps the first picture's own pixels wherever it left them unchanged (`keep.py`).
    keep_unchanged: bool = True


def label(n: int, key: str, count: int) -> str:
    """Qwen-Image-2.1's word for the Nth of `count` pictures: its template writes `<imageN>` before each image
    (Diffusers 0.41 `QwenImage21Pipeline`), and Qwen's rewriter tags only two or more; a lone picture is its name,
    `kyogre.2` as `kyogre 2`."""
    return f"<image{n}>" if count > 1 else key.replace(".", " ")


#: The decoder's alpha never quite reaches 0 or 255: a transparent background sits at 1–4 and an
#: opaque subject at 251–255 (runs 12867, 12878, 12891). Those bands snap to exact.
_ALPHA_NOISE = 4


def decoded_image(decoded: torch.Tensor) -> PILImage.Image:
    """The decoder's pixels: RGBA when its alpha has any transparency, else RGB. Quantized on the
    decoder's device, NumPy's arithmetic in Torch: a 4 MP canvas sends 16 MB of bytes to the host,
    not 67 MB of floats for six host passes."""
    # The model returns B,C,H,W in [-1,1], just as upstream VaeImageProcessor consumes.
    pixels = decoded[0].detach().float()
    if not bool(torch.isfinite(pixels).all()):
        raise OutputError(
            "Qwen Image 2.1's decoder produced non-finite pixels, so there is no picture: try another seed, "
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


def _said(prompt: Prompt[Image], adj: Adjustments, before: int = 0) -> tuple[str, list[Image]]:
    """Its text, and what each picture it mentions is, each mention in the model's words (`label`), and the pictures:
    a mention reads as `<image3>` alone, and with no words for it a third picture was dropped in 3 of 3 seeds (runs
    27130, 27161, 27174), kept with them (27190, 27200). `before`: the pictures ahead of these (an upscale's own)."""
    keys = list(prompt.references)
    room = MAX_PICTURES - before
    if len(keys) > room:
        adj.warn("prompt", f"Qwen Image 2.1 takes {MAX_PICTURES} pictures, so it goes without {', '.join(f'<{key}>' for key in keys[room:])}")
        keys = keys[:room]
    words = "\n".join(f"<{key}>: {prompt.references[key].description.strip()}" for key in keys if prompt.references[key].description.strip())
    said = "\n\n".join(filter(None, (prompt.text.strip(), words)))
    # One pass: a label never reads as another key (a box named `image1`).
    labels = {key: label(before + n, key, before + len(keys)) for n, key in enumerate(keys, 1)}
    return MENTION.sub(lambda found: labels.get(found[1], found[0]), said), [prompt.references[key] for key in keys]


def _upsampled(
    payload: GenerateInput,
    prompt: str,
    images: list[PILImage.Image],
    seed: int,
    enhancer: Enhancer,
    progress: Callable[[int, int], None],
    tel: Telemetry,
    adj: Adjustments,
) -> tuple[GenerateInput, str]:
    """The request as Qwen's enhancer rewrites it, sampled from the render's `seed`, and the canvas it chose
    when the request names none (upstream passes `wh_ratio` and `ratio_follow` to the renderer)."""
    with tel.stage("upsampling prompt"):
        found, answer, thought, said = rewrite(enhancer, prompt, images, payload.creativity, seed, progress)
    if found is None:
        adj.warn("prompt_upsampling", "Qwen's prompt enhancer gave no rewrite, so the prompt goes as written")
        tel.log("upsampled prompt", parsed=False, answer=answer[:2000], thought_tokens=thought, answer_tokens=said)
        return payload, prompt
    if transparent(prompt) and not keeps_transparency(prompt, found.prompt):
        adj.warn("prompt_upsampling", "Qwen's prompt enhancer dropped the transparent background, so the prompt goes as written")
        tel.log("upsampled prompt", kept=False, lost="transparency", rewrite=found.prompt[:2000], seed=seed,
                thought_tokens=thought, answer_tokens=said)
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
    expander_i2i: Enhancer,
    expander_t2i: Enhancer,
    model: QwenImage21DiT,
    text_encoder: QwenImage21TextEncoder,
    vae: QwenImage21VAE,
    tel: Telemetry,
    adj: Adjustments,
    turbo: FunAccLoRA | None,
    sigmas: tuple[float, ...] | None = None,
    own: bool = False,
) -> ImageResult:
    """Every function's body: with `turbo`, its overlay on the base model and its 4-step grid, else the
    base model's 40 steps of uniform spacing. Proof only (se-153): `sigmas` a distilled schedule; `own` the
    picture conditioned at its own size, at most the canvas's."""
    prompt, pictures = _said(payload.prompt, adj)
    steps = len(sigmas) if sigmas else turbo.steps if turbo else _STEPS
    tel.log("sampling", regime="turbo (Fun Acc)" if turbo else "base", steps=steps,
            schedule="PDD grid, a head per step" if turbo else "uniform, dynamic shift")
    seed = payload.drawn_seed()
    # Upstream feeds the VAE RGBA; its vision encoder sees alpha composited over white.
    reference_images = [picture.file.read(mode="RGBA") for picture in pictures] or None
    if payload.prompt_upsampling:
        enhancer = expander_i2i if reference_images else expander_t2i
        payload, prompt = _upsampled(payload, prompt, reference_images or [], seed, enhancer, ctx.progress, tel, adj)
    vae.hold(tel)
    ctx.raise_if_cancelled()
    # What is not given is the first reference's (upstream takes the LAST reference's; every Cozy
    # image package takes the first, the image being edited).
    first = reference_images[0].size if reference_images else None
    width, height = payload.size(adj, first, SIZES, megapixels=DEFAULT_MEGAPIXELS)
    reference_latents, image_pad_mask = None, None
    conditioning_images: list[PILImage.Image] = []  # the references as encoded
    if reference_images:
        # The picture being edited at the canvas's area, so the model reads it at the detail it draws (se-153; upstream
        # conditions every picture at 1 MP, `output_resolution`); the others at upstream's 1 MP.
        areas = (width * height, *(REFERENCE_AREA,) * (len(reference_images) - 1))
        if own:
            areas = (min(width * height, reference_images[0].width * reference_images[0].height), *areas[1:])
        with tel.stage("encoding reference images"):
            conditioning_images, reference_latents = vae.encode_reference_images(reference_images, areas=areas)
        ctx.raise_if_cancelled()
    with tel.stage("encoding prompt"):
        prompt, keep, given = text_encoder.fit_prompt(prompt, reference_grids(conditioning_images))
        if keep < given:
            beside = f" beside {len(pictures)} pictures" if reference_images else ""
            adj.warn("prompt", f"Qwen-Image 2.1 reads {keep} of its {given} tokens{beside}, so it goes without the rest")
        if reference_images is None:
            embeds, mask = text_encoder.encode(prompt)
        else:
            embeds, mask, image_pad_mask = text_encoder.encode_image_prompt(prompt, conditioning_images)
    with tel.stage("generating image"):
        denoise = model.denoise if turbo is None else partial(turbo.denoise, model)
        if sigmas:
            denoise = partial(model.denoise, sigmas=sigmas)
        latents = denoise(
            embeds,
            mask,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            progress=ctx.progress,
            reference_images=conditioning_images or None,
            reference_latents=reference_latents,
            image_pad_mask=image_pad_mask,
        )
    ctx.raise_if_cancelled()
    with tel.stage("decoding image"):
        decoded = vae.decode(latents, width=width, height=height)
    if reference_images and payload.keep_unchanged:
        with tel.stage("keeping what stayed"):
            decoded, kept = keep_unchanged(reference_images[0], decoded)
        tel.log("keep unchanged", kept=round(kept.fraction, 3), scale=round(kept.scale, 4),
                shift=f"{kept.shift[0]:.1f},{kept.shift[1]:.1f}", skipped=kept.skipped)
    image = decoded_image(decoded)
    with tel.stage("saving image"):
        asset = ImageFile.from_image(image)
    return ImageResult(asset, width, height, seed, prompt)


#: Qwen's prompt enhancers (QwenLM/Qwen-Image-2.1 `prompt_rewrite/`), a picture from words' (`expander_t2i`) and an
#: edit's (`expander_i2i`), and the text encoder, the `text_encoder` of a Qwen-Image-2.1 checkpoint and one for every function, are abliterated by
#: default (Paul, 11-06). Qwen's own are `qwen/qwen-image-2.1-pe-t2i@0.1.0`, `qwen/qwen-image-2.1-pe-i2i@0.1.0` and
#: any Qwen-Image-2.1 checkpoint (`model.text_encoder=qwen/qwen-image-2.1@0.1.0`). Turbo is the base model with
#: Alibaba PAI's Fun Acc over it (`turbo_lora`). `vae` encodes the pictures and decodes the image: the base model's
#: own by default, or another Qwen-Image-2.1 VAE (`model.vae=madebyollin/texture-fix-vae`).
_FULL = {"expander_i2i": "darrellbest/qwen-image-2.1-pe-i2i-abliterated@0.1.0",
         "expander_t2i": "pottokao/qwen-image-2.1-pe-t2i-abliterated@0.1.0", "model": "qwen/qwen-image-2.1@0.1.0",
         "text_encoder": "atomicchat/qwen-image-2.1-text-encoder-abliterated@0.1.0", "vae": "qwen/qwen-image-2.1@0.1.0"}
_TURBO = {"expander_i2i": "darrellbest/qwen-image-2.1-pe-i2i-abliterated@0.1.0",
          "expander_t2i": "pottokao/qwen-image-2.1-pe-t2i-abliterated@0.1.0", "model": "qwen/qwen-image-2.1@0.1.0",
          "text_encoder": "atomicchat/qwen-image-2.1-text-encoder-abliterated@0.1.0",
          "turbo_lora": "alibaba-pai/qwen-image-2.1-fun-acc@0.1.1", "vae": "qwen/qwen-image-2.1@0.1.0"}
_COMPONENTS = {"model": ("transformer",), "text_encoder": ("text_encoder",), "vae": ("vae",)}
#: Which requests need each enhancer (cl-1103): one with pictures the edit's, one without the other, and
#: `prompt_upsampling=False` neither. A machine fetches the one its request needs before it starts, never the other.
#: A picture is a reference of type `image`: Runtime leaves any other kind out of a `Prompt[Image]`.
_PICTURES = '.references.exists(r, r.type == "image")'
_NO_PICTURES = '.references.all(r, r.type != "image")'
_NEEDS = {"expander_i2i": f"request.prompt_upsampling && request.prompt{_PICTURES}",
          "expander_t2i": f"request.prompt_upsampling && request.prompt{_NO_PICTURES}"}  # fmt: skip


@app.entrypoint(defaults=_FULL, components=_COMPONENTS, needs=_NEEDS, memoize=True)  # fmt: skip
def generate(
    ctx: Context,
    payload: GenerateInput,
    expander_i2i: Enhancer,
    expander_t2i: Enhancer,
    model: QwenImage21DiT,
    text_encoder: QwenImage21TextEncoder,
    vae: QwenImage21VAE,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2.1, the 40-step base model."""
    return _generate(ctx, payload, expander_i2i, expander_t2i, model, text_encoder, vae, tel, adj, None)


@app.entrypoint(defaults=_TURBO, components=_COMPONENTS, needs=_NEEDS, memoize=True)  # fmt: skip
def generate_turbo(
    ctx: Context,
    payload: GenerateInput,
    expander_i2i: Enhancer,
    expander_t2i: Enhancer,
    model: QwenImage21DiT,
    text_encoder: QwenImage21TextEncoder,
    turbo_lora: FunAccLoRA,
    vae: QwenImage21VAE,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen-Image-2.1 with Alibaba PAI's Fun Acc: the same inputs, in 4 steps."""
    return _generate(ctx, payload, expander_i2i, expander_t2i, model, text_encoder, vae, tel, adj, turbo_lora)


#: Proof only (se-153): Qwen-Image-2.1-Turbo's own 8-step schedule, its `model_index.json` `sample_sigmas`
#: (hf://Qwen/Qwen-Image-2.1-Turbo@fc13ba19).
QWEN_TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
_QWEN_TURBO = {**_FULL, "model": "qwen/qwen-image-2.1-turbo@0.1.1"}


@app.entrypoint(defaults=_QWEN_TURBO, components=_COMPONENTS, needs=_NEEDS, memoize=True)  # fmt: skip
def generate_qwen_turbo(
    ctx: Context,
    payload: GenerateInput,
    expander_i2i: Enhancer,
    expander_t2i: Enhancer,
    model: QwenImage21DiT,
    text_encoder: QwenImage21TextEncoder,
    vae: QwenImage21VAE,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Proof only (se-153): Qwen's own Qwen-Image-2.1-Turbo checkpoint on its 8 sigmas."""
    return _generate(ctx, payload, expander_i2i, expander_t2i, model, text_encoder, vae, tel, adj, None, QWEN_TURBO_SIGMAS)


@app.entrypoint(defaults=_TURBO, components=_COMPONENTS, needs=_NEEDS, memoize=True)  # fmt: skip
def generate_turbo_own(
    ctx: Context,
    payload: GenerateInput,
    expander_i2i: Enhancer,
    expander_t2i: Enhancer,
    model: QwenImage21DiT,
    text_encoder: QwenImage21TextEncoder,
    turbo_lora: FunAccLoRA,
    vae: QwenImage21VAE,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Proof only (se-153): Turbo with the picture conditioned at its own size, at most the canvas's."""
    return _generate(ctx, payload, expander_i2i, expander_t2i, model, text_encoder, vae, tel, adj, turbo_lora, own=True)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it mentions (`_said`).
    instructions: Prompt[Image] = Prompt()
    #: The base model with its Fun Acc overlay in 4 steps, else its 40.
    turbo: bool = True
    #: `ImageRequest.target`; Qwen's rotary range holds it to 1:16 through 16:1 (`SIZES.fit`).
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `SIZES.pixels`: Qwen Image 2.1 renders at 1, 2 or 4 MP, by default 2 (Paul, 11-06).
    megapixels: Literal[1, 2, 4] = DEFAULT_MEGAPIXELS  # type: ignore[assignment]  # 2, a Literal member
    seed: Seed | None = None
    #: Qwen's prompt enhancer rewrites the request first (`_upsampled`).
    prompt_upsampling: Annotated[bool, Advanced()] = True
    #: How far that rewrite goes beyond the words (`rewrite`), so only with it.
    creativity: Annotated[Creativity, Advanced(), When("prompt_upsampling")] = "balanced"
    #: `GenerateInput.keep_unchanged`.
    keep_unchanged: Annotated[bool, Advanced()] = True


# The Turbo overlay is needed only when a box asks for it, and an enhancer only when the box upsamples: the edit's with
# pictures, else the other; a LoRA dropped on the box adapts the base model either way.
_BOX_NEEDS = {"expander_i2i": f"request.prompt_upsampling && request.instructions{_PICTURES}",
              "expander_t2i": f"request.prompt_upsampling && request.instructions{_NO_PICTURES}",
              "turbo_lora": "request.turbo"}  # fmt: skip


@app.tool(label="Image", named="image", defaults=_TURBO, components=_COMPONENTS, needs=_BOX_NEEDS, memoize=True)  # fmt: skip
def image(
    ctx: Context,
    payload: ImageBox,
    expander_i2i: Enhancer,
    expander_t2i: Enhancer,
    model: QwenImage21DiT,
    text_encoder: QwenImage21TextEncoder,
    turbo_lora: FunAccLoRA,
    vae: QwenImage21VAE,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """Qwen Image 2.1 on a canvas: Turbo (the base model and its Fun Acc overlay), or the full 40 steps."""
    request = GenerateInput(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio,
                            megapixels=payload.megapixels, seed=payload.seed,
                            prompt_upsampling=payload.prompt_upsampling, creativity=payload.creativity,
                            keep_unchanged=payload.keep_unchanged)  # fmt: skip
    return _generate(ctx, request, expander_i2i, expander_t2i, model, text_encoder, vae, tel, adj,
                     turbo_lora if payload.turbo else None)  # fmt: skip


#: What Qwen Image 2.1 is asked of a picture it upscales: a restoration, after ausboss's Qwen-Image-2.1 Restore LoRA card (se-147).
#: `{}` is the picture: `the picture`, or `<image1>` when a prompt's pictures follow it.
UPSCALE_PROMPT = ("Restore {} at a higher resolution: remove the blur, noise and JPEG compression artifacts and "
                  "recover sharp, clean detail and edges, keeping the same composition, subject, line work and colours.")
#: The base model's steps for an upscale: 20, 28 and 40 score alike against the clean pictures (se-147 round 2).
UPSCALE_STEPS = 20


class UpscaleInput(msgspec.Struct, kw_only=True):
    #: `upscale`: the picture, the box's own version.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
    #: `upscale`: what the picture should show, after the restoration request, never in place of it: text too small to
    #: read, a colour, a face. Each picture it mentions is a reference after the picture, at its 1 MP (`_said`).
    prompt: Prompt[Image] = Prompt()
    #: `upscale`: the picture's ratio at this size (`SIZES.fit`), up to Qwen's 4 MP.
    megapixels: Literal[2, 4] = 4
    seed: Seed | None = None


_UPSCALE = {"model": "qwen/qwen-image-2.1@0.1.0", "text_encoder": "atomicchat/qwen-image-2.1-text-encoder-abliterated@0.1.0",
            "vae": "qwen/qwen-image-2.1@0.1.0"}


@app.tool(label="Upscale", named="image", defaults=_UPSCALE,
          components={"model": ("transformer",), "text_encoder": ("text_encoder",), "vae": ("vae",)}, memoize=True)  # fmt: skip
def upscale(ctx: Context, payload: UpscaleInput, model: QwenImage21DiT, text_encoder: QwenImage21TextEncoder,
            vae: QwenImage21VAE, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Qwen Image 2.1 restores the picture at the size asked: the base model edits it from noise, the picture its reference
    at the canvas's size, as Diffusers sizes a reference to its output, so the edit keeps to it pixel for pixel. A start from
    the picture blown up keeps its blur and blocks at any strength (se-147). `prompt` follows the restoration request,
    and the pictures it mentions follow the picture as references, each at its 1 MP as `generate` takes them."""
    with tel.stage("reading picture"):
        picture = payload.image.read(mode="RGBA")
        said, pictures = _said(payload.prompt, adj, before=1)
        mentioned = [mention.file.read(mode="RGBA") for mention in pictures]
    width, height = SIZES.fit(adj, picture.size, payload.megapixels * MEGAPIXEL)
    seed = payload.seed if payload.seed is not None else ImageRequest(prompt="").drawn_seed()
    vae.hold(tel)
    with tel.stage("encoding reference images"):
        conditioning, latents = vae.encode_reference_images([picture], (width, height))
        if mentioned:
            more, more_latents = vae.encode_reference_images(mentioned)
            conditioning, latents = [*conditioning, *more], torch.cat([latents, more_latents], dim=1)
    request = UPSCALE_PROMPT.format("<image1>" if mentioned else "the picture")
    with tel.stage("encoding prompt"):
        prompt, keep, given = text_encoder.fit_prompt("\n\n".join(filter(None, (request, said))), reference_grids(conditioning))
        if keep < given:
            adj.warn("prompt", f"Qwen Image 2.1 reads {keep} of the upscale's {given} tokens, so it goes without the rest")
        embeds, mask, image_pad_mask = text_encoder.encode_image_prompt(prompt, conditioning)
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(embeds, mask, width=width, height=height, steps=UPSCALE_STEPS, seed=seed,
                                progress=ctx.progress, reference_images=conditioning, reference_latents=latents,
                                image_pad_mask=image_pad_mask)  # fmt: skip
    with tel.stage("decoding image"):
        image = decoded_image(vae.decode(latents, width=width, height=height))
    with tel.stage("saving image"):
        asset = ImageFile.from_image(image, format="png")
    tel.log("upscale", source=f"{picture.width}x{picture.height}", width=width, height=height, steps=UPSCALE_STEPS,
            references=len(mentioned))  # fmt: skip
    return ImageResult(asset, width, height, seed, prompt)
