"""MiniMax-H3-Image, MiniMax's image model: text to image, or an edit of up to 16 pictures its instructions mention."""

from __future__ import annotations

from typing import Annotated

import msgspec
import torch
from cozy_runtime import rng
from cozy_runtime.author import Adjustments, Context, Seed, Telemetry
from cozy_runtime.fields import Advanced, AspectRatio, Image, Prompt, Range, Ratios
from cozy_runtime.image import MEGAPIXEL, ImageRequest, ImageResult, Sizes, ratio_of
from cozy_runtime.media import ImageFile

from . import lanes as lanes  # the lane functions and the quantized lanes' recipes
from .application import app as app
from .expander import H3ImagePE
from .model import GRID, LATENT_CHANNELS, MAX_PICTURES, STEPS, H3Image, H3ImageEncoder, H3TextEncoder, latent_grid, upright

MODEL = "minimaxai/minimax-h3-image@0.1.1"
#: MiniMax-H3's text encoder, unchanged: MiniMax-H3-Image loads it from MiniMax-H3's release.
TEXT_ENCODER = "minimaxai/minimax-h3@1.0.0-rc.8"
EXPANDER = "minimaxai/minimax-h3-image-pe@0.1.0"
#: 4-20 MP, 16 MP its own (`4096x4096`, 16:9 `5472x3072`, 21:9 `6272x2688`), 1:1 through 21:9 either way.
SIZES = Sizes((4.0, 20.0), (512, 8192), GRID, "the 4-20 MP MiniMax-H3-Image makes", widest=21 / 9)
DEFAULT_MEGAPIXELS = 16.0


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and for an edit the pictures it mentions, `<Picture 1>` ... in order of first mention, up to 16.
    instructions: Prompt[Image] = Prompt()
    #: `W:H`, 1:1 through 21:9 either way (`SIZES.fit`). Unset: the expander's choice, else Picture 1's, else 1:1.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=9 / 21, max=21 / 9)] | None = None
    #: 4-20 MP (`SIZES.pixels`); unset, 16, the model's own.
    megapixels: Annotated[float, Range(4, 20, step=1, unit="MP")] | None = None
    seed: Seed | None = None
    #: MiniMax's prompt expander writes the detailed caption MiniMax-H3-Image renders best, and picks the canvas.
    prompt_upsampling: Annotated[bool, Advanced()] = True


class _Request(ImageRequest):
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text


_PICTURES = 'request.instructions.references.exists(r, r.type == "image")'


@app.tool(
    label="MiniMax-H3-Image",
    named="image",
    defaults={"model": MODEL, "vae_encoder": MODEL, "text_encoder": TEXT_ENCODER, "expander": EXPANDER},
    components={"model": ("transformer", "vae_decoder"), "vae_encoder": ("vae_encoder",),
                "text_encoder": ("text_encoder",), "expander": ("model",)},
    needs={"vae_encoder": _PICTURES, "expander": "request.prompt_upsampling"},
    memoize=True,
)  # fmt: skip
def generate(ctx: Context, payload: ImageBox, model: H3Image, vae_encoder: H3ImageEncoder, text_encoder: H3TextEncoder,
             expander: H3ImagePE, tel: Telemetry, adj: Adjustments) -> ImageResult:  # fmt: skip
    """MiniMax-H3-Image, 6 steps without CFG at 16 MP: text to image, or an edit of the pictures the instructions
    mention."""
    request = _Request(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, seed=payload.seed)
    keys = list(request.prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("instructions", f"MiniMax-H3-Image edits with {MAX_PICTURES} pictures, so it goes without "
                                 f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")  # fmt: skip
        keys = keys[:MAX_PICTURES]
    pictures = [upright(request.prompt.references[key].file.read()) for key in keys]
    text = request.prompt.text
    for n, key in enumerate(keys, 1):
        text = text.replace(f"<{key}>", f"<Picture {n}>")
    seed = request.drawn_seed()
    prompt, ratio = text, request.aspect_ratio
    if payload.prompt_upsampling and text.strip():
        with tel.stage("expanding prompt"):
            found, answers = expander.expand(text, pictures, ratio, seed=seed)
        if found is None:
            adj.warn("prompt_upsampling", "the prompt expander gave no caption twice, so the prompt renders as written")
            tel.log("expanded prompt", parsed=False, answer=answers[-1][-2000:], attempts=len(answers))
        else:
            prompt = found.caption
            ratio = ratio or found.ratio
            tel.log("expanded prompt", prompt=prompt[:4000], aspect_ratio=found.ratio or "", attempts=len(answers))
    ctx.raise_if_cancelled()
    shape = ratio_of(ratio) if ratio else pictures[0].size if pictures else (1, 1)
    pixels = (payload.megapixels or DEFAULT_MEGAPIXELS) * MEGAPIXEL
    width, height = SIZES.fit(adj, shape, SIZES.pixels(adj, request.target(None, megapixels=pixels / MEGAPIXEL)))
    # Every draw from the seed's one CPU stream, in Diffusers' order: the pictures' posterior samples, the image's
    # noise, then each picture's noise.
    generator = rng.generator(seed)
    grids = [latent_grid(picture) for picture in pictures]
    posterior = [torch.randn((1, LATENT_CHANNELS, h, w), generator=generator) for h, w in grids]
    noise = torch.randn((1, LATENT_CHANNELS, 1, height // 16, width // 16), generator=generator)
    reference_noise = [torch.randn((1, LATENT_CHANNELS, h // 2 * 2, w // 2 * 2), generator=generator) for h, w in grids]
    references: list[torch.Tensor] = []
    if pictures:
        with tel.stage("encoding pictures"):
            references = vae_encoder.encode(pictures, posterior)
    with tel.stage("encoding prompt"):
        embeds, tags = text_encoder.encode(prompt, pictures)
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = model.denoise(embeds, tags, references, noise, reference_noise, width=width, height=height,
                                on_step=lambda step: ctx.progress(step, STEPS))  # fmt: skip
    with tel.stage("decoding image"):
        picture = model.decode(latents)
    with tel.stage("saving image"):
        asset = ImageFile.from_image(picture, format="png")
    tel.log("minimax-h3-image", width=width, height=height, pictures=len(pictures), megapixels=round(width * height / MEGAPIXEL, 2))
    return ImageResult(asset, width, height, seed, prompt)
