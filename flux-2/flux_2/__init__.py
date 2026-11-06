"""FLUX.2 [dev], its Turbo LoRA, and FLUX.2 [klein] 9B and 4B: text to image, or an edit of up to ten
ordered pictures."""

from __future__ import annotations

from functools import partial
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    AspectRatio,
    Context,
    Guidance,
    Image,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    Prompt,
    Ratios,
    Seed,
    Sizes,
    Steps,
    Telemetry,
    held,
)

from .model import MAX_SIDE, MIN_SIDE, PATCH, TURBO_SIGMAS, Flux2, prepare_references, schedule
from . import quantize as quantize  # its recipes
from .turbo import Flux2TurboLoRA
from .upsampler import Flux2Upsampler

app = App()

#: 64-4096 px sides and 64² px to 4 MiP, on the 16 px grid.
MIN_MEGAPIXELS = 64 * 64 / 1048576
MAX_MEGAPIXELS = 4.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (MIN_SIDE, MAX_SIDE), PATCH, "the model's 4 MP maximum")
#: The pictures FLUX.2 takes: past them the last mentioned go, with a warning.
MAX_PICTURES = 10


class TurboInput(ImageRequest):
    #: Its text, and the pictures it edits, in order of first mention.
    prompt: Prompt[Image] = Prompt()
    guidance: Guidance = None
    prompt_upsampling: bool | None = None


class GenerateInput(TurboInput):
    steps: Steps = None


def _said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[Image]]:
    """Its text with each mention as FLUX.2's `image N`, as BFL's prompting guide writes it ("subject from
    image 1, style from image 2"; the text encoder sees no image, so this is convention, not a token), and
    the pictures."""
    keys = list(prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"FLUX.2 takes {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
        keys = keys[:MAX_PICTURES]
    said = prompt.text
    for n, key in enumerate(keys, 1):
        said = said.replace(f"<{key}>", f"image {n}")
    return said, [prompt.references[key] for key in keys]


def _generate(ctx: Context, payload: TurboInput, model: Flux2, turbo: Flux2TurboLoRA | None,
              upsampler: Flux2Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    model.hold_vae(tel)
    seed = payload.drawn_seed()
    prompt, pictures = _said(payload.prompt, adj)
    images = [picture.file.read() for picture in pictures]
    width, height = payload.size(adj, images[0].size if images else None, SIZES, megapixels=1.0)
    # The function names the regime (`generate_turbo` passes `turbo`): [dev] 50 steps at guidance 4,
    # with its Turbo fal's 8 sigmas at 2.5; a [klein] as Base 50 steps of CFG 4, or distilled 4 steps
    # without guidance (BFL `util.py`).
    distilled = turbo is not None and not model.dev
    turbo = turbo if model.dev else None
    steps = 8 if turbo else 4 if distilled else 50
    if isinstance(payload, GenerateInput) and payload.steps is not None:
        steps = held(adj, "steps", payload.steps, 1, 100)
    if distilled:
        guidance = None
        if payload.guidance is not None:
            adj.warn("guidance", "ignored: FLUX.2 [klein] distilled takes no guidance")
    else:
        default = 2.5 if turbo else 4.0
        guidance = held(adj, "guidance", default if payload.guidance is None else payload.guidance, 1.0, 20.0)
    cfg = not model.dev and guidance is not None and guidance > 1  # distilled: no guidance
    tel.log("sampling", model="[dev]" if model.dev else "[klein]",
            regime="turbo" if turbo else "distilled" if distilled else "base", steps=steps,
            guidance=guidance, cfg=cfg)
    if payload.prompt_upsampling:
        with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
            caption = upsampler.upsample(prompt, images)
        if caption:
            prompt = caption
        else:
            adj.warn("prompt_upsampling", "the upsampler returned nothing; the prompt goes as written")
        tel.log("upsampled prompt", caption=prompt[:2000])
    ctx.raise_if_cancelled()
    references = None
    if images:
        with tel.stage("encoding reference images", overall_range=(0.15, 0.2)):
            references = model.encode_references(prepare_references(images))
    with tel.stage("encoding prompt", overall_range=(0.2, 0.25)):
        embeds = model.encode(prompt)
        negative = model.encode("") if cfg else None
    ctx.raise_if_cancelled()
    tokens = (height // PATCH) * (width // PATCH)
    timesteps = schedule(steps, tokens, TURBO_SIGMAS if turbo else None)
    denoise = partial(
        model.denoise, embeds, negative, references, width=width, height=height, timesteps=timesteps,
        guidance=guidance, seed=seed, cancel=ctx.raise_if_cancelled,
        on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.25, 0.95)),
    )
    with tel.stage("generating image", overall_range=(0.25, 0.95)):
        latents = turbo.applied(model.graph.components["transformer"], denoise) if turbo else denoise()
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(latents, width=width, height=height)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


_DEV = "black-forest-labs/flux-2-dev@0.1.0"
_TURBO = "fal/flux-2-dev-turbo@0.1.0"
#: The upsampler is [dev]'s text encoder, built and fetched only when `prompt_upsampling` is on; the
#: Turbo LoRA only with [dev].
_COMPONENTS = {"model": ("transformer", "text_encoder", "vae"), "upsampler": ("text_encoder",)}
_TURBO_COMPONENTS = {"model": ("transformer", "text_encoder", "vae"), "upsampler": ("text_encoder",),
                     "turbo_lora": ("model",)}


@app.entrypoint(defaults={"model": _DEV, "upsampler": _DEV}, components=_COMPONENTS, on_demand=("upsampler",),
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, model: Flux2, upsampler: Flux2Upsampler, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The full model: FLUX.2 [dev] (50 steps, guidance 4), or a [klein] Base by `model=` (50 steps, CFG 4)."""
    return _generate(ctx, payload, model, None, upsampler, out, tel, adj)


@app.entrypoint(defaults={"model": _DEV, "turbo_lora": _TURBO, "upsampler": _DEV},
                components=_TURBO_COMPONENTS, on_demand=("upsampler", "turbo_lora"), memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: TurboInput, model: Flux2, turbo_lora: Flux2TurboLoRA,
                   upsampler: Flux2Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The fast model: FLUX.2 [dev] with fal's Turbo LoRA (8 steps, guidance 2.5), or a [klein] by
    `model=` (4 steps)."""
    return _generate(ctx, payload, model, turbo_lora, upsampler, out, tel, adj)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it edits (`_said`).
    instructions: Prompt[Image] = Prompt()
    #: [dev] with fal's Turbo LoRA, 8 steps at guidance 2.5, else 50 at guidance 4 (`_generate`).
    turbo: bool = True
    #: `ImageRequest.target`, held to the model's sizes (`SIZES.fit`).
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `SIZES.pixels`: BFL's 1 MP default up to FLUX.2's 4 MP (README, Sizes).
    megapixels: Literal[1, 2, 4] | None = None
    #: `ImageRequest.drawn_seed`.
    seed: Seed | None = None
    #: BFL's upsampler rewrites the prompt first (`_generate`).
    prompt_upsampling: bool = False


@app.tool(label="FLUX.2", named="image", defaults={"model": _DEV, "turbo_lora": _TURBO, "upsampler": _DEV},
          components=_TURBO_COMPONENTS, on_demand=("turbo_lora", "upsampler"), memoize=Memoize(requires=["seed"]))
def image(ctx: Context, payload: ImageBox, model: Flux2, turbo_lora: Flux2TurboLoRA, upsampler: Flux2Upsampler,
          out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """FLUX.2 [dev] on a canvas: with its Turbo LoRA, which loads only when asked for, or without."""
    request = GenerateInput(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=payload.megapixels,
                            seed=payload.seed, prompt_upsampling=payload.prompt_upsampling)  # fmt: skip
    return _generate(ctx, request, model, turbo_lora if payload.turbo else None, upsampler, out, tel, adj)
