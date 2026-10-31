"""FLUX.2 [dev], its Turbo LoRA, and FLUX.2 [klein] 9B and 4B: text to image, or an edit of up to ten
ordered pictures."""

from __future__ import annotations

from functools import partial
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    MEGAPIXEL,
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
    held,
)

from .model import MAX_PIXELS, MAX_SIDE, MIN_SIDE, PATCH, TURBO_SIGMAS, Flux2, prepare_references, schedule
from . import quantize as quantize  # its recipes
from .turbo import Flux2TurboLoRA
from .upsampler import Flux2Upsampler

app = App()

#: 64-4096 px sides and 4 MiP, on the 16 px grid.
SIZES = Sizes((MIN_SIDE**2 / MEGAPIXEL, MAX_PIXELS / MEGAPIXEL), (MIN_SIDE, MAX_SIDE), PATCH,
              "the model's 4 MP maximum")


class GenerateInput(ImageRequest):
    pictures: Annotated[list[Picture], msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )


def label(n: int, picture: Picture) -> str:
    """FLUX.2's word for the Nth picture, as BFL's prompting guide writes it ("subject from image 1,
    style from image 2"); the text encoder sees no image, so this is convention, not a token."""
    return f"image {n}"


def _generate(ctx: Context, payload: GenerateInput, model: Flux2, turbo: Flux2TurboLoRA | None,
              upsampler: Flux2Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    model.hold_vae(tel)
    seed = payload.drawn_seed()
    payload.unplaced(adj, "FLUX.2 has no layout control")
    images = [asset.read() for asset in payload.images()]
    width, height = payload.size(adj, images[0].size if images else None, SIZES, megapixels=1.0)
    # The function names the regime (`generate_turbo` passes `turbo`): [dev] 50 steps at guidance 4,
    # with its Turbo fal's 8 sigmas at 2.5; a [klein] as Base 50 steps of CFG 4, or distilled 4 steps
    # without guidance (BFL `util.py`).
    distilled = turbo is not None and not model.dev
    turbo = turbo if model.dev else None
    steps = 8 if turbo else 4 if distilled else 50
    if payload.steps is not None and turbo and payload.steps != steps:
        adj.clamp("steps", payload.steps, steps, "the Turbo LoRA's own 8-step schedule")
    elif payload.steps is not None:
        steps = held(adj, "steps", payload.steps, 1, 100)
    if distilled:
        guidance = None
        payload.ignored(adj, "guidance", reason="FLUX.2 [klein] distilled takes no guidance")
    else:
        default = 2.5 if turbo else 4.0
        guidance = held(adj, "guidance", default if payload.guidance is None else payload.guidance, 1.0, 20.0)
    cfg = not model.dev and guidance is not None and guidance > 1  # distilled: no guidance
    if not cfg:
        payload.ignored(adj, "negative_prompt", reason="only FLUX.2 [klein] Base runs CFG")
    tel.log("sampling", model="[dev]" if model.dev else "[klein]",
            regime="turbo" if turbo else "distilled" if distilled else "base", steps=steps,
            guidance=guidance, cfg=cfg)
    prompt = payload.rendered(adj, label)
    if payload.prompt_upsampling:
        with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
            prompt = upsampler.upsample(prompt, images)
        tel.log("upsampled prompt", caption=prompt[:2000])
    ctx.raise_if_cancelled()
    references = None
    if images:
        with tel.stage("encoding reference images", overall_range=(0.15, 0.2)):
            references = model.encode_references(prepare_references(images))
    with tel.stage("encoding prompt", overall_range=(0.2, 0.25)):
        embeds = model.encode(prompt)
        negative = model.encode(payload.negative_prompt or "") if cfg else None
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


@app.entrypoint(defaults={"model": _DEV, "turbo_lora": "fal/flux-2-dev-turbo@0.1.0", "upsampler": _DEV},
                components=_TURBO_COMPONENTS, on_demand=("upsampler", "turbo_lora"), memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: GenerateInput, model: Flux2, turbo_lora: Flux2TurboLoRA,
                   upsampler: Flux2Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The fast model: FLUX.2 [dev] with fal's Turbo LoRA (8 steps, guidance 2.5), or a [klein] by
    `model=` (4 steps)."""
    return _generate(ctx, payload, model, turbo_lora, upsampler, out, tel, adj)
