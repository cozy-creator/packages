"""Krea 2 text-to-image: Raw (the full model, CFG) and Turbo (its 8-step distillation)."""

from __future__ import annotations

from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    Guidance,
    ImageRequest,
    ImageResult,
    Memoize,
    NegativePrompt,
    Outputs,
    Sizes,
    Steps,
    Telemetry,
    held,
)

from . import quantize as quantize  # its recipes
from .model import PATCH, Krea2

app = App()

#: Krea: Raw is trained to 1 MP, Turbo makes 1-4 MP; 256 px sides at the least, on the 16 px grid.
SIZES = Sizes((0.25, 4.0), (256, 4096), PATCH, "Krea 2's 0.25-4 MP")
#: Krea's settings (steps, guidance, megapixels): Raw 52 steps at guidance 3.5 at its 1 MP, Turbo 8 steps
#: without guidance at its 2048² (`krea-ai/krea-2` README).
RAW, TURBO = (52, 3.5, 1.0), (8, 0.0, 4.0)


class TurboInput(ImageRequest):
    steps: Steps = None


class GenerateInput(TurboInput):
    guidance: Guidance = None
    negative_prompt: NegativePrompt = None


def _generate(ctx: Context, payload: TurboInput, model: Krea2, out: Outputs, tel: Telemetry,
              adj: Adjustments, *, distilled: bool) -> ImageResult:
    model.hold_vae()
    prompt = payload.rendered(adj)
    steps, guidance, megapixels = TURBO if distilled else RAW
    steps = held(adj, "steps", payload.steps or steps, 1, 100, "Krea 2's 1-100 steps")
    words = ""
    if isinstance(payload, GenerateInput):
        guidance = held(adj, "guidance", guidance if payload.guidance is None else payload.guidance, 0.0, 20.0,
                        "Krea 2's guidance 0-20")
        words = payload.negative_prompt or ""
        if guidance == 0 and words:
            adj.warn("negative_prompt", "ignored: guidance 0 runs no negative branch")
    seed = payload.drawn_seed()
    width, height = payload.size(adj, None, SIZES, megapixels=megapixels)
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt", overall_range=(0.0, 0.05)):
        positive = model.encode(prompt)
        negative = model.encode(words) if guidance > 0 else None
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.05, 0.95)):
        latents = model.denoise(
            positive, negative, width=width, height=height, steps=steps, guidance=guidance, seed=seed,
            distilled=distilled, on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.05, 0.95)),
        )
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(latents, width=width, height=height)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


_COMPONENTS = {"model": ("transformer", "text_encoder", "vae")}
_MEMO = ("krea_2", "krea_2.model")


@app.entrypoint(defaults={"model": "krea/krea-2@0.1.0"}, components=_COMPONENTS, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate(ctx: Context, payload: GenerateInput, model: Krea2, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> ImageResult:
    """The full model, Krea 2 Raw (or a fine-tune of it): 52 steps at guidance 3.5, 1 MP."""
    return _generate(ctx, payload, model, out, tel, adj, distilled=False)


@app.entrypoint(defaults={"model": "krea/krea-2-turbo@0.1.0"}, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]), memo_dependencies=_MEMO)
def generate_turbo(ctx: Context, payload: TurboInput, model: Krea2, out: Outputs, tel: Telemetry,
                   adj: Adjustments) -> ImageResult:
    """The fast model, Krea 2 Turbo (or a fine-tune of it): 8 steps without guidance at its fixed shift, 2048²."""
    return _generate(ctx, payload, model, out, tel, adj, distilled=True)
