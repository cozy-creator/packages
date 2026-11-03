"""Qwen-Image-2.1 generation and reference editing with Runtime-owned execution."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import msgspec
import torch
from cozy_runtime import stepcache
from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    Invocation,
    Operation,
    Picture,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    ProjectEntity,
    ProjectPicture,
    ProjectSetup,
    ProjectWorldState,
    Refuse,
    Sizes,
    Telemetry,
    held,
    name_label,
    spoken,
)
from PIL import Image

from .model import QwenImage21Model
from . import quantize as quantize  # its recipes

app = App()

#: The pixel counts Qwen serves: upstream's 1 MP default to its native 2K (4 MP), the default without
#: references; upstream's rotary table centres at most 2048 tokens of 16 px on an axis.
SIZES = Sizes((1.0, 4.0), (32, 2048 * 16), 32, "the 1-4 MP Qwen-Image serves")
#: Steps without a checkpoint schedule (Turbo names its own 8), and the most admitted.
_STEPS, _MAX_STEPS = 40, 100
#: Qwen-Image-2.1-Turbo's distilled schedule, its `model_index.json` `sample_sigmas`
#: (hf://Qwen/Qwen-Image-2.1-Turbo@fc13ba19). The Runtime's model config holds components only,
#: so the function, not the checkpoint, names the regime.
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


class GenerateInput(ImageRequest):
    pictures: Annotated[list[Picture], msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )
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
        raise ValueError("Qwen Image decoder produced non-finite pixels")
    pixels = ((pixels / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
    if pixels.shape[-1] == 4:
        alpha = pixels[..., 3]
        alpha[alpha <= _ALPHA_NOISE] = 0
        alpha[alpha >= 255 - _ALPHA_NOISE] = 255
        if int(alpha.min()) == 255:
            pixels = pixels[..., :3]
    return Image.fromarray(pixels.contiguous().cpu().numpy())


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
    payload.unplaced(adj, "Qwen-Image-2.1 has no layout control")
    prompt = payload.rendered(adj, label if len(payload.pictures) > 1 else name_label)
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


# --- project adapters (xs-147): Qwen's words for a project's pictures, tuned by eye on this model (cl-726) ------


def _styled(text: str, style: str) -> str:
    return f"Style: {style.strip().rstrip('.')}.\n\n{text}" if style.strip() else text


def _picture_call(function: str, prompt: str, pictures: list[ProjectPicture], aspect_ratio: str | None) -> Invocation:
    request: dict[str, Any] = {"prompt": prompt, "pictures": [{"name": p.name, "image": p.image} for p in pictures]}
    return Invocation(function=function, input=request | ({"aspect_ratio": aspect_ratio} if aspect_ratio else {}))


class PictureFields(msgspec.Struct, kw_only=True):
    """What `generate`'s adapter reads of an asset. `description` is what it is, which shots read too; `instructions` how to
    draw this picture, which nothing else reads; `camera` a setup's position; `init` and `blocking` a world state's start."""

    description: str = ""
    instructions: str = ""
    style: str = ""
    aspect_ratio: str | None = None
    camera: str = ""
    from_frame: bool = False
    init: str = ""
    blocking: str = ""


class PictureOperation(Operation, kw_only=True):
    fields: PictureFields


@app.project_adapter("generate", "generate_turbo")
def adapt(op: PictureOperation) -> Invocation:
    """An asset's picture, in the words this model was tuned by eye on (cl-726), by `kind`; any other kind says what it says.
    No layout is assumed. A character or prop is on a transparent background, in the words of Qwen's model card, and a place
    with no instructions is an establishing view of the empty set. A setup's camera is in screen terms as written: bare
    camera words return a mirror (environment-angles). A storyboard is its description."""
    f = op.fields
    description, how = f.description.strip(), f.instructions.strip()
    said = "\n\n".join(filter(None, (description, how)))
    if op.kind == "environment":
        prompt = _styled(f"{description}\n\n{how or 'An establishing view of the empty set, no people.'}", f.style)
    elif op.kind in ("character", "prop"):
        # Only the background is ours, since a reference needs its subject alone.
        prompt = (f"This is an RGBA image with transparency. {_styled(said, f.style)}\n\n"
                  "The image has alpha channel and the background is transparent.")  # fmt: skip
    elif op.kind == "storyboard":
        prompt = said  # roughly how a shot is laid out: as written, no sheet, no transparency
    elif op.kind == "setup":
        if not op.reads:
            raise Refuse(f"{op.asset} is a setup that names no place")
        place, kept = op.reads[0].name, "the picture" if f.from_frame else "the image"
        opening = f"Show the place in this picture, <{place}>, with nobody in it," if f.from_frame else f"Show this exact <{place}>"
        prompt = (f"{opening} from another camera position: {f.camera.strip().rstrip('.')}. Keep every object, material, the light "
                  f"and the weather exactly as in {kept}; add nothing new, no people.")  # fmt: skip
    else:
        prompt = _styled("\n\n".join(filter(None, (said, f.init.strip(), f.blocking.strip()))), f.style)
    pictures = [{"name": spoken(f"<{read.name}>")[1:-1], "image": read.files["image"].sha256}
                for read in op.reads if "image" in read.files]  # fmt: skip
    request: dict[str, Any] = {"prompt": spoken(prompt), "pictures": pictures}
    return Invocation(function=op.function, input=request | ({"aspect_ratio": f.aspect_ratio} if f.aspect_ratio else {}))


@app.project_adapter("entity")
def adapt_entity(payload: ProjectEntity) -> Invocation:
    """A project's character, prop or place: the style, what it is, then how to draw it (`render: instructions:`). No layout is
    assumed. A character or prop is on a transparent background, in the words of Qwen's model card, and a place with no
    instructions is an establishing view of the empty set."""
    instructions = payload.instructions.strip()
    if payload.kind == "environment":
        how = instructions or "An establishing view of the empty set, no people."
        prompt = _styled(f"{payload.description.strip()}\n\n{how}", payload.style)
        return _picture_call(payload.function, prompt, payload.pictures, payload.aspect_ratio)
    # Only the background is ours, since a reference needs its subject alone.
    said = "\n\n".join(filter(None, (payload.description.strip(), instructions)))
    prompt = (f"This is an RGBA image with transparency. {_styled(said, payload.style)}\n\n"
              "The image has alpha channel and the background is transparent.")  # fmt: skip
    return _picture_call(payload.function, prompt, payload.pictures, payload.aspect_ratio)


@app.project_adapter("setup")
def adapt_setup(payload: ProjectSetup) -> Invocation:
    """A project's camera setup, in screen terms as written: bare camera words return a mirror (environment-angles)."""
    camera = payload.camera.strip().rstrip(".")
    opening = (f"Show the place in this picture, <{payload.environment}>, with nobody in it,"
               if payload.from_frame else f"Show this exact <{payload.environment}>")  # fmt: skip
    kept = "the picture" if payload.from_frame else "the image"
    prompt = (f"{opening} from another camera position: {camera}. Keep every object, material, the light and the weather "
              f"exactly as in {kept}; add nothing new, no people.")  # fmt: skip
    return _picture_call(payload.function, prompt, payload.pictures, payload.aspect_ratio)


@app.project_adapter("world_state")
def adapt_world_state(payload: ProjectWorldState) -> Invocation:
    """A project's world state: how it starts, and who and what is where."""
    prompt = _styled("\n\n".join(filter(None, (payload.init.strip(), payload.blocking.strip()))), payload.style)
    return _picture_call(payload.function, prompt, payload.pictures, payload.aspect_ratio)
