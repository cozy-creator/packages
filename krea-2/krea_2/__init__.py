"""Krea 2: Raw (the full model, CFG) and Turbo (its 8-step distillation), text to image, and through its
community adapters (`adapters.py`) an edit of the pictures a prompt mentions, image-to-image and control maps."""

from __future__ import annotations

from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import Adjustments, App, Context, Outputs, Seed, Telemetry
from cozy_runtime.fields import MENTION, AspectRatio, FileBound, Image, Prompt, Ratios
from cozy_runtime.image import Guidance, ImageRequest, ImageResult, NegativePrompt, Sizes, Steps, held
from cozy_runtime.media import ImageFile
from PIL import Image as PILImage
from PIL import ImageOps

from . import quantize as quantize  # its recipes
from .adapters import (
    EDIT_PICTURES,
    KINDS,
    Adapter,
    armed,
    depth_map,
    edit_references,
    fitted,
    grounded,
    reference_sized,
    vlm_sized,
)
from .model import PATCH, Branch, Krea2, Reference

app = App()

#: Krea: Raw is trained to 1 MP, Turbo makes 1-4 MP; 256 px sides at the least, on the 16 px grid.
MIN_MEGAPIXELS = 0.25
MAX_MEGAPIXELS = 4.0
SIZES = Sizes((MIN_MEGAPIXELS, MAX_MEGAPIXELS), (256, 4096), PATCH, "Krea 2's 0.25-4 MP")
#: With an adapter's pictures: 2 MP at most (Identity Edit's model card: "Generate at ≤2MP").
PICTURE_SIZES = Sizes((MIN_MEGAPIXELS, 2.0), (256, 4096), PATCH, "Krea 2's adapters' 0.25-2 MP")
#: Krea's settings (steps, guidance, megapixels): Raw 52 steps at guidance 3.5 at its 1 MP, Turbo 8 steps
#: without guidance at its 2048² (`krea-ai/krea-2` README).
RAW, TURBO = (52, 3.5, 1.0), (8, 0.0, 4.0)
#: How much of a starting picture is redrawn by default (`denoise`).
DENOISE = 0.7


class Control(msgspec.Struct, kw_only=True):
    """A control map and how strongly it holds: a depth map (near is white), an OpenPose or DWPose skeleton,
    white lines on black (`lineart`), or a picture whose region to repaint is black (`inpaint`)."""

    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
    kind: Literal["depth", "pose", "lineart", "sketch", "canny", "inpaint"]
    #: The adapter's LoRA strength (Patil's `--lora-scale`, the node's LoRA weight).
    strength: Annotated[float, msgspec.Meta(ge=0, le=2)] = 1.0


class TurboInput(ImageRequest):
    #: Its text, and the pictures it mentions, in the order it first mentions them: an edit of them.
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text
    steps: Steps = None
    #: A picture to start from (image-to-image), fitted to the output.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    #: How much of `image` is redrawn: the noise it starts under; 1 redraws it from noise. None: 0.7.
    denoise: Annotated[float, msgspec.Meta(gt=0, le=1)] | None = None
    #: Control maps; Krea 2 reads one, of a kind it has an adapter for (`adapters.KINDS`).
    control: list[Control] = msgspec.field(default_factory=list)


class GenerateInput(TurboInput):
    guidance: Guidance = None
    negative_prompt: NegativePrompt = None


#: The adapters each function holds, by slot: Identity Edit V2 for mentioned pictures, a LoRA per control kind.
Adapters = dict[str, Adapter]


def _said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[PILImage.Image]]:
    """The prompt's text, each mention as its name (`kyogre.2` as `kyogre 2`: the edit adapters' captions
    name no pictures), and the pictures the edit takes, the first `EDIT_PICTURES` mentioned."""
    keys = list(prompt.references)
    if len(keys) > EDIT_PICTURES:
        adj.warn("prompt", f"Krea 2's edit takes {EDIT_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[EDIT_PICTURES:])}")
    said = MENTION.sub(lambda found: found[1].replace(".", " "), prompt.text)
    return said, [prompt.references[key].file.read() for key in keys[:EDIT_PICTURES]]


def _control(controls: list[Control], adj: Adjustments) -> Control | None:
    """The control Krea 2 reads: the first of a kind it has an adapter for, at a strength above 0."""
    usable = []
    for control in controls:
        if control.kind not in KINDS:
            adj.warn("control", f"ignored: Krea 2 has no {control.kind} adapter")
        elif control.strength > 0:
            usable.append(control)
    if len(usable) > 1:
        adj.warn("control", f"Krea 2 reads one control map, so it goes without the {len(usable) - 1} after the first")
    return usable[0] if usable else None


def _generate(ctx: Context, payload: TurboInput, model: Krea2, adapters: Adapters, out: Outputs, tel: Telemetry,
              adj: Adjustments, *, distilled: bool) -> ImageResult:
    model.hold_vae()
    prompt, pictures = _said(payload.prompt, adj)
    control = _control(payload.control, adj)
    kind = KINDS[control.kind] if control is not None else None
    steps, guidance, megapixels = TURBO if distilled else RAW
    steps = held(adj, "steps", payload.steps or steps, 1, 100, "Krea 2's 1-100 steps")
    words = ""
    if isinstance(payload, GenerateInput):
        guidance = held(adj, "guidance", guidance if payload.guidance is None else payload.guidance, 0.0, 20.0,
                        "Krea 2's guidance 0-20")
        words = payload.negative_prompt or ""
        if guidance == 0 and words:
            adj.warn("negative_prompt", "ignored: guidance 0 runs no negative branch")
    if payload.denoise is not None and payload.image is None:
        adj.warn("denoise", "ignored: there is no image to start from")
    seed = payload.drawn_seed()
    with tel.stage("reading pictures"):
        mapped = control.image.read() if control is not None else None
        start = payload.image.read() if payload.image is not None else None
    first = next((p.size for p in (*pictures, mapped, start) if p is not None), None)
    limits = PICTURE_SIZES if pictures or mapped is not None else SIZES
    width, height = payload.size(adj, first, limits, megapixels=megapixels)
    tel.log("adapters", edit=len(pictures), control=control.kind if control is not None else "",
            image_to_image=start is not None)
    ctx.raise_if_cancelled()
    with tel.stage("encoding pictures"):
        fits = [fitted(picture, width, height) for picture in pictures]
        before: tuple[Reference, ...] = ()
        if fits:
            before = edit_references([model.encode_image(fit) for fit, _ in fits], [o for _, o in fits], width // PATCH)
        after: tuple[Reference, ...] = ()
        depth = None
        if kind is not None and mapped is not None:
            if kind.protocol == "concat":
                depth = model.encode_image(depth_map(mapped, width, height))
            else:
                after = (Reference(model.encode_image(reference_sized(mapped)), len(before) + 1),)
        initial = None
        if start is not None:
            initial = model.encode_image(ImageOps.fit(start.convert("RGB"), (width, height), PILImage.Resampling.LANCZOS))
    with tel.stage("encoding prompt"):
        # Identity Edit reads its pictures with the prompt; ai-toolkit's edits their maps, each `Picture N:`.
        # Both read them with the negative prompt too, and keep their pictures in its sequence (the node's
        # grounded negative, ai-toolkit's sampler); with both, the control's map goes as VAE tokens alone.
        if pictures:
            seen, labels = tuple(grounded(picture) for picture in pictures), False
        elif after:
            assert mapped is not None
            seen, labels = (vlm_sized(mapped),), True
        else:
            seen, labels = (), False
        positive = model.encode_pictures(prompt, seen, labels) if seen else model.encode(prompt)
        negative = None
        if guidance > 0:
            negative = model.encode_pictures(words, seen, labels) if seen else model.encode(words)
    isolated = kind is not None and kind.isolated
    branches = (Branch(positive, before, after, isolated),
                None if negative is None else Branch(negative, before, after, isolated))
    armed_adapters = [(adapters["edit"], 1.0)] if pictures else []
    if kind is not None and control is not None:
        armed_adapters.append((adapters[kind.slot], control.strength))
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        transformer = model.graph.components["transformer"]
        latents = armed(armed_adapters, transformer, lambda: model.denoise(
            *branches, width=width, height=height, steps=steps, guidance=guidance, seed=seed, distilled=distilled,
            progress=ctx.progress, start=initial, strength=payload.denoise or DENOISE, control=depth,
        ))
    with tel.stage("decoding image"):
        image = model.decode(latents, width=width, height=height)
    with tel.stage("saving image"):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


_RAW_MODEL = "krea/krea-2@0.1.0"
_TURBO_MODEL = "krea/krea-2-turbo@0.1.0"
_KREA = ("transformer", "text_encoder", "vae")
#: Each adapter's mirror on the Hub, by slot (`adapters.py`); each is built and fetched only when a request
#: needs it.
_ADAPTERS = {
    "edit": "conradlocke/identity-edit@0.1.0",
    "depth": "patil/depth-control@0.1.0",
    "pose": "thedeoxen/pose-control@0.1.0",
    "lineart": "cierpliwy/lineart-edit@0.1.0",
    "inpaint": "cierpliwy/inpaint-edit@0.1.0",
}
_ADAPTER_COMPONENTS = {slot: ("transformer",) for slot in _ADAPTERS}


@app.entrypoint(defaults={"model": _RAW_MODEL, **_ADAPTERS}, components={"model": _KREA, **_ADAPTER_COMPONENTS},
                on_demand=tuple(_ADAPTERS), memoize=True)  # fmt: skip
def generate(ctx: Context, payload: GenerateInput, model: Krea2, edit: Adapter, depth: Adapter, pose: Adapter,
             lineart: Adapter, inpaint: Adapter, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The full model, Krea 2 Raw (or a fine-tune of it): 52 steps at guidance 3.5, 1 MP."""
    adapters = {"edit": edit, "depth": depth, "pose": pose, "lineart": lineart, "inpaint": inpaint}
    return _generate(ctx, payload, model, adapters, out, tel, adj, distilled=False)


@app.entrypoint(defaults={"model": _TURBO_MODEL, **_ADAPTERS}, components={"model": _KREA, **_ADAPTER_COMPONENTS},
                on_demand=tuple(_ADAPTERS), memoize=True)  # fmt: skip
def generate_turbo(ctx: Context, payload: TurboInput, model: Krea2, edit: Adapter, depth: Adapter, pose: Adapter,
                   lineart: Adapter, inpaint: Adapter, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The fast model, Krea 2 Turbo (or a fine-tune of it): 8 steps without guidance at its fixed shift, 2048²."""
    adapters = {"edit": edit, "depth": depth, "pose": pose, "lineart": lineart, "inpaint": inpaint}
    return _generate(ctx, payload, model, adapters, out, tel, adj, distilled=True)


class ImageBox(msgspec.Struct, kw_only=True):
    #: What to make, and the pictures it mentions: an edit of them (`_said`).
    instructions: Prompt[Image] = Prompt()
    #: Krea 2 Turbo's 8 steps, else Raw's 52 with CFG (`_generate`).
    turbo: bool = True
    #: `ImageRequest.size`; `SIZES.fit` holds it to 1:16 through 16:1.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `ImageRequest.size`, through `SIZES.pixels`: None is the first picture's size, else Turbo's 4 MP (README,
    #: Sizes); an edit makes 2 MP at most.
    megapixels: Literal[1, 2, 4] | None = None
    #: `_generate`'s `drawn_seed`.
    seed: Seed | None = None


# Each checkpoint is built, and its weights fetched, only when a box asks for it; the edit adapter only for a box
# with pictures.
@app.tool(label="Image", named="image", defaults={"full": _RAW_MODEL, "model": _TURBO_MODEL, **_ADAPTERS},
          components={"full": _KREA, "model": _KREA, **_ADAPTER_COMPONENTS},
          on_demand=("full", "model", *_ADAPTERS), memoize=True)  # fmt: skip
def image(ctx: Context, payload: ImageBox, full: Krea2, model: Krea2, edit: Adapter, depth: Adapter, pose: Adapter,
          lineart: Adapter, inpaint: Adapter, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Krea 2 on a canvas: Turbo, or Raw, which loads only when asked for; an edit of the pictures it mentions."""
    kind = TurboInput if payload.turbo else GenerateInput
    megapixels = payload.megapixels
    if megapixels is None and not payload.instructions.references:
        megapixels = 4
    request = kind(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=megapixels,
                   seed=payload.seed)  # fmt: skip
    adapters = {"edit": edit, "depth": depth, "pose": pose, "lineart": lineart, "inpaint": inpaint}
    return _generate(ctx, request, model if payload.turbo else full, adapters, out, tel, adj, distilled=payload.turbo)
