"""Krea 2: Raw (the full model, CFG) and Turbo (its 8-step distillation), text to image, and through its
community adapters (`adapters.py`) an edit of the pictures a prompt mentions, image-to-image and control maps."""

from __future__ import annotations

from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import Adjustments, App, Context, Seed, Telemetry
from cozy_runtime.fields import MENTION, AspectRatio, FileBound, Image, Prompt, Ratios
from cozy_runtime.image import Control, Guidance, ImageRequest, ImageResult, NegativePrompt, Sizes, Steps, held
from cozy_runtime.media import ImageFile
from PIL import Image as PILImage
from PIL import ImageOps

from . import quantize as quantize  # its recipes
from .adapters import (
    EDIT_PICTURES,
    KINDS,
    Adapter,
    armed,
    blacked,
    depth_map,
    edit_references,
    fitted,
    grounded,
    reference_sized,
    vlm_sized,
)
from .model import PATCH, Branch, Krea2DiT, Krea2TextEncoder, Krea2VAE, Reference

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
#: The control kinds Krea 2 has an adapter for (`adapters.KINDS`); the Runtime drops any other with a warning.
Kinds = Literal["pose", "lineart", "inpaint", "depth"]


class TurboInput(ImageRequest):
    #: Its text, and the pictures it mentions, in the order it first mentions them: an edit of them.
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text
    steps: Steps = None
    #: A picture to start from (image-to-image), fitted to the output.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    #: How much of `image` is redrawn: the noise it starts under; 1 redraws it from noise. None: 0.7.
    denoise: Annotated[float, msgspec.Meta(gt=0, le=1)] | None = None
    #: Control maps; Krea 2 reads the first. `inpaint`'s is the picture with the area to redraw transparent.
    control: list[Control[Kinds]] = msgspec.field(default_factory=list)


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


def _control(controls: list[Control[Kinds]], adj: Adjustments) -> Control[Kinds] | None:
    """The control Krea 2 reads: the first at a strength above 0."""
    usable = [control for control in controls if control.strength > 0]
    if len(usable) > 1:
        adj.warn("control", f"Krea 2 reads one control map, so it goes without the {len(usable) - 1} after the first")
    return usable[0] if usable else None


def _generate(ctx: Context, payload: TurboInput, model: Krea2DiT, text_encoder: Krea2TextEncoder, vae: Krea2VAE,
              adapters: Adapters,
              tel: Telemetry, adj: Adjustments, *, distilled: bool) -> ImageResult:
    vae.hold_vae()
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
        mapped = None
        if control is not None:
            mapped = blacked(control.image.read(mode="RGBA")) if control.kind == "inpaint" else control.image.read()
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
            before = edit_references([vae.encode_image(fit) for fit, _ in fits], [o for _, o in fits], width // PATCH,
                                     free=kind is not None)
        after: tuple[Reference, ...] = ()
        depth = None
        if kind is not None and mapped is not None:
            if kind.protocol == "concat":
                depth = vae.encode_image(depth_map(mapped, width, height))
            else:
                after = (Reference(vae.encode_image(reference_sized(mapped)), len(before) + 1),)
        initial = None
        if start is not None:
            initial = vae.encode_image(ImageOps.fit(start.convert("RGB"), (width, height), PILImage.Resampling.LANCZOS))
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
        positive = text_encoder.encode_pictures(prompt, seen, labels) if seen else text_encoder.encode(prompt)
        negative = None
        if guidance > 0:
            negative = text_encoder.encode_pictures(words, seen, labels) if seen else text_encoder.encode(words)
    isolated = kind is not None and kind.isolated
    branches = (Branch(positive, before, after, isolated),
                None if negative is None else Branch(negative, before, after, isolated))
    armed_adapters = [(adapters["edit"], 1.0)] if pictures else []
    if kind is not None and control is not None:
        armed_adapters.append((adapters[kind.slot], control.strength))
    ctx.raise_if_cancelled()
    with tel.stage("generating image"):
        latents = armed(armed_adapters, model.graph, lambda: model.denoise(
            *branches, width=width, height=height, steps=steps, guidance=guidance, seed=seed, distilled=distilled,
            progress=ctx.progress, start=initial, strength=payload.denoise or DENOISE, control=depth,
        ))
    with tel.stage("decoding image"):
        image = vae.decode(latents, width=width, height=height)
    with tel.stage("saving image"):
        asset = ImageFile.from_image(image, format="png")
    return ImageResult(asset, width, height, seed, prompt)


_RAW_MODEL = "krea/krea-2@0.1.0"
_TURBO_MODEL = "krea/krea-2-turbo@0.1.0"
#: Each slot reads its part of a Krea 2 checkpoint, and builds only that (xs-192): `model` the DiT, `text_encoder` the
#: Qwen3-VL-4B, the function's own by default (Raw's and Turbo's are byte-identical) or an abliterated one
#: (`model.text_encoder=quiho/qwen3-vl-4b-abliterated@0.1.0`), `vae` the VAE (Raw's and Turbo's are the same). A
#: model of one part alone serves its slot: a transformer with its scheduler, a text encoder with its tokenizer.
_KREA = ("transformer",)
_TEXT = ("text_encoder",)
_VAE = ("vae",)
#: Each adapter's mirror on the Hub, by slot (`adapters.py`), its one component; each is built and fetched only
#: when a request needs it.
_EDIT = "conradlocke/identity-edit@0.1.0"
_DEPTH = "patil/depth-control@0.1.0"
_POSE = "thedeoxen/pose-control@0.1.0"
_LINEART = "cierpliwy/lineart-edit@0.1.0"
_INPAINT = "cierpliwy/inpaint-edit@0.1.0"
_LORA = ("transformer",)
#: Which requests need each adapter (cl-1103, `_generate`): the edit one with a picture in the prompt, a control kind's
#: with a map of that kind at a strength above 0. Over-fetched: a kind's adapter when an earlier map is the one read.
_PICTURE = '.references.exists(r, r.type == "image")'
_MAP = "request.control.exists(c, c.strength > 0 && c.kind == "
_NEEDS = {"edit": f"request.prompt{_PICTURE}", "depth": f'{_MAP}"depth")', "pose": f'{_MAP}"pose")',
          "lineart": f'{_MAP}"lineart")', "inpaint": f'{_MAP}"inpaint")'}


@app.entrypoint(defaults={"model": _RAW_MODEL, "text_encoder": _RAW_MODEL, "vae": _RAW_MODEL, "edit": _EDIT,
                          "depth": _DEPTH, "pose": _POSE, "lineart": _LINEART, "inpaint": _INPAINT},
                components={"model": _KREA, "text_encoder": _TEXT, "vae": _VAE, "edit": _LORA, "depth": _LORA,
                            "pose": _LORA, "lineart": _LORA, "inpaint": _LORA},
                needs=_NEEDS, memoize=True)  # fmt: skip
def generate(ctx: Context, payload: GenerateInput, model: Krea2DiT, text_encoder: Krea2TextEncoder, vae: Krea2VAE,
             edit: Adapter, depth: Adapter, pose: Adapter, lineart: Adapter, inpaint: Adapter,
             tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The full model, Krea 2 Raw (or a fine-tune of it): 52 steps at guidance 3.5, 1 MP."""
    adapters = {"edit": edit, "depth": depth, "pose": pose, "lineart": lineart, "inpaint": inpaint}
    return _generate(ctx, payload, model, text_encoder, vae, adapters, tel, adj, distilled=False)


@app.entrypoint(defaults={"model": _TURBO_MODEL, "text_encoder": _TURBO_MODEL, "vae": _TURBO_MODEL, "edit": _EDIT,
                          "depth": _DEPTH, "pose": _POSE, "lineart": _LINEART, "inpaint": _INPAINT},
                components={"model": _KREA, "text_encoder": _TEXT, "vae": _VAE, "edit": _LORA, "depth": _LORA,
                            "pose": _LORA, "lineart": _LORA, "inpaint": _LORA},
                needs=_NEEDS, memoize=True)  # fmt: skip
def generate_turbo(ctx: Context, payload: TurboInput, model: Krea2DiT, text_encoder: Krea2TextEncoder, vae: Krea2VAE,
                   edit: Adapter, depth: Adapter, pose: Adapter, lineart: Adapter, inpaint: Adapter,
                   tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The fast model, Krea 2 Turbo (or a fine-tune of it): 8 steps without guidance at its fixed shift, 2048²."""
    adapters = {"edit": edit, "depth": depth, "pose": pose, "lineart": lineart, "inpaint": inpaint}
    return _generate(ctx, payload, model, text_encoder, vae, adapters, tel, adj, distilled=True)


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


# Each checkpoint is fetched only for a box that uses it: Turbo (`model`) with `turbo`, else Raw (`full`); the edit
# adapter only for a box with pictures. A box sends no control map, so it never reads a control adapter.
_BOX_NEEDS = {"full": "!request.turbo", "model": "request.turbo", "edit": f"request.instructions{_PICTURE}",
              "depth": "false", "pose": "false", "lineart": "false", "inpaint": "false"}


@app.tool(label="Image", named="image",
          defaults={"full": _RAW_MODEL, "model": _TURBO_MODEL, "text_encoder": _TURBO_MODEL, "vae": _TURBO_MODEL,
                    "edit": _EDIT, "depth": _DEPTH, "pose": _POSE, "lineart": _LINEART, "inpaint": _INPAINT},
          components={"full": _KREA, "model": _KREA, "text_encoder": _TEXT, "vae": _VAE, "edit": _LORA,
                      "depth": _LORA, "pose": _LORA, "lineart": _LORA, "inpaint": _LORA},
          needs=_BOX_NEEDS, memoize=True)  # fmt: skip
def image(ctx: Context, payload: ImageBox, full: Krea2DiT, model: Krea2DiT, text_encoder: Krea2TextEncoder,
          vae: Krea2VAE, edit: Adapter, depth: Adapter, pose: Adapter, lineart: Adapter, inpaint: Adapter,
          tel: Telemetry, adj: Adjustments) -> ImageResult:
    """Krea 2 on a canvas: Turbo, or Raw, which loads only when asked for; an edit of the pictures it mentions."""
    kind = TurboInput if payload.turbo else GenerateInput
    megapixels = payload.megapixels
    if megapixels is None and not payload.instructions.references:
        megapixels = 4
    request = kind(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=megapixels,
                   seed=payload.seed)  # fmt: skip
    adapters = {"edit": edit, "depth": depth, "pose": pose, "lineart": lineart, "inpaint": inpaint}
    return _generate(ctx, request, model if payload.turbo else full, text_encoder, vae, adapters, tel, adj,
                     distilled=payload.turbo)


# se-157 proof only, never merged: the upstream reference on the same weights (`parity.py`).
from .parity import RefInput as _RefInput, RefKrea2 as _RefKrea2, reference as _reference  # noqa: E402


@app.entrypoint(defaults={"model": _TURBO_MODEL}, components={"model": ("transformer", "text_encoder", "vae")})
def reference(ctx: Context, payload: _RefInput, model: _RefKrea2, tel: Telemetry) -> ImageResult:
    """se-157: Diffusers' Krea2Pipeline whole, beside the package."""
    return _reference(ctx, payload, model, tel)


from cozy_runtime.accel import hold_dtype as _hold_dtype  # noqa: E402
import torch as _torch  # noqa: E402


@app.entrypoint(defaults={"model": _TURBO_MODEL, "text_encoder": _TURBO_MODEL, "vae": _TURBO_MODEL},
                components={"model": _KREA, "text_encoder": _TEXT, "vae": _VAE})
def probe_bf16(ctx: Context, payload: TurboInput, model: Krea2DiT, text_encoder: Krea2TextEncoder, vae: Krea2VAE,
               tel: Telemetry, adj: Adjustments) -> ImageResult:
    """se-157: `generate_turbo` with the DiT served all-bf16 (as Krea's inference.py casts it), no adapters."""
    _hold_dtype(model.graph.components["transformer"], _torch.bfloat16)
    return _generate(ctx, payload, model, text_encoder, vae, {}, tel, adj, distilled=True)


from .model import Branch as _Branch  # noqa: E402
from .parity import RefProbe as _RefProbe, _gap  # noqa: E402
from cozy_runtime import rng as _rng  # noqa: E402
from cozy_runtime.author import uses_components as _uses  # noqa: E402
import numpy as _np  # noqa: E402
from PIL import Image as _PIL  # noqa: E402


class _ProbeDiT(Krea2DiT):
    @_uses("transformer")
    def velocity(self, text: _torch.Tensor, latents: _torch.Tensor, rows: int, columns: int) -> _torch.Tensor:
        device = _torch.device(str(self.device))
        with _torch.no_grad():
            stream = self._stream(_Branch(text), rows, columns, device)
            (v,) = self.graph.components["transformer"](latents.to(device, _torch.bfloat16),
                                                         _torch.ones(1, device=device, dtype=_torch.bfloat16), [stream])
        return v.cpu()


class _ProbeInput(msgspec.Struct, kw_only=True):
    prompt: str
    size: int = 1024
    seed: int = 7
    vae_dtype: str = "f32"
    ref_bf16: bool = True


@app.entrypoint(defaults={"model": _TURBO_MODEL, "reference": _TURBO_MODEL, "text_encoder": _TURBO_MODEL, "vae": _TURBO_MODEL},
                components={"model": _KREA, "reference": ("transformer", "text_encoder", "vae"), "text_encoder": _TEXT, "vae": _VAE})
def probe_stages(ctx: Context, payload: _ProbeInput, model: _ProbeDiT, reference: _RefProbe, text_encoder: Krea2TextEncoder,
                 vae: Krea2VAE, tel: Telemetry) -> ImageResult:
    """se-157: ours against Diffusers stage by stage on one input: text conditioning, the first-step velocity at
    t=1 (each pipeline on its own text, then ours on Diffusers' text), and the decode of Diffusers' x0."""
    if payload.vae_dtype == "fp16":
        vae.hold_vae()
    if payload.ref_bf16:
        _hold_dtype(reference.graph.components["transformer"], _torch.bfloat16)
    rows = columns = payload.size // 16
    noise = _rng.noise((1, 16, rows * 2, columns * 2), payload.seed, dtype=_torch.bfloat16)
    latents = noise.view(1, 16, rows, 2, columns, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, rows * columns, 64)
    ours_text = text_encoder.encode(payload.prompt)
    ref = reference.stages(payload.prompt, latents, rows, columns)
    tel.log("text", **_gap(ours_text[0], ref["text"]), tokens=int(ours_text.shape[1]), ref_tokens=int(ref["text"].shape[0]))
    v_ours = model.velocity(ours_text, latents, rows, columns)
    tel.log("velocity", **_gap(v_ours, ref["velocity"]))
    v_same_text = model.velocity(ref["text"][None].to(_torch.bfloat16), latents, rows, columns)
    tel.log("velocity on Diffusers' text", **_gap(v_same_text, ref["velocity"]))
    image = vae.decode(ref["x0"], width=payload.size, height=payload.size)
    ref_pixels = ((ref["image"][0].clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1) * 255).round().to(_torch.uint8).permute(1, 2, 0).numpy()
    ours_pixels = _np.asarray(image)
    mse = float(((ours_pixels.astype(_np.float64) - ref_pixels) ** 2).mean())
    tel.log("decode", psnr=10 * _np.log10(255**2 / max(mse, 1e-12)), max=int(_np.abs(ours_pixels.astype(int) - ref_pixels).max()))
    return ImageResult(ImageFile.from_image(_PIL.fromarray(ref_pixels), format="png"), payload.size, payload.size,
                       payload.seed, payload.prompt)
