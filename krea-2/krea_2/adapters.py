"""Krea 2's community adapters: each a LoRA over the DiT that a function arms for one request, and the way
its pictures enter the sequence, which is its own (its model card and its ComfyUI nodes).

- Identity Edit and General Edit (conradlocke, `comfyui-krea2edit` 2.0, `krea2edit-trainer` 56bf37b,
  docs/GEOMETRY.md): each reference fitted inside the output's grid and VAE-encoded, in the sequence before
  the target at RoPE frame 1, 2, with the step's modulation; the prompt read by Qwen3-VL after the pictures
  themselves, their longest side at most 768 px (`grounding_px`).
- ai-toolkit's edit LoRAs (ostris `ComfyUI-Krea2-Ostris-Edit` 7756566, `ai-toolkit` krea2 `model_kwargs.edit`):
  each picture VAE-encoded at up to 1 MP on its own grid, after the target at frame 1, 2, … and time 0; the
  prompt read after `Picture N: ` and each picture at up to 384² pixels; the pose, lineart and inpaint LoRAs'
  workflows keep the pictures attending only among themselves (the node's `kv_cache`).
- Patil's depth control LoRA (`Krea-2-controlnet` 909682a): the depth map VAE-encoded at the output's size,
  each of its tokens read beside its image token through an input projection of 128 features, not 64.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch
from cozy_runtime.accel import lora
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
from PIL import Image, ImageOps
from torch import nn
from transformers.initialization import no_init_weights

from .model import PATCH, SLOTS, Reference, Transformer, as_stored

T = TypeVar("T")
#: An uploaded Krea 2 LoRA's one component (TensorFS's `krea2.lora`), its modules under Diffusers' names.
COMPONENT = "transformer"
#: How an adapter takes its pictures (module docstring).
Protocol = Literal["identity", "ai-toolkit", "concat"]


class Factors(nn.Module):
    """One linear's `lora_A` [r, in] and `lora_B` [out, r], applied as `scale · B·A`."""

    def __init__(self, inputs: int, outputs: int, rank: int, scale: float) -> None:
        super().__init__()
        self.lora_A = nn.Linear(inputs, rank, bias=False, dtype=torch.bfloat16)
        self.lora_B = nn.Linear(rank, outputs, bias=False, dtype=torch.bfloat16)
        self.unit = scale
        self.scale = scale

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> None:
        lora.accumulate(x, out, self.lora_A.weight, self.lora_B.weight, self.scale)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self.lora_A.weight, self.lora_B.weight, self.scale


class AdapterGraph:
    """An uploaded Krea 2 LoRA: factors at each of its `target_modules` (PEFT's config, which TensorFS
    writes with `lora_alpha` the commonest α), and a control LoRA's expanded input projection, which
    TensorFS keeps whole as `img_in.modules_to_save`."""

    def __init__(self, config: Config) -> None:
        adapter = config.section(COMPONENT) if COMPONENT in config.keys() else config
        settings = adapter.mapping()
        rank = adapter.as_int("r")
        scale = adapter.as_float("lora_alpha", float(rank)) / rank
        with torch.device("meta"):  # the class defaults are Krea 2's geometry
            dit = Krea2Transformer2DModel()
        linears = {name: module for name, module in dit.named_modules() if isinstance(module, nn.Linear)}
        named = [str(target) for target in settings["target_modules"]]
        targets = sorted({name for name in linears for target in named if name == target or name.endswith("." + target)})
        root = nn.Module()
        self.sites: dict[str, Factors] = {}
        with no_init_weights():
            for path in targets:
                linear = linears[path]
                factors = Factors(linear.in_features, linear.out_features, rank, scale)
                _add(root, path, factors)
                self.sites[path] = factors
            self.projection: nn.Linear | None = None
            if _stored(config, f"{COMPONENT}.img_in.modules_to_save.weight"):
                image = linears["img_in"]
                self.projection = nn.Linear(2 * image.in_features, image.out_features, dtype=torch.float32)
                _add(root, "img_in.modules_to_save", self.projection)
        as_stored(config, COMPONENT, root)  # factors in their stored dtype (bf16, or Patil's F32)
        self.components: dict[str, nn.Module] = {COMPONENT: root}


def _stored(config: Config, key: str) -> bool:
    """Whether the checkpoint stores `key`: the Runtime refuses a missing one (`config_tensor_missing`)."""
    try:
        return bool(config.tensor_dtype(key, default=""))
    except Exception as exc:  # the Runtime's ConformanceError, which `cozy_runtime.author` doesn't export
        if getattr(exc, "code", None) == "config_tensor_missing":
            return False
        raise


def _add(root: nn.Module, path: str, module: nn.Module) -> None:
    parent = root
    *parents, leaf = path.split(".")
    for name in parents:
        if name not in parent._modules:
            parent.add_module(name, nn.Module())
        parent = parent.get_submodule(name)
    parent.add_module(leaf, module)


class Adapter(Model[AdapterGraph]):
    """A Krea 2 adapter's LoRA, kept at its stored dtype."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: AdapterGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(AdapterGraph)

    @uses_components(COMPONENT)
    def applied(self, transformer: Transformer, slot: int, strength: float, body: Callable[[], T]) -> T:
        """`body` with this LoRA armed on the DiT in `slot` at `strength`: Runtime bakes it into the resident
        weights, else its hooks add it. A control LoRA's projection reads the control map at full weight
        (Patil's `--lora-scale` scales the LoRA alone)."""
        for factors in self.graph.sites.values():
            factors.scale = factors.unit * strength
        overlays = transformer.adapter_overlays
        overlays[slot] = self.graph.sites
        control = self.graph.projection is not None
        if control:  # the graph, one object per adapter: a recorded step's read guard holds across calls
            transformer.control_adapter = self.graph
        try:
            return body()
        finally:
            overlays[slot] = None
            if control:
                transformer.control_adapter = None


def armed(adapters: list[tuple[Adapter, float]], transformer: Transformer, body: Callable[[], T]) -> T:
    """`body` with each adapter armed in its own slot (at most `SLOTS`)."""
    if not adapters:
        return body()
    (adapter, strength), rest = adapters[0], adapters[1:]
    slot = SLOTS - len(adapters)
    return adapter.applied(transformer, slot, strength, lambda: armed(rest, transformer, body))


# Identity Edit (krea2edit) -----------------------------------------------------------------------------

#: The longest side Qwen3-VL reads a reference at (the node's `grounding_px`; trained at 384-768).
GROUNDING = 768
#: Identity Edit V2's fidelity dial for the person (the last reference): the target's attention to its
#: tokens doubled (model card: `ref_boost` 2 for one person, 2-3 beside a scene at 1).
BOOST = 2.0
#: References it was trained with (`krea2edit-trainer` refuses more).
EDIT_PICTURES = 2


def grounded(picture: Image.Image) -> Image.Image:
    """The reference as Qwen3-VL reads it: its longest side at most `GROUNDING` (Lanczos, the trainer's)."""
    longest = max(picture.size)
    if longest <= GROUNDING:
        return picture
    scale = GROUNDING / longest
    return picture.resize((max(1, round(picture.width * scale)), max(1, round(picture.height * scale))),
                          Image.Resampling.LANCZOS)


def fitted(picture: Image.Image, width: int, height: int) -> tuple[Image.Image, tuple[float, float]]:
    """The node's `fit` protocol (`_fit_encode_image`, the trainer's `_fit_prep`): within 8% of the output's
    ratio, the least center crop to it at the output's size; else the picture inside the output's grid,
    center-cropped so its fitted side lands on the 16 px grid at the same scale. Returns the picture to
    encode and its grid's origin in the output's, in tokens (centered, fractional)."""
    ih, iw = picture.height, picture.width
    scale = min(height / ih, width / iw)
    if ih * scale >= height * 0.92 and iw * scale >= width * 0.92:
        cover = max(height / ih, width / iw)
        crop_h, crop_w = min(ih, round(height / cover)), min(iw, round(width / cover))
        new_h, new_w = height, width
    else:
        new_h = min(max(PATCH, int(ih * scale) // PATCH * PATCH), max(PATCH, height // PATCH * PATCH))
        new_w = min(max(PATCH, int(iw * scale) // PATCH * PATCH), max(PATCH, width // PATCH * PATCH))
        crop_h, crop_w = min(ih, max(1, round(new_h / scale))), min(iw, max(1, round(new_w / scale)))
    top, left = (ih - crop_h) // 2, (iw - crop_w) // 2
    picture = picture.crop((left, top, left + crop_w, top + crop_h))
    picture = picture.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return picture, (max(0.0, (height - new_h) / PATCH / 2), max(0.0, (width - new_w) / PATCH / 2))


def edit_references(latents: list[torch.Tensor], origins: list[tuple[float, float]], columns: int,
                    free: bool) -> tuple[Reference, ...]:
    """Identity Edit V2's placement (model card): one reference pinned on the output (`aligned`, trained), its
    fidelity at `BOOST`, or beside the canvas when the layout is the control map's (`free`: the person restaged
    freely); with two, the scene (the first) pinned and the person (the second) beside the canvas (`beside`,
    the most natural composites), boosted."""
    if len(latents) == 1:
        origin = (origins[0][0], float(columns)) if free else origins[0]
        return (Reference(latents[0], 1, origin, BOOST),)
    scene, person = latents
    return (Reference(scene, 1, origins[0]), Reference(person, 2, (origins[1][0], float(columns)), BOOST))


# ai-toolkit's edits (ostris) ---------------------------------------------------------------------------

#: Pixels Qwen3-VL reads each picture at, at most (`vlm_max_pixels`), and its shortest side at least.
VLM_PIXELS, VLM_SIDE = 384 * 384, 28
#: Pixels each picture's VAE tokens carry, at most (`control_image_max_pixels`).
REFERENCE_PIXELS = 1024 * 1024


def blacked(picture: Image.Image) -> Image.Image:
    """An inpaint map (the picture with the area to redraw transparent) as Cierpliwy's inpaint LoRA reads it: that
    area black. An opaque picture is unchanged."""
    black = Image.new("RGBA", picture.size, (0, 0, 0, 255))
    return Image.alpha_composite(black, picture.convert("RGBA")).convert("RGB")


def vlm_sized(picture: Image.Image) -> Image.Image:
    """`_prep_vlm_images`: down (never up) to `VLM_PIXELS`, bicubic."""
    scale = min(1.0, math.sqrt(VLM_PIXELS / (picture.width * picture.height)))
    size = (max(round(picture.width * scale), VLM_SIDE), max(round(picture.height * scale), VLM_SIDE))
    return picture if size == picture.size else picture.resize(size, Image.Resampling.BICUBIC)


def reference_sized(picture: Image.Image) -> Image.Image:
    """`_encode_ref_latents`: down to `REFERENCE_PIXELS` when larger, each side then rounded to 16 px."""
    width, height = float(picture.width), float(picture.height)
    if width * height > REFERENCE_PIXELS:
        height = math.sqrt(REFERENCE_PIXELS * height / width)
        width = height * picture.width / picture.height
    size = (max(PATCH, round(width / PATCH) * PATCH), max(PATCH, round(height / PATCH) * PATCH))
    return picture if size == picture.size else picture.resize(size, Image.Resampling.BILINEAR)


# Patil's depth control ---------------------------------------------------------------------------------


def depth_map(picture: Image.Image, width: int, height: int) -> Image.Image:
    """The depth map as the LoRA was trained on it (`DepthLoRAPipeline`): one channel stretched to 0-1
    (near is white), on three, at the output's size (resized to cover, centered)."""
    gray = ImageOps.autocontrast(picture.convert("L"), cutoff=0)
    return ImageOps.fit(gray, (width, height), Image.Resampling.LANCZOS).convert("RGB")


@dataclass(frozen=True)
class Kind:
    """A control kind's adapter: its slot, how it takes its map, and whether its pictures attend only among
    themselves."""

    slot: str
    protocol: Protocol
    isolated: bool = False


#: Each control kind Krea 2 has an adapter for (the request's `Kinds`); `sketch` and `canny` have none yet. Each
#: ai-toolkit adapter's own workflow runs it with `Krea2OstrisEditModelPatch`'s `kv_cache` on: its map attends only
#: to itself.
KINDS = {
    "depth": Kind("depth", "concat"),
    "pose": Kind("pose", "ai-toolkit", isolated=True),
    "lineart": Kind("lineart", "ai-toolkit", isolated=True),
    "inpaint": Kind("inpaint", "ai-toolkit", isolated=True),
}
