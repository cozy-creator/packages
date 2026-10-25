"""HiDream-O1-Image: Qwen3-VL-8B as a pixel-space flow model, no VAE and no separate text encoder.

The text model, vision tower and processor are Transformers' Qwen3-VL; the 32 px patch embedder,
the timestep token and the pixel head are HiDream's. Sequences, positions, reference preparation
and sampling follow upstream's `models/pipeline.py`, `models/utils.py` and
`models/qwen3_vl_transformers.py` (HiDream-ai/HiDream-O1-Image, main 2c2d29f). Text and condition
tokens attend causally among themselves and never to what follows them, so their keys and values
are computed once a request; the timestep token and the target's and references' patches attend to
everything, every step. Runtime constructs and fills the modules and owns their residency.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cozy_runtime import rng, vlm
from cozy_runtime.attention import Attention
from cozy_runtime.author import Config, Loader, Model, uses_components
from PIL import Image, ImageDraw
from transformers import (
    Qwen3VLConfig,
)
from transformers.initialization import no_init_weights
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLVisionModel

from .sampling import Euler, Flash, UniPC

COMPONENT = "model"
PATCH = 32
#: Upstream's noise scale: the initial noise is this many standard normals, and the Dev sampler
#: noises each prediction again at its own scale.
NOISE_SCALE = 8.0
#: Where the target image's positions start (`get_rope_index_fix_point`).
FIX_POINT = 4096
CONDITION_IMAGE_SIZE = 384
T_EPS = 1e-3
#: `generate`'s CFG stops for the last fifth of the steps (limited-interval guidance,
#: Kynkäänniemi et al. 2024). There each patch's colour settles, and CFG 5 multiplies the pixel
#: head's small per-patch colour error into a visible 32 px grid (upstream #1).
GUIDED = 0.8
BOI, TMS = "<|boi_token|>", "<|tms_token|>"


class TimestepEmbedder(torch.nn.Module):
    def __init__(self, hidden: int, frequencies: int = 256) -> None:
        super().__init__()
        self.frequencies = frequencies
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequencies, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequencies // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = (t * 1000)[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class PatchEmbed(torch.nn.Module):
    """A raw 32x32 RGB patch, through a 1024-wide bottleneck, into the LM's width."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj1 = torch.nn.Linear(PATCH * PATCH * 3, hidden // 4, bias=False)
        self.proj2 = torch.nn.Linear(hidden // 4, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class PixelHead(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden, PATCH * PATCH * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Body(torch.nn.Module):
    """The checkpoint's `model.*`: Qwen3-VL's vision tower and text model, and HiDream's three
    additions. Its `lm_head` is never built, so never read."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        hidden = config.text_config.hidden_size
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config, dtype=torch.bfloat16)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config, dtype=torch.bfloat16)
        self.t_embedder1 = TimestepEmbedder(hidden)
        self.x_embedder = PatchEmbed(hidden)
        self.final_layer2 = PixelHead(hidden)


class Root(torch.nn.Module):
    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.model = Body(config)


class O1Graph:
    def __init__(self, config: Config) -> None:
        mapping = {key: value for key, value in config.mapping().items() if not key.startswith("_")}
        with no_init_weights():
            root = Root(Qwen3VLConfig(**mapping))
        for module in (root.model.t_embedder1, root.model.x_embedder, root.model.final_layer2):
            module.to(torch.bfloat16)
        text = root.model.language_model.config
        for layer in root.model.language_model.layers:
            # The denoising steps' attention, a site whose kernel the Runtime picks.
            layer.self_attn.site = Attention(text.num_attention_heads, text.head_dim)
        self.components: dict[str, torch.nn.Module] = {COMPONENT: root.eval()}


# ---------------------------------------------------------------- references and sizes


def grid_size(width: float, height: float, budget: float, patch: int = PATCH) -> tuple[int, int]:
    """Upstream's `resize_pilimage` size: `width`:`height` scaled to `budget` pixels, each side
    rounded or floored to the `patch` grid, the largest of those within the budget (the smallest
    if none is); at least one patch a side."""
    scale = math.sqrt(budget / (width * height))
    w, h = width * scale, height * scale
    candidates = sorted(
        [(round(w) // patch * patch, round(h) // patch * patch), (round(w) // patch * patch, math.floor(h) // patch * patch),
         (math.floor(w) // patch * patch, round(h) // patch * patch),
         (math.floor(w) // patch * patch, math.floor(h) // patch * patch)],
        key=lambda c: c[0] * c[1],
        reverse=True,
    )
    chosen = next((c for c in candidates if c[0] * c[1] <= budget), candidates[-1])
    return max(chosen[0], patch), max(chosen[1], patch)


def area_resize(image: Image.Image, size: float, patch: int = PATCH) -> Image.Image:
    """Upstream's `resize_pilimage`: halved by 2x2 boxes while its short side is at least twice
    `size`, then bicubic to `grid_size` of `size`² pixels, centre-cropped to it."""
    while min(*image.size) >= 2 * size:
        image = image.resize((image.width // 2, image.height // 2), resample=Image.Resampling.BOX)
    width, height = image.size
    w, h = grid_size(width, height, size * size, patch)
    if width / w < height / h:
        tall = round(height / (width / w))
        image = image.resize((w, tall), resample=Image.Resampling.BICUBIC)
        top = (tall - h) // 2
        return image.crop((0, top, w, top + h))
    wide = round(width / (height / h))
    image = image.resize((wide, h), resample=Image.Resampling.BICUBIC)
    left = (wide - w) // 2
    return image.crop((left, 0, left + w, h))


def reference_size(target_side: int, count: int) -> int:
    """The side of the square area each of `count` references is resized to, by the target's
    longer side."""
    if count == 1:
        return target_side
    if count == 2:
        return target_side * 48 // 64
    if count <= 4:
        return target_side // 2
    if count <= 8:
        return target_side * 24 // 64
    return target_side // 4


#: Upstream's layout (`models/utils.py`): the five largest boxes are drawn, each in its colour, on a
#: black canvas of the output's size, which follows the references; each reference is framed in its
#: box's colour.
MAX_BOXES = 5
COLOURS = ((255, 0, 0), (0, 180, 0), (0, 0, 255), (204, 180, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0),
           (0, 128, 0), (0, 0, 128), (128, 128, 0))


def canvas_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    """`[x0, y0, x1, y1]` (0-1) in pixels as upstream rounds and clamps them; None if it vanishes."""
    x0, x1 = (max(0, min(width - 1, round(v * width))) for v in (box[0], box[2]))
    y0, y1 = (max(0, min(height - 1, round(v * height))) for v in (box[1], box[3]))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def placed(references: list[Image.Image], boxes: list[tuple[int, int, int, int] | None], width: int,
           height: int, side: float) -> list[Image.Image]:
    """Upstream's `create_layout_reference_images` and `generate_image`'s resize: each reference
    resized to `side`² and framed in its box's colour (a reference without a drawn box takes an
    unused one), then the layout canvas. `boxes` are pixel boxes on the `width` x `height` output."""
    ranked = sorted((i for i, box in enumerate(boxes) if box), key=lambda i: _area(boxes[i]), reverse=True)
    line = int(math.sqrt(width * height) * 0.05)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    colours: dict[int, tuple[int, int, int]] = {}
    for rank, index in enumerate(ranked[:MAX_BOXES]):
        colours[index] = COLOURS[rank % len(COLOURS)]
        draw.rectangle(boxes[index], outline=colours[index], width=max(line - rank * max(1, line // MAX_BOXES), 5))
    used = set(colours.values())
    framed = []
    for index, reference in enumerate(references):
        colour = colours.get(index)
        if colour is None:
            colour = next(c for c in (COLOURS[index % len(COLOURS)], *COLOURS) if c not in used)
            used.add(colour)
        image = area_resize(reference.convert("RGB"), side)
        frame = ImageDraw.Draw(image)
        w, h = image.size
        for t in range(int(math.sqrt(w * h) * 0.04)):
            frame.rectangle([t, t, w - 1 - t, h - 1 - t], outline=colour)
        framed.append(area_resize(image, side))
    return [*framed, area_resize(canvas, side)]


def _area(box: tuple[int, int, int, int] | None) -> int:
    return (box[2] - box[0]) * (box[3] - box[1]) if box else 0


def condition_image(reference: Image.Image, count: int) -> Image.Image:
    """The vision tower's copy of a prepared reference: its ratio at 384² (288², 192² for more
    than four or eight), sides floored to 32, LANCZOS."""
    side = CONDITION_IMAGE_SIZE if count <= 4 else CONDITION_IMAGE_SIZE * 48 // 64 if count <= 8 else 192
    ratio = reference.width / reference.height
    width = math.sqrt(side * side * ratio)
    w, h = max(PATCH, int(width / 32) * 32), max(PATCH, int(width / ratio / 32) * 32)
    return reference.resize((w, h), resample=Image.Resampling.LANCZOS)


def to_rows(x: torch.Tensor) -> torch.Tensor:
    """[3, height, width] as [1, n, 3·32·32] rows of patches, row-major, (C, p, p) within a row."""
    c, h, w = x.shape
    return x.view(c, h // PATCH, PATCH, w // PATCH, PATCH).permute(1, 3, 0, 2, 4).reshape(1, -1, c * PATCH * PATCH)


def patches(image: Image.Image) -> torch.Tensor:
    """[-1, 1] RGB as rows of 32x32x3 patches."""
    pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1)
    return to_rows(pixels.float().mul(1.0 / 255).sub(0.5).div(0.5))[0]  # torchvision's ToDtype(scale), Normalize


def unpatch(rows: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[1, n, 3·32·32] patches back to [3, height, width]."""
    h, w = height // PATCH, width // PATCH
    return rows[0].view(h, w, 3, PATCH, PATCH).permute(2, 0, 3, 1, 4).reshape(3, height, width)


# ---------------------------------------------------------------- sequences


def rope_index(tokens: list[int], grids: list[tuple[int, int, int]], skips: list[int], image: int) -> torch.Tensor:
    """Upstream's `get_rope_index_fix_point` over one sequence of text and images: [3, T]. Text
    counts up; a vision-tower image sits on its own (t, h, w) grid after the text before it; the
    first image whose `skip` is 1 (the target) starts at `FIX_POINT`, and what follows it counts on
    from its largest position."""
    pieces: list[torch.Tensor] = []
    fix, st = FIX_POINT, 0
    for index, (t, h, w) in enumerate(grids):
        ed = tokens.index(image, st)
        text = max(0, ed - st - skips[index])
        at = int(pieces[-1].max()) + 1 if pieces else 0
        pieces.append(torch.arange(text).view(1, -1).expand(3, -1) + at)
        grid = torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")).flatten(1)
        if skips[index]:
            if fix > 0:
                fix -= at
            pieces.append(grid + fix + at)
            fix = 0
        else:
            pieces.append(grid + text + at)
        st = ed + t * h * w
    if st < len(tokens):
        at = int(pieces[-1].max()) + 1 if pieces else 0
        pieces.append(torch.arange(len(tokens) - st).view(1, -1).expand(3, -1) + at)
    return torch.cat(pieces, dim=1)


@dataclass
class Sequence:
    """One caption's sequence: its text ids (the timestep token last), the vision tower's inputs,
    the positions of every token (text, then target, then references) and the causal prefix's
    length (everything before the timestep token)."""

    ids: torch.Tensor
    pixel_values: torch.Tensor | None
    grid: torch.Tensor | None
    positions: torch.Tensor
    prefix: int


def sequence(
    processor: Any, caption: str, conditions: list[Image.Image], references: list[tuple[int, int]],
    rows: int, columns: int,
) -> Sequence:
    """`references` are the prepared references' patch grids (rows, columns); `conditions` the
    vision tower's copies of them."""
    tokenizer = processor.tokenizer
    image, start = tokenizer.convert_tokens_to_ids("<|image_pad|>"), tokenizer.convert_tokens_to_ids("<|vision_start|>")
    tail = tokenizer.encode(BOI + TMS, add_special_tokens=False, return_tensors="pt")
    pixel_values = grid = None
    if conditions:
        content: Any = [{"type": "image"} for _ in conditions] + [{"type": "text", "text": caption}]
        text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True)
        encoded = processor(text=[text], images=conditions, padding="longest", return_tensors="pt")
        ids = torch.cat([encoded.input_ids, tail], dim=-1)
        pixel_values, grid = encoded.pixel_values, encoded.image_grid_thw
        merge = processor.image_processor.merge_size
        grids = [(int(t), int(h) // merge, int(w) // merge) for t, h, w in grid.tolist()]
    else:
        text = processor.apply_chat_template([{"role": "user", "content": caption}], tokenize=False,
                                             add_generation_prompt=True)
        ids = torch.cat([tokenizer.encode(text, add_special_tokens=False, return_tensors="pt"), tail], dim=-1)
        grids = []
    vision = [(1, rows, columns), *((1, h, w) for h, w in references)]
    padded = ids[0].tolist()
    for _, h, w in vision:
        padded += [start] + [image] * (h * w - 1)
    skips = [0] * len(grids) + [1] * len(vision)
    positions = rope_index(padded, grids + vision, skips, image)
    return Sequence(ids, pixel_values, grid, positions, ids.shape[1] - 1)


# ---------------------------------------------------------------- the transformer


def _rope(config: Any, positions: torch.Tensor, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL's interleaved M-RoPE in f32 from an f32 inverse frequency (upstream's
    `original_inv_freq`), as (cos, sin) of [1, T, head_dim] in `like`'s dtype."""
    device = like.device
    dim = config.head_dim
    theta = config.rope_parameters["rope_theta"]
    section = config.rope_parameters["mrope_section"]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    freqs = positions.to(device).float()[:, None, :, None] * inv.to(device)[None, None, None, :]
    mixed = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        span = slice(offset, section[axis] * 3, 3)
        mixed[..., span] = freqs[axis, ..., span]
    emb = torch.cat((mixed, mixed), dim=-1)
    return emb.cos().to(like.dtype), emb.sin().to(like.dtype)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos[:, None] + rotated * sin[:, None]


def _layer(
    layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    past: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One decoder layer: causal over the prefix itself (`past` None), else every token attends to
    the cached prefix and to all of `h`. Returns the hidden states and this pass's keys and values."""
    attention = layer.self_attn
    x = layer.input_layernorm(h)
    shape = (*x.shape[:-1], -1, attention.head_dim)
    q = _rotate(attention.q_norm(attention.q_proj(x).view(shape)).transpose(1, 2), cos, sin)
    k = _rotate(attention.k_norm(attention.k_proj(x).view(shape)).transpose(1, 2), cos, sin)
    v = attention.v_proj(x).view(shape).transpose(1, 2)
    keys, values = (k, v) if past is None else (torch.cat([past[0], k], 2), torch.cat([past[1], v], 2))
    groups = q.shape[1] // keys.shape[1]  # grouped-query heads, repeated so any SDPA backend serves them
    keys, values = keys.repeat_interleave(groups, dim=1), values.repeat_interleave(groups, dim=1)
    if past is None:
        out = F.scaled_dot_product_attention(q, keys, values, is_causal=True).transpose(1, 2)
    else:
        out = attention.site(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2))
    h = h + attention.o_proj(out.reshape(*x.shape[:-1], -1))
    return h + layer.mlp(layer.post_attention_layernorm(h)), k, v


@dataclass
class Prefix:
    """A caption's causal prefix, computed once: each layer's keys and values, and the bf16 RoPE
    tables of the tokens after it."""

    kv: list[tuple[torch.Tensor, torch.Tensor]]
    cos: torch.Tensor
    sin: torch.Tensor


def _prefix(body: Body, seq: Sequence, vision: tuple[torch.Tensor, list[torch.Tensor]] | None,
            image_token: int, device: torch.device) -> Prefix:
    """The prefix's keys and values, the vision tower's tokens in place of its image pads and its
    deepstack features added after the first layers, as Qwen3-VL's text model adds them."""
    text = body.language_model
    ids = seq.ids[:, : seq.prefix].to(device)
    h = text.embed_tokens(ids)
    mask = ids[0] == image_token
    if vision is not None:
        h[0, mask] = vision[0].to(h.dtype)
    cos, sin = _rope(text.config, seq.positions, h)
    kv = []
    for index, layer in enumerate(text.layers):
        h, k, v = _layer(layer, h, cos[:, : seq.prefix], sin[:, : seq.prefix], None)
        kv.append((k, v))
        if vision is not None and index < len(vision[1]):
            h[0, mask] = h[0, mask] + vision[1][index].to(h.dtype)
    return Prefix(kv, cos[:, seq.prefix :], sin[:, seq.prefix :])


def _predict(body: Body, prefix: Prefix, t: torch.Tensor, inputs: torch.Tensor, n: int) -> torch.Tensor:
    """The pixel head's prediction of the target's clean patches (bf16 [1, n, 3072]) from the
    timestep `t` and the noisy target's and the references' patches."""
    text = body.language_model
    h = torch.cat([body.t_embedder1(t.view(1))[:, None], body.x_embedder(inputs)], dim=1)
    for layer, past in zip(text.layers, prefix.kv, strict=True):
        h = _layer(layer, h, prefix.cos, prefix.sin, past)[0]
    return body.final_layer2(text.norm(h[:, 1 : 1 + n]))


@dataclass(frozen=True)
class Recipe:
    """How one checkpoint samples: `unipc` (shift, CFG), `flash` (noise scale, clip) or `euler`."""

    sampler: str
    shift: float = 3.0
    noise: float = NOISE_SCALE
    clip: float = 0.0


class HiDreamO1(Model[O1Graph], encoded_leaves="accept"):
    """One HiDream-O1-Image checkpoint, full or Dev."""

    graph: O1Graph
    processor: Any

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(O1Graph, factory=O1Graph)
        self.processor = vlm.processor(loader.assets)

    @property
    def body(self) -> Body:
        body: Body = self.graph.components[COMPONENT].model
        return body

    @uses_components(COMPONENT)
    def generate(
        self,
        caption: str,
        negative: str | None,
        references: list[Image.Image],
        *,
        width: int,
        height: int,
        recipe: Recipe,
        steps: int,
        guidance: float,
        seed: int,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
        interval: bool,
    ) -> Image.Image:
        """Upstream's `generate_image` from prepared references (each on the 32 px grid): CFG
        against `negative` when it is given, over the first `GUIDED` of the steps if `interval`."""
        body = self.body
        image_token = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        n = rows * columns
        conditions = [condition_image(r, len(references)) for r in references]
        grids = [(r.height // PATCH, r.width // PATCH) for r in references]
        captions = [caption] if negative is None else [caption, negative]
        with torch.inference_mode():
            vision = None
            prefixes = []
            for text in captions:
                seq = sequence(self.processor, text, conditions, grids, rows, columns)
                if seq.pixel_values is not None and seq.grid is not None and vision is None:
                    out = body.visual(seq.pixel_values.to(device, torch.bfloat16), grid_thw=seq.grid.to(device))
                    vision = (out.pooler_output, out.deepstack_features)
                prefixes.append(_prefix(body, seq, vision, image_token, device))
            refs = (torch.cat([patches(r) for r in references]).to(device, torch.bfloat16)[None]
                    if references else None)
            # One stream: the flash sampler's noise follows the initial noise's.
            generator = rng.generator(seed + 1)
            noise = recipe.noise * rng.noise((1, 3, height, width), generator)
            z = to_rows(noise[0].to(device, torch.bfloat16))
            sampler: Euler | UniPC
            if recipe.sampler == "unipc":
                sampler = UniPC(steps, recipe.shift)
            elif recipe.sampler == "flash":
                sampler = Flash(generator, recipe.noise, recipe.clip)
            else:
                sampler = Euler()
            count = len(sampler.timesteps)
            for index, step in enumerate(sampler.timesteps):
                cancel()
                t = torch.tensor(float(step), device=device) / 1000.0
                sigma = t.clamp_min(T_EPS)
                inputs = z if refs is None else torch.cat([z, refs], dim=1)
                guided = prefixes if not interval or index < round(GUIDED * count) else prefixes[:1]
                velocities = [
                    (_predict(body, prefix, 1.0 - t, inputs, n).float() - z.float()) / sigma for prefix in guided
                ]
                velocity = velocities[0]
                if len(velocities) == 2:
                    velocity = velocities[1] + guidance * (velocities[0] - velocities[1])
                z = sampler.step(index, -velocity, z.float()).to(torch.bfloat16)
                on_step(index)
            image = unpatch((z + 1) / 2, height, width).cpu().float()
        pixels = np.round(np.clip(image.numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        return Image.fromarray(pixels)
