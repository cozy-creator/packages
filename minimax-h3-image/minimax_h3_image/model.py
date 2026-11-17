"""MiniMax-H3-Image: its DiT and VAE decoder (`H3Image`), its VAE encoder (`H3ImageEncoder`, read only for pictures)
and MiniMax-H3's text encoder (`H3TextEncoder`, the `text_encoder` of a MiniMax-H3 checkpoint, unchanged).

A port of Diffusers 0.43's `modular_pipelines/minimax_h3_image`. The text and each reference's rows are packed with
the image's in one sequence, `[text | pictures | image]`: no audio rows. Six Euler steps on the checkpoint's fixed
sigma grid, each through its own output head (PDD), no CFG. The pictures are noised to t = 0.999 and held there.
Every draw is the request's one CPU stream, in Diffusers' order: each picture's posterior sample, the image's noise,
then each picture's noise.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

import numpy as np
import torch
from cozy_runtime import vlm
from cozy_runtime.accel.attention import AttentionLayout, attention_scope
from cozy_runtime.author import Config, Loader, Model, Serving, memoize, preview, uses_components
from diffusers.modular_pipelines.minimax_h3.before_denoise import _frame_position_grid, patchify_video_latents
from PIL import Image, ImageOps
from torch import nn
from transformers import ProcessorMixin, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from .transformer import ImageTransformer, TabledTransformer, build_transformer, compute_dtypes, parse_tables
from .vae import PIXEL_MEAN, PIXEL_STD, ImageDecoder, ImageEncoder

#: The checkpoint's sampling grid (`pdd_config.json`), one output head per step.
SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)
STEPS = len(SIGMAS) - 1
IMAGE_TAG, TEXT_TAG = 0, 1
#: The pictures' rows are noised to t = 0.999 and held there every step.
NOISE_AUG = 0.999
PATCH = (1, 2, 2)
#: Canvas sides are whole patches of whole VAE cells: 16 x 2.
GRID = 32
LATENT_CHANNELS = 24
#: What the presentation keeps of the prompt, and of each picture's label, in tokens.
MAX_TOKENS = 4096
#: The text encoder's hidden state the DiT reads: after decoder layer 50 of 64, before any norm.
TEXT_LAYER = 50
MAX_PICTURES = 16
#: A picture's short edge as the VAE reads it (kept within, else scaled to the nearer bound), and as the text
#: encoder's vision tower reads it.
VAE_EDGES, VISION_EDGES = (1024, 2048), (1024, 1024)
#: The transformer config's labels of the AdaLN rows a `bf16` checkpoint stores (`lanes.prepare`).
TABLES = "adaln_tables"

#: The DiT's attention holds at 8 bits; the VAE decoder's tiles (261 tokens) and the text encoder's stay exact.
ATTENTION_BITS: dict[str, Literal[16, 8, 4]] = {"transformer": 8}
#: Dense every step; the token refiner and blocks 0-1 dense, the refiner and block 1 exact (H3's key outliers).
_DENSE_PATHS = ("token_refiner", "transformer_blocks.0", "transformer_blocks.1")
_EXACT_PATHS = ("token_refiner", "transformer_blocks.1")


def f32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def timesteps() -> tuple[float, ...]:
    """Each step's timestep, `1 - sigma` in float32 (Diffusers' `MiniMaxH3ImageSetTimestepsStep`)."""
    return tuple(f32(1.0 - sigma) for sigma in SIGMAS[:-1])


def sigmas() -> tuple[float, ...]:
    """The grid the Euler steps walk: each sigma through its float32 timestep and back, then 0."""
    return (*(f32(1.0 - t) for t in timesteps()), 0.0)


def table_keys() -> dict[str, list[dict[str, object]]]:
    """The AdaLN rows the 6 steps read, in first use: each step's image and text rows, and the pictures' rows at
    t = 0.999. `bf16` stores tables of exactly these."""
    blocks: list[dict[str, object]] = []
    finals: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    names = ("video", "text")
    for t in timesteps():
        for timestep, tag in ((t, IMAGE_TAG), (t, TEXT_TAG), (f32(NOISE_AUG), IMAGE_TAG)):
            label = f32(timestep).hex()
            if (label, tag) not in seen:
                seen.add((label, tag))
                blocks.append({"index": len(blocks), "timestep": label, "modality": names[tag], "modality_tag": tag})
            if all(row["timestep"] != label for row in finals):
                finals.append({"index": len(finals), "timestep": label})
    return {"block_modulation": blocks, "final_normalization": finals}


def bounded(width: int, height: int, edges: tuple[int, int]) -> tuple[int, int]:
    """Diffusers' `bound_short_edge`: a short edge within `edges` keeps the size; else the picture is scaled to the
    nearer bound, both sides on the grid."""
    short = min(width, height)
    if edges[0] <= short <= edges[1]:
        return width, height
    scale = (edges[0] if short < edges[0] else edges[1]) / short
    return max(GRID, round(width * scale / GRID) * GRID), max(GRID, round(height * scale / GRID) * GRID)


def upright(picture: Image.Image) -> Image.Image:
    """A picture as Diffusers reads a reference: its EXIF orientation applied, RGB."""
    return ImageOps.exif_transpose(picture).convert("RGB")


def resized(picture: Image.Image, edges: tuple[int, int]) -> Image.Image:
    size = bounded(*picture.size, edges)
    return picture if picture.size == size else picture.resize(size, Image.Resampling.LANCZOS)


def latent_grid(picture: Image.Image) -> tuple[int, int]:
    """A picture's latent `(h, w)` as the VAE encodes it: cropped to whole 16 px cells."""
    width, height = resized(picture, VAE_EDGES).size
    return height // 16, width // 16


def section(config: Config, name: str) -> Config:
    """`name`'s config: a section of a checkpoint with several components, else the one component's own."""
    return config.section(name) if name in config.keys() else config


def stored[M: nn.Module](module: M, component: str, config: Config) -> M:
    """Every parameter in its stored dtype: the fill refuses one that differs."""
    dtypes = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
    for name, parameter in module.named_parameters():
        dtype = dtypes.get(config.tensor_dtype(f"{component}.{name}", default="f32"))
        if dtype is not None and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype)
    return module


def _dit(section_: Mapping[str, object]) -> ImageTransformer:
    """As uploaded (`bf16-full`), or over the AdaLN tables its `adaln_tables` labels (`bf16` and its lanes)."""
    config = dict(section_)
    labels = config.pop(TABLES, None)
    if labels is None:
        return compute_dtypes(build_transformer(config))
    tables = parse_tables(labels)
    held = {(tables[0][row], tag) for row, tag in tables[1]}
    if any((t, tag) not in held for t in timesteps() for tag in (IMAGE_TAG, TEXT_TAG)):
        raise ValueError("model modulation tables do not cover MiniMax-H3-Image's 6 steps")
    return compute_dtypes(build_transformer(config, tables))


class ImageGraph:
    def __init__(self, config: Config) -> None:
        decoder = ImageDecoder(section(config, "vae_decoder").mapping())
        self.vae_config = section(config, "vae_decoder").mapping()
        self.components: dict[str, nn.Module] = {
            "transformer": _dit(section(config, "transformer").mapping()),
            "vae_decoder": stored(decoder, "vae_decoder", config).eval(),
        }


class H3Image(Model[ImageGraph]):
    """MiniMax-H3-Image's DiT and VAE decoder; one GPU."""

    serving = Serving(attention_bits=ATTENTION_BITS, loras=("transformer",))

    graph: ImageGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(ImageGraph)

    def unload(self, loader: Loader) -> None:
        return None

    @uses_components("transformer")
    def denoise(
        self,
        embeds: torch.Tensor,
        tags: torch.Tensor,
        references: list[torch.Tensor],
        noise: torch.Tensor,
        reference_noise: list[torch.Tensor],
        *,
        width: int,
        height: int,
        on_step: Callable[[int], None],
    ) -> torch.Tensor:
        """The image's normalized latents `(1, 24, h, w)` on the CPU, from its `noise` `(1, 24, 1, h, w)` and each
        picture's normalized latents and noise (`reference_noise`, their shapes): six Euler steps along the
        data-ward velocity, `x += (sigma - sigma_next) * v` in float32 (Diffusers' `euler_update_form="velocity"`)."""
        device = torch.device(str(self.device))
        latent_height, latent_width = height // 16, width // 16
        grids = [tuple(int(s) for s in r.shape[-2:]) for r in references]
        fields, reference_rows = layout(tags, grids, latent_height, latent_width)
        level = torch.tensor(NOISE_AUG, device=device)
        rows = []
        for latents, drawn in zip(references, reference_noise, strict=True):
            mixed = level * latents.to(device, torch.float32) + (1.0 - level) * drawn.to(device, torch.float32)
            rows.append(patchify_video_latents(mixed[:, :, None], PATCH))
        rows.append(patchify_video_latents(noise.to(device, torch.float32), PATCH))
        latents = torch.cat(rows)
        del rows
        grid = sigmas()
        image_indices = fields["image_indices"].to(device)
        output = image_indices[reference_rows:]
        tokens = fields["token_tags"].to(device)
        texts = int(fields["text_indices"].numel())
        plan = []
        for t in timesteps():
            row_timesteps = torch.full((tokens.numel(),), t, dtype=torch.float32)
            row_timesteps[fields["image_indices"][:reference_rows]] = NOISE_AUG
            unique, indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
            plan.append((unique.to(device), indices.to(device)))
        transformer = cast(ImageTransformer, self.graph.components["transformer"])
        layout_ = AttentionLayout(
            live_tokens=tokens.numel(), protected_prefix=texts + reference_rows, step=0, dense_until_step=STEPS,
            dense_paths=_DENSE_PATHS, exact_paths=_EXACT_PATHS, tau=1.0,
        )  # fmt: skip
        positions = fields["position_ids"].to(device)
        text_indices = fields["text_indices"].to(device)
        with torch.no_grad():
            for times, indices in plan:  # an uncovered row refuses here, never inside the loop
                transformer.require_rows(times, indices, tokens)
            rotary = transformer.rope(positions)
            with attention_scope(layout_):
                text = transformer.text_rows(embeds.to(device))
                for i, (times, indices) in enumerate(plan):
                    velocity = transformer(
                        hidden_states=latents[None], encoder_hidden_states=embeds, timestep=times,
                        timestep_indices=indices, token_tags=tokens, position_ids=positions,
                        image_indices=image_indices, text_indices=text_indices, output_indices=output, step=i,
                        rotary=rotary, text=text,
                    )[0].float()  # fmt: skip
                    sigma, after = grid[i], grid[i + 1]
                    target = latents[reference_rows:]
                    preview(lambda: _unpatchify(target + sigma * velocity, latent_height, latent_width), "minimax-h3")  # noqa: B023  # fmt: skip
                    latents[reference_rows:] = target + (f32(sigma) - f32(after)) * velocity
                    del velocity
                    on_step(i)
        return _unpatchify(latents[reference_rows:], latent_height, latent_width)[:, :, 0].cpu()

    @uses_components("vae_decoder")
    def decode(self, latents: torch.Tensor) -> Image.Image:
        """Normalized latents to the picture: ImageNet-normalized pixels reverted, clamped, and quantized to 8 bits by
        truncation, as the released pipeline does."""
        decoder = cast(ImageDecoder, self.graph.components["vae_decoder"])
        device = torch.device(str(self.device))
        mean = torch.tensor(self.graph.vae_config["latents_mean"], device=device).view(1, -1, 1, 1)
        std = torch.tensor(self.graph.vae_config["latents_std"], device=device).view(1, -1, 1, 1)
        with torch.no_grad():
            pixels = decoder(latents.to(device) * std + mean).float()
            pixels = (pixels * torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
                      + torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)).clamp(0, 1)  # fmt: skip
            array = (pixels * 255).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array)


class EncoderGraph:
    def __init__(self, config: Config) -> None:
        self.vae_config = section(config, "vae_encoder").mapping()
        self.components: dict[str, nn.Module] = {
            "vae_encoder": stored(ImageEncoder(self.vae_config), "vae_encoder", config).eval()
        }


class H3ImageEncoder(Model[EncoderGraph]):
    """The VAE's encoder: each picture's conditioning latents."""

    graph: EncoderGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(EncoderGraph)

    def unload(self, loader: Loader) -> None:
        return None

    @uses_components("vae_encoder")
    def encode(self, pictures: list[Image.Image], noise: list[torch.Tensor]) -> list[torch.Tensor]:
        """Each picture's normalized latents `(1, 24, h, w)` on the CPU (Diffusers' `MiniMaxH3ImageEditVaeEncoderStep`):
        at its `VAE_EDGES` size, cropped to whole cells, ImageNet-normalized, a posterior sample with its `noise`
        rounded to float16, cropped to whole patches."""
        encoder = cast(ImageEncoder, self.graph.components["vae_encoder"])
        device = torch.device(str(self.device))
        pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
        mean = torch.tensor(self.graph.vae_config["latents_mean"], device=device).view(1, -1, 1, 1)
        std = torch.tensor(self.graph.vae_config["latents_std"], device=device).view(1, -1, 1, 1)
        out = []
        with torch.no_grad():
            for picture, drawn in zip(pictures, noise, strict=True):
                pixels = torch.from_numpy(np.array(resized(picture, VAE_EDGES))).to(device).permute(2, 0, 1)[None]
                pixels = pixels[..., : pixels.shape[-2] // 16 * 16, : pixels.shape[-1] // 16 * 16]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                moments = encoder(pixels)
                mu, logvar = torch.chunk(moments, 2, dim=1)
                sample = mu + torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0)) * drawn.to(device, moments.dtype)
                latents = sample.to(torch.float16).float()
                latents = latents[..., : latents.shape[-2] // 2 * 2, : latents.shape[-1] // 2 * 2]
                out.append(((latents - mean) / std).cpu())
        return out


def _forget_rope_deltas(module: nn.Module, _args: tuple[object, ...], _output: object) -> None:
    """Qwen3-VL stores `rope_deltas` on its first picture input though nothing reads it with `use_cache=False`; one
    int64 kept past the forward trips the Runtime's strict memory ledger."""
    module.rope_deltas = None  # type: ignore[assignment]  # Qwen3VLModel's cache attribute, not a tensor slot


#: A MiniMax-H3 checkpoint's text encoder declares this truncation (`cozy_h3`): 50 of 64 layers, no final norm or head.
_CONDITIONER = {"source_architecture": "Qwen3VLForConditionalGeneration", "retained_decoder_layers": TEXT_LAYER,
                "conditioning_hidden_state": TEXT_LAYER, "output": "pre_norm", "language_model_head": False}  # fmt: skip


def conditioner(config: Mapping[str, object]) -> Qwen3VLForConditionalGeneration:
    """MiniMax-H3's text encoder as its checkpoint stores it: Qwen3-VL with the language stack cut after layer 50,
    the final norm and the head removed, so its last hidden state is `hidden_states[50]`."""
    source = dict(config)
    if source.pop("cozy_h3", None) != _CONDITIONER:
        raise ValueError("text_encoder is not MiniMax-H3's 50-layer pre-norm conditioner (`cozy_h3`)")
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**source))  # type: ignore[no-untyped-call]  # Transformers' __init__ is untyped
    language = model.model.language_model
    language.layers = nn.ModuleList(list(language.layers[:TEXT_LAYER]))
    language.norm = nn.Identity()  # type: ignore[assignment]  # the final norm, removed
    model.lm_head = nn.Identity()  # the head, never built
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    model.model.register_forward_hook(_forget_rope_deltas, always_call=True)
    return cast(Qwen3VLForConditionalGeneration, model.eval())  # type: ignore[no-untyped-call]  # Transformers' eval is untyped


class TextGraph:
    def __init__(self, config: Config) -> None:
        self.components: dict[str, nn.Module] = {"text_encoder": conditioner(section(config, "text_encoder").mapping())}


class H3TextEncoder(Model[TextGraph]):
    """MiniMax-H3's text encoder, which MiniMax-H3-Image uses unchanged."""

    graph: TextGraph
    processor: ProcessorMixin

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(TextGraph)
        self.processor = vlm.processor(loader.assets, "text_encoder/processor/")

    def unload(self, loader: Loader) -> None:
        return None

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str, pictures: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """The DiT's text rows `(1, n, 5120)` and each row's modality tag. Text to image: the prompt verbatim, no
        template or special tokens. An edit: `Picture i: ` and its vision block for each picture, then the prompt."""
        tokenizer = self.processor.tokenizer
        token_ids: list[int] = []
        tags: list[int] = []
        vision: dict[str, torch.Tensor] = {}
        if pictures:
            image_processor = self.processor.image_processor
            vision = dict(image_processor(images=[resized(p, VISION_EDGES) for p in pictures], return_tensors="pt"))
            merge = int(image_processor.merge_size) ** 2
            start, pad, end = (tokenizer.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"))
            for index, grid in enumerate(vision["image_grid_thw"]):
                label = tokenizer(f"Picture {index + 1}: ", add_special_tokens=False)["input_ids"][:MAX_TOKENS]
                block = [start, *[pad] * (int(grid.prod()) // merge), end]
                token_ids += [*label, *block]
                tags += [TEXT_TAG] * len(label) + [IMAGE_TAG] * len(block)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:MAX_TOKENS]
        token_ids += ids
        tags += [TEXT_TAG] * len(ids)
        encoder = cast(Qwen3VLForConditionalGeneration, self.graph.components["text_encoder"])
        device = torch.device(str(self.device))
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        types = torch.tensor(self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
        extra = {name: value.to(device, encoder.dtype if name.startswith("pixel_") else value.dtype)
                 for name, value in vision.items()}  # fmt: skip
        with torch.no_grad():
            out = encoder.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), mm_token_type_ids=types,
                                use_cache=False, output_hidden_states=False, **extra)  # fmt: skip
        return out.last_hidden_state.to(torch.bfloat16).cpu(), torch.tensor(tags, dtype=torch.long)


def _unpatchify(rows: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The image's rows back to `(1, 24, 1, h, w)`: `patchify_video_latents` inverted."""
    patch_t, patch_h, patch_w = PATCH
    channels = rows.shape[-1] // (patch_t * patch_h * patch_w)
    value = rows.reshape(1, 1, height // patch_h, width // patch_w, channels, patch_t, patch_h, patch_w)
    return value.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, channels, 1, height, width)


def layout(
    tags: torch.Tensor, grids: Sequence[tuple[int, int]], latent_height: int, latent_width: int
) -> tuple[dict[str, torch.Tensor], int]:
    """Diffusers' `build_image_packed_sequence`: `[text | pictures | image]`, the float64 rotary grid (text on
    the clock's first slots, each picture one integer slot after them, the image the slot after the last), and the
    picture row count."""
    _, patch_h, patch_w = PATCH
    texts = int(tags.shape[0])
    reference_rows = sum((h // patch_h) * (w // patch_w) for h, w in grids)
    total = texts + reference_rows + (latent_height // patch_h) * (latent_width // patch_w)
    position_ids = torch.zeros(total, 3, dtype=torch.float64)
    position_ids[:texts, 0] = torch.arange(texts, dtype=torch.float64)
    cursor, time = texts, float(texts)
    for h, w in grids:
        grid, _ = _frame_position_grid(h, w, patch_h, patch_w)
        rows = slice(cursor, cursor + grid.shape[0])
        position_ids[rows, 0] = time
        position_ids[rows, 1:] = grid
        cursor, time = rows.stop, time + 1.0
    grid, _ = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    position_ids[cursor:, 0] = time
    position_ids[cursor:, 1:] = grid
    token_tags = torch.empty(total, dtype=torch.long)
    token_tags[:texts] = tags
    token_tags[texts:] = IMAGE_TAG
    fields = {"position_ids": position_ids, "token_tags": token_tags, "image_indices": torch.arange(texts, total),
              "text_indices": torch.arange(texts)}  # fmt: skip
    return fields, reference_rows


def canvas(ratio: tuple[int, int], megapixels: float) -> tuple[int, int]:
    """`ratio` at `megapixels` on the grid, as `Sizes.fit` snaps it (no clamp)."""
    pixels, r = megapixels * (1 << 20), ratio[0] / ratio[1]
    return (max(GRID, round(math.sqrt(pixels * r) / GRID) * GRID), max(GRID, round(math.sqrt(pixels / r) / GRID) * GRID))
