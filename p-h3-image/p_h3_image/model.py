"""P-H3-Image (Pruna AI): its DiT, text encoder, text connector and VAE, and the 8-step denoise.

A port of Diffusers' `modular_pipelines/p_h3_image` (main f598e6e6, not in a release). The image is one
latent frame denoised in one packed sequence beside two audio latents it discards: `[text | audio | image]`
for text-to-image, `[text | pictures | audio | image]` for an edit. Eight Euler steps on the checkpoint's
fixed sigma grid, no CFG. The text is Qwen3-VL-4B's decoder layers 12/18/24/30/35 (hidden states 13/19/25/31/36,
the last normed) through Pruna's connector; an edit's pictures are vision tokens in it too.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Literal, cast

import numpy as np
import torch
from cozy_runtime import rng, vlm
from cozy_runtime.accel.attention import AttentionLayout, attention_scope
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import Config, Loader, Model, Serving, memoize, preview, uses_components
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    _fill_audio_positions,
    _frame_position_grid,
    patchify_video_latents,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch import nn
from transformers import ProcessorMixin, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from . import fused
from .transformer import MODALITIES, PH3ImageTransformer, build_transformer, compute_dtypes, parse_tables
from .vae import ImageVAE, encode_picture

#: The checkpoint's sampling grid (`modular_model_index.json` `sample_sigmas`), used verbatim.
SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
SHIFT = 12.0
STEPS = len(SIGMAS) - 1
#: Audio latents per channel denoised beside the image, then dropped: the DiT's soundtrack is stereo.
AUDIO_LATENTS, AUDIO_CHANNELS = 2, 2
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
#: An edit's picture rows are noised to t = 0.999 and held there; the posterior is sampled under seed 42.
NOISE_AUG, ENCODE_SEED = 0.999, 42
#: The DiT's `(frames, height, width)` patch.
PATCH = (1, 2, 2)
MAX_PICTURES = 8
GRID = 32
MEGAPIXEL = 1 << 20
#: An edit's pictures keep their size between 1 and 8 MP; its canvas defaults to Picture 1's, at least 4 MP.
PICTURE_PIXELS = (MEGAPIXEL, 8 * MEGAPIXEL)
CANVAS_MIN_PIXELS = 4 * MEGAPIXEL
#: What the text encoder and the upsampler see of a picture: about one megapixel.
VISION_PIXELS = 1_000_000
#: The Qwen3-VL decoder layers whose outputs the connector reads: Pruna's hidden states 13/19/25/31/36, the last
#: normed (Transformers ties the last layer's to `last_hidden_state`).
TEXT_LAYERS = (12, 18, 24, 30, 35)
#: The transformer config's labels of the AdaLN rows a `bf16` checkpoint stores (`lanes.pruned`).
TABLES = "adaln_tables"

#: The DiT's attention holds at 8 bits. The VAE decoder's stays exact: its tiles are 261-token sequences, where
#: Kitchen INT8 decoded 8 MP in 41 s on a 4090 against SDPA's 4.3 s.
ATTENTION_BITS: dict[str, Literal[16, 8, 4]] = {"transformer": 8}
#: The model's components, by slot.
IMAGE_COMPONENTS = ("processor", "text_connector", "text_encoder", "transformer", "vae")
#: Dense from the first step; the token refiner and blocks 0-1 dense, the refiner and block 1 exact (block 1's
#: key outliers are in its activations: 8-bit rows there lose 0.1+ rel L2).
_DENSE_PATHS = ("token_refiner", "transformer_blocks.0", "transformer_blocks.1")
_EXACT_PATHS = ("token_refiner", "transformer_blocks.1")


def f32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def timesteps() -> tuple[float, ...]:
    """Each step's timestep, `1 - sigma` in float32, as the scheduler's grid gives it."""
    return tuple(f32(1.0 - f32(sigma)) for sigma in SIGMAS[:-1])


def table_keys() -> dict[str, list[dict[str, object]]]:
    """The AdaLN rows the 8 steps read, in first use: each step's image, text and audio row, an edit's pictures'
    (t = 0.999) and clean audio (t = 1). `bf16` stores tables of exactly these."""
    blocks: list[dict[str, object]] = []
    finals: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for t in timesteps():
        for timestep, tag in ((t, VIDEO_TAG), (t, TEXT_TAG), (t, AUDIO_TAG), (max(t, f32(NOISE_AUG)), VIDEO_TAG),
                              (f32(1.0), AUDIO_TAG)):  # fmt: skip
            label = f32(timestep).hex()
            if (label, tag) not in seen:
                seen.add((label, tag))
                blocks.append({"index": len(blocks), "timestep": label, "modality": MODALITIES[tag], "modality_tag": tag})
            if all(row["timestep"] != label for row in finals):
                finals.append({"index": len(finals), "timestep": label})
    return {"block_modulation": blocks, "final_normalization": finals}


def size_at_area(width: int, height: int, pixels: int) -> tuple[int, int]:
    """`(width, height)` at about `pixels` at its own ratio, sides on the grid; one within 20% keeps its size."""
    if width % GRID == 0 and height % GRID == 0 and 0.8 * pixels <= width * height <= 1.2 * pixels:
        return width, height
    ratio = width / height
    return (max(GRID, round(math.sqrt(pixels * ratio) / GRID) * GRID),
            max(GRID, round(math.sqrt(pixels / ratio) / GRID) * GRID))  # fmt: skip


def size_in_range(width: int, height: int, low: int, high: int) -> tuple[int, int]:
    """`(width, height)` kept, or scaled at its ratio into `low..high` pixels, sides rounded to the grid."""
    area = width * height
    scale = math.sqrt(min(max(area, low), high) / area)
    return (max(GRID, round(width * scale / GRID) * GRID), max(GRID, round(height * scale / GRID) * GRID))


def resized(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def stored[M: nn.Module](module: M, component: str, config: Config) -> M:
    """Every parameter in its stored dtype: the fill refuses one that differs."""
    dtypes = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
    for name, parameter in module.named_parameters():
        dtype = dtypes.get(config.tensor_dtype(f"{component}.{name}", default="bf16"))
        if dtype is not None and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype)
    return module


class TextConnector(nn.Module):
    """Pruna's `PH3ImageTextConnector`: the encoder's `TEXT_LAYERS` outputs concatenated, standardized,
    projected to the DiT's 5120 text channels and destandardized; the first token, an attention sink, is a learned
    row. Float32, as stored."""

    def __init__(self, text_hidden_size: int = 2560, out_channels: int = 5120, **_: object) -> None:
        super().__init__()
        width = len(TEXT_LAYERS) * text_hidden_size
        self.input_mean = nn.Parameter(torch.zeros(width))
        self.input_std = nn.Parameter(torch.ones(width))
        self.proj = nn.Linear(width, out_channels, bias=False)
        self.output_mean = nn.Parameter(torch.zeros(out_channels))
        self.output_std = nn.Parameter(torch.ones(out_channels))
        self.sink_embedding = nn.Parameter(torch.zeros(out_channels))

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        dtype = self.proj.weight.dtype
        rows = torch.cat([hidden_states[index] for index in TEXT_LAYERS], dim=-1)
        rows = (rows.to(dtype) - self.input_mean) / self.input_std
        rows = self.proj(rows) * self.output_std + self.output_mean
        return torch.cat([self.sink_embedding.expand(rows.shape[0], 1, -1), rows[:, 1:]], dim=1)


#: A forward pre-hook's answer: the call's positional and keyword arguments.
_Call = tuple[tuple[object, ...], dict[str, object]]


@contextmanager
def _hidden_states_of(encoder: Qwen3VLForConditionalGeneration, layers: Sequence[int]) -> Iterator[None]:
    """While open, a forward of `encoder` keeps its language model's `layers` outputs alone (Transformers' list
    form of `output_hidden_states`, by decoder layer, the last tied to `last_hidden_state`), and none of its vision
    tower's. `output_hidden_states=True` held all 37 (x tokens x 2560) through the forward, and every vision
    block's beside them (se-122)."""
    language: nn.Module = encoder.model.language_model
    kept = list(layers)

    def tower(_: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> _Call:
        return args, {**kwargs, "output_hidden_states": False}

    def text(_: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> _Call:
        return args, {**kwargs, "output_hidden_states": kept}

    handles = (
        encoder.register_forward_pre_hook(tower, with_kwargs=True),
        language.register_forward_pre_hook(text, with_kwargs=True),
    )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _forget_rope_deltas(module: nn.Module, _args: tuple[object, ...], _output: object) -> None:
    """Qwen3-VL stores `rope_deltas` on its first picture input though nothing reads it with `use_cache=False`;
    one int64 kept past the forward trips the Runtime's strict memory ledger."""
    module.rope_deltas = None  # type: ignore[assignment]  # Qwen3VLModel's cache attribute, not a tensor slot


def _dit(section: Mapping[str, object]) -> PH3ImageTransformer:
    """As uploaded (`bf16-full`), or over the AdaLN tables its `adaln_tables` labels (`bf16` and its lanes)."""
    config = dict(section)
    labels = config.pop(TABLES, None)
    tables = None
    if labels is not None:
        tables = parse_tables(labels)
        held = {(tables[0][row], tag) for row, tag in tables[1]}
        if any((t, tag) not in held for t in timesteps() for tag in (VIDEO_TAG, TEXT_TAG, AUDIO_TAG)):
            raise ValueError("model modulation tables do not cover P-H3-Image's 8 steps")
    return compute_dtypes(build_transformer(config, tables))


class ImageGraph:
    def __init__(self, config: Config) -> None:
        text = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config.section("text_encoder").mapping()))  # type: ignore[no-untyped-call]  # Transformers' __init__ is untyped
        text.lm_head = nn.Identity()  # tied to the embeddings, not stored, never run
        text.model.register_forward_hook(_forget_rope_deltas, always_call=True)
        connector = TextConnector(**config.section("text_connector").mapping())
        vae = ImageVAE.build(config.section("vae").mapping())
        self.components: dict[str, nn.Module] = {
            "transformer": _dit(config.section("transformer").mapping()),
            "text_encoder": stored(text, "text_encoder", config).eval(),  # type: ignore[no-untyped-call]  # Transformers' eval is untyped
            "text_connector": stored(connector, "text_connector", config).eval(),
            "vae": stored(cast(nn.Module, vae), "vae", config).eval(),
        }
        self.processor_config = config.section("processor").mapping()


class PH3Image(Model[ImageGraph]):
    """P-H3-Image's DiT, text encoder, connector and VAE; one GPU."""

    serving = Serving(attention_bits=ATTENTION_BITS, loras=("transformer",))

    graph: ImageGraph
    processor: ProcessorMixin

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(ImageGraph)
        self.processor = vlm.processor(loader.assets, "processor/", config=self.graph.processor_config)

    def unload(self, loader: Loader) -> None:
        return None

    def execution_plan(self, plan: Plan) -> str:
        return fused.install(self, self.graph, plan)

    @memoize
    @uses_components("text_encoder", "text_connector")
    def encode_text(self, prompt: str, pictures: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """The DiT's text rows and each row's modality tag. Text-to-image: the prompt verbatim, no template.
        An edit: `<Picture i>: ` and its vision tokens for each picture, then the instruction after
        `Edit <Picture 1>. ` (a brief the upsampler wrote, or a prompt already so opened, as written)."""
        tokenizer = self.processor.tokenizer
        token_ids: list[int] = []
        tags: list[int] = []
        vision: dict[str, torch.Tensor] = {}
        if pictures:
            sized = [resized(p, size_at_area(*p.size, VISION_PIXELS)) for p in pictures]
            image_processor = self.processor.image_processor
            vision = dict(image_processor(images=sized, return_tensors="pt"))
            merge = int(image_processor.merge_size) ** 2
            start, pad, end = (tokenizer.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"))
            for index, grid in enumerate(vision["image_grid_thw"]):
                label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                block = [start, *[pad] * (int(grid.prod()) // merge), end]
                token_ids += [*label, *block]
                tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(block)
            if not prompt.lstrip().startswith(("Edit <Picture 1>.", "subject_definitions:")):
                prompt = f"Edit <Picture 1>. {prompt}"
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += ids
        tags += [TEXT_TAG] * len(ids)
        encoder = cast(Qwen3VLForConditionalGeneration, self.graph.components["text_encoder"])
        device = torch.device(str(self.device))
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        types = torch.tensor(self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
        extra = {name: value.to(device, encoder.dtype if name.startswith("pixel_") else value.dtype)
                 for name, value in vision.items()}  # fmt: skip
        with torch.no_grad(), _hidden_states_of(encoder, TEXT_LAYERS):
            out = encoder(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), mm_token_type_ids=types,
                          use_cache=False, logits_to_keep=1, **extra)  # fmt: skip
            embeds = self.graph.components["text_connector"](out.hidden_states).to(torch.bfloat16)
        return embeds, torch.tensor(tags, dtype=torch.long)

    @uses_components("vae")
    def encode_pictures(self, pictures: list[Image.Image]) -> list[torch.Tensor]:
        """Each picture's normalized conditioning latents, `(1, 24, 1, h, w)` on the CPU."""
        vae = cast(ImageVAE, self.graph.components["vae"])
        mean = torch.tensor(vae.config["latents_mean"]).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config["latents_std"]).view(1, -1, 1, 1, 1)
        latents = []
        with torch.no_grad():
            for picture in pictures:
                pixels = torch.from_numpy(np.array(picture.convert("RGB")))
                latents.append((encode_picture(vae, pixels, ENCODE_SEED).cpu() - mean) / std)
        return latents

    @uses_components("transformer")
    def denoise(
        self,
        embeds: torch.Tensor,
        tags: torch.Tensor,
        conditions: list[torch.Tensor],
        *,
        width: int,
        height: int,
        seed: int,
        on_step: Callable[[int], None],
        start: torch.Tensor | None = None,
        first: int = 0,
    ) -> torch.Tensor:
        """The image's normalized latents, `(1, 24, 1, h, w)`: each picture's noise drawn first, then the image's,
        then the audio's, from the seed's one stream (Diffusers' order), 8 dense Euler steps. With `start` (the
        canvas's normalized latents, `encode_pictures`), the steps before `first` are skipped and the image begins
        as `start` noised to `SIGMAS[first]`."""
        device = torch.device(str(self.device))
        generator = rng.generator(seed)
        latent_height, latent_width = height // 16, width // 16
        held, condition_rows = _layout(tags, conditions, latent_height, latent_width)
        fields = {name: value.to(device) for name, value in held.items()}
        embeds = embeds.to(device)
        packed = []
        for condition in conditions:
            noise = randn_tensor(tuple(condition.shape), generator=generator, device=device, dtype=torch.float32)
            level = torch.tensor(NOISE_AUG, device=device)
            packed.append(patchify_video_latents(level * condition.to(device) + (1.0 - level) * noise, PATCH))
        noise = randn_tensor((1, 24, 1, latent_height, latent_width), generator=generator, device=device, dtype=torch.float32)
        if start is not None:
            noise = (1.0 - SIGMAS[first]) * start.to(device) + SIGMAS[first] * noise
        video = patchify_video_latents(noise, PATCH)
        if packed:
            video = torch.cat([*packed, video])
        audio = randn_tensor((AUDIO_LATENTS * AUDIO_CHANNELS, 32), generator=generator, device=device, dtype=torch.float32)
        scheduler, audio_scheduler = MiniMaxH3Scheduler(shift=SHIFT), MiniMaxH3Scheduler(shift=SHIFT)
        for each in (scheduler, audio_scheduler):
            each.set_timesteps(sigmas=list(SIGMAS), device=device)
            if first:
                each.set_begin_index(first)
        texts = int(held["text_indices"].numel())
        rows = [
            tuple(value.to(device) for value in MiniMaxH3SetTimestepsStep.build_row_timesteps(
                held["video_indices"], held["audio_indices"], condition_rows, 0, texts,
                float(t), float(t), max(float(t), NOISE_AUG), 1.0,
            ))
            for t in scheduler.timesteps
        ]  # fmt: skip
        transformer = cast(PH3ImageTransformer, self.graph.components["transformer"])
        tokens = fields["token_tags"]
        layout = AttentionLayout(
            live_tokens=tokens.numel(), protected_prefix=texts + condition_rows + int(fields["audio_indices"].numel()),
            step=0, dense_until_step=STEPS, dense_paths=_DENSE_PATHS, exact_paths=_EXACT_PATHS, tau=1.0,
        )  # fmt: skip
        with torch.no_grad():
            for times, indices in rows:  # an uncovered row refuses here, never inside the loop
                transformer.require_rows(times, indices, tokens)
            rotary = transformer.rope(fields["position_ids"])
            with attention_scope(layout):
                text = transformer.text_rows(embeds)
                for i, (times, indices) in enumerate(rows):
                    if i < first:
                        continue
                    velocity, audio_velocity = transformer(
                        hidden_states=video[None], audio_hidden_states=audio[None],
                        encoder_hidden_states=embeds, timestep=times, timestep_indices=indices, token_tags=tokens,
                        position_ids=fields["position_ids"], video_indices=fields["video_indices"],
                        audio_indices=fields["audio_indices"], text_indices=fields["text_indices"],
                        rotary=rotary, text=text,
                    )  # fmt: skip
                    preview(lambda: _unpatchify(video[condition_rows:] + (1 - scheduler.timesteps[i]) * velocity[0, condition_rows:].float(), latent_height, latent_width), "minimax-h3")  # noqa: F821  # fmt: skip
                    video[condition_rows:] = scheduler.step(
                        velocity[0, condition_rows:].float(), scheduler.timesteps[i], video[condition_rows:], return_dict=False
                    )[0]
                    audio = audio_scheduler.step(
                        audio_velocity[0].float(), audio_scheduler.timesteps[i], audio, return_dict=False
                    )[0]
                    del velocity, audio_velocity
                    on_step(i)
        return _unpatchify(video[condition_rows:], latent_height, latent_width).cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> Image.Image:
        vae = cast(ImageVAE, self.graph.components["vae"])
        with torch.no_grad():
            pixels = vae.decode_image(latents.to(torch.device(str(self.device))))
        array = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(array)


def _unpatchify(rows: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The image's rows back to `(1, 24, 1, H, W)`: `patchify_video_latents` inverted."""
    patch_t, patch_h, patch_w = PATCH
    channels = rows.shape[-1] // (patch_t * patch_h * patch_w)
    value = rows.reshape(1, 1, height // patch_h, width // patch_w, channels, patch_t, patch_h, patch_w)
    return value.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, channels, 1, height, width)


def _layout(
    tags: torch.Tensor, conditions: Sequence[torch.Tensor], latent_height: int, latent_width: int
) -> tuple[dict[str, torch.Tensor], int]:
    """The packed sequence's fields and its picture row count: text, audio, image for text-to-image (Diffusers'
    layout for one frame), and for an edit the pictures between the text and the audio
    (`PH3ImageEditPrepareLayoutStep`)."""
    names = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")
    if not conditions:
        values = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            tags, 1, latent_height, latent_width, AUDIO_LATENTS, PATCH, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG
        )
        return dict(zip(names, values[:5], strict=True)), 0
    _, patch_h, patch_w = PATCH
    texts = int(tags.shape[0])
    condition_rows = sum((c.shape[3] // patch_h) * (c.shape[4] // patch_w) for c in conditions)
    audio_rows = AUDIO_LATENTS * AUDIO_CHANNELS
    total = texts + condition_rows + audio_rows + (latent_height // patch_h) * (latent_width // patch_w)
    position_ids = torch.zeros(total, 3, dtype=torch.float64)
    position_ids[:texts, 0] = torch.arange(texts, dtype=torch.float64)
    cursor, time = texts, float(texts)
    for condition in conditions:  # each picture takes one integer slot on the rotary clock
        grid, _ = _frame_position_grid(condition.shape[3], condition.shape[4], patch_h, patch_w)
        rows = slice(cursor, cursor + grid.shape[0])
        position_ids[rows, 0] = time
        position_ids[rows, 1:] = grid
        cursor, time = rows.stop, time + 1.0
    audio_start, video_start = cursor, cursor + audio_rows
    grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    _fill_audio_positions(position_ids, slice(audio_start, video_start), AUDIO_LATENTS, time, width_grid, AUDIO_CHANNELS)
    position_ids[video_start:, 0] = time
    position_ids[video_start:, 1:] = grid
    video_indices = torch.cat([torch.arange(texts, audio_start), torch.arange(video_start, total)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(texts)
    token_tags = torch.empty(total, dtype=torch.long)
    token_tags[text_indices] = tags
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return dict(zip(names, (position_ids, token_tags, video_indices, audio_indices, text_indices), strict=True)), condition_rows
