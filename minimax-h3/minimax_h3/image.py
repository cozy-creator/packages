"""P-H3-Image (PrunaAI): text-to-image and 1-8 picture edits from a fine-tune of H3's FL2VA DiT.

A port of Diffusers' `modular_pipelines/p_h3_image` (main f598e6e6, not in a release). The image is one
latent frame denoised in H3's packed sequence beside two audio latents it discards: `[text | audio | image]`
for text-to-image (H3's T2VA layout), `[text | pictures | audio | image]` for an edit (H3's Ref2VA layout with
image references only). Eight Euler steps on the checkpoint's fixed sigma grid, no CFG. The text is
Qwen3-VL-4B's hidden states 13/19/25/31/36 through Pruna's connector; an edit's pictures are vision tokens in
it too. The VAE is H3's encoder with a single-frame ViT decoder. The DiT, its loop, kernels and LoRA consumers
are H3's own (`adaln_pruned`, `steps.denoise_official`, `glue`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import torch
from cozy_runtime import lm, rng, vlm
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import Config, Loader, Model, Serving, memoize, uses_components
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoViTDecoder3d
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
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedModel, ProcessorMixin
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.initialization import no_init_weights

from . import glue, producer
from .adaln_pruned import H3Transformer, build_transformer
from .conditioner import release_position_cache
from .official import PATCH_SIZE, Schedule, TimestepPlan, _apply_transformer_dtype, _attention_layout
from .preparation.table_layout import TableLayout
from .regenerate import unpatchify
from .steps import denoise_official
from .vae_tiles import TileBatchedVideoVAE, encode_condition

#: The checkpoint's sampling grid (`modular_model_index.json` `sample_sigmas`), used verbatim.
SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
SHIFT = 12.0
STEPS = len(SIGMAS) - 1
#: Audio latents per channel denoised beside the image, then dropped; MiniMax-H3's soundtrack is stereo.
AUDIO_LATENTS, AUDIO_CHANNELS = 2, 2
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
#: An edit's picture rows are noised to t = 0.999 and held there; the posterior is sampled under seed 42.
NOISE_AUG, ENCODE_SEED = 0.999, 42
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_PICTURES = 8
GRID = 32
MEGAPIXEL = 1 << 20
#: An edit's pictures keep their size between 1 and 8 MP; its canvas defaults to Picture 1's, at least 4 MP.
PICTURE_PIXELS = (MEGAPIXEL, 8 * MEGAPIXEL)
CANVAS_MIN_PIXELS = 4 * MEGAPIXEL
#: What the text encoder and the upsampler see of a picture: about one megapixel.
VISION_PIXELS = 1_000_000
HIDDEN_STATES = (13, 19, 25, 31, 36)
#: Rotary time of every token of the image decoder's one frame (`PH3ImageViTDecoder`).
FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0

#: The DiT's attention holds at 8 bits. The VAE decoder's stays exact: its tiles are 261-token sequences, where
#: Kitchen INT8 decoded 8 MP in 41 s on a 4090 against SDPA's 4.3 s (crona runs 7, 11).
ATTENTION_BITS: dict[str, Literal[16, 8, 4]] = {"transformer": 8}
#: The model's components and the upsampler's, by slot.
IMAGE_COMPONENTS = ("processor", "text_connector", "text_encoder", "transformer", "vae")
UPSAMPLER_COMPONENTS = ("prompt_upsampler", "prompt_upsampler_processor")


def _f32(value: float) -> float:
    return float(np.float32(value))


def plan() -> TimestepPlan:
    """The one schedule P-H3-Image serves, as H3's plan: its AdaLN rows are the pruned lane's tables."""
    sigmas = tuple(_f32(sigma) for sigma in SIGMAS)
    schedule = Schedule(sigma_grid_points=len(sigmas), video_sigmas=sigmas, audio_sigmas=sigmas)
    return TimestepPlan(task="fl2va", video_shift=SHIFT, audio_shift=SHIFT, schedules=(schedule,))


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


class TextConnector(nn.Module):
    """Pruna's `PH3ImageTextConnector`: the encoder's hidden states `HIDDEN_STATES` concatenated, standardized,
    projected to the DiT's 5120 text channels and destandardized; the first token, an attention sink, is a learned
    row. Float32, as stored."""

    def __init__(self, text_hidden_size: int = 2560, out_channels: int = 5120, **_: object) -> None:
        super().__init__()
        width = len(HIDDEN_STATES) * text_hidden_size
        self.input_mean = nn.Parameter(torch.zeros(width))
        self.input_std = nn.Parameter(torch.ones(width))
        self.proj = nn.Linear(width, out_channels, bias=False)
        self.output_mean = nn.Parameter(torch.zeros(out_channels))
        self.output_std = nn.Parameter(torch.ones(out_channels))
        self.sink_embedding = nn.Parameter(torch.zeros(out_channels))

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        dtype = self.proj.weight.dtype
        rows = torch.cat([hidden_states[index] for index in HIDDEN_STATES], dim=-1)
        rows = (rows.to(dtype) - self.input_mean) / self.input_std
        rows = self.proj(rows) * self.output_std + self.output_mean
        return torch.cat([self.sink_embedding.expand(rows.shape[0], 1, -1), rows[:, 1:]], dim=1)


class ImageDecoder(MiniMaxH3VideoViTDecoder3d):
    """H3's ViT decoder for one latent frame: each token a 16x16 pixel block (`PH3ImageViTDecoder`)."""

    def __init__(self, **config: object) -> None:
        super().__init__(patch_size_t=1, **config)  # type: ignore[arg-type]  # the VAE's decoder keywords


class ImageVAE(TileBatchedVideoVAE):
    """`AutoencoderKLPH3Image`: H3's encoder and latent space with `ImageDecoder`, tiles decoded as one batch."""

    @classmethod
    def build(cls, config: Mapping[str, object]) -> ImageVAE:
        vae = cast(ImageVAE, cls.from_config(dict(config)))
        c = vae.config
        vae.decoder = ImageDecoder(
            in_channels=c["latent_channels"], out_channels=c["out_channels"], patch_size=vae.spatial_compression_ratio,
            num_layers=c["decoder_num_layers"], num_attention_heads=c["decoder_num_attention_heads"],
            attention_head_dim=c["decoder_attention_head_dim"], num_register_tokens=c["decoder_num_register_tokens"],
            ffn_mult=c["decoder_ffn_mult"], rope_theta=c["decoder_rope_theta"], rope_dim_ratio=c["decoder_rope_dim_ratio"],
            norm_eps=c["decoder_norm_eps"],
        )  # fmt: skip
        return vae

    def time_grid(self, frames: int, device: torch.device) -> torch.Tensor:
        return torch.full((1,), FRAME_POSITION, device=device)

    def decode_image(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized `(1, 24, 1, h, w)` latents to `(1, 3, H, W)` RGB in 0..1."""
        mean = torch.tensor(self.config["latents_mean"], device=latents.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.config["latents_std"], device=latents.device).view(1, -1, 1, 1, 1)
        z = (latents * std + mean).to(self.post_quant_conv.weight.dtype)
        pixels = self._decode_clip(z)[:, :, 0].float()
        mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(1, -1, 1, 1)
        deviation = torch.tensor(PIXEL_STD, device=pixels.device).view(1, -1, 1, 1)
        return (pixels * deviation + mean).clamp(0, 1)


def _stored[M: nn.Module](module: M, component: str, config: Config) -> M:
    """Every parameter in its stored dtype: the fill refuses one that differs."""
    dtypes = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
    for name, parameter in module.named_parameters():
        stored = dtypes.get(config.tensor_dtype(f"{component}.{name}", default="bf16"))
        if stored is not None and parameter.dtype != stored:
            parameter.data = parameter.data.to(stored)
    return module


def _dit(section: Mapping[str, object]) -> H3Transformer:
    """FULL as uploaded, or AdaLN-pruned over the tables its `cozy_h3` labels (`p-h3-image-pruned`)."""
    config = dict(section)
    extension = config.pop("cozy_h3", None)
    tables = None
    if isinstance(extension, Mapping) and extension.get("modulation") == "adaln-pruned":
        layout = TableLayout.parse(extension.get("table_keys"))
        layout.require(plan().schedule().video_timesteps, plan().schedule().audio_timesteps)
        tables = (layout.timesteps, layout.block_keys)
    transformer = _apply_transformer_dtype(build_transformer(config, tables))
    transformer.install_lora_consumers()
    return transformer


class ImageGraph:
    def __init__(self, config: Config) -> None:
        text = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config.section("text_encoder").mapping()))  # type: ignore[no-untyped-call]  # Transformers' __init__ is untyped
        text.lm_head = nn.Identity()  # tied to the embeddings, not stored, never run
        release_position_cache(text)
        connector = TextConnector(**config.section("text_connector").mapping())
        vae = ImageVAE.build(config.section("vae").mapping())
        self.components: dict[str, nn.Module] = {
            "transformer": _dit(config.section("transformer").mapping()),
            "text_encoder": _stored(text, "text_encoder", config).eval(),  # type: ignore[no-untyped-call]  # Transformers' eval is untyped
            "text_connector": _stored(connector, "text_connector", config).eval(),
            "vae": _stored(cast(nn.Module, vae), "vae", config).eval(),
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
        producer.retain_rotary(self.graph.components["transformer"])
        return glue.install(self, self.graph, plan)

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
        with torch.no_grad():
            out = encoder(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), mm_token_type_ids=types,
                          use_cache=False, output_hidden_states=True, logits_to_keep=1, **extra)  # fmt: skip
            embeds = self.graph.components["text_connector"](out.hidden_states).to(torch.bfloat16)
        return embeds, torch.tensor(tags, dtype=torch.long)

    @uses_components("vae")
    def encode_pictures(self, pictures: list[Image.Image]) -> list[torch.Tensor]:
        """Each picture's normalized conditioning latents, `(1, 24, 1, h, w)` on the CPU: MiniMax-H3's recipe, a
        posterior sample under seed 42 rounded to float16."""
        vae = cast(ImageVAE, self.graph.components["vae"])
        mean = torch.tensor(vae.config["latents_mean"]).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config["latents_std"]).view(1, -1, 1, 1, 1)
        latents = []
        with torch.no_grad():
            for picture in pictures:
                pixels = torch.from_numpy(np.array(picture.convert("RGB")))
                sampled = encode_condition(vae, pixels, PIXEL_MEAN, PIXEL_STD, ENCODE_SEED).cpu()
                latents.append((sampled - mean) / std)
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
    ) -> torch.Tensor:
        """The image's normalized latents, `(1, 24, 1, h, w)`: each picture's noise drawn first, then the image's,
        then the audio's, from the seed's one stream (Diffusers' order), 8 dense Euler steps."""
        device = torch.device(str(self.device))
        generator = rng.generator(seed)
        latent_height, latent_width = height // 16, width // 16
        fields, condition_rows = _layout(tags, conditions, latent_height, latent_width)
        packed = []
        for condition in conditions:
            noise = randn_tensor(tuple(condition.shape), generator=generator, device=device, dtype=torch.float32)
            level = torch.tensor(NOISE_AUG, device=device)
            packed.append(patchify_video_latents(level * condition.to(device) + (1.0 - level) * noise, PATCH_SIZE))
        noise = randn_tensor((1, 24, 1, latent_height, latent_width), generator=generator, device=device, dtype=torch.float32)
        video = patchify_video_latents(noise, PATCH_SIZE)
        if packed:
            video = torch.cat([*packed, video])
        audio = randn_tensor((AUDIO_LATENTS * AUDIO_CHANNELS, 32), generator=generator, device=device, dtype=torch.float32)
        schedulers = SimpleNamespace(scheduler=MiniMaxH3Scheduler(shift=SHIFT), audio_scheduler=MiniMaxH3Scheduler(shift=SHIFT))
        for scheduler in (schedulers.scheduler, schedulers.audio_scheduler):
            scheduler.set_timesteps(sigmas=list(SIGMAS), device=device)
        timesteps = schedulers.scheduler.timesteps
        count = int(fields["text_indices"].numel())
        row_plan = [
            tuple(value.to(device) for value in MiniMaxH3SetTimestepsStep.build_row_timesteps(
                fields["video_indices"], fields["audio_indices"], condition_rows, 0, count,
                float(t), float(t), max(float(t), NOISE_AUG), 1.0,
            ))
            for t in timesteps
        ]  # fmt: skip
        state = SimpleNamespace(
            row_timestep_plan=row_plan, denoiser_input_fields={k: v.to(device) for k, v in fields.items()},
            latents=video, audio_latents=audio, num_condition_video_rows=condition_rows, num_condition_audio_rows=0,
            prompt_embeds=embeds.to(device), timesteps=timesteps, audio_timesteps=schedulers.audio_scheduler.timesteps,
        )  # fmt: skip
        transformer = cast(H3Transformer, self.graph.components["transformer"])
        denoise_official(transformer, schedulers, state, lambda step: _attention_layout(state, step, sol_dense_steps=STEPS),
                         STEPS, on_step)  # fmt: skip
        return unpatchify(state.latents[condition_rows:], 1, latent_height, latent_width).cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> Image.Image:
        vae = cast(ImageVAE, self.graph.components["vae"])
        with torch.no_grad():
            pixels = vae.decode_image(latents.to(torch.device(str(self.device))))
        array = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(array)


def _layout(
    tags: torch.Tensor, conditions: Sequence[torch.Tensor], latent_height: int, latent_width: int
) -> tuple[dict[str, torch.Tensor], int]:
    """The packed sequence's fields and its picture row count: H3's T2VA layout for one frame, or for an edit
    its Ref2VA layout with image references alone (`PH3ImageEditPrepareLayoutStep`)."""
    names = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")
    if not conditions:
        values = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            tags, 1, latent_height, latent_width, AUDIO_LATENTS, PATCH_SIZE, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG
        )
        return dict(zip(names, values[:5], strict=True)), 0
    _, patch_h, patch_w = PATCH_SIZE
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


# ------------------------------------------------------------------------------------------- upsampler

ASPECT_RATIOS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}
MAX_NEW_TOKENS = {"t2i": 1024, "edit": 1536}
_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
             "non_diegetic_music")  # fmt: skip
_SECTION = re.compile(r"(?m)^(" + "|".join(_SECTIONS) + r"):[ \t]*$")
_RETENTION = re.compile(
    r"^<(Picture|Subject) (\d+)>(?: \(([^)]*)\))?:\s*(fully_preserved|partially_preserved|attribute_transfer|"
    r"weak_reference)\s*[-–—]\s*(.*)$"
)


def aspect_label(width: int, height: int) -> str:
    """The aspect ratio the upsampler knows closest to `width / height`."""
    return min(ASPECT_RATIOS, key=lambda label: abs(math.log(width / height / ASPECT_RATIOS[label])))


def is_edit_prompt(prompt: str) -> bool:
    """An edit prompt in a form P-H3-Image edits from: the six-section brief, or one `Edit <Picture 1>.` paragraph
    that says what it keeps."""
    text = prompt.strip()
    names = _SECTION.findall(text)
    if names:
        parts = _SECTION.split(text)
        body = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        retention = [line.strip() for line in body.get("retention_analysis", "").splitlines() if line.strip()]
        return (tuple(names) == _SECTIONS and text.count("[keyframe completion]") == 1 and bool(retention)
                and all(_RETENTION.match(line) for line in retention))  # fmt: skip
    return text.startswith("Edit <Picture 1>.") and "Keep" in text and "\n" not in text


class UpsamplerGraph:
    def __init__(self, config: Config) -> None:
        section = config.section("prompt_upsampler")
        with no_init_weights():
            model: PreTrainedModel = AutoModelForImageTextToText.from_config(  # type: ignore[no-untyped-call]  # unannotated in Transformers
                AutoConfig.for_model(**section.mapping()), dtype=torch.bfloat16
            )
            model.tie_weights()
        self.components: dict[str, nn.Module] = {"prompt_upsampler": _stored(model, "prompt_upsampler", config).eval()}  # type: ignore[no-untyped-call]  # Transformers' eval is untyped
        self.processor_config = config.section("prompt_upsampler_processor").mapping()


class PH3Upsampler(Model[UpsamplerGraph]):
    """Pruna's prompt upsampler (a Qwen3.5-9B fine-tune): a short request to the full prompt P-H3-Image renders
    best, or for an edit a brief of what to keep and what to change. Greedy; it reasons first."""

    serving = Serving(min_bits=4, phase="decode")

    graph: UpsamplerGraph
    processor: ProcessorMixin

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpsamplerGraph)
        self.processor = vlm.processor(loader.assets, "prompt_upsampler_processor/", config=self.graph.processor_config)

    def unload(self, loader: Loader) -> None:
        return None

    def execution_plan(self, plan: Plan) -> str:
        return lm.install(plan, cast(PreTrainedModel, self.graph.components["prompt_upsampler"]))

    @memoize
    @uses_components("prompt_upsampler")
    def upsample(self, request: str, pictures: list[Image.Image], aspect: str) -> tuple[str, str]:
        """The prompt for `request` and "", or `request` itself and why, when the answer is cut off or not in the
        form P-H3-Image reads."""
        model = cast(PreTrainedModel, self.graph.components["prompt_upsampler"])
        tokenizer = self.processor.tokenizer
        task = "edit" if pictures else "t2i"
        shown = [resized(p.convert("RGB"), size_at_area(*p.size, VISION_PIXELS)) for p in pictures]
        content: str | list[dict[str, str]]
        if shown:
            content = [part for index in range(len(shown))
                       for part in ({"type": "text", "text": f"<Picture {index + 1}>: "}, {"type": "image"})]  # fmt: skip
            content.append({"type": "text", "text": f"Edit <Picture 1>. {request.strip()}"})
        else:
            content = f"Aspect ratio: {aspect}\nRequest: {request.strip()}"
        text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=shown or None, return_tensors="pt").to(torch.device(str(self.device)))
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        out = lm.generate(model, inputs, max_new_tokens=MAX_NEW_TOKENS[task], stop=[end], pad=tokenizer.pad_token_id)
        ids = [int(token) for token in out[0].tolist()]
        ended = end in ids
        answer = str(tokenizer.decode(ids[: ids.index(end)] if ended else ids, skip_special_tokens=False))
        prompt = answer.partition("</think>")[2].strip()
        valid = ended and answer.count("</think>") == 1 and len(prompt.split()) >= 5
        if valid and (not shown or is_edit_prompt(prompt)):
            return prompt, ""
        return request, "cut off at its token limit" if not ended else "not in the form P-H3-Image reads"
