"""FLUX.2 [dev] and [klein] (distilled and Base), component-staged over Diffusers' modules.

The modules are Diffusers' (`Flux2Transformer2DModel`, `AutoencoderKLFlux2`) and Transformers'
(Mistral-Small-3.2 for [dev], Qwen3 for [klein]); prompt encoding is Diffusers' (`Flux2Pipeline`,
`Flux2KleinPipeline`). Reference preparation, the schedule and the sampler follow BFL's reference
code (`black-forest-labs/flux2` 50fe516: `sampling.py`, `scripts/cli.py`). Runtime constructs and
fills the modules and owns their residency.
"""

from __future__ import annotations

import json
from functools import partial
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from cozy_runtime import lora, rng
from cozy_runtime.author import Config, Loader, Model, ModelAssets, UnsupportedInput, uses_components
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Pipeline, Flux2Transformer2DModel
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE
from PIL import Image, ImageOps
from torch import nn
from tokenizers import Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PixtralImageProcessor,
    PixtralProcessor,
    PreTrainedTokenizerFast,
)
from transformers import initialization as transformer_init

#: Pixels per latent token: the VAE's 8x, then 2x2 packing.
PATCH = 16
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_PIXELS = 4 << 20
#: BFL's reference caps: one reference at 2024², each of several at 1024² (`encode_image_refs`).
SINGLE_REFERENCE_PIXELS, MULTI_REFERENCE_PIXELS = 2024**2, 1024**2
#: Reference i sits at time 10 + 10i (`encode_image_refs`).
REFERENCE_TIME = 10
MAX_TOKENS = 512
#: The hidden states each text encoder conditions on (BFL `text_encoder.py`), by model type.
TEXT_LAYERS = {"mistral3": (10, 20, 30), "qwen3": (9, 18, 27)}
#: fal's Turbo schedule (`fal/FLUX.2-dev-Turbo` card); Diffusers' `Flux2Pipeline`, which the card
#: runs, time-shifts these by the resolution's mu like any schedule.
TURBO_SIGMAS = (1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031)
COMPONENTS = ("transformer", "text_encoder", "vae")


def _read(assets: ModelAssets, name: str) -> bytes:
    return assets.read(f"tokenizer/{name}")


def tokenizer(assets: ModelAssets) -> PreTrainedTokenizerFast:
    """The checkpoint's `tokenizer/` (Qwen3's, or Mistral's under Pixtral's processor), parsed in memory."""
    config = json.loads(_read(assets, "tokenizer_config.json"))
    for key in ("added_tokens_decoder", "tokenizer_file", "name_or_path", "tokenizer_class", "processor_class",
                "chat_template", "auto_map"):
        config.pop(key, None)
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(_read(assets, "tokenizer.json").decode()),
        chat_template=_read(assets, "chat_template.jinja").decode(), **config,
    )


def processor(assets: ModelAssets) -> PixtralProcessor:
    """[dev]'s Pixtral processor (Mistral's tokenizer and image processor), parsed in memory."""
    images = json.loads(_read(assets, "preprocessor_config.json"))
    folder = json.loads(_read(assets, "processor_config.json"))
    for config in (images, folder):
        for key in ("processor_class", "image_processor_type"):
            config.pop(key, None)
    return PixtralProcessor(image_processor=PixtralImageProcessor(**images), tokenizer=tokenizer(assets),
                            chat_template=_read(assets, "chat_template.jinja").decode(), **folder)


def text_encoder(config: Mapping[str, Any]) -> nn.Module:
    """A FLUX.2 text encoder, as the whole model: Mistral-Small-3.2 for [dev], Qwen3 for [klein]."""
    built = AutoModelForImageTextToText if config["model_type"] == "mistral3" else AutoModelForCausalLM
    return built.from_config(AutoConfig.for_model(**config), dtype=torch.bfloat16)


def _conditioning_only(encoder: Any, depth: int) -> None:
    """FLUX.2 reads its text encoder's hidden states up to `depth`: the later layers, the final norm
    and the head are never run, so they are not built (their stored bytes are skipped). With the norm
    an identity, hidden state `depth` stays the pre-norm output. [dev]'s prompts are text only, so
    Mistral3's vision tower and projector are not built either."""
    inner = encoder.model
    language = inner.language_model if hasattr(inner, "language_model") else inner
    language.layers = nn.ModuleList(list(language.layers)[:depth])
    language.norm = nn.Identity()
    encoder.lm_head = nn.Identity()
    if language is not inner:
        inner.vision_tower = inner.multi_modal_projector = None


def _armed(transformer: nn.Module, path: str) -> nn.Module | None:
    """The Turbo overlay's factors at `path` while `turbo.applied` arms it (`_cozy_turbo`)."""
    armed = getattr(transformer, "_cozy_turbo", None)
    return None if armed is None else armed.get(path)


class Flux2Graph:
    def __init__(self, config: Config) -> None:
        text = config.section("text_encoder").mapping()
        family = str(text["model_type"])
        if family not in TEXT_LAYERS:
            raise ValueError(f"FLUX.2 has no text encoder of type {family!r}")
        with transformer_init.no_init_weights():
            transformer = Flux2Transformer2DModel.from_config(config.section("transformer").mapping())
            encoder = text_encoder(text)
            vae = AutoencoderKLFlux2.from_config(config.section("vae").mapping())
        _conditioning_only(encoder, max(TEXT_LAYERS[family]))
        if transformer.config.guidance_embeds:  # [dev]: its Turbo LoRA's sites
            transformer._cozy_turbo = None
            for path, module in transformer.named_modules():
                if isinstance(module, nn.Linear):
                    module.register_forward_hook(lora.Hook(partial(_armed, transformer, path)))
        self.family = family
        # The VAE runs in its stored precision: bf16 in Diffusers' repositories, f32 in BFL's `ae`.
        stored = config.tensor_dtype("vae.encoder.conv_in.weight", default="f32")
        self.components: dict[str, nn.Module] = {
            "transformer": transformer.to(dtype=torch.bfloat16),
            "text_encoder": encoder.eval(),
            "vae": vae.to(dtype=torch.float32 if stored == "f32" else torch.bfloat16),
        }


def schedule(steps: int, tokens: int, sigmas: Sequence[float] | None = None) -> list[float]:
    """BFL's `get_schedule`: `steps` linear times (or the given sigmas), time-shifted by the
    empirical mu of the image's token count, ending at 0."""
    mu = compute_empirical_mu(tokens, steps)
    t = torch.linspace(1, 0, steps + 1) if sigmas is None else torch.tensor([*sigmas, 0.0])
    return [float(v) for v in (math.exp(mu) / (math.exp(mu) + (1 / t - 1))).tolist()]


def prepare_references(images: Sequence[Image.Image]) -> list[Image.Image]:
    """BFL's `default_prep`: RGB, at least 64 px a side and at most 8:1, capped at 2024² pixels
    (1024² each of several), centre-cropped to the 16 px grid."""
    limit = SINGLE_REFERENCE_PIXELS if len(images) == 1 else MULTI_REFERENCE_PIXELS
    prepared = []
    for index, image in enumerate(images):
        image = ImageOps.exif_transpose(image).convert("RGB")
        try:
            Flux2ImageProcessor.check_image_input(image)
        except ValueError as exc:
            raise UnsupportedInput(f"references.{index}.image: {exc}") from None
        image = Flux2ImageProcessor._resize_if_exceeds_area(image, limit)
        w, h = image.width // PATCH * PATCH, image.height // PATCH * PATCH
        left, top = (image.width - w) // 2, (image.height - h) // 2
        prepared.append(image.crop((left, top, left + w, top + h)))
    return prepared


def grid_ids(rows: int, columns: int, time: int = 0) -> torch.Tensor:
    """(t, h, w, l) position ids of a latent grid."""
    return torch.cartesian_prod(torch.tensor([time]), torch.arange(rows), torch.arange(columns), torch.arange(1))


def text_ids(length: int) -> torch.Tensor:
    return torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length))


def _bn(vae: Any, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(like)
    return mean, std


class Flux2(Model[Flux2Graph], encoded_leaves="accept"):
    """One FLUX.2 checkpoint: [dev] (embedded guidance) or a [klein], which runs as Base (CFG) or
    distilled (no guidance) as the function names it: the pipeline's `is_distilled` never reaches
    the model's config."""

    graph: Flux2Graph
    tokenizer: Any

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Flux2Graph, factory=Flux2Graph)
        self.tokenizer = processor(loader.assets) if self.dev else tokenizer(loader.assets)

    @property
    def dev(self) -> bool:
        """[dev]: Mistral-conditioned, guidance-distilled, with an embedded guidance input."""
        return self.graph.family == "mistral3"

    @uses_components("text_encoder", memoize=True, memo_dependencies=(tokenizer, processor, text_encoder, _conditioning_only))
    def encode(self, prompt: str) -> torch.Tensor:
        encoder = self.graph.components["text_encoder"]
        device = torch.device(str(self.device))
        layers = TEXT_LAYERS[self.graph.family]
        with torch.inference_mode():
            if self.dev:
                embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                    encoder, self.tokenizer, prompt, dtype=torch.bfloat16, device=device,
                    max_sequence_length=MAX_TOKENS, system_message=SYSTEM_MESSAGE, hidden_states_layers=layers,
                )
            else:
                embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                    encoder, self.tokenizer, prompt, dtype=torch.bfloat16, device=device,
                    max_sequence_length=MAX_TOKENS, hidden_states_layers=layers,
                )
        return embeds.cpu()

    @uses_components("vae", memoize=True)
    def encode_references(self, references: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized packed latents of the prepared references as one token sequence, with ids."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        tokens, ids = [], []
        with torch.inference_mode():
            for index, reference in enumerate(references):
                pixels = torch.from_numpy(np.array(reference)).permute(2, 0, 1).float() / 127.5 - 1.0
                latents = vae.encode(pixels[None].to(device=device, dtype=vae.dtype)).latent_dist.mode()
                latents = Flux2Pipeline._patchify_latents(latents)
                mean, std = _bn(vae, latents)
                latents = (latents - mean) / std
                tokens.append(latents.flatten(2).transpose(1, 2).to(torch.bfloat16).cpu())
                ids.append(grid_ids(latents.shape[2], latents.shape[3], REFERENCE_TIME * (1 + index)))
        return torch.cat(tokens, dim=1), torch.cat(ids)

    @uses_components("transformer")
    def denoise(
        self,
        embeds: torch.Tensor,
        negative: torch.Tensor | None,
        references: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        width: int,
        height: int,
        timesteps: Sequence[float],
        guidance: float | None,
        seed: int,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
    ) -> torch.Tensor:
        """BFL's Euler sampler in bf16 (`denoise`, `denoise_cfg`): [dev] embeds `guidance`, [klein]
        Base takes CFG against `negative`, [klein] distilled takes neither."""
        tf = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        noise = rng.noise((1, tf.config.in_channels, rows, columns), seed)
        x = noise.to(torch.bfloat16).flatten(2).transpose(1, 2).to(device)
        size = x.shape[1]
        ids = grid_ids(rows, columns)
        text = embeds if negative is None else torch.cat([negative, embeds])
        text = text.to(device=device, dtype=torch.bfloat16)
        refs = None
        if references is not None:
            refs = references[0].to(device).expand(text.shape[0], -1, -1)
            ids = torch.cat([ids, references[1]])
        ids, txt_ids = ids.to(device), text_ids(text.shape[1]).to(device)
        batch = text.shape[0]
        embedded = None if guidance is None or negative is not None else torch.full((batch,), guidance, device=device)
        with torch.inference_mode():
            for step, (t, nxt) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
                cancel()
                tokens = x.expand(batch, -1, -1)
                if refs is not None:
                    tokens = torch.cat([tokens, refs], dim=1)
                pred = tf(
                    hidden_states=tokens, encoder_hidden_states=text,
                    timestep=torch.full((batch,), t, device=device, dtype=torch.bfloat16),
                    img_ids=ids, txt_ids=txt_ids, guidance=embedded, return_dict=False,
                )[0][:, :size]
                if negative is not None:
                    unconditional, conditional = pred.chunk(2)
                    pred = unconditional + guidance * (conditional - unconditional)
                x = x + (nxt - t) * pred
                on_step(step)
        return x.cpu()

    @uses_components("vae")
    def decode(self, tokens: torch.Tensor, *, width: int, height: int) -> Image.Image:
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        with torch.inference_mode():
            latents = tokens.to(device=device, dtype=vae.dtype).view(1, rows, columns, -1).permute(0, 3, 1, 2)
            mean, std = _bn(vae, latents)
            latents = Flux2Pipeline._unpatchify_latents(latents * std + mean)
            image = vae.decode(latents, return_dict=False)[0][0]
            if not torch.isfinite(image).all():
                raise ValueError("FLUX.2's decoder produced non-finite pixels")
            pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)
