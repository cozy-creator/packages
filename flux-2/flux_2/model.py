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
from itertools import pairwise
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import torch
from cozy_runtime import rng
from cozy_runtime.accel import hold_dtype, lora
from cozy_runtime.author import (
    Config,
    Loader,
    Model,
    ModelAssets,
    OutputError,
    Telemetry,
    memoize,
    preview,
    uses_components,
)
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
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

if TYPE_CHECKING:
    from diffusers.configuration_utils import ConfigMixin

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
#: The VAE's dtype on a GPU, as ComfyUI runs FLUX.2's VAE: its activations fit fp16 (dtype audit M1).
VAE_DTYPE = torch.float16


def _read(assets: ModelAssets, name: str) -> bytes:
    return assets.read(f"tokenizer/{name}")


def tokenizer(assets: ModelAssets) -> PreTrainedTokenizerFast:
    """The checkpoint's `tokenizer/` (Qwen3's, or Mistral's under Pixtral's processor), parsed in memory."""
    config = json.loads(_read(assets, "tokenizer_config.json"))
    for key in ("added_tokens_decoder", "tokenizer_file", "name_or_path", "tokenizer_class", "processor_class",
                "chat_template", "auto_map"):
        config.pop(key, None)
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # Transformers' tokenizer __init__ is unannotated
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


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def _to[M: nn.Module](module: M, dtype: torch.dtype) -> M:
    """`module.to(dtype=dtype)`, typed: Diffusers' `to` is `functools.wraps`'d, which the checkers
    bind without `self`."""
    return module.to(dtype=dtype)


def text_encoder(config: Mapping[str, Any]) -> nn.Module:  # type: ignore[explicit-any]  # the checkpoint's JSON config
    """A FLUX.2 text encoder, as the whole model: Mistral-Small-3.2 for [dev], Qwen3 for [klein]."""
    built = AutoModelForImageTextToText if config["model_type"] == "mistral3" else AutoModelForCausalLM
    model: nn.Module = built.from_config(AutoConfig.for_model(**config), dtype=torch.bfloat16)  # type: ignore[no-untyped-call]  # Transformers' Auto classes are unannotated
    return model


def _conditioning_only(encoder: nn.Module, depth: int) -> None:
    """FLUX.2 reads its text encoder's hidden states up to `depth`: the later layers, the final norm
    and the head are never run, so they are not built (their stored bytes are skipped). With the norm
    an identity, hidden state `depth` stays the pre-norm output. [dev]'s prompts are text only, so
    Mistral3's vision tower and projector are not built either. `encoder` is Mistral3's or Qwen3's
    causal LM, by family."""
    inner = encoder.get_submodule("model")
    language = inner.get_submodule("language_model") if hasattr(inner, "language_model") else inner
    language.set_submodule("layers", nn.ModuleList(list(language.get_submodule("layers").children())[:depth]))
    language.set_submodule("norm", nn.Identity())
    encoder.set_submodule("lm_head", nn.Identity())
    if language is not inner:
        setattr(inner, "vision_tower", None)  # noqa: B010  # a module slot emptied: nn.Module's __setattr__ types no None
        setattr(inner, "multi_modal_projector", None)  # noqa: B010


def _armed(transformer: nn.Module, path: str) -> lora.Factors | None:
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
            transformer = construct(Flux2Transformer2DModel, config.section("transformer").mapping())
            encoder = text_encoder(text)
            vae = construct(AutoencoderKLFlux2, config.section("vae").mapping())
        _conditioning_only(encoder, max(TEXT_LAYERS[family]))
        if transformer.config["guidance_embeds"]:  # [dev]: its Turbo LoRA's sites
            transformer._cozy_turbo = None
            for path, module in transformer.named_modules():
                if isinstance(module, nn.Linear):
                    module.register_forward_hook(lora.Hook(partial(_armed, transformer, path)))
        self.family = family
        # The VAE is built as stored (f32 in BFL's `ae`, bf16 in Diffusers' repositories); a GPU
        # serves it at `VAE_DTYPE` (`Flux2.hold_vae`).
        stored = config.tensor_dtype("vae.encoder.conv_in.weight", default="f32")
        self.components: Components = {
            "transformer": _to(transformer, torch.bfloat16),
            "text_encoder": encoder.eval(),
            "vae": _to(vae, torch.float32 if stored == "f32" else torch.bfloat16),
        }


class Components(TypedDict):
    transformer: Flux2Transformer2DModel
    text_encoder: nn.Module
    vae: AutoencoderKLFlux2


def schedule(steps: int, tokens: int, sigmas: Sequence[float] | None = None) -> list[float]:
    """BFL's `get_schedule`: `steps` linear times (or the given sigmas), time-shifted by the
    empirical mu of the image's token count, ending at 0."""
    mu = compute_empirical_mu(tokens, steps)
    t = torch.linspace(1, 0, steps + 1) if sigmas is None else torch.tensor([*sigmas, 0.0])
    return [float(v) for v in (math.exp(mu) / (math.exp(mu) + (1 / t - 1))).tolist()]


def prepare_references(images: Sequence[Image.Image]) -> list[Image.Image]:
    """BFL's `default_prep`: RGB, its middle at most 8:1, scaled up to at least 64 px a side, capped at
    2024² pixels (1024² each of several), centre-cropped to the 16 px grid."""
    limit = SINGLE_REFERENCE_PIXELS if len(images) == 1 else MULTI_REFERENCE_PIXELS
    prepared = []
    for image in images:
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = min(image.width, 8 * image.height), min(image.height, 8 * image.width)
        image = image.crop(((image.width - w) // 2, (image.height - h) // 2, (image.width + w) // 2, (image.height + h) // 2))
        if min(image.size) < 64:
            scale = 64 / min(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
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


def _bn(vae: AutoencoderKLFlux2, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # FLUX.2's VAE batch norm tracks its running statistics.
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like)  # type: ignore[union-attr]
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config["batch_norm_eps"]).to(like)  # type: ignore[union-attr]
    return mean, std


class Flux2(Model[Flux2Graph]):
    """One FLUX.2 checkpoint: [dev] (embedded guidance) or a [klein], which runs as Base (CFG) or
    distilled (no guidance) as the function names it: the pipeline's `is_distilled` never reaches
    the model's config."""

    graph: Flux2Graph
    tokenizer: PixtralProcessor | PreTrainedTokenizerFast

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Flux2Graph)
        self.tokenizer = processor(loader.assets) if self.dev else tokenizer(loader.assets)

    @property
    def dev(self) -> bool:
        """[dev]: Mistral-conditioned, guidance-distilled, with an embedded guidance input."""
        return self.graph.family == "mistral3"

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str) -> torch.Tensor:
        encoder = self.graph.components["text_encoder"]
        device = torch.device(str(self.device))
        layers = TEXT_LAYERS[self.graph.family]
        with torch.inference_mode():
            # Diffusers types these by its own loaders' classes; the family picks the encoder.
            embeds: torch.Tensor
            if self.dev:
                embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                    encoder, self.tokenizer, prompt, dtype=torch.bfloat16, device=device,  # type: ignore[arg-type]
                    max_sequence_length=MAX_TOKENS, system_message=SYSTEM_MESSAGE, hidden_states_layers=layers,  # type: ignore[arg-type]
                )
            else:
                embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                    encoder, self.tokenizer, prompt, dtype=torch.bfloat16, device=device,  # type: ignore[arg-type]
                    max_sequence_length=MAX_TOKENS, hidden_states_layers=layers,  # type: ignore[arg-type]
                )
        return embeds.cpu()

    def hold_vae(self, tel: Telemetry) -> None:
        """Serve the VAE at `VAE_DTYPE` on a GPU from now on, before any VAE scope opens: the Runtime
        casts the stored bytes when it next fills them. A CPU keeps the stored dtype."""
        vae = self.graph.components["vae"]
        if torch.device(str(self.device)).type == "cuda" and next(vae.parameters()).dtype != VAE_DTYPE:
            hold_dtype(vae, VAE_DTYPE)
            tel.log("vae representation", dtype=str(VAE_DTYPE).removeprefix("torch."))

    @memoize
    @uses_components("vae")
    def encode_references(self, references: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized packed latents of the prepared references as one token sequence, with ids."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        tokens, ids = [], []
        with torch.inference_mode():
            for index, reference in enumerate(references):
                pixels = torch.from_numpy(np.array(reference)).permute(2, 0, 1).float() / 127.5 - 1.0
                latents = vae.encode(pixels[None].to(device=device, dtype=vae.dtype)).latent_dist.mode()
                latents = Flux2Pipeline._patchify_latents(latents.float())  # type: ignore[no-untyped-call]  # Diffusers' helper is unannotated
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
        progress: Callable[[int, int], None],
        start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BFL's Euler sampler in bf16 (`denoise`, `denoise_cfg`): [dev] embeds `guidance`, [klein] Base
        takes CFG against `negative`, [klein] distilled takes neither. With `start` (the canvas's packed
        normalized latents), the state begins as `start` noised to the first timestep."""
        tf = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        noise = rng.noise((1, int(tf.config["in_channels"]), rows, columns), seed)
        cfg = negative is not None
        steps = len(timesteps) - 1
        with torch.no_grad():
            x = noise.to(torch.bfloat16).flatten(2).transpose(1, 2).to(device)
            if start is not None:
                x = ((1.0 - timesteps[0]) * start.to(device).float() + timesteps[0] * x.float()).to(torch.bfloat16)
            ids = grid_ids(rows, columns)
            text = embeds if negative is None else torch.cat([negative, embeds])
            text = text.to(device=device, dtype=torch.bfloat16)
            batch = text.shape[0]
            reference = None
            if references is not None:
                reference = references[0].to(device).expand(batch, -1, -1)
                ids = torch.cat([ids, references[1]])
            ids, txt_ids = ids.to(device), text_ids(text.shape[1]).to(device)
            embedded = None if guidance is None or cfg else torch.full((batch,), guidance, device=device)
            # BFL's timestep is the bf16 of each time; its Euler adds the bf16 product of the
            # velocity and the f32 of the time delta to the bf16 state.
            times = torch.tensor(timesteps[:-1], dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
            for i, (now, then) in enumerate(pairwise(timesteps)):
                tokens = x.expand(batch, -1, -1)
                if reference is not None:
                    tokens = torch.cat([tokens, reference], dim=1)
                pred = tf(
                    hidden_states=tokens, encoder_hidden_states=text, timestep=times[i].expand(batch),
                    img_ids=ids, txt_ids=txt_ids, guidance=embedded, return_dict=False,
                )[0][:, : x.shape[1]]
                if cfg:
                    unconditional, conditional = pred.chunk(2)
                    pred = unconditional + guidance * (conditional - unconditional)
                preview(lambda: x - now * pred, "flux2", grid=(rows, columns))
                x = x + (pred.float() * (then - now)).to(pred.dtype)
                progress(i, steps)
        return x.cpu()

    @uses_components("vae")
    def decode(self, tokens: torch.Tensor, *, width: int, height: int) -> Image.Image:
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        with torch.inference_mode():
            latents = tokens.to(device=device, dtype=torch.float32).view(1, rows, columns, -1).permute(0, 3, 1, 2)
            mean, std = _bn(vae, latents)
            latents = Flux2Pipeline._unpatchify_latents(latents * std + mean)  # type: ignore[no-untyped-call]  # Diffusers' helper is unannotated
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0][0]
            if not torch.isfinite(image).all():
                raise OutputError(
                    "FLUX.2's decoder produced non-finite pixels, so there is no picture: try another seed, "
                    "and if it repeats another lane with `--lane bf16`",
                    code="output_integrity",
                )
            pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)
