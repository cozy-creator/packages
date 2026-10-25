"""Component-staged Qwen Image 2.1 over Diffusers' own pipeline.

The architecture and sampling are Diffusers' (`QwenImage21Pipeline`, Diffusers 0.41.0). Runtime
constructs and fills the modules and owns their residency; no Diffusers weight loader, device
placement, or offload hook takes part in serving.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import torch
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from transformers import (
    AutoModelForImageTextToText,
    Qwen3VLConfig,
)
from transformers import initialization as transformer_init

from cozy_runtime import fusion, rng, vlm
from cozy_runtime.author import (
    Config,
    ConformanceError,
    Loader,
    MemoDistribution,
    Model,
    uses_components,
)
from cozy_runtime.author.kernels import Plan


def _modulation(block: Any, arguments: Mapping[str, Any]) -> fusion.Modulation:
    """Diffusers' `_select_modulation_rows` as a row per token: a target-image token reads its
    sample's row, the text and condition-image prefix the trailing t=0 row."""
    table, mask = arguments["modulation"], arguments["target_token_mask"]
    batch, tokens = arguments["hidden_states"].shape[:2]
    samples = torch.arange(batch, device=table.device)[:, None]
    if mask is None:
        index = samples.expand(batch, tokens)
    else:
        index = torch.where(mask[None, :], samples, table.shape[0] - 1)
    return fusion.Modulation(index.reshape(-1), *table.chunk(4, dim=-1))


#: The 2.1 block as Runtime fuses it (scale, gate | scale, gate; tanh gates; no shift).
BLOCK = fusion.DiTBlock(
    QwenImage21TransformerBlock,
    width=4096,
    head_dim=128,
    modulation=_modulation,
    norms=("img_norm1", "img_norm2"),
    norm="layer",
    attention="attn",
    arguments=(
        "attention_mask",
        "rotary_emb",
        "layer_cache",
        "kv_cache_mode",
        "cache_write_slice",
        "segments",
        "key_valid",
    ),
    mlp=fusion.GatedMLP("img_mlp", gate="gate_layer", up="proj", down="out"),
    rope=fusion.Rope("rotary_emb", interleaved=True),
    gate="tanh",
)


class _Progress[T]:
    """Pipeline iteration without tqdm: a disabled bar still starts a monitor thread, and
    Runtime reports progress through Telemetry."""

    def __init__(self, iterable: Iterable[T] | None = None, total: int | None = None) -> None:
        self.iterable = iterable

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            raise TypeError("progress has no iterable")
        return iter(self.iterable)

    def update(self, n: int = 1) -> None:
        pass

    def __enter__(self) -> _Progress[T]:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _section(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Qwen Image {name} configuration must be a mapping")
    return dict(value)


_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}


def _stored(config: Config, component: str, module: torch.nn.Module) -> torch.dtype:
    """The one dtype the checkpoint stores `component`'s parameters in; f32 built from config alone.
    Qwen-Image-2.1 stores its VAE in f32, Turbo in bf16."""
    stored = {config.tensor_dtype(f"{component}.{name}", default="f32") for name, _ in module.named_parameters()}
    dtype = _DTYPES.get(next(iter(stored))) if len(stored) == 1 else None
    if dtype is None:
        raise ConformanceError(
            f"{component} stores {sorted(stored)}; Qwen Image builds it in one of {sorted(_DTYPES)}",
            code="artifact_dtype",
            fields=[component],
        )
    return dtype


class QwenImage21Graph:
    """One native construction census, with source checkpoint precision preserved."""

    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            self.components: dict[str, Any] = {
                "transformer": QwenImage21Transformer2DModel.from_config(
                    _section(mapping["transformer"], "transformer")
                ).to(dtype=torch.bfloat16),
                # Construction dtype leaves derived RoPE buffers in Transformers' F32.
                # Casting the completed module would round those frequencies to BF16.
                "text_encoder": AutoModelForImageTextToText.from_config(
                    Qwen3VLConfig(**_section(mapping["text_encoder"], "text_encoder")),
                    dtype=torch.bfloat16,
                ),
                "vae": AutoencoderKLQwenImage21.from_config(_section(mapping["vae"], "vae")),
            }
        vae = self.components["vae"]
        vae.to(dtype=_stored(config, "vae", vae))
        # The pipeline reads hidden states only, so the 1.24 GB language-model head is not
        # built; Runtime skips its stored weight.
        self.components["text_encoder"].lm_head = torch.nn.Identity()
        self.scheduler_config = _section(mapping["scheduler"], "scheduler")
        # The processor folder's config file, which TensorFS keeps as its component's config.
        self.processor_config = _section(mapping.get("processor", {}), "processor")
        # A distilled checkpoint (Turbo) names its own schedule; its length is the step count.
        sigmas = _section(mapping.get("model_index", {}), "model_index").get("sample_sigmas")
        self.sample_sigmas = [float(sigma) for sigma in sigmas] if sigmas else None


def build_qwen_image21(config: Config) -> QwenImage21Graph:
    return QwenImage21Graph(config)


class _StagedPipeline(QwenImage21Pipeline):
    reference_latents: torch.Tensor | None = None
    reference_pad_mask: torch.Tensor | None = None
    #: The device the Runtime runs the model on (`Model.device`).
    device_placed: torch.device = torch.device("cpu")

    def progress_bar[T](
        self, iterable: Iterable[T] | None = None, total: int | None = None
    ) -> _Progress[T]:
        return _Progress(iterable, total)

    @property
    def _execution_device(self) -> torch.device:
        # Diffusers reads the first registered module's (the VAE's) parameters, which may sit
        # paged on the CPU while the Runtime runs the model: the Runtime's device instead.
        return self.device_placed

    def encode_prompt(self, *args: Any, **kwargs: Any) -> Any:
        # The denoiser must reuse the vision token positions from the separately
        # leased text encoder. Diffusers otherwise rejects precomputed image embeds.
        if self.reference_pad_mask is not None:
            kwargs["image_pad_mask"] = self.reference_pad_mask
        return super().encode_prompt(*args, **kwargs)

    def prepare_latents(self, images: Any, *args: Any, **kwargs: Any) -> Any:
        # References were encoded while the VAE had its own residency lease.
        # Never invoke that parked component from the transformer-only lease.
        latents, image_latents = super().prepare_latents(
            None if self.reference_latents is not None else images, *args, **kwargs
        )
        if self.reference_latents is not None:
            image_latents = self.reference_latents.to(device=latents.device, dtype=latents.dtype)
        return latents, image_latents


#: Code the encoders read beyond their own bodies: their memo key binds it.
_ENCODER_CODE = (
    _StagedPipeline,
    vlm.processor,
    MemoDistribution("diffusers"),
    MemoDistribution("transformers"),
)


class QwenImage21Model(Model[QwenImage21Graph], encoded_leaves="accept"):
    graph: QwenImage21Graph
    processor: Any

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(QwenImage21Graph, factory=build_qwen_image21)
        self.processor = vlm.processor(loader.assets, "processor/", self.graph.processor_config)

    def execution_plan(self, plan: Plan) -> str:
        return fusion.install(plan, BLOCK, self.graph.components["transformer"])

    @property
    def sample_sigmas(self) -> list[float] | None:
        return self.graph.sample_sigmas

    def _pipeline(self) -> _StagedPipeline:
        pipeline = _StagedPipeline(
            **self.graph.components,
            processor=self.processor,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            sample_sigmas=self.graph.sample_sigmas,
        )
        pipeline.device_placed = torch.device(str(self.device))
        return pipeline

    @uses_components("text_encoder", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode(self, prompt: str) -> tuple[Any, Any]:
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            embeds, mask, _ = self._pipeline().encode_prompt(
                prompt, device=torch.device(str(self.device))
            )
        return embeds, mask

    @uses_components("vae", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode_reference_images(self, images: list[Any]) -> tuple[list[Any], torch.Tensor]:
        """Encode each reference once, using Diffusers' 1 MP conditioning size."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        pipeline = self._pipeline()
        resized, packed = [], []
        with torch.inference_mode():
            for image in images:
                image = image.convert("RGBA")
                width, height, _ = calculate_dimensions(1024 * 1024, image.width / image.height)
                resized.append(pipeline.image_processor.resize(image, width=width, height=height))
                pixels = (
                    pipeline.image_processor.preprocess(image, width=width, height=height)
                    .unsqueeze(2)
                    .to(device=device, dtype=vae.dtype)
                )
                # argmax encoding is deterministic and consumes no sampler RNG.
                encoded = pipeline._encode_vae_image(pixels, generator=None)
                packed.append(
                    pipeline._pack_latents(
                        encoded, 1, pipeline.latent_channels, encoded.shape[3], encoded.shape[4]
                    ).cpu()
                )
        return resized, torch.cat(packed, dim=1)

    @uses_components("text_encoder", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode_image_prompt(self, prompt: str, images: list[Any]) -> tuple[Any, Any, Any]:
        """Encode the instruction and ordered reference images as one condition."""
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            encoded: tuple[Any, Any, Any] = self._pipeline().encode_prompt(
                prompt, image=images, device=torch.device(str(self.device))
            )
        return encoded

    @uses_components("transformer")
    def denoise(
        self,
        embeds: Any,
        mask: Any,
        *,
        width: int,
        height: int,
        steps: int,
        seed: int,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
        reference_images: list[Any] | None = None,
        reference_latents: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ) -> Any:
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        pipeline = self._pipeline()
        pipeline.reference_latents = reference_latents
        if image_pad_mask is not None:
            pipeline.reference_pad_mask = image_pad_mask.to(device=device)

        def observed(_pipeline: Any, step: int, _timestep: Any, values: Any) -> Any:
            cancel()
            on_step(step)
            return values

        cancel()
        with torch.inference_mode():
            return pipeline(
                image=reference_images,
                prompt_embeds=embeds.to(device=device, dtype=transformer.dtype),
                prompt_embeds_mask=None if mask is None else mask.to(device=device),
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=rng.generator(seed),
                output_type="latent",
                callback_on_step_end=observed,
            ).images

    @uses_components("vae")
    def decode(self, latents: Any, *, width: int, height: int) -> Any:
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        with torch.inference_mode():
            latents = QwenImage21Pipeline._unpack_latents(latents, height, width, 16)
            latents = latents.to(device=device, dtype=vae.dtype)
            mean = latents.new_tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
            std = latents.new_tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
            # The pipeline's final decode, with no cast of F32 source weights, as one frame:
            # Diffusers' `decode` keeps every convolution's input for a next frame an image
            # never has (its feature cache, half the decode's memory); the same bits without.
            x = vae.post_quant_conv(latents * std + mean)
            return vae.decoder(x, first_chunk=True).clamp(-1.0, 1.0)[:, :, 0]
