"""Component-staged Qwen Image 2.1 over Diffusers' own pipeline.

The architecture and sampling are Diffusers' (`QwenImage21Pipeline`, Diffusers 0.41.0). Runtime
constructs and fills the modules and owns their residency; no Diffusers weight loader, device
placement, or offload hook takes part in serving.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21ResidualUpBlock
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21TransformerBlock,
)
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from transformers import (
    AutoModelForImageTextToText,
    Qwen3VLConfig,
)
from transformers import initialization as transformer_init

from cozy_runtime import fusion, rng, tiles, vlm
from cozy_runtime.author import (
    Config,
    ConformanceError,
    Loader,
    MemoDistribution,
    Model,
    Telemetry,
    denoise_loop,
    hold_dtype,
    uses_components,
)
from cozy_runtime.author.kernels import Plan

from ._prepared import _FlowEuler, _Layout


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
        with transformer_init.no_init_weights():
            self.components: dict[str, Any] = {
                "transformer": QwenImage21Transformer2DModel.from_config(
                    config.section("transformer").mapping()
                ).to(dtype=torch.bfloat16),
                # Construction dtype leaves derived RoPE buffers in Transformers' F32.
                # Casting the completed module would round those frequencies to BF16.
                "text_encoder": AutoModelForImageTextToText.from_config(
                    Qwen3VLConfig(**config.section("text_encoder").mapping()),
                    dtype=torch.bfloat16,
                ),
                "vae": AutoencoderKLQwenImage21.from_config(config.section("vae").mapping()),
            }
        vae = self.components["vae"]
        vae.to(dtype=_stored(config, "vae", vae))
        # Runtime pages a model by its `_no_split_modules`; Diffusers names none for this VAE, so
        # its 1.35 GB of F32 weights, encoder included, stayed resident through a decode and left
        # a 3 GB card short for the decode's activations. Each residual, attention and resample
        # block is one page instead (every one is called as a module, also by `_decoded`).
        vae._no_split_modules = [
            "QwenImage21ResidualBlock",
            "QwenImage21AttentionBlock",
            "QwenImage21Resample",
        ]
        # The pipeline reads hidden states only, so the 1.24 GB language-model head is not
        # built; Runtime skips its stored weight.
        self.components["text_encoder"].lm_head = torch.nn.Identity()
        self.scheduler_config = config.section("scheduler").mapping()
        # The processor folder's config file, which TensorFS keeps as its component's config.
        self.processor_config = config.section("processor").mapping() if "processor" in config.keys() else {}


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

    def hold_vae(self, tel: Telemetry) -> None:
        """Run the VAE in bf16 where the GPU computes it natively, as ComfyUI stages it and as Turbo
        stores it; the base release stores F32, whose decode took 2.0 s on an 8 GB RTX 3070 against
        ComfyUI's 0.93 s. The Runtime casts the stored bytes when it next fills them; call this
        before any VAE scope opens."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        if device.type != "cuda" or not torch.cuda.is_bf16_supported(including_emulation=False):
            return
        if next(vae.parameters()).dtype != torch.bfloat16:
            hold_dtype(vae, torch.bfloat16)
            tel.log("vae representation", dtype="bfloat16")

    def _pipeline(self, sigmas: Sequence[float] | None = None) -> _StagedPipeline:
        pipeline = _StagedPipeline(
            **self.graph.components,
            processor=self.processor,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            sample_sigmas=None if sigmas is None else list(sigmas),
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
        sigmas: Sequence[float] | None = None,
        reference_images: list[Any] | None = None,
        reference_latents: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ) -> Any:
        """Text-to-image with an unpadded prompt runs the cached loop; edits and padded prompts
        run the Diffusers pipeline. `sigmas`: a distilled schedule (its length is `steps`), else
        the pipeline's uniform spacing."""
        cancel()
        if reference_latents is None and reference_images is None and mask is None:
            return self._denoise_cached(
                embeds, width=width, height=height, steps=steps, seed=seed, on_step=on_step,
                sigmas=sigmas,
            )
        return self._denoise_eager(
            embeds,
            mask,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            sigmas=sigmas,
            on_step=on_step,
            cancel=cancel,
            reference_images=reference_images,
            reference_latents=reference_latents,
            image_pad_mask=image_pad_mask,
        )

    def _denoise_cached(
        self,
        embeds: Any,
        *,
        width: int,
        height: int,
        steps: int,
        seed: int,
        on_step: Callable[[int], None],
        sigmas: Sequence[float] | None = None,
    ) -> Any:
        """Extract step 0 eagerly, then steps ``1..N-1`` in cached mode as one `denoise_loop`.

        The preparation mirrors `QwenImage21Pipeline.__call__` for the text-to-image case,
        reusing the pipeline's own `prepare_latents`/`calculate_shift`/`retrieve_timesteps` so
        the schedule, noise and layout match the Diffusers loop exactly.
        """
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        pipeline = self._pipeline(sigmas)
        scheduler = pipeline.scheduler
        multiple_of = pipeline.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        with torch.inference_mode(False), torch.no_grad():
            embeds = embeds.to(device=device, dtype=transformer.dtype)
            latents, _ = pipeline.prepare_latents(
                None,
                1,
                transformer.config.in_channels,
                height,
                width,
                transformer.dtype,
                device,
                rng.generator(seed),
                None,
            )
            # The distilled schedule when given (Turbo), else the uniform spacing the pipeline uses
            # when no `sigmas` is passed. The dynamic shift follows the canvas.
            spacing = (
                np.array(sigmas, dtype=np.float32) if sigmas else np.linspace(1.0, 1 / steps, steps)
            )
            mu = calculate_shift(
                latents.shape[1],
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )
            timesteps, steps = retrieve_timesteps(scheduler, steps, device, sigmas=spacing, mu=mu)
            scheduler.set_begin_index(0)

            # The joint-sequence image mask: no condition images for text-to-image, then one
            # vision-language slot per 2x2 group of target latents, as the pipeline appends.
            image_pad_mask = embeds.new_zeros((1, embeds.shape[1]), dtype=torch.bool)
            image_pad_mask = torch.cat(
                [image_pad_mask, image_pad_mask.new_ones(1, latents.shape[1] // 4)], dim=1
            )
            img_shapes = [[(1, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor)]]
            layout = _Layout(transformer, img_shapes, image_pad_mask, None, device)

            # Step 0 prefills the prefix K/V over the full joint sequence.
            kv_cache = QwenImage21KVCache(len(transformer.transformer_blocks))
            timestep = timesteps[0:1].to(transformer.dtype) / 1000
            noise = transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=None,
                img_shapes=img_shapes,
                img_mask=image_pad_mask,
                attention_kwargs={},
                kv_cache=kv_cache,
                kv_cache_mode="extract",
                return_dict=False,
            )[0]
            noise = noise[:, -latents.shape[1] :]
            latents = scheduler.step(noise, timesteps[0], latents, return_dict=False)[0]
            on_step(0)
            if steps < 2:
                return latents
            euler = _FlowEuler(scheduler, device)
        blocks = transformer.transformer_blocks
        caches = [kv_cache.get_layer(index) for index in range(len(blocks))]
        rotary, target, attention = layout.rotary_emb, layout.modulation_mask, layout.attention_mask

        def step(i: torch.Tensor, latents: torch.Tensor) -> None:
            with torch.no_grad():
                hidden = transformer.img_in(latents)
                temb = transformer.time_text_embed(euler.timestep(i), hidden)
                modulation = transformer.modulation(temb)
                for block, cache in zip(blocks, caches, strict=True):
                    hidden = block(
                        hidden_states=hidden,
                        modulation=modulation,
                        rotary_emb=rotary,
                        attention_mask=attention,
                        target_token_mask=target,
                        layer_cache=cache,
                        kv_cache_mode="cached",
                        cache_write_slice=None,
                        segments=None,
                        key_valid=None,
                    )
                hidden = transformer.norm_out(hidden, temb, target)
                noise = transformer.proj_out(hidden)
                latents.copy_((latents.float() + euler.delta(i) * noise).to(latents.dtype))

        denoise_loop(step, range(1, steps), state=(latents,), on_step=on_step)
        return latents

    def _denoise_eager(
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
        sigmas: Sequence[float] | None = None,
        reference_images: list[Any] | None = None,
        reference_latents: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ) -> Any:
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        pipeline = self._pipeline(sigmas)
        pipeline.reference_latents = reference_latents
        if image_pad_mask is not None:
            pipeline.reference_pad_mask = image_pad_mask.to(device=device)

        def observed(_pipeline: Any, step: int, _timestep: Any, values: Any) -> Any:
            cancel()
            on_step(step)
            return values

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
    def decode(self, latents: Any, *, width: int, height: int, tel: Telemetry) -> Any:
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        with torch.inference_mode():
            latents = QwenImage21Pipeline._unpack_latents(latents, height, width, 16)
            latents = latents.to(device=device, dtype=vae.dtype)
            mean = latents.new_tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
            std = latents.new_tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
            z = latents * std + mean
            decoded = _decoded_whole(vae, z)
            if decoded is None:
                decoded = _decoded_tiled(vae, z, tel)
            return decoded.clamp(-1.0, 1.0)[:, :, 0]


#: Tile sides (px) for a canvas the whole decode cannot fit, largest first. 512 px tiles err 7 dB
#: less against the whole decode than Diffusers' 256 px default (42.9 vs 35.2 dB at 4 MP), whose
#: seams show in flat sky, but they do not fit an RTX 3070 with 3 GB free.
_TILES = (512, 256)


def _decoded_tiled(vae: Any, z: torch.Tensor, tel: Telemetry) -> torch.Tensor:
    """Diffusers' tiled decode in the largest tiles that fit."""
    *larger, smallest = _TILES
    for tile in larger:
        try:
            return _tiled(vae, z, tile, tel)
        except torch.OutOfMemoryError as error:
            error.__traceback__ = None  # this attempt's tiles go with its frames
    return _tiled(vae, z, smallest, tel)


def _tiled(vae: Any, z: torch.Tensor, tile: int, tel: Telemetry) -> torch.Tensor:
    """F32 like the whole decode, `tile` px tiles blended with their neighbours over a quarter of
    their side: Diffusers' bits, each tile decoded as the whole canvas is (`_decoded`, no feature
    cache), a few kernels a blend (`cozy_runtime.tiles`)."""
    decoded = tiles.tiled(
        lambda t: _decoded(vae.decoder, vae.post_quant_conv(t)),
        z,
        scale=vae.spatial_compression_ratio,
        side=tile,
        stride=tile * 3 // 4,
    )
    tel.log("tiled vae decode", tile_px=tile)
    return decoded


def _decoded_whole(vae: Any, z: torch.Tensor) -> torch.Tensor | None:
    """The pipeline's final decode, with no cast of F32 source weights, as one frame: Diffusers'
    `decode` keeps every convolution's input for a next frame an image never has (its feature
    cache, half the decode's memory); the same bits without. None when the whole canvas does
    not fit the device (Qwen at 1 MP with 3 GB free); the caller then tiles."""
    try:
        return _decoded(vae.decoder, vae.post_quant_conv(z))
    except torch.OutOfMemoryError as error:
        error.__traceback__ = None  # the whole decode's activations go with its frames
        return None


def _decoded(decoder: Any, x: torch.Tensor) -> torch.Tensor:
    """`decoder(x, first_chunk=True)` without its residual up-blocks' input copies: each block's
    shortcut is its input itself (nothing writes it in place), and the last, with no shortcut,
    keeps none. Diffusers clones it regardless, 4.8 GB at 2048² F32 in the last block alone,
    which put a 4 MP decode past a 24 GB card (Runtime 0.34, run 6: 24.2 GB live)."""
    x = decoder.mid_block(decoder.conv_in(x))
    for block in decoder.up_blocks:
        if not isinstance(block, QwenImage21ResidualUpBlock):
            x = block(x, first_chunk=True)
            continue
        shortcut = x if block.avg_shortcut is not None else None
        for resnet in block.resnets:
            x = resnet(x)
        if block.upsampler is not None:
            x = block.upsampler(x)
        if shortcut is not None:
            x = x + block.avg_shortcut(shortcut, first_chunk=True)
        del shortcut
    return decoder.conv_out(decoder.nonlinearity(decoder.norm_out(x)))
