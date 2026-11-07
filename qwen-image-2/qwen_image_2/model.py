"""Component-staged Qwen Image 2.1 over Diffusers' own pipeline.

The architecture and sampling are Diffusers' (`QwenImage21Pipeline`, Diffusers 0.41.0). Runtime
constructs and fills the modules and owns their residency; no Diffusers weight loader, device
placement, or offload hook takes part in serving.

A caller's LoRAs adapt the DiT through Runtime's PEFT layers (`__adapter_compatibility__`). The
Turbo overlay (`turbo.FunAccLoRA`) reaches the DiT's linears through permanent hooks
(`cozy_runtime.lora.Hook`), which Runtime bakes into the resident weights while it is armed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import (
    QwenImage21CausalConv3d,
    QwenImage21Decoder3d,
    QwenImage21Encoder3d,
    QwenImage21ResidualBlock,
    QwenImage21ResidualUpBlock,
    QwenImage21RMS_norm,
    QwenImage21Upsample,
    _patchify,
)
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21KVLayerCache,
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
from transformers.modeling_utils import local_torch_dtype

from cozy_runtime import fusion, lora, rng, tiles, vlm
from cozy_runtime.author import (
    AdapterCompatibility,
    Config,
    ConformanceError,
    Loader,
    MemoDistribution,
    Model,
    Telemetry,
    denoise_loop,
    hold_dtype,
    retained,
    uses_components,
)
from cozy_runtime.author.kernels import Plan

from ._prepared import _FlowEuler, _full_step, _Joint, _Layout, _cached_step


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
#: The VAE's dtype on CUDA with native bf16, never fp16: on a transparent background the decoder's
#: last up-blocks carry activations up to 3.7e6 (F32, the release gate's RGBA request), far past
#: fp16's 65504 (0.1.12 decoded NaN); bf16 holds them at 53.8 dB against F32.
VAE_DTYPE = torch.bfloat16
#: The VAE's RMS norms in one kernel each, with the SiLU each residual block and each head applies
#: after them; Diffusers computes them in F32 copies of every activation.
VAE_NORM = fusion.ChannelNorm(
    QwenImage21RMS_norm,
    silu=(
        (QwenImage21ResidualBlock, ("norm1", "norm2")),
        (QwenImage21Decoder3d, ("norm_out",)),
        (QwenImage21Encoder3d, ("norm_out",)),
    ),
)
#: A lone frame's convolutions without the padded copy before each, and its nearest upsampling in
#: bf16 (Diffusers upsamples in F32).
VAE_FRAME = fusion.LoneFrame(QwenImage21CausalConv3d, upsample=QwenImage21Upsample)


class _RetainedLayer(QwenImage21KVLayerCache):
    """A layer's prefix K/V held by the Runtime (`retained`): on the card while there is room,
    else in pinned host RAM, filled ahead of the layer's next step."""

    def __init__(self, held: Any) -> None:
        super().__init__()
        self.held, self.state = held, None

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.state = self.held.put(k, v)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.state.get()


class _RetainedKV(QwenImage21KVCache):
    """Diffusers' prefix cache, its layers held by the Runtime. Two references' 8k conditioning
    tokens keep 132 MiB a block, 7.9 GB over the stack: kept as tensors they outgrew an 8 GB card
    in step 1 (runs 39611, 39615)."""

    def __init__(self, num_layers: int, held: Any) -> None:
        self.layer_caches = [_RetainedLayer(held) for _ in range(num_layers)]


_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}

#: The DiT's linears an overlay adapts, (in, out): every one but the output projection, which a
#: parallel decoder replaces per step (Fun Acc adapts all 231).
_BLOCK_SITES = {
    "attn.to_q": (4096, 4096),
    "attn.to_k": (4096, 4096),
    "attn.to_v": (4096, 4096),
    "attn.to_out.0": (4096, 4096),
    "img_mlp.gate_layer": (4096, 12288),
    "img_mlp.proj": (4096, 12288),
    "img_mlp.out": (12288, 4096),
}
SITES: dict[str, tuple[int, int]] = {
    "img_in": (64, 4096),
    "txt_in.in_layer": (4096, 4096),
    "txt_in.out_layer": (4096, 4096),
    "modulation.1": (4096, 16384),
    "time_text_embed.timestep_embedder.linear_1": (256, 4096),
    "time_text_embed.timestep_embedder.linear_2": (4096, 4096),
    "norm_out.linear": (4096, 4096),
    **{f"transformer_blocks.{i}.{site}": io for i in range(32) for site, io in _BLOCK_SITES.items()},
}
#: The output projection's (out, in).
PROJ_OUT = (64, 4096)


def _armed(transformer: nn.Module, path: str) -> nn.Module | None:
    """The overlay's factors at `path` while one is armed (`_cozy_overlay`), else None."""
    armed = getattr(transformer, "_cozy_overlay", None)
    return None if armed is None else armed.get(path)


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
        # Construction dtype leaves derived buffers F32, as upstream loads them: the DiT's
        # timestep `freqs` and the text encoder's RoPE. A cast of the built module would round them.
        with transformer_init.no_init_weights():
            with local_torch_dtype(torch.bfloat16):
                transformer = QwenImage21Transformer2DModel.from_config(
                    config.section("transformer").mapping()
                )
            transformer._cozy_overlay = None
            for path in SITES:
                transformer.get_submodule(path).register_forward_hook(lora.Hook(partial(_armed, transformer, path)))
            self.components: dict[str, Any] = {
                "transformer": transformer,
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
    #: The device the Runtime runs the model on (`Model.device`).
    device_placed: torch.device = torch.device("cpu")

    @property
    def _execution_device(self) -> torch.device:
        # Diffusers reads the first registered module's (the VAE's) parameters, which may sit
        # paged on the CPU while the Runtime runs the model: the Runtime's device instead.
        return self.device_placed


#: Code the encoders read beyond their own bodies: their memo key binds it.
_ENCODER_CODE = (
    _StagedPipeline,
    vlm.processor,
    MemoDistribution("diffusers"),
    MemoDistribution("transformers"),
)


class QwenImage21Model(
    Model[QwenImage21Graph],
    encoded_leaves="accept",
    # The 32 DiT blocks attend at 8 bits (comfy-kitchen INT8 on RTX 40-class cards): attention is
    # two thirds of a 4 MP step. A 16-bit request attends exactly.
    attention_bits={"transformer.transformer_blocks.": 8},
):
    #: Caller LoRAs on the DiT, at any strength: the Runtime bakes them into the weights.
    __adapter_compatibility__ = (AdapterCompatibility("lora", "", ("transformer",), -math.inf, math.inf),)

    graph: QwenImage21Graph
    processor: Any

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(QwenImage21Graph, factory=build_qwen_image21)
        self.processor = vlm.processor(loader.assets, "processor/", self.graph.processor_config)

    def execution_plan(self, plan: Plan) -> str:
        vae = self.graph.components["vae"]
        installed = (
            fusion.install(plan, BLOCK, self.graph.components["transformer"]),
            fusion.install_norms(plan, VAE_NORM, self.graph.components["vae"]),
            fusion.install_lone_frame(plan, VAE_FRAME, vae.decoder, vae.post_quant_conv),
        )
        return "+".join(identity for identity in installed if identity)

    def hold_vae(self, tel: Telemetry) -> None:
        """Run the VAE in bf16 where the GPU computes it natively, as ComfyUI stages it and as Turbo
        stores it; the base release stores F32, whose decode took 2.0 s on an 8 GB RTX 3070 against
        ComfyUI's 0.93 s. The Runtime casts the stored bytes when it next fills them; call this
        before any VAE scope opens."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        if device.type != "cuda" or not torch.cuda.is_bf16_supported(including_emulation=False):
            return
        if next(vae.parameters()).dtype != VAE_DTYPE:
            hold_dtype(vae, VAE_DTYPE)
            tel.log("vae representation", dtype="bfloat16")

    def _pipeline(
        self, sigmas: Sequence[float] | None = None, scheduler: Mapping[str, Any] | None = None
    ) -> _StagedPipeline:
        """`scheduler`: fields an overlay's sampling sets over the checkpoint's scheduler."""
        pipeline = _StagedPipeline(
            **self.graph.components,
            processor=self.processor,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(
                {**self.graph.scheduler_config, **(scheduler or {})}
            ),
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
                encoded = _encoded(vae, pixels)
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
        sigmas: Sequence[float] | None = None,
        scheduler: Mapping[str, Any] | None = None,
        heads: torch.Tensor | None = None,
        reference_images: list[Any] | None = None,
        reference_latents: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ) -> Any:
        """`QwenImage21Pipeline.__call__`'s loop, the references' latent tokens before the target's.
        Its preparation mirrors the pipeline's, reusing its `prepare_latents`, `calculate_shift`
        and `retrieve_timesteps`, so the schedule, noise and layout match the Diffusers loop.

        Text and reference tokens modulate from t = 0, so their K/V are every step's: step 0
        extracts them into the Runtime's store (`retained`) and steps 1..N-1 run the cached pass
        over the target tokens as one `denoise_loop`. With no room for them, or a padded prompt,
        every step is the whole joint pass. `sigmas`: a distilled schedule (its length is `steps`),
        else the pipeline's uniform spacing; `scheduler`, an overlay's fields over the checkpoint's.
        `heads`: a parallel decoder's output projection per step (PDD), whose prediction, and so the
        state, is F32."""
        transformer = self.graph.components["transformer"]
        if not transformer.config.causal_condition:
            raise ConformanceError("Qwen Image 2.1 denoises with causal conditioning", code="artifact_config")
        device = torch.device(str(self.device))
        pipeline = self._pipeline(sigmas, scheduler)
        scheduler = pipeline.scheduler
        blocks = transformer.transformer_blocks
        multiple_of = pipeline.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        factor = pipeline.vae_scale_factor
        sizes = [calculate_dimensions(1024 * 1024, image.width / image.height)[:2] for image in reference_images or []]

        with torch.inference_mode(False), torch.no_grad():
            embeds = embeds.to(device=device, dtype=transformer.dtype)
            mask = None if mask is None else mask.to(device=device)
            references = () if reference_latents is None else (reference_latents.to(device=device, dtype=transformer.dtype),)
            latents, _ = pipeline.prepare_latents(
                None, 1, transformer.config.in_channels, height, width, transformer.dtype, device,
                rng.generator(seed), None,
            )
            # The distilled schedule when given (Turbo), else the uniform spacing the pipeline uses
            # when no `sigmas` is passed. The dynamic shift follows the canvas.
            spacing = np.array(sigmas, dtype=np.float32) if sigmas else np.linspace(1.0, 1 / steps, steps)
            mu = calculate_shift(
                latents.shape[1],
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )
            timesteps, steps = retrieve_timesteps(scheduler, steps, device, sigmas=spacing, mu=mu)
            scheduler.set_begin_index(0)
            euler = _FlowEuler.from_scheduler(scheduler, device)
            # The joint sequence's image slots: the text encoder's for each reference, then one per
            # 2x2 group of target latents, as the pipeline appends them.
            slots = image_pad_mask if image_pad_mask is not None else embeds.new_zeros((1, embeds.shape[1]), dtype=torch.bool)
            img_mask = torch.cat([slots.to(device), slots.new_ones(1, latents.shape[1] // 4, device=device)], dim=1)
            img_shapes = [[*((1, h // factor, w // factor) for w, h in sizes), (1, height // factor, width // factor)]]
            layout = _Layout(transformer, img_shapes, img_mask, device)
            config = transformer.config
            nbytes = 2 * layout.prefix_len * config.num_attention_heads * config.attention_head_dim
            held = None if mask is not None else retained(nbytes * transformer.dtype.itemsize * len(blocks))
            if held is None:
                joint = _Joint(transformer, img_shapes, img_mask, mask, device)
                # Under a parallel decoder the state is F32 from the first step, as `scheduler.step` makes it.
                latents = latents if heads is None else latents.float()
            else:
                # Step 0 prefills each layer's prefix K/V into the store over the full joint sequence.
                kv_cache = _RetainedKV(len(blocks), held)
                first = None
                if heads is not None:
                    first = transformer.proj_out.register_forward_hook(lambda _m, a, _o: F.linear(a[0], heads[0]))
                try:
                    noise = transformer(
                        hidden_states=torch.cat([*references, latents], dim=1),
                        timestep=timesteps[0:1].to(transformer.dtype) / 1000,
                        encoder_hidden_states=embeds,
                        encoder_hidden_states_mask=None,
                        img_shapes=img_shapes,
                        img_mask=img_mask,
                        attention_kwargs={},
                        kv_cache=kv_cache,
                        kv_cache_mode="extract",
                        return_dict=False,
                    )[0]
                finally:
                    if first is not None:
                        first.remove()
                noise = noise[:, -latents.shape[1] :]
                # A parallel decoder's prediction is F32, and so is the state from here (PDD's
                # `native_time_fp32_state`); the DiT reads it in its own dtype.
                noise = noise if heads is None else noise.float()
                latents = scheduler.step(noise, timesteps[0], latents, return_dict=False)[0]
                del noise
                on_step(0)
        projections = () if heads is None else (heads,)
        if held is None:
            denoise_loop(
                _full_step(transformer, joint.segments, joint.length, latents.shape[1], mask is not None,
                           heads=heads is not None),
                range(steps), state=(latents,),
                inputs=(euler.ts, euler.dt, *joint.inputs(), *projections, embeds, *references),
                on_step=on_step,
            )
            return latents
        with held:
            if steps > 1:
                denoise_loop(
                    _cached_step(transformer, heads=heads is not None), range(1, steps), state=(latents,),
                    inputs=(euler.ts, euler.dt, layout.rotary_emb, layout.modulation_mask, *projections,
                            *(layer.state for layer in kv_cache.layer_caches)),
                    key=("qwen-image-2.1.cached/1" if heads is None else "qwen-image-2.1.cached-heads/1", len(blocks)),
                    on_step=on_step,
                )
        return latents

    @uses_components("vae")
    def decode(self, latents: Any, *, width: int, height: int) -> Any:
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        with torch.inference_mode():
            latents = QwenImage21Pipeline._unpack_latents(latents, height, width, 16)
            latents = latents.to(device=device, dtype=vae.dtype)
            mean = latents.new_tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
            std = latents.new_tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
            return _decoded(vae, latents * std + mean).clamp(-1.0, 1.0)[:, :, 0]


def _encoded(vae: Any, pixels: torch.Tensor) -> torch.Tensor:
    """The pipeline's `_encode_vae_image` of one frame (argmax, normalized): Diffusers' `encode`
    without its feature cache (each convolution's input kept on the VAE for a next frame an image
    never has)."""
    x = pixels if vae.config.patch_size is None else _patchify(pixels, patch_size=vae.config.patch_size)
    z = vae.config.z_dim
    latents = vae.quant_conv(vae.encoder(x))[:, :z]
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(latents)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(latents)
    return (latents - mean) / std


def _decoded(vae: Any, z: torch.Tensor) -> torch.Tensor:
    """The pipeline's decode of one frame, the whole canvas: Diffusers' `decode` without its
    feature cache (each convolution's input kept for a next frame an image never has), every
    residual and resample block over the canvas in strips of rows (`tiles.rows`: exact, a strip's
    working set). Each residual up-block's shortcut is its input itself (nothing writes it in
    place), added into the upsampled rows; Diffusers clones it."""
    decoder = vae.decoder
    x = decoder.conv_in(vae.post_quant_conv(z))
    mid = decoder.mid_block
    x = tiles.rows(mid.resnets[0], x, halo=2)
    for attention, resnet in zip(mid.attentions, mid.resnets[1:], strict=True):
        x = tiles.rows(resnet, attention(x), halo=2)
    for block in decoder.up_blocks:
        if not isinstance(block, QwenImage21ResidualUpBlock):
            raise TypeError(f"Qwen Image 2.1's decoder has residual up-blocks, not {type(block)}")
        shortcut = x if block.avg_shortcut is not None else None
        for resnet in block.resnets:
            x = tiles.rows(resnet, x, halo=2)
        if block.upsampler is not None:
            x = tiles.rows(block.upsampler, x, halo=1, scale=2)
        if shortcut is not None:
            duplicate = block.avg_shortcut
            x = tiles.rows(
                lambda s, up=duplicate: up(s, first_chunk=True), shortcut, halo=0, scale=2, out=x
            )
        del shortcut
    return tiles.rows(
        lambda s: decoder.conv_out(decoder.nonlinearity(decoder.norm_out(s))), x, halo=1
    )
