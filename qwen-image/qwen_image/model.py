"""Qwen-Image-2512 and Qwen-Image-Edit-2511, component-staged over Diffusers' modules.

Both are Qwen-Image's 20B dual-stream DiT beside one Qwen2.5-VL-7B text encoder and one VAE, which the
two checkpoints store byte for byte alike. `QwenImage` is a whole checkpoint, the text-to-image DiT
with the encoder and VAE; `QwenImageTransformer` is a DiT alone, the edit one, bound in its own slot.
A request denoises on one of them and Runtime owns each one's residency, so going from one to the
other moves pages, never reloads.

Conditioning and sampling are Diffusers' `QwenImagePipeline` and `QwenImageEditPlusPipeline` (0.41.0),
Qwen's own inference code. Text alone is read with the text-to-image template; with references, the
edit template reads each at a 384² area, the VAE encodes each at a 1024² area, and the edit DiT
denoises the target beside their tokens (`zero_cond_t`: reference tokens at t = 0).

A LoRA bound in its own slot (`QwenImageLoRA`, a Lightning distillation) reaches its DiT's linears
through permanent forward hooks. Runtime bakes the armed LoRA into the resident weights of whatever
lane serves (`weights.LoRASource`); unbaked, the hook adds it each forward.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import torch
from cozy_runtime import rng, tiles
from cozy_runtime.accel import fusion, hold_dtype, lora
from cozy_runtime.author import (
    Config,
    Loader,
    Model,
    ModelAssets,
    OutputError,
    Serving,
    Telemetry,
    memoize,
    uses_components,
)
from cozy_runtime.accel.kernels import Plan
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
    QwenImageCausalConv3d,
    QwenImageDecoder3d,
    QwenImageEncoder3d,
    QwenImageResample,
    QwenImageResidualBlock,
    QwenImageRMS_norm,
    QwenImageUpsample,
)
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    QwenImageEditPlusPipeline,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from PIL import Image
from torch import nn
from transformers import (
    AddedToken,
    AutoModelForImageTextToText,
    Qwen2_5_VLConfig,
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)
from transformers import initialization as transformer_init
from tqdm import tqdm

if TYPE_CHECKING:
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageUpBlock
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

#: Pixels per latent token: the VAE's 8x, then 2x2 packing.
PATCH = 16
#: The text-to-image prompt's token budget, as `QwenImagePipeline` runs it.
TEXT_TOKENS = 512
#: The DiT linears a Qwen-Image LoRA adapts in every block (lightx2v's Lightning: all twelve).
LORA_SITES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
)
#: The VAE's dtype on CUDA. The lane stores bf16, which fp16 holds exactly: against an F32 decode, fp16
#: measures 62.9 dB and bf16 53.4 (dtype audit M1), with about 427x headroom below fp16's range.
VAE_DTYPE = torch.float16
#: The VAE's RMS norms in one kernel each, with the SiLU each residual block, the decoder's head and
#: the encoder's apply after them; Diffusers computes them in F32 copies of every activation.
VAE_NORM = fusion.ChannelNorm(
    QwenImageRMS_norm,
    silu=(
        (QwenImageResidualBlock, ("norm1", "norm2")),
        (QwenImageDecoder3d, ("norm_out",)),
        (QwenImageEncoder3d, ("norm_out",)),
    ),
    dtype="fp16",
)
#: A lone frame's 3x3x3 causal convolutions on their last temporal tap, the only one it meets (Diffusers
#: pads two zero frames and convolves all 27 taps), and its nearest upsampling in its own dtype, not F32.
VAE_FRAME = fusion.LoneFrame(QwenImageCausalConv3d, upsample=QwenImageUpsample)


_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16, "i64": torch.int64}


class Arming:
    """The LoRA a denoise has armed on this construction's DiT, else None (a base forward)."""

    __slots__ = ("lora",)

    def __init__(self) -> None:
        self.lora: nn.Module | None = None


def install_lora_hooks(transformer: QwenImageTransformer2DModel, arming: Arming) -> None:
    """A permanent hook on each LoRA site: the armed LoRA's factors there, else nothing."""
    for index, _ in enumerate(transformer.transformer_blocks):
        for site in LORA_SITES:
            path = f"transformer_blocks.{index}.{site}"
            transformer.get_submodule(path).register_forward_hook(lora.Hook(partial(_armed, arming, path)))


def _armed(arming: Arming, path: str) -> LoRAFactors | None:
    return None if arming.lora is None else arming.lora.get_submodule(path)  # type: ignore[return-value]


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def _to[M: nn.Module](module: M, dtype: torch.dtype) -> M:
    """`module.to(dtype=dtype)`, typed: Diffusers' and Transformers' `to` is `functools.wraps`'d,
    which the checkers bind without `self`."""
    return module.to(dtype=dtype)


class TransformerComponents(TypedDict):
    transformer: QwenImageTransformer2DModel


class QwenImageComponents(TransformerComponents):
    text_encoder: Qwen2_5_VLForConditionalGeneration
    vae: AutoencoderKLQwenImage


class TransformerGraph:
    """A DiT construction in bf16, with its LoRA hooks and its checkpoint's scheduler."""

    components: TransformerComponents

    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = construct(QwenImageTransformer2DModel, config.section("transformer").mapping())
        self.components = {"transformer": _to(transformer, torch.bfloat16)}
        self.scheduler_config = config.section("scheduler").mapping()
        self.arming = Arming()
        install_lora_hooks(transformer, self.arming)


class QwenImageGraph(TransformerGraph):
    """A whole checkpoint: the DiT, Qwen2.5-VL and the VAE, in the checkpoint's precision."""

    components: QwenImageComponents

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.text_encoder_config = config.section("text_encoder").mapping()
        vae_dtype = _DTYPES[config.tensor_dtype("vae.encoder.conv_in.weight", default="bf16")]
        with transformer_init.no_init_weights():
            # Construction dtype leaves Qwen2.5-VL's RoPE buffers in F32, as upstream loads.
            self.components["text_encoder"] = AutoModelForImageTextToText.from_config(  # type: ignore[no-untyped-call]  # Transformers' Auto classes are untyped
                Qwen2_5_VLConfig(**self.text_encoder_config), dtype=torch.bfloat16
            )
            self.components["vae"] = _to(construct(AutoencoderKLQwenImage, config.section("vae").mapping()), vae_dtype)
        # The pipelines read the last hidden state only: the 1.1 GB head is never built.
        self.components["text_encoder"].lm_head = nn.Identity()


def build_processor(  # type: ignore[explicit-any]  # text_encoder: the checkpoint's JSON, as Config.mapping() gives it
    assets: ModelAssets, text_encoder: Mapping[str, Any]
) -> Qwen2VLProcessor:
    """Qwen2.5-VL's processor: the checkpoint's `tokenizer/` files parsed in memory, and the image
    processor its vision tower reads (Transformers' defaults at the tower's patch, merge and frame
    sizes, which is what Qwen-Image-Edit's `processor/` folder states)."""

    def read(name: str) -> bytes:
        return assets.read(f"tokenizer/{name}")

    config = json.loads(read("tokenizer_config.json"))
    added = config.pop("added_tokens_decoder", {})
    for key in ("tokenizer_class", "name_or_path", "chat_template", "processor_class"):
        config.pop(key, None)
    merges = [
        tuple(line.split(" "))
        for line in read("merges.txt").decode().splitlines()
        if line and not line.startswith("#version")
    ]
    tokenizer = Qwen2Tokenizer(
        vocab=json.loads(read("vocab.json")),
        merges=merges,  # type: ignore[arg-type]  # Transformers types merges as strings; Tokenizers' BPE takes these pairs
        added_tokens_decoder={int(i): AddedToken(**token) for i, token in added.items()},
        **config,
    )
    vision = text_encoder["vision_config"]
    image = Qwen2VLImageProcessor(
        patch_size=vision["patch_size"],
        temporal_patch_size=vision["temporal_patch_size"],
        merge_size=vision["spatial_merge_size"],
    )
    return Qwen2VLProcessor(  # type: ignore[no-untyped-call]  # Transformers' processor __init__ is untyped
        image_processor=image, tokenizer=tokenizer, video_processor=Qwen2VLVideoProcessor()
    )


def reference_sizes(image: Image.Image) -> tuple[tuple[int, int], tuple[int, int]]:
    """Upstream's (width, height) of a reference for the vision encoder (384² area) and the
    VAE (1024² area), each side a multiple of 32."""
    ratio = image.width / image.height
    return (
        calculate_dimensions(CONDITION_IMAGE_SIZE, ratio),  # type: ignore[no-untyped-call]  # Diffusers' helper is untyped
        calculate_dimensions(VAE_IMAGE_SIZE, ratio),  # type: ignore[no-untyped-call]
    )


#: Encoded references: their packed VAE latents as one token sequence, and each one's token grid.
References = tuple[torch.Tensor, list[tuple[int, int, int]]]


def sample(
    transformer: QwenImageTransformer2DModel,
    device: torch.device,
    scheduler_config: Mapping[str, object],
    embeds: torch.Tensor,
    mask: torch.Tensor | None,
    negative: tuple[torch.Tensor, torch.Tensor | None] | None,
    references: References | None,
    *,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
) -> torch.Tensor:
    """Upstream's sampling loop (`QwenImageEditPlusPipeline.__call__`): seeded noise drawn on the CPU in
    the DiT's dtype, `FlowMatchEulerDiscreteScheduler` on `linspace(1, 1/steps)` shifted by the canvas,
    the references' tokens beside the target, and true CFG with the conditional norm restored when
    `negative` is given."""
    dtype = transformer.dtype
    rows, columns = height // PATCH, width // PATCH
    scheduler = construct(FlowMatchEulerDiscreteScheduler, dict(scheduler_config))
    config = scheduler.config
    channels: int = transformer.config["in_channels"] // 4
    img_shapes = [[(1, rows, columns), *(references[1] if references else [])]]
    with torch.no_grad():
        noise = rng.noise((1, 1, channels, rows * 2, columns * 2), seed, dtype=dtype)
        latents: torch.Tensor = QwenImagePipeline._pack_latents(  # type: ignore[no-untyped-call]  # Diffusers' helper is untyped
            noise.to(device), 1, channels, rows * 2, columns * 2
        )
        mu = calculate_shift(
            latents.shape[1],
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 4096),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.15),
        )
        # Diffusers types sigmas as a list; the scheduler takes the array as it is.
        retrieve_timesteps(scheduler, steps, device, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)  # type: ignore[arg-type]
        image = None if references is None else references[0].to(device=device, dtype=dtype)
        texts = [
            (text.to(device=device, dtype=dtype), None if text_mask is None else text_mask.to(device))
            for text, text_mask in [(embeds, mask), *([negative] if negative is not None else [])]
        ]
        for t in tqdm(scheduler.timesteps):
            model_input = latents if image is None else torch.cat([latents, image], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            predictions: list[torch.Tensor] = [
                transformer(
                    hidden_states=model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states_mask=text_mask,
                    encoder_hidden_states=text,
                    img_shapes=img_shapes,
                    attention_kwargs={},
                    return_dict=False,
                )[0][:, : latents.shape[1]]
                for text, text_mask in texts
            ]
            prediction = predictions[0]
            if negative is not None:
                combined = predictions[1] + guidance * (prediction - predictions[1])
                prediction = combined * (
                    torch.norm(prediction, dim=-1, keepdim=True) / torch.norm(combined, dim=-1, keepdim=True)
                )
            latents = scheduler.step(prediction, t, latents, return_dict=False)[0]
    return latents.cpu()


class QwenImageTransformer(Model[TransformerGraph]):
    """A Qwen-Image DiT alone (Qwen-Image-Edit-2511's): it denoises what a `QwenImage` encodes."""

    graph: TransformerGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(TransformerGraph)

    @uses_components("transformer")
    def denoise(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
        negative: tuple[torch.Tensor, torch.Tensor | None] | None,
        references: References | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        scheduler_config: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        """`sample` on this DiT, at `scheduler_config` or its checkpoint's."""
        return sample(
            self.graph.components["transformer"],
            self.device,
            scheduler_config or self.graph.scheduler_config,
            embeds, mask, negative, references,
            width=width, height=height, steps=steps, guidance=guidance, seed=seed,
        )


class QwenImage(Model[QwenImageGraph]):
    """A whole Qwen-Image checkpoint (Qwen-Image-2512): it encodes text and references for either
    DiT, denoises on its own and decodes."""

    graph: QwenImageGraph
    processor: Qwen2VLProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(QwenImageGraph)
        self.processor = build_processor(loader.assets, self.graph.text_encoder_config)

    def execution_plan(self, plan: Plan) -> str:
        vae = self.graph.components["vae"]
        encoder = vae.encoder
        # Not the encoder's temporal downsamplers: their `time_conv` (stride 2, no causal padding)
        # only ever runs on a next frame's cache, never on a lone frame.
        frame = (
            *(block for block in encoder.down_blocks if not isinstance(block, QwenImageResample)),
            encoder.conv_in, encoder.mid_block, encoder.conv_out, vae.quant_conv,
            vae.decoder, vae.post_quant_conv,
        )
        installed = (
            fusion.install_norms(plan, VAE_NORM, vae),
            fusion.install_lone_frame(plan, VAE_FRAME, *frame),
        )
        return "+".join(identity for identity in installed if identity)

    def hold_vae(self, tel: Telemetry) -> None:
        """Encode and decode in `VAE_DTYPE` on CUDA; the CPU keeps the stored bf16. The Runtime casts
        the stored bytes when it next fills them; called before any VAE scope opens."""
        vae = self.graph.components["vae"]
        if self.device.type != "cuda" or next(vae.parameters()).dtype == VAE_DTYPE:
            return
        hold_dtype(vae, VAE_DTYPE)
        tel.log("vae representation", dtype="float16")

    def _pipeline(self, edit: bool) -> QwenImagePipeline | QwenImageEditPlusPipeline:
        scheduler = construct(FlowMatchEulerDiscreteScheduler, self.graph.scheduler_config)
        vae, text_encoder = self.graph.components["vae"], self.graph.components["text_encoder"]
        tokenizer, transformer = self.processor.tokenizer, self.graph.components["transformer"]
        if edit:
            return QwenImageEditPlusPipeline(
                processor=self.processor, scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                tokenizer=tokenizer, transformer=transformer,
            )
        return QwenImagePipeline(
            scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer
        )

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The prompt as upstream reads it, its embeddings and mask (None when every token is
        valid): with references, the edit template with each at its 384² area; without, the
        text-to-image template, at most 512 tokens."""
        device = self.device
        pipeline = self._pipeline(edit=bool(images))
        with torch.inference_mode():
            if images:
                conditions = []
                for image in images:
                    (width, height), _ = reference_sizes(image)
                    conditions.append(pipeline.image_processor.resize(image, height, width))
                # With references `_pipeline` is the edit one; Diffusers types its image as one tensor.
                embeds, mask = pipeline.encode_prompt(prompt, image=conditions, device=device)  # type: ignore[call-arg,arg-type]
            else:
                embeds, mask = pipeline.encode_prompt(prompt, device=device, max_sequence_length=TEXT_TOKENS)
        return embeds.cpu(), None if mask is None else mask.cpu()

    @memoize
    @uses_components("vae")
    def encode_references(self, images: list[Image.Image]) -> References:
        """Each reference at its 1024² area through the VAE (argmax), normalized and packed: the
        pipeline's `_encode_vae_image` of one frame without Diffusers' feature cache (each
        convolution's input kept for a next frame an image never has)."""
        vae = self.graph.components["vae"]
        device = self.device
        pipeline = self._pipeline(edit=True)
        z = vae.config["z_dim"]
        mean = torch.tensor(vae.config["latents_mean"]).view(1, z, 1, 1, 1)
        std = torch.tensor(vae.config["latents_std"]).view(1, z, 1, 1, 1)
        packed: list[torch.Tensor] = []
        grids: list[tuple[int, int, int]] = []
        with torch.inference_mode():
            for image in images:
                width, height = reference_sizes(image)[1]
                pixels = pipeline.image_processor.preprocess(image, height, width).unsqueeze(2)
                moments = vae.quant_conv(vae.encoder(pixels.to(device=device, dtype=vae.dtype)))
                latents = moments[:, :z]
                latents = (latents - mean.to(latents)) / std.to(latents)
                rows, columns = latents.shape[3:]
                packed.append(
                    pipeline._pack_latents(latents, 1, latents.shape[1], rows, columns).cpu()  # type: ignore[no-untyped-call]  # Diffusers' helper is untyped
                )
                grids.append((1, rows // 2, columns // 2))
        return torch.cat(packed, dim=1), grids

    @uses_components("transformer")
    def denoise(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
        negative: tuple[torch.Tensor, torch.Tensor | None] | None,
        references: References | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        scheduler_config: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        """`sample` on this checkpoint's DiT, at `scheduler_config` or the checkpoint's."""
        return sample(
            self.graph.components["transformer"],
            self.device,
            scheduler_config or self.graph.scheduler_config,
            embeds, mask, negative, references,
            width=width, height=height, steps=steps, guidance=guidance, seed=seed,
        )

    @uses_components("vae")
    def decode(self, latents: torch.Tensor, *, width: int, height: int) -> Image.Image:
        """Upstream's decode of the one frame (`_decoded`), its pixels quantized on the device as
        `VaeImageProcessor` quantizes them: a 4 MP canvas sends 12 MB of bytes to the host, not
        50 MB of floats for host passes."""
        vae = self.graph.components["vae"]
        device = self.device
        with torch.inference_mode():
            latents = QwenImagePipeline._unpack_latents(latents, height, width, PATCH // 2)  # type: ignore[no-untyped-call]  # Diffusers' helper is untyped
            latents = latents.to(device=device, dtype=vae.dtype)
            z = vae.config["z_dim"]
            mean = torch.tensor(vae.config["latents_mean"]).view(1, z, 1, 1, 1).to(latents)
            inverse = 1.0 / torch.tensor(vae.config["latents_std"]).view(1, z, 1, 1, 1).to(latents)
            image = _decoded(vae, latents / inverse + mean)[0, :, 0].float()
            if not torch.isfinite(image).all():
                raise OutputError(
                    "Qwen-Image's decoder produced non-finite pixels, so there is no picture: try another seed, "
                    "and if it repeats another lane with `--lane bf16`",
                    code="output_integrity",
                )
            pixels = ((image.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
            return Image.fromarray(pixels.permute(1, 2, 0).contiguous().cpu().numpy())


def _decoded(vae: AutoencoderKLQwenImage, z: torch.Tensor) -> torch.Tensor:
    """The pipeline's decode of one frame, the whole canvas: Diffusers' `decode` without its
    feature cache (each convolution's input kept for a next frame an image never has), every
    residual and resample block over the canvas in strips of rows (`tiles.rows`: exact, a strip's
    working set)."""
    decoder = vae.decoder
    x = decoder.conv_in(vae.post_quant_conv(z))
    mid = decoder.mid_block
    x = tiles.rows(mid.resnets[0], x, halo=2)
    for attention, resnet in zip(mid.attentions, mid.resnets[1:], strict=True):
        x = tiles.rows(resnet, attention(x), halo=2)
    for block in cast("list[QwenImageUpBlock]", decoder.up_blocks):  # a ModuleList of these
        for resnet in block.resnets:
            x = tiles.rows(resnet, x, halo=2)
        if block.upsamplers is not None:
            x = tiles.rows(block.upsamplers[0], x, halo=1, scale=2)
    return tiles.rows(lambda s: decoder.conv_out(decoder.nonlinearity(decoder.norm_out(s))), x, halo=1)


#: lightx2v's Lightning sampling: a constant shift of 3, the one it was distilled at
#: (`Qwen-Image-Lightning/generate_with_diffusers.py`).
LIGHTNING_SCHEDULER: dict[str, object] = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

#: The 20B DiT's width, MLP width and depth, and the Lightning LoRA's rank.
WIDTH, HIDDEN, BLOCKS, RANK = 3072, 12288, 60, 64
#: The LoRA checkpoint's one component (a single-file upload).
LORA_COMPONENT = "model"


class LoRAFactors(nn.Module):
    """One linear's rank-r factors and alpha as TensorFS stores a kohya LoRA (`lora_down`/`lora_up`
    as PEFT's `lora_A`/`lora_B`), in its dtype."""

    alpha: torch.Tensor

    def __init__(
        self, in_features: int, out_features: int, rank: int, dtype: torch.dtype, alpha: torch.dtype
    ) -> None:
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        self.register_buffer("alpha", torch.zeros((), dtype=alpha))
        self._scale: float | None = None

    @property
    def scale(self) -> float:
        """`alpha / rank`, as Diffusers reads the kohya file (lightx2v's own script); read once
        filled. The stored alpha, not the PEFT config TensorFS writes beside it (`lora_alpha` = r)."""
        if self._scale is None:
            self._scale = float(self.alpha) / self.lora_A.weight.shape[0]
        return self._scale

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> None:
        lora.accumulate(x, out, self.lora_A.weight, self.lora_B.weight, self.scale)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self.lora_A.weight, self.lora_B.weight, self.scale


class LoRAGraph:
    """The LoRA's factors at the DiT's own module paths, in the checkpoint's dtypes."""

    def __init__(
        self, config: Config, *, width: int = WIDTH, hidden: int = HIDDEN, blocks: int = BLOCKS,
        rank: int = RANK,
    ) -> None:
        def dtype(name: str) -> torch.dtype:
            key = f"{LORA_COMPONENT}.transformer_blocks.0.{LORA_SITES[0]}.{name}"
            return _DTYPES[config.tensor_dtype(key, default="f32")]

        weights, alpha = dtype("lora_A.weight"), dtype("alpha")
        root = nn.Module()
        root.transformer_blocks = nn.ModuleList()
        for _ in range(blocks):
            block = nn.Module()
            for site in LORA_SITES:
                parent = block
                *path, leaf = site.split(".")
                for name in path:
                    if not hasattr(parent, name):
                        parent.add_module(name, nn.Module())
                    parent = getattr(parent, name)
                sizes = (
                    (width, hidden) if site.endswith("net.0.proj")
                    else (hidden, width) if site.endswith("net.2") else (width, width)
                )
                parent.add_module(leaf, LoRAFactors(*sizes, rank, weights, alpha))
            root.transformer_blocks.append(block)
        self.components: dict[str, nn.Module] = {LORA_COMPONENT: root}


def build_lora(config: Config) -> LoRAGraph:
    return LoRAGraph(config)


class QwenImageLoRA(Model[LoRAGraph]):
    """A Qwen-Image LoRA on all twelve linears of every DiT block (the Lightning distillation),
    with the sampling it was distilled for."""

    serving = Serving(encoded_leaves=False)

    graph: LoRAGraph
    scheduler_config: Mapping[str, object] = LIGHTNING_SCHEDULER

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(build_lora)

    @uses_components(LORA_COMPONENT)
    def denoise(
        self,
        base: QwenImage | QwenImageTransformer,
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
        negative: tuple[torch.Tensor, torch.Tensor | None] | None,
        references: References | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
    ) -> torch.Tensor:
        """`base.denoise` with this LoRA armed on its DiT and this LoRA's sampling."""
        arming = base.graph.arming
        arming.lora = self.graph.components[LORA_COMPONENT]
        try:
            return base.denoise(
                embeds, mask, negative, references, scheduler_config=self.scheduler_config,
                width=width, height=height, steps=steps, guidance=guidance, seed=seed,
            )
        finally:
            arming.lora = None
