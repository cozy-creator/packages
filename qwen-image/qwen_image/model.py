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
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cozy_runtime import rng
from cozy_runtime.author import Config, Loader, Model, ModelAssets, uses_components
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
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


_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16, "i64": torch.int64}


class Arming:
    """The LoRA a denoise has armed on this construction's DiT, else None (a base forward)."""

    __slots__ = ("lora",)

    def __init__(self) -> None:
        self.lora: nn.Module | None = None


class LoRAHook:
    """A permanent forward hook on one DiT linear (`weights.LoRASource`). Runtime bakes the armed
    LoRA's factors into the linear's resident bytes and sets `_cozy_baked`; otherwise the hook
    adds `scale * up(down(x))` to the output, as PEFT does."""

    __slots__ = ("_cozy_baked", "arming", "path")

    def __init__(self, arming: Arming, path: str) -> None:
        self.arming = arming
        self.path = path
        self._cozy_baked = False

    def __call__(self, module: nn.Module, args: tuple[Any, ...], out: torch.Tensor) -> None:
        factors = self._factors()
        if factors is None or self._cozy_baked:
            return
        down, up = factors.lora_A.weight, factors.lora_B.weight
        delta = F.linear(F.linear(args[0].to(down.dtype), down), up) * factors.scale
        out.add_(delta.to(out.dtype))

    def _factors(self) -> LoRAFactors | None:
        lora = self.arming.lora
        return None if lora is None else lora.get_submodule(self.path)  # type: ignore[return-value]

    def lora_factors(self) -> tuple[object, torch.Tensor, torch.Tensor, float] | None:
        factors = self._factors()
        if factors is None:
            return None
        return factors, factors.lora_A.weight, factors.lora_B.weight, factors.scale


def install_lora_hooks(transformer: nn.Module, arming: Arming) -> None:
    for index, _ in enumerate(transformer.transformer_blocks):
        for site in LORA_SITES:
            path = f"transformer_blocks.{index}.{site}"
            transformer.get_submodule(path).register_forward_hook(LoRAHook(arming, path))


class TransformerGraph:
    """A DiT construction in bf16, with its LoRA hooks and its checkpoint's scheduler."""

    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = QwenImageTransformer2DModel.from_config(config.section("transformer").mapping())
        self.components: dict[str, Any] = {"transformer": transformer.to(dtype=torch.bfloat16)}
        self.scheduler_config = config.section("scheduler").mapping()
        self.arming = Arming()
        install_lora_hooks(transformer, self.arming)


class QwenImageGraph(TransformerGraph):
    """A whole checkpoint: the DiT, Qwen2.5-VL and the VAE, in the checkpoint's precision."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.text_encoder_config = config.section("text_encoder").mapping()
        vae_dtype = _DTYPES[config.tensor_dtype("vae.encoder.conv_in.weight", default="bf16")]
        with transformer_init.no_init_weights():
            # Construction dtype leaves Qwen2.5-VL's RoPE buffers in F32, as upstream loads.
            self.components["text_encoder"] = AutoModelForImageTextToText.from_config(
                Qwen2_5_VLConfig(**self.text_encoder_config), dtype=torch.bfloat16
            )
            self.components["vae"] = AutoencoderKLQwenImage.from_config(config.section("vae").mapping()).to(
                dtype=vae_dtype
            )
        # The pipelines read the last hidden state only: the 1.1 GB head is never built.
        self.components["text_encoder"].lm_head = nn.Identity()


def build_processor(assets: ModelAssets, text_encoder: Mapping[str, Any]) -> Qwen2VLProcessor:
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
        merges=merges,
        added_tokens_decoder={int(i): AddedToken(**token) for i, token in added.items()},
        **config,
    )
    vision = text_encoder["vision_config"]
    image = Qwen2VLImageProcessor(
        patch_size=vision["patch_size"],
        temporal_patch_size=vision["temporal_patch_size"],
        merge_size=vision["spatial_merge_size"],
    )
    return Qwen2VLProcessor(
        image_processor=image, tokenizer=tokenizer, video_processor=Qwen2VLVideoProcessor()
    )


def reference_sizes(image: Image.Image) -> tuple[tuple[int, int], tuple[int, int]]:
    """Upstream's (width, height) of a reference for the vision encoder (384² area) and the
    VAE (1024² area), each side a multiple of 32."""
    ratio = image.width / image.height
    return (
        calculate_dimensions(CONDITION_IMAGE_SIZE, ratio),
        calculate_dimensions(VAE_IMAGE_SIZE, ratio),
    )


#: Encoded references: their packed VAE latents as one token sequence, and each one's token grid.
References = tuple[torch.Tensor, list[tuple[int, int, int]]]


#: Code the encoders read beyond their own bodies: their memo keys bind it.
_ENCODER_CODE = (build_processor, reference_sizes, QwenImagePipeline, QwenImageEditPlusPipeline)


def sample(
    transformer: nn.Module,
    device: torch.device,
    scheduler_config: Mapping[str, Any],
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
    on_step: Callable[[int], None],
    cancel: Callable[[], None],
) -> torch.Tensor:
    """Upstream's sampling loop: seeded noise drawn on the CPU in the DiT's dtype, Euler steps on
    `linspace(1, 1/steps)` shifted by `scheduler_config`, the references' tokens beside the target,
    and true CFG with the conditional norm restored when `negative` is given."""
    # Upstream's latents take the prompt's dtype, the DiT's (bf16).
    dtype = transformer.dtype
    rows, columns = height // PATCH, width // PATCH
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(scheduler_config))
    channels = transformer.config.in_channels // 4
    noise = rng.noise((1, 1, channels, rows * 2, columns * 2), seed, dtype=dtype)
    latents = QwenImagePipeline._pack_latents(noise.to(device), 1, channels, rows * 2, columns * 2)
    grids = [(1, rows, columns), *(references[1] if references else [])]
    conditioning = None if references is None else references[0].to(device=device, dtype=dtype)
    sigmas = np.linspace(1.0, 1 / steps, steps)
    config = scheduler.config
    mu = calculate_shift(
        latents.shape[1],
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(scheduler, steps, device, sigmas=sigmas, mu=mu)
    texts = [(embeds, mask), *([negative] if negative is not None else [])]
    texts = [
        (e.to(device=device, dtype=dtype), None if m is None else m.to(device)) for e, m in texts
    ]
    scheduler.set_begin_index(0)
    with torch.inference_mode():
        for index, t in enumerate(timesteps):
            cancel()
            model_input = latents if conditioning is None else torch.cat([latents, conditioning], dim=1)
            timestep = t.expand(1).to(dtype)
            predictions = [
                transformer(
                    hidden_states=model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states_mask=text_mask,
                    encoder_hidden_states=text,
                    img_shapes=[grids],
                    attention_kwargs={},
                    return_dict=False,
                )[0][:, : latents.size(1)]
                for text, text_mask in texts
            ]
            prediction = predictions[0]
            if negative is not None:
                combined = predictions[1] + guidance * (prediction - predictions[1])
                prediction = combined * (
                    torch.norm(prediction, dim=-1, keepdim=True)
                    / torch.norm(combined, dim=-1, keepdim=True)
                )
            latents = scheduler.step(prediction, t, latents, return_dict=False)[0]
            on_step(index)
    return latents.cpu()


class QwenImageTransformer(Model[TransformerGraph], encoded_leaves="accept"):
    """A Qwen-Image DiT alone (Qwen-Image-Edit-2511's): it denoises what a `QwenImage` encodes."""

    graph: TransformerGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(TransformerGraph, factory=TransformerGraph)

    @uses_components("transformer")
    def denoise(self, *args: Any, scheduler_config: Mapping[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        """`sample` on this DiT, at `scheduler_config` or its checkpoint's."""
        return sample(
            self.graph.components["transformer"],
            torch.device(str(self.device)),
            scheduler_config or self.graph.scheduler_config,
            *args,
            **kwargs,
        )


class QwenImage(Model[QwenImageGraph], encoded_leaves="accept"):
    """A whole Qwen-Image checkpoint (Qwen-Image-2512): it encodes text and references for either
    DiT, denoises on its own and decodes."""

    graph: QwenImageGraph
    processor: Qwen2VLProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(QwenImageGraph, factory=QwenImageGraph)
        self.processor = build_processor(loader.assets, self.graph.text_encoder_config)

    def _pipeline(self, edit: bool) -> QwenImagePipeline | QwenImageEditPlusPipeline:
        parts = {
            "scheduler": FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            "vae": self.graph.components["vae"],
            "text_encoder": self.graph.components["text_encoder"],
            "tokenizer": self.processor.tokenizer,
            "transformer": self.graph.components["transformer"],
        }
        if edit:
            return QwenImageEditPlusPipeline(processor=self.processor, **parts)
        return QwenImagePipeline(**parts)

    @uses_components("text_encoder", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode(self, prompt: str, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The prompt as upstream reads it, its embeddings and mask (None when every token is
        valid): with references, the edit template with each at its 384² area; without, the
        text-to-image template, at most 512 tokens."""
        device = torch.device(str(self.device))
        pipeline = self._pipeline(edit=bool(images))
        with torch.inference_mode():
            if images:
                conditions = []
                for image in images:
                    (width, height), _ = reference_sizes(image)
                    conditions.append(pipeline.image_processor.resize(image, height, width))
                embeds, mask = pipeline.encode_prompt(prompt, image=conditions, device=device)
            else:
                embeds, mask = pipeline.encode_prompt(prompt, device=device, max_sequence_length=TEXT_TOKENS)
        return embeds.cpu(), None if mask is None else mask.cpu()

    @uses_components("vae", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode_references(self, images: list[Image.Image]) -> References:
        """Each reference at its 1024² area through the VAE (argmax), normalized and packed."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        pipeline = self._pipeline(edit=True)
        packed, grids = [], []
        with torch.inference_mode():
            for image in images:
                width, height = reference_sizes(image)[1]
                pixels = pipeline.image_processor.preprocess(image, height, width).unsqueeze(2)
                latents = pipeline._encode_vae_image(
                    pixels.to(device=device, dtype=vae.dtype), generator=None
                )
                rows, columns = latents.shape[3:]
                packed.append(pipeline._pack_latents(latents, 1, latents.shape[1], rows, columns).cpu())
                grids.append((1, rows // 2, columns // 2))
        return torch.cat(packed, dim=1), grids

    @uses_components("transformer")
    def denoise(self, *args: Any, scheduler_config: Mapping[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        """`sample` on this checkpoint's DiT, at `scheduler_config` or the checkpoint's."""
        return sample(
            self.graph.components["transformer"],
            torch.device(str(self.device)),
            scheduler_config or self.graph.scheduler_config,
            *args,
            **kwargs,
        )

    @uses_components("vae")
    def decode(self, latents: torch.Tensor, *, width: int, height: int) -> Image.Image:
        """Upstream's decode of the one frame. The decoder runs without Diffusers' feature cache,
        which only keeps each convolution's input for a next frame an image never has."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        with torch.inference_mode():
            latents = QwenImagePipeline._unpack_latents(latents, height, width, PATCH // 2)
            latents = latents.to(device=device, dtype=vae.dtype)
            z = vae.config.z_dim
            mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(latents)
            inverse = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(latents)
            decoded = vae.decoder(vae.post_quant_conv(latents / inverse + mean))
            image = decoded.clamp(-1.0, 1.0)[0, :, 0].float()
            if not torch.isfinite(image).all():
                raise ValueError("Qwen-Image's decoder produced non-finite pixels")
            pixels = ((image / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round()
        return Image.fromarray(pixels.astype(np.uint8))


#: lightx2v's Lightning sampling: a constant shift of 3, the one it was distilled at
#: (`Qwen-Image-Lightning/generate_with_diffusers.py`).
LIGHTNING_SCHEDULER: dict[str, Any] = {
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
        self.components: dict[str, Any] = {LORA_COMPONENT: root}


def build_lora(config: Config) -> LoRAGraph:
    return LoRAGraph(config)


class QwenImageLoRA(Model[LoRAGraph]):
    """A Qwen-Image LoRA on all twelve linears of every DiT block (the Lightning distillation),
    with the sampling it was distilled for."""

    graph: LoRAGraph
    scheduler_config: Mapping[str, Any] = LIGHTNING_SCHEDULER

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(LoRAGraph, factory=build_lora)

    @uses_components(LORA_COMPONENT)
    def denoise(self, base: QwenImage | QwenImageTransformer, *args: Any, **kwargs: Any) -> torch.Tensor:
        """`base.denoise` with this LoRA armed on its DiT and this LoRA's sampling."""
        arming = base.graph.arming
        arming.lora = self.graph.components[LORA_COMPONENT]
        try:
            return base.denoise(*args, scheduler_config=self.scheduler_config, **kwargs)
        finally:
            arming.lora = None
