"""Krea 2 (Raw and Turbo), component-staged over Diffusers' modules.

The DiT and VAE are Diffusers' (`Krea2Transformer2DModel`, `AutoencoderKLQwenImage`, 0.42.1); the text
encoder is Transformers' Qwen3-VL-4B text model. Prompt conditioning, the schedule and the sampler are
Diffusers' `Krea2Pipeline`, Krea's reference code (`krea-ai/krea-2` db3984f) in its form. Runtime
constructs and fills the modules and owns their residency.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from cozy_runtime import rng
from cozy_runtime.author import (
    AdapterCompatibility,
    Config,
    Loader,
    MemoDistribution,
    Model,
    ModelAssets,
    uses_components,
)
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline, calculate_shift, retrieve_timesteps
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast, Qwen3VLConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

#: Pixels per DiT token: the VAE's 8x, then 2x2 patches.
PATCH = 16
VAE_SCALE = 8
#: The DiT's activations; its RMSNorm scales stay F32 (Diffusers' `_keep_in_fp32_modules`).
DTYPE = torch.bfloat16
#: The text encoder's hidden states the DiT reads when the checkpoint's index names none.
TAPS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
#: Diffusers' fixed text length: at most 512 tokens after the dropped system prefix.
TEXT_TOKENS = 512
#: `Krea2Pipeline`'s template (Qwen-Image's): the prompt between a system prefix of 34 tokens, which the
#: DiT never reads, and a 5-token assistant suffix.
PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS, SUFFIX_TOKENS = 34, 5
#: The turbo checkpoint's fixed timestep shift (`is_distilled`); Raw's follows the token count.
DISTILLED_MU = 1.15
#: Lightweight modules Diffusers keeps in F32 under a bf16 load.
_F32 = frozenset(Krea2Transformer2DModel._keep_in_fp32_modules)


class TextEncoder(nn.Module):
    """Qwen3-VL-4B's language model (`language_model.*`) up to the deepest tapped layer. The vision
    tower, the later layers and the final norm are never built, so their bytes are never read; with the
    norm an identity, hidden state `depth` stays the pre-norm output Diffusers taps."""

    def __init__(self, config: Qwen3VLConfig, depth: int) -> None:
        super().__init__()
        self.language_model = Qwen3VLTextModel._from_config(config.text_config, dtype=DTYPE)
        self.language_model.layers = nn.ModuleList(list(self.language_model.layers)[:depth])
        self.language_model.norm = nn.Identity()


class Krea2Graph:
    def __init__(self, config: Config) -> None:
        index = config.section("model_index").mapping() if "model_index" in config.keys() else {}
        taps: Any = index.get("text_encoder_select_layers") or TAPS
        self.taps = tuple(int(i) for i in taps)
        self.distilled = bool(index.get("is_distilled", False))
        with no_init_weights():
            transformer = Krea2Transformer2DModel.from_config(config.section("transformer").mapping())
            encoder = TextEncoder(Qwen3VLConfig(**config.section("text_encoder").mapping()), max(self.taps))
            vae = AutoencoderKLQwenImage.from_config(config.section("vae").mapping())
        transformer.to(DTYPE)
        for name, module in transformer.named_modules():
            if name.rsplit(".", 1)[-1] in _F32:
                module.float()
        self.components: dict[str, nn.Module] = {
            "transformer": transformer.eval(),
            "text_encoder": encoder.eval(),
            # bf16, as Diffusers' and Krea's pipelines run it; the checkpoint stores F32.
            "vae": vae.to(DTYPE).eval(),
        }
        self.scheduler_config: Mapping[str, Any] = config.section("scheduler").mapping()


def build_tokenizer(assets: ModelAssets) -> PreTrainedTokenizerFast:
    """The checkpoint's Qwen2 tokenizer (`tokenizer/`), parsed in memory."""
    config = json.loads(assets.read("tokenizer/tokenizer_config.json"))
    for key in ("tokenizer_class", "backend", "is_local", "name_or_path", "tokenizer_file",
                "added_tokens_decoder", "chat_template", "extra_special_tokens", "errors"):
        config.pop(key, None)
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer/tokenizer.json").decode()), **config
    )


def unpack(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Diffusers' `_unpack_latents`: DiT tokens [1, (H/16)(W/16), 64] as one VAE frame [1, 16, 1, H/8, W/8]."""
    rows, columns = height // PATCH, width // PATCH
    channels = latents.shape[-1] // 4
    latents = latents.view(1, rows, columns, channels, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(1, channels, 1, rows * 2, columns * 2)


#: Code the memoized encode reads beyond its own body.
_CODE = (build_tokenizer, TextEncoder, MemoDistribution("transformers"), MemoDistribution("tokenizers"))


class Krea2(Model[Krea2Graph], encoded_leaves="accept"):
    """One Krea 2 checkpoint: Raw (CFG, the token count's shift) or Turbo (distilled: no CFG, shift 1.15)."""

    #: Caller LoRAs on the DiT, at any strength: the Runtime bakes them into the weights.
    __adapter_compatibility__ = (AdapterCompatibility("lora", "", ("transformer",), -np.inf, np.inf),)

    graph: Krea2Graph
    tokenizer: PreTrainedTokenizerFast

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Krea2Graph, factory=Krea2Graph)
        self.tokenizer = build_tokenizer(loader.assets)

    @property
    def distilled(self) -> bool:
        return self.graph.distilled

    @uses_components("text_encoder", memoize=True, memo_dependencies=_CODE)
    def encode(self, prompt: str) -> torch.Tensor:
        """The DiT's text conditioning [1, S, 12, 2560] (bf16, on the CPU): Diffusers' Qwen-Image chat
        template, at most 512 tokens after its 34-token system prefix, the 12 tapped hidden states.

        Diffusers pads the prompt to 512 tokens and masks the padding, with positions counting real
        tokens only; the DiT reads text keys at rotary position 0 and masks padded ones. Unpadded, the
        same tokens get the same positions and every real token the same values, without the padding's
        compute or a mask (which leaves only SDPA among attention kernels)."""
        encoder = self.graph.components["text_encoder"].language_model
        device = torch.device(str(self.device))
        tokens = self.tokenizer(
            PROMPT_PREFIX + prompt, truncation=True,
            max_length=TEXT_TOKENS + PREFIX_TOKENS - SUFFIX_TOKENS,
        ).input_ids + self.tokenizer(PROMPT_SUFFIX).input_ids
        ids = torch.tensor([tokens], device=device)
        positions = torch.arange(ids.shape[1], device=device).view(1, 1, -1).expand(3, 1, -1)
        with torch.inference_mode():
            hidden = encoder(input_ids=ids, position_ids=positions, output_hidden_states=True,
                             use_cache=False).hidden_states
            stacked = torch.stack([hidden[i] for i in self.graph.taps], dim=2)[:, PREFIX_TOKENS:]
        return stacked.to(DTYPE).cpu()

    @uses_components("transformer")
    def denoise(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
    ) -> torch.Tensor:
        """`Krea2Pipeline.__call__`'s loop: bf16 noise from `seed` drawn on the CPU, Euler on
        `linspace(1, 1/steps)` time-shifted by mu, and Krea's guidance `cond + g(cond - uncond)` when
        `negative` is given. Returns the packed latents [1, (H/16)(W/16), 64] on the CPU.

        Each step is one pure function of (latents, step index) and the request's tensors: the place
        `author.denoise_loop` (xs-127) takes it once a Runtime release carries it."""
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        channels = transformer.config.in_channels // 4
        noise = rng.noise((1, channels, rows * 2, columns * 2), seed, dtype=DTYPE)
        latents = noise.view(1, channels, rows, 2, columns, 2).permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(1, rows * columns, channels * 4).to(device)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(self.graph.scheduler_config))
        config = scheduler.config
        mu = DISTILLED_MU if self.distilled else calculate_shift(
            rows * columns, config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 6400),
            config.get("base_shift", 0.5), config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            scheduler, steps, device, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu
        )
        texts = [positive, *([] if negative is None else [negative])]
        texts = [(text.to(device=device, dtype=DTYPE),
                  Krea2Pipeline.prepare_position_ids(text.shape[1], rows, columns, device)) for text in texts]
        scheduler.set_begin_index(0)
        with torch.inference_mode():
            for index, t in enumerate(timesteps):
                cancel()
                timestep = (t / config.num_train_timesteps).expand(1).to(DTYPE)
                predictions = [
                    transformer(hidden_states=latents, encoder_hidden_states=text, timestep=timestep,
                                position_ids=ids, return_dict=False)[0]
                    for text, ids in texts
                ]
                prediction = predictions[0]
                if negative is not None:
                    prediction = prediction + guidance * (prediction - predictions[1])
                latents = scheduler.step(prediction, t, latents, return_dict=False)[0]
                on_step(index)
        return latents.cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor, *, width: int, height: int) -> Image.Image:
        """Diffusers' decode of the one frame. The decoder runs without Diffusers' feature cache, which
        only keeps each convolution's input for a next frame an image never has."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        with torch.inference_mode():
            z = unpack(latents, height, width).to(device=device, dtype=vae.dtype)
            mean = z.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            inverse = 1.0 / z.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            image = vae.decoder(vae.post_quant_conv(z / inverse + mean)).clamp(-1, 1)[:, :, 0]
            if not torch.isfinite(image).all():
                raise ValueError("Krea 2's decoder produced non-finite pixels")
            (picture,) = VaeImageProcessor(vae_scale_factor=PATCH).postprocess(image.float(), output_type="pil")
        return picture  # type: ignore[no-any-return]
