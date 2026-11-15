"""LTX-2.5 Distilled over official Diffusers components, with Runtime-owned residency."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from cozy_runtime.accel import hold_dtype
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import Config, Loader, Model, ModelAssets, Serving, device, uses_components
from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2ImageToVideoPipeline,
    LTX2LatentUpsamplePipeline,
    LTX2Pipeline,
    LTX2TextConnectors,
    LTX2VideoTransformer3DModel,
)
from diffusers.models.latent_upscaler.latent_upscaler_ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from transformers import AddedToken, Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration, GemmaTokenizer

FPS = 24
SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)
REFINE_SIGMAS = SIGMAS[-3:]
COMPONENTS = ("transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder", "latent_upsampler")


def stored(module: nn.Module, component: str, config: Config) -> nn.Module:
    dtypes = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
    for name, parameter in module.named_parameters():
        dtype = dtypes.get(config.tensor_dtype(f"{component}.{name}", default="bf16"))
        if dtype is not None:
            parameter.data = parameter.data.to(dtype)
    return module.eval()


def tokenizer(assets: ModelAssets) -> GemmaTokenizer:
    settings = json.loads(assets.read("tokenizer/tokenizer_config.json"))
    added = settings.pop("added_tokens_decoder", {})
    for key in ("tokenizer_class", "auto_map", "name_or_path", "tokenizer_file", "vocab_file"):
        settings.pop(key, None)
    made = GemmaTokenizer(
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer/tokenizer.json").decode()),
        added_tokens_decoder={int(index): AddedToken(**value) for index, value in added.items()},
        **settings,
    )
    if "tokenizer/chat_template.jinja" in assets.names():
        made.chat_template = assets.read("tokenizer/chat_template.jinja").decode()
    return made


class Graph:
    def __init__(self, config: Config) -> None:
        text = Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(**config.section("text_encoder").mapping()))
        # Prompt encoding uses hidden states, never vocabulary logits; this tied projection
        # is not a separate stored tensor and would otherwise allocate [tokens,vocabulary].
        text.lm_head = nn.Identity()
        vae = AutoencoderKLLTX2Video.from_config(config.section("vae").mapping())
        vae.enable_tiling()
        self.components = {
            "transformer": stored(LTX2VideoTransformer3DModel.from_config(config.section("transformer").mapping()), "transformer", config),
            "text_encoder": stored(text, "text_encoder", config),
            "connectors": stored(LTX2TextConnectors.from_config(config.section("connectors").mapping()), "connectors", config),
            "vae": stored(vae, "vae", config),
            "audio_vae": stored(AutoencoderKLLTX2Audio.from_config(config.section("audio_vae").mapping()), "audio_vae", config),
            "vocoder": stored(LTX2VocoderWithBWE.from_config(config.section("vocoder").mapping()), "vocoder", config),
            "latent_upsampler": stored(LTX2LatentUpsamplerModel.from_config(config.section("latent_upsampler").mapping()), "latent_upsampler", config),
        }
        self.scheduler_config = config.section("scheduler").mapping()


class TextPipeline(LTX2Pipeline):
    @property
    def _execution_device(self) -> torch.device:
        # The earlier stage's text root may be parked while the DiT is active. The first
        # component in Diffusers' registry is not the placement authority.
        return torch.device(str(device()))


class ImagePipeline(LTX2ImageToVideoPipeline):
    @property
    def _execution_device(self) -> torch.device:
        return torch.device(str(device()))


class Ltx25(Model[Graph]):
    serving = Serving(min_bits=16, attention_bits=16)
    graph: Graph
    tokenizer: GemmaTokenizer

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)
        self.tokenizer = tokenizer(loader.assets)

    def execution_plan(self, plan: Plan) -> str:
        """The reference vocoder computes in F32; the Runtime casts at its next fill."""
        hold_dtype(self.graph.components["vocoder"], torch.float32)
        return "ltx25-vocoder-f32"

    def _pipeline(self, image: bool = False) -> LTX2Pipeline | LTX2ImageToVideoPipeline:
        c = self.graph.components
        cls = ImagePipeline if image else TextPipeline
        pipe = cls(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            vae=c["vae"], audio_vae=c["audio_vae"], text_encoder=c["text_encoder"],
            tokenizer=self.tokenizer, connectors=c["connectors"], transformer=c["transformer"], vocoder=c["vocoder"],
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe

    @uses_components("text_encoder")
    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        pipe = self._pipeline()
        with torch.inference_mode():
            embeds, mask, _, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False, max_sequence_length=1024,
                                                     device=torch.device(str(self.device)))
        return embeds, mask

    @uses_components("transformer", "connectors", "vae", "audio_vae")
    def denoise(self, embeds: tuple[torch.Tensor, torch.Tensor], image: Image.Image | None, *, width: int, height: int,
                frames: int, generator: torch.Generator, check: Callable[[], None], progress: Callable[[int], None],
                latents: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Both returns are already denormalized by Diffusers' output_type='latent'.
        # The VAE roots are also leased: their latent-statistic buffers participate here.
        pipe = self._pipeline(image=image is not None)
        sigmas = list(SIGMAS if latents is None else REFINE_SIGMAS)

        def callback(_pipe: object, index: int, _t: torch.Tensor, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            check()
            progress(index)
            return values

        kwargs: dict[str, object] = {
            "prompt_embeds": embeds[0], "prompt_attention_mask": embeds[1],
            "height": height, "width": width, "num_frames": frames, "frame_rate": float(FPS),
            "sigmas": sigmas, "num_inference_steps": len(sigmas), "generator": generator,
            "guidance_scale": 1.0, "audio_guidance_scale": 1.0,
            "stg_scale": 0.0, "audio_stg_scale": 0.0,
            "modality_scale": 1.0, "audio_modality_scale": 1.0,
            "output_type": "latent", "return_dict": False, "callback_on_step_end": callback,
            "callback_on_step_end_tensor_inputs": [],
        }
        if image is not None:
            kwargs["image"] = image
        if latents is not None:
            kwargs.update(latents=latents[0], audio_latents=latents[1], noise_scale=REFINE_SIGMAS[0])
        check()
        try:
            with torch.inference_mode():
                video, audio = pipe(**kwargs)
            return video, audio
        finally:
            pipe.transformer._reset_stateful_cache()

    @uses_components("latent_upsampler", "vae")
    def upscale(self, latents: torch.Tensor) -> torch.Tensor:
        pipe = LTX2LatentUpsamplePipeline(vae=self.graph.components["vae"], latent_upsampler=self.graph.components["latent_upsampler"])
        pipe.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            return cast(torch.Tensor, pipe(latents=latents, latents_normalized=False, output_type="latent", return_dict=False)[0])

    @uses_components("vae")
    def decode_video(self, latents: torch.Tensor) -> npt.NDArray[np.uint8]:
        vae = cast(AutoencoderKLLTX2Video, self.graph.components["vae"])
        with torch.inference_mode():
            pixels = vae.decode(latents.to(vae.dtype), None, return_dict=False)[0]
        pixels = pixels[0].permute(1, 2, 3, 0).float().add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
        return pixels.contiguous().cpu().numpy()

    @uses_components("audio_vae", "vocoder")
    def decode_audio(self, latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        vae = cast(AutoencoderKLLTX2Audio, self.graph.components["audio_vae"])
        vocoder = cast(LTX2VocoderWithBWE, self.graph.components["vocoder"])
        with torch.inference_mode():
            mel = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
            audio = vocoder(mel.to(vocoder.dtype))[0].float().cpu()
        return audio, int(vocoder.config.output_sampling_rate)

