"""LTX-2.5 Distilled over official Diffusers components, with Runtime-owned residency."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypedDict, cast

import numpy as np
import torch
from cozy_runtime.accel import sequence_parallel
from cozy_runtime.author import Config, Loader, Model, ModelAssets, Serving, device, memoize, uses_components
from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
from diffusers.models.condition_embedders.condition_embedder_ltx2 import LTX2TextConnectors
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_ltx_euler_ancestral_rf import LTXEulerAncestralRFScheduler
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import LTX2ImageToVideoPipeline, retrieve_latents
from diffusers.pipelines.ltx2.utils import apply_image_conditioning_crf, resolve_default_image_crf
from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline
from diffusers.models.latent_upscaler.latent_upscaler_ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from transformers import AddedToken, Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration, GemmaTokenizer

from cozy_runtime import rng

from .vae import VideoVAE
from .parallel import install

FPS = 24
SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)
REFINE_SIGMAS = SIGMAS[-3:]
COMPONENTS = ("transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder", "latent_upsampler")
VideoPixels = np.ndarray[tuple[int, int, int, int], np.dtype[np.uint8]]


def stored[T: nn.Module](module: T, component: str, config: Config) -> T:
    dtypes = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
    for name, parameter in module.named_parameters():
        dtype = dtypes.get(config.tensor_dtype(f"{component}.{name}", default="bf16"))
        if dtype is not None:
            parameter.data = parameter.data.to(dtype)
    for prefix, child in module.named_modules():
        for name, buffer in child.named_buffers(recurse=False):
            if name in child._non_persistent_buffers_set or not buffer.is_floating_point():
                continue
            key = ".".join(part for part in (prefix, name) if part)
            default = {torch.float32: "f32", torch.float64: "f64", torch.float16: "f16", torch.bfloat16: "bf16"}.get(buffer.dtype, "f32")
            dtype = dtypes.get(config.tensor_dtype(f"{component}.{key}", default=default))
            if dtype is not None:
                buffer.data = buffer.data.to(dtype)
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


class Components(TypedDict):
    transformer: LTX2VideoTransformer3DModel
    text_encoder: Gemma4UnifiedForConditionalGeneration
    connectors: LTX2TextConnectors
    vae: AutoencoderKLLTX2Video
    audio_vae: AutoencoderKLLTX2Audio
    vocoder: LTX2VocoderWithBWE
    latent_upsampler: LTX2LatentUpsamplerModel


class Graph:
    components: Components
    def __init__(self, config: Config) -> None:
        text_settings = config.section("text_encoder").mapping()
        language_settings = config.section("text_encoder").section("text_config").mapping()
        softcap = language_settings.get("final_logit_softcapping")
        # Canonical JSON writes 30.0 as 30. Gemma's strict float-only field rejects
        # that equivalent integer spelling; restore its Python type at this boundary.
        if type(softcap) is int:
            language_settings["final_logit_softcapping"] = float(softcap)
        text_settings["text_config"] = language_settings
        text = Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(**text_settings))
        # Prompt encoding uses hidden states, never vocabulary logits; this tied projection
        # is not a separate stored tensor and would otherwise allocate [tokens,vocabulary].
        text.lm_head = nn.Identity()
        vae = VideoVAE.from_config(config.section("vae").mapping())
        vae.enable_tiling()
        self.components = {
            "transformer": stored(LTX2VideoTransformer3DModel.from_config(config.section("transformer").mapping()), "transformer", config),
            "text_encoder": stored(text, "text_encoder", config),
            "connectors": stored(LTX2TextConnectors.from_config(config.section("connectors").mapping()), "connectors", config),
            "vae": stored(vae, "vae", config),
            "audio_vae": stored(AutoencoderKLLTX2Audio.from_config(config.section("audio_vae").mapping()), "audio_vae", config),
            # The reference vocoder computes in F32. Its Cozy checkpoint stores that exact,
            # losslessly widened representation before Runtime places any weights or buffers.
            "vocoder": LTX2VocoderWithBWE.from_config(config.section("vocoder").mapping()).float().eval(),
            "latent_upsampler": stored(LTX2LatentUpsamplerModel.from_config(config.section("latent_upsampler").mapping()), "latent_upsampler", config),
        }
        install(self.components["transformer"])
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


class _Stream:
    """One noise stream shared by the video and audio schedulers (the pipeline deep-copies its scheduler)."""

    def __init__(self, generator: torch.Generator) -> None:
        self.generator = generator

    def __deepcopy__(self, memo: object) -> _Stream:
        return self


class Ancestral(LTXEulerAncestralRFScheduler):
    """proof/se-161: native LTX-2.5's Euler ancestral (eta 1, s_noise 1; ltx_pipelines/distilled.py), video's
    noise drawn before audio's each step, from its own stream."""

    stream: _Stream

    def set_timesteps(self, num_inference_steps: int | None = None, device: str | torch.device | None = None,
                      sigmas: list[float] | torch.Tensor | None = None, **kwargs: object) -> None:
        super().set_timesteps(device=device, sigmas=[*(float(s) for s in (sigmas or [])), 0.0])
        self.timesteps = self.timesteps[:-1]  # the terminal zero is the last step's target, not a step

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):  # type: ignore[no-untyped-def,override]
        return super().step(model_output, timestep, sample, generator=self.stream.generator, return_dict=return_dict)


@sequence_parallel(degrees=(2, 4, 8))
class Ltx25(Model[Graph]):
    serving = Serving(attention_bits={"transformer.transformer_blocks.": 8})
    graph: Graph
    tokenizer: GemmaTokenizer

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)
        self.tokenizer = tokenizer(loader.assets)

    def _pipeline(self, image: bool = False, noise: int | None = None) -> LTX2Pipeline | LTX2ImageToVideoPipeline:
        c = self.graph.components
        cls = ImagePipeline if image else TextPipeline
        scheduler: FlowMatchEulerDiscreteScheduler | Ancestral
        if noise is None:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config)
        else:
            scheduler = Ancestral(eta=1.0, s_noise=1.0)
            scheduler.stream = _Stream(rng.generator(noise))
        pipe = cls(
            scheduler=scheduler,  # type: ignore[arg-type]
            vae=c["vae"], audio_vae=c["audio_vae"], text_encoder=c["text_encoder"],
            tokenizer=self.tokenizer, connectors=c["connectors"], transformer=c["transformer"], vocoder=c["vocoder"],
        )
        pipe.set_progress_bar_config(disable=True)  # type: ignore[no-untyped-call]  # Diffusers leaves progress configuration unannotated.
        return pipe

    @memoize
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
                latents: tuple[torch.Tensor, torch.Tensor] | None = None, noise: int | None = None,
                encoded: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        # Both returns are already denormalized by Diffusers' output_type='latent'.
        # The VAE roots are also leased: their latent-statistic buffers participate here.
        pipe = self._pipeline(image=image is not None, noise=noise)
        sigmas = list(SIGMAS if latents is None else REFINE_SIGMAS)
        if latents is not None and image is not None and encoded:
            # Native LTX-2.5 and its ComfyUI template condition stage 2 on the picture encoded at stage 2's size;
            # Diffusers' pipeline given `latents` keeps the upsampled stage-1 encode instead.
            first = self._first_frame(pipe, image, width=width, height=height)
            with torch.inference_mode():
                video = latents[0].clone()
                video[:, :, :1] = first.to(video.dtype)
            latents = (video, latents[1])

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

    def _first_frame(self, pipe: LTX2Pipeline | LTX2ImageToVideoPipeline, image: Image.Image, *, width: int,
                     height: int) -> torch.Tensor:
        """The picture's latent at `width`x`height`, as Diffusers' pipeline encodes a first frame (CRF re-compression,
        resize, the posterior's mode), unnormalized."""
        vae = self.graph.components["vae"]
        crf = resolve_default_image_crf(pipe.text_encoder)
        if crf:
            image = Image.fromarray(apply_image_conditioning_crf(np.asarray(image.convert("RGB")), crf))
        pixels = pipe.video_processor.preprocess(image, height=height, width=width)
        with torch.inference_mode():
            pixels = pixels.to(torch.device(str(self.device)), vae.dtype)
            return cast(torch.Tensor, retrieve_latents(vae.encode(pixels.unsqueeze(2)), None, "argmax"))

    @uses_components("latent_upsampler", "vae")
    def upscale(self, latents: torch.Tensor) -> torch.Tensor:
        pipe = LTX2LatentUpsamplePipeline(vae=self.graph.components["vae"], latent_upsampler=self.graph.components["latent_upsampler"])
        pipe.set_progress_bar_config(disable=True)  # type: ignore[no-untyped-call]  # Diffusers leaves progress configuration unannotated.
        with torch.inference_mode():
            return cast(torch.Tensor, pipe(latents=latents, latents_normalized=False, output_type="latent", return_dict=False)[0])

    @uses_components("vae")
    def decode_video(self, latents: torch.Tensor) -> VideoPixels:
        vae = self.graph.components["vae"]
        with torch.inference_mode():
            pixels = vae.decode(latents.to(vae.dtype), None, return_dict=False)[0]
        pixels = pixels[0].permute(1, 2, 3, 0)
        result = np.empty(tuple(pixels.shape), dtype=np.uint8)
        # A full 20-second 1080p F32 copy is 12 GB. Pointwise conversion needs only a
        # small frame batch; keeping it bounded preserves the exact rounded RGB8 bytes.
        for start in range(0, len(pixels), 16):
            chunk = pixels[start:start + 16].to(dtype=torch.float32, copy=True).add_(1).mul_(127.5).round_().clamp_(0, 255)
            result[start:start + 16] = chunk.to(torch.uint8).cpu().numpy()
        return cast(VideoPixels, result)

    @uses_components("audio_vae", "vocoder")
    def decode_audio(self, latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        vae = self.graph.components["audio_vae"]
        vocoder = self.graph.components["vocoder"]
        with torch.inference_mode():
            mel = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
            audio = vocoder(mel.to(vocoder.dtype))[0].float().cpu()
        return audio, int(vocoder.config["output_sampling_rate"])

