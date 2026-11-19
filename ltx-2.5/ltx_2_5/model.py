"""LTX-2.5 Distilled over official Diffusers components, with Runtime-owned residency."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from contextlib import nullcontext
from typing import TypedDict, cast

import numpy as np
import torch
from cozy_runtime import rng
from cozy_runtime.accel import sequence_parallel
from cozy_runtime.author import Config, Loader, Model, ModelAssets, Serving, device, memoize, uses_components
from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
from diffusers.models.condition_embedders.condition_embedder_ltx2 import LTX2TextConnectors
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
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

from . import mel
from .frames import install as install_frames
from .frames import per_frame
from .vae import VideoPixels, VideoVAE
from .parallel import install

FPS = 24
SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)
REFINE_SIGMAS = SIGMAS[-3:]
#: A canvas's ancestral first stage draws its step noise from the seed plus this (ltx_pipelines.distilled
#: `ANCESTRAL_NOISE_SEED_OFFSET`), apart from its starting noise.
ANCESTRAL_SEED = 10000
#: Seconds of a canvas, as each modality's mask reads them: [start, end).
type Span = tuple[float, float]


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
    vae: VideoVAE
    audio_vae: AutoencoderKLLTX2Audio
    vocoder: LTX2VocoderWithBWE
    latent_upsampler: LTX2LatentUpsamplerModel


def text_encoder(config: Config) -> Gemma4UnifiedForConditionalGeneration:
    """Gemma 4 12B from an LTX-2.5 checkpoint's `text_encoder` section, or from the whole config of a checkpoint of the
    encoder alone (Runtime gives one component's config unsectioned)."""
    section = config.section("text_encoder") if "text_encoder" in config.keys() else config  # noqa: SIM118  # Config has no __contains__
    text_settings = section.mapping()
    language_settings = section.section("text_config").mapping()
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
    return stored(text, "text_encoder", config)


class Graph:
    components: Components
    def __init__(self, config: Config) -> None:
        vae = VideoVAE.from_config(config.section("vae").mapping())
        vae.enable_tiling()
        self.components = {
            "transformer": stored(LTX2VideoTransformer3DModel.from_config(config.section("transformer").mapping()), "transformer", config),
            "text_encoder": text_encoder(config),
            "connectors": stored(LTX2TextConnectors.from_config(config.section("connectors").mapping()), "connectors", config),
            "vae": stored(vae, "vae", config),
            "audio_vae": stored(AutoencoderKLLTX2Audio.from_config(config.section("audio_vae").mapping()), "audio_vae", config),
            # The reference vocoder computes in F32. Its Cozy checkpoint stores that exact,
            # losslessly widened representation before Runtime places any weights or buffers.
            "vocoder": LTX2VocoderWithBWE.from_config(config.section("vocoder").mapping()).float().eval(),
            "latent_upsampler": stored(LTX2LatentUpsamplerModel.from_config(config.section("latent_upsampler").mapping()), "latent_upsampler", config),
        }
        install(self.components["transformer"])
        install_frames(self.components["transformer"])
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


class TextEncoderComponents(TypedDict):
    text_encoder: Gemma4UnifiedForConditionalGeneration


class TextEncoderGraph:
    """The `text_encoder` of an LTX-2.5 checkpoint, or of a checkpoint of the encoder alone."""

    components: TextEncoderComponents

    def __init__(self, config: Config) -> None:
        self.components = {"text_encoder": text_encoder(config)}


def encode_prompt(encoder: Gemma4UnifiedForConditionalGeneration, tokens: GemmaTokenizer, prompt: str,
                  at: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffusers' LTX-2 prompt encoding: Gemma's hidden states packed, and their mask; the connectors run in `denoise`."""
    # Diffusers registers an absent module as None; prompt encoding reads only the encoder and the tokenizer.
    absent: dict[str, None] = dict.fromkeys(("scheduler", "vae", "audio_vae", "connectors", "transformer", "vocoder"))
    pipe = TextPipeline(text_encoder=encoder, tokenizer=tokens, **absent)  # type: ignore[arg-type]
    pipe.set_progress_bar_config(disable=True)  # type: ignore[no-untyped-call]  # Diffusers leaves progress configuration unannotated.
    with torch.inference_mode():
        embeds, mask, _, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False, max_sequence_length=1024,
                                                 device=at)
    return embeds, mask


@sequence_parallel(degrees=(2, 4, 8))
class Ltx25TextEncoder(Model[TextEncoderGraph]):
    """LTX-2.5's Gemma 4 12B, the `text_encoder` slot: an LTX-2.5 checkpoint's encoder and tokenizer, or a checkpoint
    of the encoder alone. Only the encoder is built. Its degrees are `Ltx25`'s, so a group serves both slots."""

    graph: TextEncoderGraph
    tokenizer: GemmaTokenizer

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(TextEncoderGraph)
        self.tokenizer = tokenizer(loader.assets)

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        return encode_prompt(self.graph.components["text_encoder"], self.tokenizer, prompt, torch.device(str(self.device)))


@sequence_parallel(degrees=(2, 4, 8))
class Ltx25(Model[Graph]):
    serving = Serving(attention_bits={"transformer.transformer_blocks.": 8})
    graph: Graph
    tokenizer: GemmaTokenizer

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)
        self.tokenizer = tokenizer(loader.assets)

    def _pipeline(self, image: bool = False) -> LTX2Pipeline | LTX2ImageToVideoPipeline:
        c = self.graph.components
        cls = ImagePipeline if image else TextPipeline
        pipe = cls(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            vae=c["vae"], audio_vae=c["audio_vae"], text_encoder=c["text_encoder"],
            tokenizer=self.tokenizer, connectors=c["connectors"], transformer=c["transformer"], vocoder=c["vocoder"],
        )
        pipe.set_progress_bar_config(disable=True)  # type: ignore[no-untyped-call]  # Diffusers leaves progress configuration unannotated.
        return pipe

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The checkpoint's own encoder; `generate` encodes with its `text_encoder` slot (`Ltx25TextEncoder`). The graph
        keeps the encoder because the int8 recipe quantizes it through this class and `_first_frame` reads its config for
        the CRF (18, not 33); this method declares it (`undeclared_construction`)."""
        return encode_prompt(self.graph.components["text_encoder"], self.tokenizer, prompt, torch.device(str(self.device)))

    @uses_components("transformer", "connectors", "vae", "audio_vae")
    def denoise(self, embeds: tuple[torch.Tensor, torch.Tensor], image: Image.Image | None, *, width: int, height: int,
                frames: int, generator: torch.Generator, check: Callable[[], None], progress: Callable[[int], None],
                latents: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Both returns are already denormalized by Diffusers' output_type='latent'.
        # The VAE roots are also leased: their latent-statistic buffers participate here.
        pipe = self._pipeline(image=image is not None)
        sigmas = list(SIGMAS if latents is None else REFINE_SIGMAS)
        if latents is not None and image is not None:
            # Stage 2 conditions on the picture encoded at its own size, as native LTX-2.5 (ltx_pipelines'
            # image_conditionings_for_chunk) and its ComfyUI template do; Diffusers' pipeline given `latents` would keep
            # the upsampled half-size encode, a blurred first frame.
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
        # Diffusers' image-to-video timestep is `t * (1 - mask)` with the mask set on whole latent frames (frame 0,
        # pipeline_ltx2_image2video.py:806-807 and 850-851), so its video modulation embeds once a frame.
        try:
            with torch.inference_mode(), per_frame() if image is not None else nullcontext():
                video, audio = pipe(**kwargs)
            return video, audio
        finally:
            pipe.transformer._reset_stateful_cache()

    def _first_frame(self, pipe: LTX2Pipeline | LTX2ImageToVideoPipeline, image: Image.Image, *, width: int,
                     height: int) -> torch.Tensor:
        """The picture's unnormalized latent at `width`x`height`, encoded as Diffusers' pipeline encodes a first frame:
        CRF re-compression, resize, the posterior's mode."""
        vae = self.graph.components["vae"]
        crf = resolve_default_image_crf(pipe.text_encoder)
        if crf:
            image = Image.fromarray(apply_image_conditioning_crf(np.asarray(image.convert("RGB")), crf))
        pixels = pipe.video_processor.preprocess(image, height=height, width=width)
        with torch.inference_mode():
            pixels = pixels.to(torch.device(str(self.device)), vae.dtype)
            return cast(torch.Tensor, retrieve_latents(vae.encode(pixels.unsqueeze(2)), None, "argmax"))

    @uses_components("transformer", "connectors", "vae", "audio_vae")
    def canvas(self, embeds: tuple[torch.Tensor, torch.Tensor], video: torch.Tensor, audio: torch.Tensor | None, *,
               video_span: Span | None, audio_span: Span | None, seed: int, check: Callable[[], None],
               progress: Callable[[int], None]) -> tuple[torch.Tensor, torch.Tensor]:
        """A canvas of held frames and sound with spans made again: upstream's retake (`TemporalRegionMask`: every
        token whose seconds meet the span is made, the rest is held clean), sampled as LTX Desktop samples distilled
        2.5's retake and extend: an ancestral stage over the 8 distilled sigmas, then a same-size Euler refine over the
        last 3 ("Euler-only on that 8-step schedule never lets the prompt take").

        `video` [1, C, f, h, w] and `audio` [1, C, L, M] are the VAEs' unnormalized latents; no `audio` makes all of
        the sound. A span of None holds that modality whole, as frozen conditioning. Returns both, denormalized."""
        pipe = self._pipeline()
        transformer, vae, sound = pipe.transformer, pipe.vae, pipe.audio_vae
        device = torch.device(str(self.device))
        _, _, f, h, w = video.shape
        frames = (f - 1) * pipe.vae_temporal_compression_ratio + 1
        length = round(frames / FPS * pipe.audio_sampling_rate / pipe.audio_hop_length
                       / pipe.audio_vae_temporal_compression_ratio)
        bins = int(sound.config["mel_bins"]) // pipe.audio_vae_mel_compression_ratio
        patch, patch_t = pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
        scaling = float(vae.config["scaling_factor"])
        check()
        try:
            with torch.inference_mode():
                texts, audio_texts, text_mask = pipe.connectors(embeds[0], embeds[1],
                                                                padding_side=getattr(pipe.tokenizer, "padding_side", "left"))
                clean_v = pipe._pack_latents(pipe._normalize_latents(video.to(device, torch.float32), vae.latents_mean,
                                                                     vae.latents_std, scaling), patch, patch_t)
                if audio is None:
                    clean_a = torch.zeros(1, length, int(sound.config["latent_channels"]) * bins, device=device)
                else:
                    held = audio.to(device, torch.float32)[:, :, :length]
                    held = torch.nn.functional.pad(held, (0, 0, 0, length - held.shape[2]))
                    clean_a = pipe._normalize_audio_latents(pipe._pack_audio_latents(held), sound.latents_mean,
                                                            sound.latents_std)
                video_coords = transformer.rope.prepare_video_coords(1, f, h, w, device, fps=float(FPS))
                audio_coords = transformer.audio_rope.prepare_audio_coords(1, length, device)
                mask_v = _inside(video_coords, video_span)
                mask_a = _inside(audio_coords, (0.0, math.inf) if audio is None else audio_span)
                # The causal first latent frame encodes one pixel frame; upstream marks it for 2.5's keyframe embedding
                # whatever conditions it (ltx_core/tools.py `_first_frame_keyframes_mask`).
                marker = torch.zeros_like(mask_v)
                marker[:, : h * w] = 1.0
                noise = rng.generator(seed)
                states = (_noised(clean_v, mask_v, 1.0, noise), _noised(clean_a, mask_a, 1.0, noise))
                done = 0

                def stage(states: tuple[torch.Tensor, torch.Tensor], cleans: tuple[torch.Tensor, torch.Tensor],
                          sigmas: tuple[float, ...], eta: float) -> tuple[torch.Tensor, torch.Tensor]:
                    nonlocal done
                    x_v, x_a = states
                    for index, (sigma, next_sigma) in enumerate(zip(sigmas, (*sigmas[1:], 0.0), strict=True)):
                        check()
                        t = torch.full((1,), sigma * 1000.0, device=device)
                        with transformer.cache_context("cond_uncond"), per_frame():
                            velocity_v, velocity_a = transformer(
                                hidden_states=x_v.to(texts.dtype), audio_hidden_states=x_a.to(texts.dtype),
                                encoder_hidden_states=texts, audio_encoder_hidden_states=audio_texts,
                                timestep=t[:, None] * mask_v[..., 0], audio_timestep=t[:, None] * mask_a[..., 0],
                                # A modality held whole is clean conditioning: sigma 0 for its prompt AdaLN and the
                                # cross-modal gates (ltx_pipelines/utils/helpers.py `modality_from_latent_state`).
                                sigma=t * float(mask_v.max()), audio_sigma=t * float(mask_a.max()),
                                encoder_attention_mask=text_mask, audio_encoder_attention_mask=text_mask,
                                num_frames=f, height=h, width=w, fps=float(FPS), audio_num_frames=length,
                                video_coords=video_coords, audio_coords=audio_coords, video_keyframes_mask=marker,
                                isolate_modalities=False, spatio_temporal_guidance_blocks=None, perturbation_mask=None,
                                use_cross_timestep=True, return_dict=False,
                            )
                        # Each step's ancestral noise is its own counter draw, video then audio.
                        x_v = _step(x_v, velocity_v, cleans[0], mask_v, sigma, next_sigma, eta,
                                    (seed + ANCESTRAL_SEED, done + index, 0))
                        x_a = _step(x_a, velocity_a, cleans[1], mask_a, sigma, next_sigma, eta,
                                    (seed + ANCESTRAL_SEED, done + index, 1))
                        progress(done + index)
                    done += len(sigmas)
                    return x_v, x_a

                made = stage(states, (clean_v, clean_a), SIGMAS, 1.0)
                # The refine starts from what the first stage made, its made part renoised to the refine's first sigma
                # (Desktop `_with_stage2_latent`); the held part is the take's, already clean.
                states = (_noised(made[0], mask_v, REFINE_SIGMAS[0], noise),
                          _noised(made[1], mask_a, REFINE_SIGMAS[0], noise))
                x_v, x_a = stage(states, made, REFINE_SIGMAS, 0.0)
                video = pipe._denormalize_latents(pipe._unpack_latents(x_v, f, h, w, patch, patch_t), vae.latents_mean,
                                                  vae.latents_std, scaling)
                audio = pipe._unpack_audio_latents(pipe._denormalize_audio_latents(x_a, sound.latents_mean,
                                                                                   sound.latents_std), length,
                                                   num_mel_bins=bins)
            return video, audio
        finally:
            transformer._reset_stateful_cache()

    @uses_components("vae")
    def encode_video(self, frames: VideoPixels) -> torch.Tensor:
        """RGB8 frames (`8k + 1`, sides on 32) as the video VAE's latents [1, C, f, h, w], unnormalized, as upstream
        encodes a retake's source."""
        vae = self.graph.components["vae"]
        with torch.inference_mode():
            return vae.latents(frames, torch.device(str(self.device)))

    @uses_components("audio_vae")
    def encode_audio(self, waveform: torch.Tensor, rate: int) -> torch.Tensor:
        """A waveform [channels, samples] as the audio VAE's latents [1, C, L, M], unnormalized, as upstream encodes
        a soundtrack (`encode_audio`: log-mel at the VAE's rate, the posterior's mode)."""
        vae = self.graph.components["audio_vae"]
        config = vae.config
        with torch.inference_mode():
            spectrogram = mel.log_mel(waveform.to(torch.device(str(self.device))), rate, int(config["sample_rate"]),
                                      int(config["mel_bins"]), int(config["mel_hop_length"]),
                                      channels=int(config["in_channels"]))
            posterior = cast(tuple[DiagonalGaussianDistribution], vae.encode(spectrogram.to(vae.dtype), return_dict=False))
            return posterior[0].mode().float()

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
            return vae.frames(latents.to(vae.dtype))

    @uses_components("audio_vae", "vocoder")
    def decode_audio(self, latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        vae = self.graph.components["audio_vae"]
        vocoder = self.graph.components["vocoder"]
        with torch.inference_mode():
            spectrogram = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
            audio = vocoder(spectrogram.to(vocoder.dtype))[0].float().cpu()
        return audio, int(vocoder.config["output_sampling_rate"])


def _inside(coords: torch.Tensor, span: Span | None) -> torch.Tensor:
    """[1, tokens, 1]: 1 on each token whose seconds meet `span` (upstream's `TemporalRegionMask`), from RoPE's
    [1, axes, tokens, 2] bounds, time first. A boundary that only touches the span stays out."""
    if span is None:
        return torch.zeros(1, coords.shape[2], 1, device=coords.device)
    start, end = span
    return ((coords[0, 0, :, 1] > start + 1e-4) & (coords[0, 0, :, 0] < end - 1e-4)).float()[None, :, None]


def _noised(latent: torch.Tensor, mask: torch.Tensor, scale: float, generator: torch.Generator) -> torch.Tensor:
    """`latent` with its masked tokens at noise level `scale` (upstream's `GaussianNoiser`); the rest stays clean."""
    if not bool(mask.any()):
        return latent
    noise = rng.noise(latent.shape, generator, dtype=latent.dtype, device=latent.device)
    return torch.lerp(latent, torch.lerp(latent, noise, scale), mask)


def _step(sample: torch.Tensor, velocity: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor, sigma: float,
          next_sigma: float, eta: float, draw: tuple[int, int, int]) -> torch.Tensor:
    """One rectified-flow Euler step on the masked tokens, ancestral at `eta` > 0 (ltx_core's
    `EulerAncestralDiffusionStep`: to `sigma_down`, then renoised to `next_sigma`, variance-preserving), its noise
    the Runtime's counter draw for (seed, step, stream) `draw`; held tokens stay clean."""
    if not bool(mask.any()):
        return sample
    denoised = torch.lerp(clean, sample - velocity.float() * sigma, mask)
    if next_sigma == 0:
        return denoised
    down = next_sigma * (1.0 + (next_sigma / sigma - 1.0) * eta)
    ratio = down / sigma
    stepped = ratio * sample + (1.0 - ratio) * denoised
    if eta <= 0:
        return stepped
    kept, lowered = 1.0 - next_sigma, 1.0 - down
    renoise = math.sqrt(max(next_sigma**2 - down**2 * kept**2 / lowered**2, 0.0))
    noise = rng.counter_noise(sample.shape, draw[0], draw[1], stream=draw[2], dtype=sample.dtype, device=sample.device)
    return torch.lerp(clean, (kept / lowered) * stepped + noise * renoise, mask)

