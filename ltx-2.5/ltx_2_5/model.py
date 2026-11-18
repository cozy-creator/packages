"""LTX-2.5 Distilled over official Diffusers components, with Runtime-owned residency."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
import torch
from cozy_runtime.accel import lora, sequence_parallel
from cozy_runtime.author import Config, Loader, Model, ModelAssets, Serving, device, memoize, uses_components
from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
from diffusers.models.condition_embedders.condition_embedder_ltx2 import LTX2TextConnectors
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import (
    LTX2InContextPipeline, LTX2ReferenceCondition, calculate_shift, retrieve_timesteps,
)
from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline
from diffusers.models.latent_upscaler.latent_upscaler_ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from transformers import AddedToken, Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration, GemmaTokenizer

from . import mel
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
        text = Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(**config.section("text_encoder").mapping()))
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
        #: Where an adapter's LoRA arms on the DiT (Lightricks' Ingredients, `adapter.Ingredients`): every linear.
        self.sites = lora.Sites(self.components["transformer"])
        install(self.components["transformer"])
        self.scheduler_config = config.section("scheduler").mapping()


#: A picture, or a clip's frames, and the frame of the video it is: 0 opens the video on it (a clip's frames open it in
#: turn, as a pickup's carry does); a later frame is a guide there.
Frame = tuple[Image.Image | list[Image.Image], int]


@dataclass(frozen=True)
class Window:
    """A retake's seconds `[start, end)`: inside them the video, the sound or both are redone, and the rest is kept from
    the given latents (upstream's `TemporalRegionMask`)."""

    start: float
    end: float
    video: bool
    sound: bool

    def inside(self, coords: torch.Tensor) -> torch.Tensor:
        """1 on each token whose seconds meet the window; `coords` [1, axes, tokens, 2], time first, as RoPE takes them."""
        return ((coords[0, 0, :, 1] > self.start) & (coords[0, 0, :, 0] < self.end)).float()[None, :, None]


class Pipeline(LTX2InContextPipeline):
    """Diffusers' LTX-2 in-context pipeline's conditioning, driven by one distilled loop of our own: pictures at frames,
    an IC-LoRA's reference, and sound held as it is, the way upstream's stages take them."""

    #: Each guide's own frame, in the order Diffusers places them: its `index` is a latent's, whose first frame is
    #: `8k+1`, while upstream's `ImageGuide` takes any frame (ltx_pipelines/utils/types.py `ImageGuide`).
    guides: list[int]

    @property
    def _execution_device(self) -> torch.device:
        # The earlier stage's text root may be parked while the DiT is active. The first
        # component in Diffusers' registry is not the placement authority.
        return torch.device(str(device()))

    def conditions(self, frames: Sequence[Frame]) -> list[LTX2VideoCondition]:
        """Each picture at its frame, at full strength (the model card's `--image PATH FRAME_IDX 1.0`)."""
        scale = self.vae_temporal_compression_ratio
        self.guides = []
        made: list[LTX2VideoCondition] = []
        for picture, frame in frames:
            index = 0 if frame == 0 else (frame - 1) // scale + 1
            if index:
                self.guides.append(frame)
            made.append(LTX2VideoCondition(frames=picture, index=index, strength=1.0))
        return made

    def _prepare_keyframe_coords(self, keyframe_latent_num_frames: int, keyframe_latent_height: int,
                                 keyframe_latent_width: int, pixel_frame_idx: int, num_pixel_frames: int, fps: float,
                                 device: torch.device) -> torch.Tensor:
        return super()._prepare_keyframe_coords(
            keyframe_latent_num_frames, keyframe_latent_height, keyframe_latent_width,
            self.guides.pop(0) if self.guides else pixel_frame_idx, num_pixel_frames, fps, device)

    def run(self, embeds: tuple[torch.Tensor, torch.Tensor], pictures: Sequence[Frame], *, frames: int,
            sigmas: Sequence[float], generator: torch.Generator, check: Callable[[], None],
            progress: Callable[[int], None], width: int = 0, height: int = 0, reference: list[Image.Image] | None = None,
            downscale: int = 1, held: torch.Tensor | None = None,
            latents: tuple[torch.Tensor, torch.Tensor | None] | None = None,
            window: Window | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One distilled stage (no guidance): Diffusers' in-context latents (pictures, then an IC-LoRA's reference
        video, at 1 / `downscale` of the stage's size),
        upstream's first-frame keyframe marker, and `held` audio latents ([1, C, L, M], not normalized) kept clean
        over the clip's first L audio latents: all of them is a soundtrack (upstream's `freeze_audio`), fewer a carry
        (`AudioConditionByLatentIndex`). `window` redoes only its seconds of `latents` and `held` (a retake). Width 0 is
        sound alone. Returns the video's and the audio's latents, denormalized."""
        device = self._execution_device
        fps = float(FPS)
        noise_scale = sigmas[0]
        texts, audio_texts, text_mask = self.connectors(embeds[0], embeds[1], padding_side=self.tokenizer_padding_side)
        alone = width == 0
        if alone:
            # Lightricks' audio-only text-to-audio (ComfyUI-LTXVideo `LTXVAudioOnlyModel`, `LTXVAudioOnlyEmptyVideoLatent`):
            # a one-frame stub video nothing attends to, the modalities isolated. Its 8 tokens split whole at every
            # sequence-parallel degree.
            f, h, w = 1, 2, 4
            video = torch.zeros(1, f * h * w, self.transformer.config.in_channels, device=device, dtype=torch.float32)
            mask, clean, appended = torch.ones_like(video[..., :1]), torch.zeros_like(video), None
        else:
            f = (frames - 1) // self.vae_temporal_compression_ratio + 1
            h, w = height // self.vae_spatial_compression_ratio, width // self.vae_spatial_compression_ratio
            references = [LTX2ReferenceCondition(frames=reference, strength=1.0)] if reference else None
            video, mask, clean, appended, _, _, _ = self.prepare_latents(
                conditions=self.conditions(pictures) or None, reference_conditions=references,
                reference_downscale_factor=downscale, batch_size=1,
                num_channels_latents=self.transformer.config.in_channels, height=height, width=width,
                num_frames=frames, frame_rate=fps, noise_scale=noise_scale, dtype=torch.float32, device=device,
                generator=generator, latents=None if latents is None else latents[0])
        # The causal first latent frame encodes one pixel frame; upstream marks it for LTX-2.5's keyframe embedding
        # whatever conditions it (ltx_core/tools.py `_first_frame_keyframes_mask`); appended tokens are unmarked.
        marker = torch.zeros(video.shape[0], video.shape[1], 1, device=device, dtype=torch.float32)
        marker[:, : h * w] = 1.0
        video_coords = self.transformer.rope.prepare_video_coords(1, f, h, w, device, fps=fps)
        if window is not None and latents is not None:
            # Outside the window the take's own latents stay clean; inside, the noise prepared above is redone.
            n = f * h * w
            source = self._pack_latents(self._normalize_latents(
                latents[0].to(device=device, dtype=torch.float32), self.vae.latents_mean, self.vae.latents_std,
                self.vae.config.scaling_factor), self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
            keep = 1 - window.inside(video_coords) if window.video else torch.ones_like(mask[:, :n])
            video[:, :n] = keep * source + (1 - keep) * video[:, :n]
            clean[:, :n] = keep * source + (1 - keep) * clean[:, :n]
            mask[:, :n] = torch.maximum(mask[:, :n], keep)
        if appended is not None:
            video_coords = torch.cat([video_coords, appended], dim=2)
        length = round(frames / fps * self.audio_sampling_rate / self.audio_hop_length / self.audio_vae_temporal_compression_ratio)
        bins = self.audio_mel_bins // self.audio_vae_mel_compression_ratio
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(1, length, device)
        given = None if latents is None else latents[1]
        if given is None:
            audio = randn_tensor((1, length, self.audio_latent_channels * bins), generator=generator, device=device,
                                 dtype=torch.float32)
        else:
            audio = self._create_noised_state(self._normalize_audio_latents(
                self._pack_audio_latents(given), self.audio_vae.latents_mean, self.audio_vae.latents_std),
                noise_scale, generator).to(device=device, dtype=torch.float32)
        open_ = torch.ones(1, length, 1, device=device, dtype=torch.float32)
        fixed = torch.zeros_like(audio)
        if held is not None:
            kept = self._normalize_audio_latents(self._pack_audio_latents(held.to(device=device, dtype=torch.float32)),
                                                 self.audio_vae.latents_mean, self.audio_vae.latents_std)[:, :length]
            fixed[:, : kept.shape[1]] = kept
            open_[:, : kept.shape[1]] = 0.0
        if window is not None and window.sound:
            open_ = torch.maximum(open_, window.inside(audio_coords))
        audio = audio * open_ + fixed * (1 - open_)
        frozen = bool(open_.max() == 0)
        # A video kept whole (sound alone, or a retake of the sound only) is frozen conditioning too.
        still = bool(mask[:, : f * h * w].min() == 1)
        audio_scheduler = copy.deepcopy(self.scheduler)
        mu = calculate_shift(video.shape[1], self.scheduler.config.get("base_image_seq_len", 1024),
                             self.scheduler.config.get("max_image_seq_len", 4096),
                             self.scheduler.config.get("base_shift", 0.95), self.scheduler.config.get("max_shift", 2.05))
        audio_steps, _ = retrieve_timesteps(audio_scheduler, len(sigmas), device, None, sigmas=list(sigmas), mu=mu)
        steps, _ = retrieve_timesteps(self.scheduler, len(sigmas), device, None, sigmas=list(sigmas), mu=mu)
        dtype = embeds[0].dtype
        for i, t in enumerate(steps):
            check()
            sigma = t.expand(1)
            audio_sigma = audio_steps[i].expand(1)
            with self.transformer.cache_context("cond_uncond"):
                velocity, audio_velocity = self.transformer(
                    hidden_states=video.to(dtype), audio_hidden_states=audio.to(dtype),
                    encoder_hidden_states=texts, audio_encoder_hidden_states=audio_texts,
                    timestep=sigma.unsqueeze(-1) * (1 - mask.squeeze(-1)),
                    audio_timestep=audio_sigma.unsqueeze(-1) * open_.squeeze(-1),
                    # A frozen modality is clean conditioning: its sigma is 0 for the prompt AdaLN and the
                    # cross-modal gates (ltx_pipelines/utils/helpers.py `modality_from_latent_state`).
                    sigma=torch.zeros_like(sigma) if still else sigma,
                    audio_sigma=torch.zeros_like(audio_sigma) if frozen else audio_sigma,
                    encoder_attention_mask=text_mask, audio_encoder_attention_mask=text_mask,
                    num_frames=f, height=h, width=w, fps=fps, audio_num_frames=length,
                    video_coords=video_coords, audio_coords=audio_coords, video_keyframes_mask=marker,
                    isolate_modalities=alone, spatio_temporal_guidance_blocks=None, perturbation_mask=None,
                    use_cross_timestep=True, return_dict=False,
                )
            if not still:
                denoised = self.convert_velocity_to_x0(video, velocity.float(), i, self.scheduler)
                denoised = denoised * (1 - mask) + clean * mask
                video = self.scheduler.step(self.convert_x0_to_velocity(video, denoised, i, self.scheduler), t, video,
                                            return_dict=False)[0]
            if not frozen:
                sound = self.convert_velocity_to_x0(audio, audio_velocity.float(), i, audio_scheduler)
                sound = sound * open_ + fixed * (1 - open_)
                audio = audio_scheduler.step(self.convert_x0_to_velocity(audio, sound, i, audio_scheduler), t, audio,
                                             return_dict=False)[0]
            progress(i)
        video = self._unpack_latents(video[:, : f * h * w], f, h, w, self.transformer_spatial_patch_size,
                                     self.transformer_temporal_patch_size)
        if not alone:
            video = self._denormalize_latents(video, self.vae.latents_mean, self.vae.latents_std,
                                              self.vae.config.scaling_factor)
        if frozen:
            audio = fixed
        audio = self._denormalize_audio_latents(audio, self.audio_vae.latents_mean, self.audio_vae.latents_std)
        return video, self._unpack_audio_latents(audio, length, num_mel_bins=bins)


@sequence_parallel(degrees=(2, 4, 8))
class Ltx25(Model[Graph]):
    # 8-bit attention in the DiT's blocks (se-139); a request's LoRAs on both stages, as upstream's distilled stages take
    # them (ltx_pipelines/distilled.py `loras` on its one `DiffusionStage`).
    serving = Serving(attention_bits={"transformer.transformer_blocks.": 8}, loras=("transformer",))
    graph: Graph
    tokenizer: GemmaTokenizer

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)
        self.tokenizer = tokenizer(loader.assets)

    def pipeline(self) -> Pipeline:
        c = self.graph.components
        pipe = Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            vae=c["vae"], audio_vae=c["audio_vae"], text_encoder=c["text_encoder"],
            tokenizer=self.tokenizer, connectors=c["connectors"], transformer=c["transformer"], vocoder=c["vocoder"],
        )
        pipe.set_progress_bar_config(disable=True)  # type: ignore[no-untyped-call]  # Diffusers leaves progress configuration unannotated.
        return pipe

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        pipe = self.pipeline()
        with torch.inference_mode():
            embeds, mask, _, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False, max_sequence_length=1024,
                                                     device=torch.device(str(self.device)))
        return embeds, mask

    @uses_components("transformer", "connectors", "vae", "audio_vae")
    def denoise(self, embeds: tuple[torch.Tensor, torch.Tensor], pictures: Sequence[Frame], *, width: int, height: int,
                frames: int, sigmas: Sequence[float], generator: torch.Generator, check: Callable[[], None],
                progress: Callable[[int], None], reference: list[Image.Image] | None = None, downscale: int = 1,
                held: torch.Tensor | None = None, latents: tuple[torch.Tensor, torch.Tensor | None] | None = None,
                window: Window | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # The VAE roots are also leased: their latent-statistic buffers participate here, and they encode the pictures
        # and the reference at each stage's size, as upstream's stages do (ltx_pipelines/distilled.py `_video_conditionings`).
        pipe = self.pipeline()
        check()
        try:
            with torch.inference_mode():
                return pipe.run(embeds, pictures, reference=reference, downscale=downscale, held=held, width=width,
                                height=height, frames=frames, sigmas=sigmas, generator=generator, check=check,
                                progress=progress, latents=latents, window=window)
        finally:
            pipe.transformer._reset_stateful_cache()

    @uses_components("transformer", "connectors", "audio_vae")
    def sound(self, embeds: tuple[torch.Tensor, torch.Tensor], *, frames: int, generator: torch.Generator,
              check: Callable[[], None], progress: Callable[[int], None]) -> torch.Tensor:
        """Sound alone for a clip of `frames`, in one distilled stage (Lightricks' LTX-2.5 text-to-audio template):
        the audio VAE's latents [1, C, L, M]."""
        pipe = self.pipeline()
        check()
        try:
            with torch.inference_mode():
                return pipe.run(embeds, (), frames=frames, sigmas=SIGMAS, generator=generator, check=check,
                                progress=progress)[1]
        finally:
            pipe.transformer._reset_stateful_cache()

    @uses_components("vae")
    def encode_video(self, frames: Sequence[Image.Image]) -> torch.Tensor:
        """RGB frames (8k + 1, sides on 32) as the video VAE's latents [1, C, F, H, W], not normalized, as upstream
        encodes a retake's source (ltx_pipelines/utils/helpers.py `video_latent_from_file`)."""
        vae = self.graph.components["vae"]
        pixels = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames])).permute(3, 0, 1, 2)[None]
        with torch.inference_mode():
            pixels = pixels.to(device=torch.device(str(self.device)), dtype=vae.dtype) / 127.5 - 1
            return cast(torch.Tensor, vae.encode(pixels).latent_dist.mode()).float()

    @uses_components("audio_vae")
    def encode_audio(self, waveform: torch.Tensor, rate: int) -> torch.Tensor:
        """A waveform [channels, samples] as the audio VAE's latents [1, C, L, M], as upstream encodes a soundtrack
        (ltx_core/model/audio_vae `encode_audio`: log-mel at the VAE's rate, its mode)."""
        vae = self.graph.components["audio_vae"]
        config = vae.config
        with torch.inference_mode():
            spectrogram = mel.log_mel(waveform.to(torch.device(str(self.device))), rate, int(config["sample_rate"]),
                                      int(config["mel_bins"]), int(config["mel_hop_length"]), channels=int(config["in_channels"]))
            return cast(torch.Tensor, vae.encode(spectrogram.to(vae.dtype)).latent_dist.mode()).float()

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

