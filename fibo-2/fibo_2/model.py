"""Bria Fibo 2.0, Dev and Edit Turbo, component-staged over Diffusers' own `BriaFibo2Pipeline`.

The architecture, caption handling and sampling are Diffusers' (main `48cdbaef`, #14991, verified by
Bria against its reference code). Runtime constructs and fills the modules and owns their residency:
the pipeline's own run encodes the prompts, and `Fibo2.denoise` is its denoising loop. A caller's LoRAs
adapt the DiT through Runtime's PEFT layers (`serving.loras`).
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import TypedDict, cast

import numpy as np
import torch
from cozy_runtime import rng, vlm
from cozy_runtime.accel import hold_dtype
from cozy_runtime.author import (
    Config,
    Loader,
    Model,
    OutputError,
    Serving,
    Telemetry,
    memoize,
    preview,
    uses_components,
)
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_bria_fibo2 import BriaFibo2Transformer2DModel
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import BriaFibo2Pipeline, calculate_shift, retrieve_timesteps
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from transformers import initialization as transformer_init

#: The fonts Bria's training drew the caption's text images with, after Pillow's built-in font: the checkpoint's
#: copy of `fonts/` of `briaai/Fibo-2.0-dev`, held by its root component.
FONTS = ("model/fonts/Aileron-Regular.otf", "model/fonts/DejaVuSans.ttf")
#: The VAE's dtype on a GPU, as ComfyUI runs FLUX.2's VAE: its activations fit fp16 (dtype audit M1).
VAE_DTYPE = torch.float16


class _Encoded(Exception):
    """The recording run's stop: every prompt is encoded, and the latents are the transformer's."""


#: An encoding: the embeddings and their mask (None when nothing is padded).
Encoding = tuple[torch.Tensor, torch.Tensor | None]


class _Staged(BriaFibo2Pipeline):
    """`BriaFibo2Pipeline` run per component. The text encoder's run records each prompt the call
    encodes, in order, and stops before the latents."""

    stage_device: torch.device
    encoded: list[Encoding]
    transformer: BriaFibo2Transformer2DModel
    scheduler: FlowMatchEulerDiscreteScheduler
    vae: AutoencoderKLFlux2

    @property
    def _execution_device(self) -> torch.device:
        return self.stage_device

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: list[Image.Image] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        text_images: list[Image.Image] | list[list[Image.Image]] | None = None,
    ) -> Encoding:
        embeds, mask = super().encode_prompt(prompt, image, device, 1, None, None, text_images)
        self.encoded.append((embeds.cpu(), None if mask is None else mask.cpu()))
        return embeds, mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise _Encoded


class FiboGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            text_encoder: Qwen3VLForConditionalGeneration = AutoModelForImageTextToText.from_config(  # type: ignore[no-untyped-call]  # Transformers' auto classes are unannotated
                Qwen3VLConfig(**config.section("text_encoder").mapping()), dtype=torch.bfloat16
            )
            self.components: Components = {
                "transformer": _to(construct(BriaFibo2Transformer2DModel, config.section("transformer").mapping()), torch.bfloat16),
                "text_encoder": text_encoder,
                "vae": _to(construct(AutoencoderKLFlux2, config.section("vae").mapping()), torch.float32),
            }
        # The pipeline reads hidden states only: the language-model head is never built.
        text_encoder.lm_head = torch.nn.Identity()
        self.scheduler_config: dict[str, object] = config.section("scheduler").mapping()
        # The processor folder's config file, which TensorFS keeps as its component's config.
        self.processor_config: dict[str, object] = config.section("processor").mapping()


class Components(TypedDict):
    transformer: BriaFibo2Transformer2DModel
    text_encoder: Qwen3VLForConditionalGeneration
    vae: AutoencoderKLFlux2


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def _to[M: torch.nn.Module](module: M, dtype: torch.dtype) -> M:
    """`module.to(dtype=dtype)`, typed: Diffusers' `to` is `functools.wraps`'d, which the checkers bind without `self`."""
    return module.to(dtype=dtype)


class Fibo2(Model[FiboGraph]):
    """One Fibo 2.0 checkpoint: Dev (CFG) or Edit Turbo (4 steps, no guidance)."""

    #: Caller LoRAs on the DiT, at any strength: the Runtime bakes them into the weights.
    serving = Serving(loras=("transformer",))

    graph: FiboGraph
    processor: Qwen3VLProcessor
    fonts: tuple[bytes, ...]

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(FiboGraph)
        self.processor = cast(Qwen3VLProcessor, vlm.processor(loader.assets, "processor/", self.graph.processor_config))  # Qwen3-VL's folder
        self.fonts = tuple(loader.assets.read(name) for name in FONTS)

    def _pipeline(self, component: str) -> _Staged:
        pipe = _Staged(
            **self.graph.components,
            processor=self.processor,
            scheduler=construct(FlowMatchEulerDiscreteScheduler, self.graph.scheduler_config),
        )
        pipe.stage_device = torch.device(str(self.device))
        pipe.encoded = []
        return pipe

    def hold_vae(self, tel: Telemetry) -> None:
        """Serve the VAE at `VAE_DTYPE` on a GPU from now on, before any VAE scope opens: the Runtime
        casts the stored float32 bytes when it next fills them. A CPU keeps float32."""
        vae = self.graph.components["vae"]
        if torch.device(str(self.device)).type == "cuda" and next(vae.parameters()).dtype != VAE_DTYPE:
            hold_dtype(vae, VAE_DTYPE)
            tel.log("vae representation", dtype=str(VAE_DTYPE).removeprefix("torch."))

    @memoize
    @uses_components("text_encoder")
    def encode(
        self, prompt: str, images: list[Image.Image], *, width: int, height: int, guidance: float, negative: str | None
    ) -> dict[str, Encoding]:
        """Every encoding the pipeline's call makes, by its part in the sampler: `prompt` (with the
        images to edit and the caption's text images), `base` (a generation's caption without its
        text images) and `negative` (with CFG). `__call__` encodes them in that order."""
        pipe = self._pipeline("text_encoder")
        try:
            with torch.inference_mode():
                pipe(prompt=prompt, image=images or None, width=width, height=height, guidance_scale=guidance,
                     negative_prompt=negative, fonts=[io.BytesIO(font) for font in self.fonts],  # type: ignore[arg-type]  # Pillow's truetype reads a file object as well as a path
                     output_type="latent")
        except _Encoded:
            first, *rest = pipe.encoded
            roles = {"prompt": first}
            if guidance > 1.0:
                roles["negative"] = rest.pop()
            if rest:
                roles["base"] = rest.pop()
            return roles
        raise RuntimeError("the pipeline never reached its latents")

    @memoize
    @uses_components("vae")
    def encode_images(self, images: list[Image.Image], *, width: int, height: int) -> list[torch.Tensor]:
        """The images to edit as the pipeline crops and normalizes them, as latents."""
        pipe = self._pipeline("vae")
        with torch.inference_mode():
            crops, _ = pipe.image_processor.preprocess(
                images, height=height, width=width, area=(pipe.default_sample_size * pipe.vae_scale_factor) ** 2,
                text_encoder_image_size=pipe.text_encoder_image_size,
            )
            latents = pipe.prepare_image_latents(crops, 1, pipe.stage_device, torch.bfloat16)  # type: ignore[no-untyped-call]  # Diffusers' own, unannotated
        return [tensor.cpu() for tensor in latents]

    @uses_components("transformer")
    def denoise(
        self,
        encoded: dict[str, Encoding],
        image_latents: list[torch.Tensor],
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        progress: Callable[[int, int], None],
    ) -> torch.Tensor:
        """`BriaFibo2Pipeline.__call__`'s loop: `FlowMatchEulerDiscreteScheduler` from sigma 1 to 1/steps,
        shifted by the image size, the noise drawn on the CPU from `seed` and kept in float32 across
        steps, CFG as a second pass. A generation with text images reads `base` while the noise level
        is above `text_images_max_noise_level`."""
        pipe = self._pipeline("transformer")
        transformer, device = pipe.transformer, pipe.stage_device
        scheduler = pipe.scheduler
        cfg = guidance > 1.0
        batch = encoded["prompt"][0].shape[0]
        with torch.no_grad():
            shape = (batch, transformer.config["in_channels"], height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
            latents = rng.noise(shape, seed, device=device)
            patch = transformer.config["patch_size"]
            config = scheduler.config
            mu = calculate_shift(
                (latents.shape[2] // patch) * (latents.shape[3] // patch),
                config["base_image_seq_len"], config["max_image_seq_len"], config["base_shift"], config["max_shift"],
            )
            # Diffusers types sigmas as a list; the scheduler takes the array as it is.
            retrieve_timesteps(scheduler, steps, device, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)  # type: ignore[arg-type]
            timesteps = scheduler.timesteps
            texts = {
                role: (embeds.to(device), None if mask is None else mask.to(device))
                for role, (embeds, mask) in encoded.items()
            }
            images = [tensor.to(device=device, dtype=transformer.dtype) for tensor in image_latents] or None
            for i, t in enumerate(timesteps.tolist()):
                timestep = timesteps[i].expand(latents.shape[0]) / 1000
                hidden = latents.to(transformer.dtype)
                text_images = "base" not in texts or t / 1000 <= pipe.text_images_max_noise_level
                roles = ["prompt" if text_images else "base", *(["negative"] if cfg else [])]
                predictions = [
                    transformer(
                        hidden_states=hidden, timestep=timestep, encoder_hidden_states=texts[role][0],
                        encoder_attention_mask=texts[role][1], context_latents=images, return_dict=False,
                    )[0].float()
                    for role in roles
                ]
                noise_pred = predictions[0]
                if cfg:
                    noise_pred = predictions[1] + guidance * (noise_pred - predictions[1])
                preview(lambda: latents - scheduler.sigmas[i] * noise_pred, "flux2")
                latents = scheduler.step(noise_pred, timesteps[i], latents, return_dict=False)[0]  # type: ignore[arg-type]  # Diffusers annotates the legacy FloatTensor
                progress(i, steps)
        return latents.cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> Image.Image:
        """The pipeline's decode: the latents' BatchNorm normalization undone, then the decoder."""
        pipe = self._pipeline("vae")
        vae = pipe.vae
        with torch.inference_mode():
            latents = pipe._patchify_latents(latents.to(pipe.stage_device, torch.float32))  # type: ignore[no-untyped-call]  # Diffusers' own, unannotated
            mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)  # type: ignore[union-attr]  # tracked: it keeps its statistics
            scale = (1 / torch.sqrt(vae.bn.running_var + vae.config["batch_norm_eps"])).view(1, -1, 1, 1)
            latents = pipe._unpatchify_latents(latents / scale.to(latents) + mean)  # type: ignore[no-untyped-call]  # Diffusers' own, unannotated
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
            if not torch.isfinite(image).all():
                raise OutputError(
                    "Fibo 2.0's decoder produced non-finite pixels, so there is no picture: try another seed, "
                    "and if it repeats another lane with `--lane bf16`",
                    code="output_integrity",
                )
            (picture,) = pipe.image_processor.postprocess(image.float(), output_type="pil")  # type: ignore[misc]  # a list, for "pil"
        return picture  # type: ignore[no-any-return]
