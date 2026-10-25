"""Bria Fibo 2.0, Dev and Edit Turbo, component-staged over Diffusers' own `BriaFibo2Pipeline`.

The architecture, caption handling and sampling are Diffusers' (main `48cdbaef`, #14991, verified by
Bria against its reference code). Runtime constructs and fills the modules and owns their residency:
the pipeline runs once per component, its text encoder's and VAE's results replayed into the
transformer's run.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from importlib.resources import files
from typing import Any

import torch
from cozy_runtime import rng, vlm
from cozy_runtime.author import Config, Loader, MemoDistribution, Model, uses_components
from diffusers import (
    AutoencoderKLFlux2,
    BriaFibo2Pipeline,
    BriaFibo2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    Qwen3VLConfig,
    Qwen3VLProcessor,
)
from transformers import initialization as transformer_init

#: The fonts Bria's training drew the caption's text images with, after Pillow's built-in font
#: (`fonts/` of `briaai/Fibo-2.0-dev`).
FONTS = tuple(str(files(__package__).joinpath("fonts", name)) for name in ("Aileron-Regular.otf", "DejaVuSans.ttf"))


class _Progress[T]:
    """Pipeline iteration without tqdm; Runtime reports progress through Telemetry."""

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


class _Encoded(Exception):
    """The recording run's stop: every prompt is encoded, and the latents are the transformer's."""


def _key(prompt: str | list[str], image: object, text_images: object) -> str:
    texts = [len(own) for own in text_images or []] if isinstance(text_images, list) else []
    return json.dumps([prompt, image is not None, texts], ensure_ascii=False)


class _Staged(BriaFibo2Pipeline):
    """`BriaFibo2Pipeline` run per component. Recording, it encodes each prompt the call encodes and
    stops before the latents; replaying, it takes those encodings and the encoded images to edit."""

    stage_device: torch.device
    recording: bool = False
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor | None]]
    image_latents: list[torch.Tensor] | None = None

    def progress_bar[T](self, iterable: Iterable[T] | None = None, total: int | None = None) -> _Progress[T]:
        return _Progress(iterable, total)

    @property
    def _execution_device(self) -> torch.device:
        return self.stage_device

    def encode_prompt(  # type: ignore[override]
        self,
        prompt: str | list[str],
        image: list[Image.Image] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        text_images: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        key = _key(prompt, image, text_images)
        if self.recording:
            embeds, mask = super().encode_prompt(prompt, image, device, 1, None, None, text_images)
            self.encoded[key] = (embeds.cpu(), None if mask is None else mask.cpu())
        embeds, mask = self.encoded[key]
        device = device or self.stage_device
        return embeds.to(device), None if mask is None else mask.to(device)

    def prepare_latents(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        if self.recording:
            raise _Encoded
        return super().prepare_latents(*args, **kwargs)  # type: ignore[no-any-return]

    def prepare_image_latents(  # type: ignore[override]
        self, image: Any, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> list[torch.Tensor]:
        if self.image_latents is None:
            return super().prepare_image_latents(image, batch_size, device, dtype)  # type: ignore[no-any-return]
        return [latents.to(device=device, dtype=dtype) for latents in self.image_latents]


class FiboGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            text_encoder = AutoModelForImageTextToText.from_config(
                Qwen3VLConfig(**config.section("text_encoder").mapping()), dtype=torch.bfloat16
            )
            self.components: dict[str, Any] = {
                "transformer": BriaFibo2Transformer2DModel.from_config(config.section("transformer").mapping()).to(
                    dtype=torch.bfloat16
                ),
                "text_encoder": text_encoder,
                "vae": AutoencoderKLFlux2.from_config(config.section("vae").mapping()).to(dtype=torch.float32),
            }
        # The pipeline reads hidden states only: the language-model head is never built.
        text_encoder.lm_head = torch.nn.Identity()
        self.scheduler_config = config.section("scheduler").mapping()
        # The processor folder's config file, which TensorFS keeps as its component's config.
        self.processor_config = config.section("processor").mapping() if "processor" in config.keys() else {}


#: Code the stages read beyond their own bodies: their memo keys bind it.
_CODE = (_Staged, vlm.processor, MemoDistribution("diffusers"), MemoDistribution("transformers"))


class Fibo2(Model[FiboGraph], encoded_leaves="accept"):
    """One Fibo 2.0 checkpoint: Dev (CFG) or Edit Turbo (4 steps, no guidance)."""

    graph: FiboGraph
    processor: Qwen3VLProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(FiboGraph, factory=FiboGraph)
        self.processor = vlm.processor(loader.assets, "processor/", self.graph.processor_config)

    def _pipeline(self, component: str) -> _Staged:
        pipe = _Staged(
            **self.graph.components,
            processor=self.processor,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
        )
        pipe.stage_device = torch.device(str(self.device))
        pipe.encoded = {}
        return pipe

    @uses_components("text_encoder", memoize=True, memo_dependencies=_CODE)
    def encode(
        self, prompt: str, images: list[Image.Image], *, width: int, height: int, guidance: float, negative: str | None
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
        """Every encoding the pipeline's call makes: the prompt (with the images to edit and the
        caption's text images), the prompt without its text images, and the negative prompt."""
        pipe = self._pipeline("text_encoder")
        pipe.recording = True
        try:
            with torch.inference_mode():
                pipe(prompt=prompt, image=images or None, width=width, height=height, guidance_scale=guidance,
                     negative_prompt=negative, fonts=list(FONTS), output_type="latent")
        except _Encoded:
            return pipe.encoded
        raise RuntimeError("the pipeline never reached its latents")

    @uses_components("vae", memoize=True, memo_dependencies=_CODE)
    def encode_images(self, images: list[Image.Image], *, width: int, height: int) -> list[torch.Tensor]:
        """The images to edit as the pipeline crops and normalizes them, as latents."""
        pipe = self._pipeline("vae")
        with torch.inference_mode():
            crops, _ = pipe.image_processor.preprocess(
                images, height=height, width=width, area=(pipe.default_sample_size * pipe.vae_scale_factor) ** 2,
                text_encoder_image_size=pipe.text_encoder_image_size,
            )
            latents = pipe.prepare_image_latents(crops, 1, pipe.stage_device, torch.bfloat16)
        return [tensor.cpu() for tensor in latents]

    @uses_components("transformer")
    def denoise(
        self,
        prompt: str,
        images: list[Image.Image],
        encoded: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
        image_latents: list[torch.Tensor],
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        negative: str | None,
        seed: int,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
    ) -> torch.Tensor:
        """Diffusers' sampler: Euler from sigma 1 to 1/steps, shifted by the image size, the noise
        drawn on the CPU from `seed` and kept in float32 across steps."""
        pipe = self._pipeline("transformer")
        pipe.encoded = encoded
        pipe.image_latents = image_latents

        def observed(_pipe: Any, step: int, _timestep: Any, values: Any) -> Any:
            cancel()
            on_step(step)
            return values

        cancel()
        with torch.inference_mode():
            latents = pipe(
                prompt=prompt, image=images or None, width=width, height=height, num_inference_steps=steps,
                guidance_scale=guidance, negative_prompt=negative, fonts=list(FONTS),
                generator=rng.generator(seed), output_type="latent", callback_on_step_end=observed,
            ).images
        return latents.cpu()  # type: ignore[no-any-return]

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> Image.Image:
        """The pipeline's decode: the latents' BatchNorm normalization undone, then the decoder."""
        pipe = self._pipeline("vae")
        vae = pipe.vae
        with torch.inference_mode():
            latents = pipe._patchify_latents(latents.to(pipe.stage_device, torch.float32))
            mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
            scale = (1 / torch.sqrt(vae.bn.running_var + vae.config.batch_norm_eps)).view(1, -1, 1, 1)
            latents = pipe._unpatchify_latents(latents / scale.to(latents) + mean)
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
            if not torch.isfinite(image).all():
                raise ValueError("Fibo 2.0's decoder produced non-finite pixels")
            (picture,) = pipe.image_processor.postprocess(image, output_type="pil")
        return picture  # type: ignore[no-any-return]
