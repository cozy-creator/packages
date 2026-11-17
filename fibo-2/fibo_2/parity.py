"""se-157 proof only, never merged: Diffusers' `BriaFibo2Pipeline` (0.42.1) whole, as its docstring runs it
(`from_pretrained(dtype=torch.bfloat16)`), on the same weights, beside the package.

Unmodified modules (the whole Qwen3-VL with its head, the DiT, the FLUX.2 VAE in bf16), the checkpoint's
processor and fonts, and one `__call__`.
The noise is the package's (`rng.noise`, f32 on the CPU), passed as `latents`.
"""

from __future__ import annotations

import hashlib
import io
from typing import Annotated, cast

import msgspec
import torch
from cozy_runtime import rng, vlm
from cozy_runtime.author import Config, Context, Loader, Model, Telemetry, uses_components
from cozy_runtime.fields import FileBound
from cozy_runtime.image import ImageResult
from cozy_runtime.media import ImageFile
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_bria_fibo2 import BriaFibo2Transformer2DModel
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import BriaFibo2Pipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers import AutoModelForImageTextToText, Qwen3VLConfig
from transformers import initialization as transformer_init

from .model import FONTS, construct


class RefGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = construct(BriaFibo2Transformer2DModel, config.section("transformer").mapping())
            encoder = AutoModelForImageTextToText.from_config(  # type: ignore[no-untyped-call]
                Qwen3VLConfig(**config.section("text_encoder").mapping()), dtype=torch.bfloat16
            )
            vae = construct(AutoencoderKLFlux2, config.section("vae").mapping())
        encoder.lm_head = torch.nn.Identity()  # tied, not stored; the pipeline reads hidden states only
        self.components: dict[str, torch.nn.Module] = {
            "transformer": transformer.to(dtype=torch.bfloat16).eval(), "text_encoder": encoder.eval(),
            "vae": vae.to(dtype=torch.bfloat16).eval(),
        }
        self.scheduler_config = dict(config.section("scheduler").mapping())
        self.processor_config = dict(config.section("processor").mapping())


class RefFibo2(Model[RefGraph]):
    graph: RefGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(RefGraph)
        self.processor = vlm.processor(loader.assets, "processor/", self.graph.processor_config)
        self.fonts = tuple(loader.assets.read(name) for name in FONTS)
        self.styles = {name: loader.assets.read(name).decode() for name in loader.assets.names()
                       if name.endswith("negative_prompt.json")}

    @uses_components("transformer", "text_encoder", "vae")
    def sample(self, prompt: str, images: list, width: int, height: int, steps: int, guidance: float,
               negative: str | None, seed: int) -> object:
        device = torch.device(str(self.device))
        c = self.graph.components
        pipe = BriaFibo2Pipeline(
            transformer=c["transformer"], scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),  # type: ignore[arg-type]
            vae=c["vae"], text_encoder=c["text_encoder"], processor=self.processor,  # type: ignore[arg-type]
        )
        tf = cast(BriaFibo2Transformer2DModel, c["transformer"])
        latents = rng.noise((1, int(tf.config["in_channels"]), height // pipe.vae_scale_factor, width // pipe.vae_scale_factor), seed)
        with torch.inference_mode():
            out = pipe(prompt=prompt, image=images or None, height=height, width=width, num_inference_steps=steps,
                       guidance_scale=guidance, negative_prompt=negative, fonts=[io.BytesIO(font) for font in self.fonts], latents=latents.to(device),
                       output_type="pil")
        return out.images[0]


class RefInput(msgspec.Struct, kw_only=True):
    prompt: str
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    negative_prompt: str | None = None
    seed: int = 7


def reference(ctx: Context, payload: RefInput, model: RefFibo2, tel: Telemetry) -> ImageResult:
    """The pipeline's own call: its defaults (30 steps, guidance 5, its empty-caption negative) unless given; the
    size given (the package's output size, to compare like for like), else 1024²."""
    images = [payload.image.read()] if payload.image is not None else []
    steps = payload.steps or 30
    guidance = 5.0 if payload.guidance is None else payload.guidance
    width, height = payload.width or 1024, payload.height or 1024
    for name, text in model.styles.items():
        tel.log("style negative", name=name, sha=hashlib.sha256(text.encode()).hexdigest()[:16], text=text)
    with tel.stage("generating image"):
        image = model.sample(payload.prompt, images, width, height, steps, guidance, payload.negative_prompt, payload.seed)
    return ImageResult(ImageFile.from_image(image, format="png"), image.width, image.height, payload.seed, payload.prompt)  # type: ignore[attr-defined]
