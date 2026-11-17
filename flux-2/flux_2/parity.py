"""se-157 proof only, never merged: Diffusers' `Flux2Pipeline` (0.42.1) as fal's Turbo card runs it, on the same
[dev] weights, beside the package.

The full Mistral-Small-3.2 (every layer, its final norm, its vision tower) encodes through the pipeline's own
`_get_mistral_3_small_prompt_embeds` (the processor parsed from the checkpoint's files, as the package does: a
package may not call `from_pretrained`); the pipeline's `__call__` samples (its scheduler, its 1 MP reference cap,
its guidance embedding) and decodes with the VAE in bf16, the card's `torch_dtype` (BFL runs it in float32). fal's
Turbo LoRA is added unfused, as PEFT does after `load_lora_weights` (scale alpha/r = 1), by forward hooks over
fal's stored factors, so the weights stay untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, TypeVar, cast

import msgspec
import torch
from cozy_runtime import rng
from cozy_runtime.author import Config, Context, Loader, Model, Telemetry, uses_components
from cozy_runtime.fields import FileBound
from cozy_runtime.image import ImageResult
from cozy_runtime.media import ImageFile
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import initialization as transformer_init

from .model import MAX_TOKENS, PATCH, TEXT_LAYERS, TURBO_SIGMAS, construct, processor, text_encoder
from .turbo import COMPONENT, Flux2TurboLoRA

T = TypeVar("T")


class RefGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = construct(Flux2Transformer2DModel, config.section("transformer").mapping())
            encoder = text_encoder(config.section("text_encoder").mapping())
            vae = construct(AutoencoderKLFlux2, config.section("vae").mapping())
        encoder.set_submodule("lm_head", torch.nn.Identity())  # the pipeline reads hidden states only
        self.components: dict[str, torch.nn.Module] = {
            "transformer": transformer.to(dtype=torch.bfloat16), "text_encoder": encoder.eval(),
            "vae": vae.to(dtype=torch.bfloat16),
        }
        self.scheduler_config = dict(config.section("scheduler").mapping())


class RefFlux2(Model[RefGraph]):
    graph: RefGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(RefGraph)
        self.processor = processor(loader.assets)

    @uses_components("text_encoder")
    def encode(self, prompt: str) -> torch.Tensor:
        with torch.inference_mode():
            embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                self.graph.components["text_encoder"], self.processor, prompt, dtype=torch.bfloat16,  # type: ignore[arg-type]
                device=torch.device(str(self.device)), max_sequence_length=MAX_TOKENS, system_message=SYSTEM_MESSAGE,
                hidden_states_layers=TEXT_LAYERS["mistral3"],  # type: ignore[arg-type]
            )
        return embeds.cpu()

    @uses_components("transformer", "vae")
    def sample(self, embeds: torch.Tensor, images: list[Image.Image], width: int, height: int, steps: int,
               guidance: float, seed: int, turbo: bool) -> Image.Image:
        tf = cast(Flux2Transformer2DModel, self.graph.components["transformer"])
        device = torch.device(str(self.device))
        pipe = Flux2Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            vae=self.graph.components["vae"], text_encoder=None, tokenizer=None, transformer=tf,  # type: ignore[arg-type]
        )
        noise = rng.noise((1, int(tf.config["in_channels"]), height // PATCH, width // PATCH), seed).to(torch.bfloat16)
        with torch.inference_mode():
            out = pipe(
                image=images or None, prompt_embeds=embeds.to(device), height=height, width=width,
                num_inference_steps=steps, sigmas=list(TURBO_SIGMAS) if turbo else None, guidance_scale=guidance,
                latents=noise.to(device), output_type="pil",
            )
        return cast(Image.Image, out.images[0])


class RefTurbo(Flux2TurboLoRA):
    """fal's factors, added to each target linear's output by a forward hook: PEFT's unfused LoRA."""

    @uses_components(COMPONENT)
    def hooked(self, model: RefFlux2, body: Callable[[], T]) -> T:
        tf = model.graph.components["transformer"]
        handles = []
        for path, pair in self.graph.factors.items():
            down, up, scale = pair.factors()

            def hook(_m: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor,
                     down: torch.Tensor = down, up: torch.Tensor = up, scale: float = scale) -> torch.Tensor:
                x = inputs[0].to(device=down.device, dtype=down.dtype)
                return output + scale * ((x @ down.T) @ up.T).to(output.device, output.dtype)

            handles.append(tf.get_submodule(path).register_forward_hook(hook))
        try:
            return body()
        finally:
            for handle in handles:
                handle.remove()


class RefInput(msgspec.Struct, kw_only=True):
    prompt: str
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    turbo: bool = True
    seed: int = 7


def reference(ctx: Context, payload: RefInput, model: RefFlux2, turbo_lora: RefTurbo, tel: Telemetry) -> ImageResult:
    """The card's call: [dev] Turbo, 8 sigmas at guidance 2.5, or [dev] 50 steps at 4."""
    images = [payload.image.read().convert("RGB")] if payload.image is not None else []
    width = payload.width or (images[0].width if images else 1024)
    height = payload.height or (images[0].height if images else 1024)
    steps = payload.steps or (8 if payload.turbo else 50)
    guidance = payload.guidance or (2.5 if payload.turbo else 4.0)
    with tel.stage("encoding prompt"):
        embeds = model.encode(payload.prompt)

    def run() -> Image.Image:
        return model.sample(embeds, images, width, height, steps, guidance, payload.seed, payload.turbo)

    with tel.stage("generating image"):
        image = turbo_lora.hooked(model, run) if payload.turbo else run()
    tel.log("reference", steps=steps, guidance=guidance, width=width, height=height, turbo=payload.turbo,
            text_tokens=int(embeds.shape[1]))
    return ImageResult(ImageFile.from_image(image, format="png"), image.width, image.height, payload.seed, payload.prompt)
