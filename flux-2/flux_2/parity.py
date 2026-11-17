"""se-157 proof only, never merged: Diffusers' `Flux2Pipeline` (0.42.1) as fal's Turbo card and BFL run it, on
the same [dev] weights, beside the package.

The full Mistral-Small-3.2 (every layer, its final norm, its vision tower) encodes through the pipeline's own
`_get_mistral_3_small_prompt_embeds` with the processor `AutoProcessor` loads from the checkpoint's files; the
pipeline's `__call__` samples (its scheduler, its 1 MP reference cap, its guidance embedding) and decodes with
the VAE in bf16, the card's `torch_dtype` (BFL runs it in float32). fal's Turbo LoRA comes from its Hugging Face file and is added unfused, as
PEFT does after `load_lora_weights` (scale alpha/r = 1), by forward hooks, so the shared weights are untouched.
"""

from __future__ import annotations

import os
import tempfile
from typing import Annotated, cast

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
from transformers import AutoProcessor
from transformers import initialization as transformer_init

from .model import MAX_TOKENS, PATCH, TEXT_LAYERS, TURBO_SIGMAS, construct, text_encoder

TURBO_REPO, TURBO_REV, TURBO_FILE = "fal/FLUX.2-dev-Turbo", "9ee51cd87578162cf8d02355a870bc5f4570045c", "flux.2-turbo-lora.safetensors"


class RefGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = construct(Flux2Transformer2DModel, config.section("transformer").mapping())
            encoder = text_encoder(config.section("text_encoder").mapping())
            vae = construct(AutoencoderKLFlux2, config.section("vae").mapping())
        self.components: dict[str, torch.nn.Module] = {
            "transformer": transformer.to(dtype=torch.bfloat16), "text_encoder": encoder.eval(),
            "vae": vae.to(dtype=torch.bfloat16),  # the card's torch_dtype; the pipeline decodes in the VAE's dtype
        }
        self.scheduler_config = dict(config.section("scheduler").mapping())


class RefFlux2(Model[RefGraph]):
    graph: RefGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(RefGraph)
        folder = tempfile.mkdtemp(prefix="flux2-tokenizer-")
        for name in loader.assets.names():
            if name.startswith("tokenizer/"):
                path = os.path.join(folder, name.removeprefix("tokenizer/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as file:
                    file.write(loader.assets.read(name))
        self.processor = AutoProcessor.from_pretrained(folder)

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
    def sample(self, embeds: torch.Tensor, images: list, width: int, height: int, steps: int, guidance: float,
               seed: int, turbo: bool) -> object:
        tf = cast(Flux2Transformer2DModel, self.graph.components["transformer"])
        device = torch.device(str(self.device))
        pipe = Flux2Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            vae=self.graph.components["vae"], text_encoder=None, tokenizer=None, transformer=tf,  # type: ignore[arg-type]
        )
        noise = rng.noise((1, int(tf.config["in_channels"]), height // PATCH, width // PATCH), seed).to(torch.bfloat16)
        handles = _turbo_hooks(tf, device) if turbo else []
        try:
            out = pipe(
                image=images or None, prompt_embeds=embeds.to(device), height=height, width=width,
                num_inference_steps=steps, sigmas=list(TURBO_SIGMAS) if turbo else None, guidance_scale=guidance,
                latents=noise.to(device), output_type="pil",
            )
        finally:
            for handle in handles:
                handle.remove()
        return out.images[0]


def _turbo_hooks(tf: torch.nn.Module, device: torch.device) -> list:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights = load_file(hf_hub_download(TURBO_REPO, TURBO_FILE, revision=TURBO_REV))
    handles = []
    for key in weights:
        if not key.endswith(".lora_A.weight"):
            continue
        path = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
        down = weights[key].to(device=device, dtype=torch.bfloat16)
        up = weights[key.replace("lora_A", "lora_B")].to(device=device, dtype=torch.bfloat16)
        module = tf.get_submodule(path)

        def hook(_module, inputs, output, down=down, up=up):
            return output + (inputs[0].to(down.dtype) @ down.T) @ up.T

        handles.append(module.register_forward_hook(hook))
    return handles


class RefInput(msgspec.Struct, kw_only=True):
    prompt: str
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    turbo: bool = True
    seed: int = 7


def reference(ctx: Context, payload: RefInput, model: RefFlux2, tel: Telemetry) -> ImageResult:
    """The card's call: [dev] Turbo 8 sigmas at guidance 2.5, or [dev] 50 steps at 4."""
    images = [payload.image.read().convert("RGB")] if payload.image is not None else []
    width = payload.width or (images[0].width if images else 1024)
    height = payload.height or (images[0].height if images else 1024)
    steps = payload.steps or (8 if payload.turbo else 50)
    guidance = payload.guidance or (2.5 if payload.turbo else 4.0)
    with tel.stage("encoding prompt"):
        embeds = model.encode(payload.prompt)
    with tel.stage("generating image"):
        image = model.sample(embeds, images, width, height, steps, guidance, payload.seed, payload.turbo)
    tel.log("reference", steps=steps, guidance=guidance, width=width, height=height, turbo=payload.turbo)
    return ImageResult(ImageFile.from_image(image, format="png"), image.width, image.height, payload.seed, payload.prompt)  # type: ignore[attr-defined]
