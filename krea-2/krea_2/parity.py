"""se-157 proof only, never merged: Diffusers' `Krea2Pipeline` (0.42.1) as `from_pretrained(torch_dtype=bf16)`
builds it, on the same weights, beside the package.

Unmodified modules: `Krea2Transformer2DModel` all in bf16 (Krea's `inference.py`; Diffusers' bf16 load keeps
its norms F32), the whole
`Qwen3VLModel` (every layer, its final norm), the Qwen-Image VAE in bf16; the checkpoint's tokenizer; the pipeline's own `__call__` encodes (padded to 512 and masked), samples and
decodes. The noise is the package's (`rng.noise`, bf16 on the CPU), passed as `latents`.
"""

from __future__ import annotations

from typing import cast

import msgspec
import torch
from cozy_runtime import rng
from cozy_runtime.accel import hold_dtype
from cozy_runtime.author import Config, Context, Loader, Model, Telemetry, uses_components
from cozy_runtime.image import ImageResult
from cozy_runtime.media import ImageFile
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLConfig, Qwen3VLModel
from transformers import initialization as transformer_init

from .model import PATCH, as_stored, build_tokenizer, construct


class RefGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = construct(Krea2Transformer2DModel, config.section("transformer").mapping())
            encoder = Qwen3VLModel._from_config(Qwen3VLConfig(**config.section("text_encoder").mapping()), dtype=torch.bfloat16)  # type: ignore[no-untyped-call]
            vae = construct(AutoencoderKLQwenImage, config.section("vae").mapping())
        # Built as stored (the Runtime fills without a cast), then served in bf16 (`reference`): Krea's own
        # inference.py casts the DiT and the VAE to bf16 whole.
        as_stored(config, "transformer", transformer)
        as_stored(config, "vae", vae)
        self.components: dict[str, torch.nn.Module] = {
            "transformer": transformer.eval(), "text_encoder": encoder.eval(), "vae": vae.eval(),
        }
        self.scheduler_config = dict(config.section("scheduler").mapping())


class RefKrea2(Model[RefGraph]):
    graph: RefGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(RefGraph)
        self.tokenizer = build_tokenizer(loader.assets)  # the checkpoint's tokenizer (no from_pretrained in a package)

    @uses_components("transformer", "text_encoder", "vae")
    def sample(self, prompt: str, negative: str, width: int, height: int, steps: int, guidance: float, seed: int,
               distilled: bool) -> object:
        device = torch.device(str(self.device))
        components = self.graph.components
        pipe = Krea2Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config), vae=components["vae"],  # type: ignore[arg-type]
            text_encoder=components["text_encoder"], tokenizer=self.tokenizer, transformer=components["transformer"],  # type: ignore[arg-type]
            is_distilled=distilled,
        )
        rows, columns = height // PATCH, width // PATCH
        channels = int(cast(Krea2Transformer2DModel, components["transformer"]).config["in_channels"]) // 4
        noise = rng.noise((1, channels, rows * 2, columns * 2), seed, dtype=torch.bfloat16)
        noise = noise.view(1, channels, rows, 2, columns, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, rows * columns, channels * 4)
        with torch.inference_mode():
            out = pipe(prompt=prompt, negative_prompt=negative, height=height, width=width, num_inference_steps=steps,
                       guidance_scale=guidance, latents=noise.to(device), output_type="pil")
        return out.images[0]


class RefInput(msgspec.Struct, kw_only=True):
    prompt: str
    negative_prompt: str = ""
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    turbo: bool = True
    seed: int = 7


def reference(ctx: Context, payload: RefInput, model: RefKrea2, tel: Telemetry) -> ImageResult:
    """Krea's README settings: Raw 52 steps at cfg 3.5 at 1024², Turbo 8 steps at cfg 0, mu 1.15, 2048²."""
    side = 2048 if payload.turbo else 1024
    width, height = payload.width or side, payload.height or side
    steps = payload.steps or (8 if payload.turbo else 52)
    guidance = payload.guidance if payload.guidance is not None else (0.0 if payload.turbo else 3.5)
    for part in ("transformer", "vae"):
        hold_dtype(model.graph.components[part], torch.bfloat16)
    with tel.stage("generating image"):
        image = model.sample(payload.prompt, payload.negative_prompt, width, height, steps, guidance, payload.seed,
                             payload.turbo)
    tel.log("reference", steps=steps, guidance=guidance, width=width, height=height, turbo=payload.turbo)
    return ImageResult(ImageFile.from_image(image, format="png"), image.width, image.height, payload.seed, payload.prompt)  # type: ignore[attr-defined]
