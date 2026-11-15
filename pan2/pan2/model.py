"""Official PAN2 components and modular stages; Runtime owns their weights and residency."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from cozy_runtime import rng
from cozy_runtime.author import Config, InvalidRequest, Loader, Model, ModelAssets, Serving, uses_components
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, PAN2Transformer3DModel
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines.modular_pipeline import BlockState, ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.pan2.decoders import PAN2VaeDecoderStep
from diffusers.modular_pipelines.pan2.encoders import PAN2TextEncoderStep, PAN2VaeEncoderStep, get_pan2_default_negative_prompt
from diffusers.modular_pipelines.pan2.modular_blocks_pan2 import PAN2CoreDenoiseStep, PAN2Image2VideoCoreDenoiseStep
from diffusers.modular_pipelines.pan2.modular_pipeline import PAN2ModularPipeline
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from transformers import AddedToken, Qwen2Tokenizer, Qwen3_5TextConfig, Qwen3_5TextModel

FPS = 24
MAX_FRAMES = 481
DEFAULT_STEPS = 50
COMPONENTS = ("transformer", "text_encoder", "vae")


def stored(module: nn.Module, component: str, config: Config) -> nn.Module:
    """Construct each destination at its stored logical dtype, including PAN2's F32 modulation."""
    dtypes = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
    for name, parameter in module.named_parameters():
        dtype = dtypes.get(config.tensor_dtype(f"{component}.{name}", default="bf16"))
        if dtype is not None:
            parameter.data = parameter.data.to(dtype)
    return module.eval()


def tokenizer(assets: ModelAssets, folder: str = "tokenizer/") -> Qwen2Tokenizer:
    """Build from embedded bytes: no Hugging Face cache or serving-time filesystem view."""
    settings = json.loads(assets.read(folder + "tokenizer_config.json"))
    added = settings.pop("added_tokens_decoder", {})
    for key in ("tokenizer_class", "auto_map", "name_or_path", "tokenizer_file", "vocab_file", "merges_file"):
        settings.pop(key, None)
    made = Qwen2Tokenizer(
        tokenizer_object=Tokenizer.from_str(assets.read(folder + "tokenizer.json").decode()),
        added_tokens_decoder={int(index): AddedToken(**value) for index, value in added.items()},
        **settings,
    )
    if folder + "chat_template.jinja" in assets.names():
        made.chat_template = assets.read(folder + "chat_template.jinja").decode()
    return made


class Graph:
    def __init__(self, config: Config) -> None:
        self.components = {
            "transformer": stored(PAN2Transformer3DModel.from_config(config.section("transformer").mapping()), "transformer", config),
            "text_encoder": stored(Qwen3_5TextModel(Qwen3_5TextConfig(**config.section("text_encoder").mapping())), "text_encoder", config),
            "vae": stored(AutoencoderKLWan.from_config(config.section("vae").mapping()), "vae", config),
        }
        self.scheduler_config = config.section("scheduler").mapping()


def stage(block: ModularPipelineBlocks, components: Mapping[str, object], values: dict[str, object]) -> PipelineState:
    """A request-local stage with just its active weighted roots; no tensor deepcopy between stages."""
    pipe = block.init_pipeline()
    pipe.update_components(**components, enable_safety_checker=False)
    state = PipelineState(values=values)
    pipe.blocks(pipe, state)
    return state


class Pan2(Model[Graph]):
    serving = Serving(min_bits=16, attention_bits=16)
    graph: Graph
    tokenizer: Qwen2Tokenizer

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)
        self.tokenizer = tokenizer(loader.assets)

    @uses_components("text_encoder")
    def encode(self, prompt: str, frames: int, guidance: float, negative: str | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The assistant suffix also consumes the encoder's 1536-token capacity. The released
        # prompt-writing policy reserves this margin rather than truncating timed events.
        tokens = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        if tokens > 1480:
            raise InvalidRequest(f"PAN2's caption has {tokens} tokens; shorten it to at most 1480", fields=["prompt"])
        if guidance > 1 and not negative:
            negative = get_pan2_default_negative_prompt(frames, FPS)
        state = stage(PAN2TextEncoderStep(), {"text_encoder": self.graph.components["text_encoder"], "tokenizer": self.tokenizer},
                      {"prompt": prompt, "negative_prompt": negative if guidance > 1 else None,
                       "num_frames": frames, "fps": float(FPS)})
        return cast(torch.Tensor, state.values["prompt_embeds"]), cast(torch.Tensor | None, state.values.get("negative_prompt_embeds"))

    @uses_components("vae")
    def encode_image(self, image: Image.Image, width: int, height: int) -> torch.Tensor:
        vae = cast(AutoencoderKLWan, self.graph.components["vae"])
        try:
            state = stage(PAN2VaeEncoderStep(), {"vae": vae}, {"image": image, "width": width, "height": height})
            return cast(torch.Tensor, state.values["image_latents"])
        finally:
            vae.clear_cache()

    @uses_components("transformer")
    def denoise(self, embeds: tuple[torch.Tensor, torch.Tensor | None], image: torch.Tensor | None, *,
                width: int, height: int, frames: int, steps: int, guidance: float, seed: int,
                check: Callable[[], None], progress: Callable[[int], None]) -> torch.Tensor:
        block = PAN2CoreDenoiseStep() if image is None else PAN2Image2VideoCoreDenoiseStep()
        pipe = block.init_pipeline()
        pipe.update_components(
            transformer=self.graph.components["transformer"],
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            guider=ClassifierFreeGuidance(guidance_scale=guidance),
        )
        loop = pipe.blocks.sub_blocks["denoise"]
        original = loop.loop_step

        def step(components: PAN2ModularPipeline, state: BlockState, *, i: int, t: torch.Tensor) -> tuple[PAN2ModularPipeline, BlockState]:
            check()
            result = original(components, state, i=i, t=t)
            progress(i)
            return result

        loop.loop_step = step
        values: dict[str, object] = {
            "prompt_embeds": embeds[0], "negative_prompt_embeds": embeds[1], "width": width, "height": height,
            "num_frames": frames, "num_inference_steps": steps, "generator": rng.generator(seed),
        }
        if image is not None:
            values["image_latents"] = image
        state = PipelineState(values=values)
        try:
            with torch.inference_mode():
                pipe.blocks(pipe, state)
            return cast(torch.Tensor, state.values["latents"])
        finally:
            pipe.transformer._reset_stateful_cache()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> npt.NDArray[np.uint8]:
        vae = cast(AutoencoderKLWan, self.graph.components["vae"])
        try:
            state = stage(PAN2VaeDecoderStep(), {"vae": vae}, {"latents": latents, "output_type": "np"})
            video = cast(npt.NDArray[np.float32], state.values["videos"])[0]
            return np.rint(video.clip(0, 1) * 255).astype(np.uint8)
        finally:
            vae.clear_cache()
