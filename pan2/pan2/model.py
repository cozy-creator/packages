"""Official PAN2 components and modular stages; Runtime owns their weights and residency."""

from __future__ import annotations

import json
import weakref
from collections.abc import Callable, Mapping
from typing import TypedDict, cast

import numpy as np
import torch
from cozy_runtime.accel import sequence_parallel
from cozy_runtime.vlm import VisionLM
from cozy_runtime import rng
from cozy_runtime.author import Config, InvalidRequest, Loader, Model, ModelAssets, Serving, uses_components
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.transformers.transformer_pan2 import PAN2Transformer3DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.guiders.classifier_free_guidance import ClassifierFreeGuidance
from diffusers.modular_pipelines.modular_pipeline import BlockState, PipelineState
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
VideoPixels = np.ndarray[tuple[int, int, int, int], np.dtype[np.uint8]]


def stored[T: nn.Module](module: T, component: str, config: Config) -> T:
    """Construct each destination at its stored logical dtype, including PAN2's F32 modulation."""
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


class Components(TypedDict):
    transformer: PAN2Transformer3DModel
    text_encoder: Qwen3_5TextModel
    vae: AutoencoderKLWan


#: The curve form's config (Comfy-Org's names): the time embedding's SiLU, `vec(t)`, as coordinates in a basis of
#: `adaln_curve_dim` (192) lerped from `adaln_t_table`, one row per `adaln_curve_grid` (4097) points of t in [0, 1]; and
#: every modulation `linear` over those coordinates. It replaces the time embedder and the 4096-wide modulation weights
#: (17.7 GB F32) with 0.8 GB (se-140); `lanes.curve` writes it from the full form.
CURVE = ("adaln_curve_grid", "adaln_curve_dim")


class _CurveCoordinates(nn.Module):
    """`time_proj` of the curve form: each timestep (0-1000) as its lerped row of the transformer's table."""

    def __init__(self, owner: PAN2Transformer3DModel) -> None:
        super().__init__()
        self._owner = weakref.ref(owner)  # not a submodule: the table is the transformer's own buffer

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        table = cast(torch.Tensor, cast(PAN2Transformer3DModel, self._owner()).adaln_t_table)
        position = (timestep.float() / 1000).clamp(0, 1) * (table.shape[0] - 1)
        low = position.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[low], table[low + 1], (position - low).unsqueeze(1))


class _Float32(nn.Module):
    """Stands where the time embedder's `linear_1` was: Diffusers casts the time projection to its weight's dtype."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.empty(0), persistent=False)


class _Coordinates(nn.Module):
    """`time_embedder` of the curve form: the coordinates are already `vec(t)`."""

    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = _Float32()

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates


def curved(model: PAN2Transformer3DModel, grid: int, dim: int) -> PAN2Transformer3DModel:
    """`model` in the curve form: its table, coordinate time embedding and `dim`-wide modulation linears."""
    inner = int(model.config["num_attention_heads"]) * int(model.config["attention_head_dim"])
    model.register_buffer("adaln_t_table", torch.empty(grid, dim, dtype=torch.float32))
    model.time_proj = _CurveCoordinates(model)
    model.time_embedder = _Coordinates()
    for block in model.transformer_blocks:
        block.norm1.silu = nn.Identity()
        block.norm1.linear = nn.Linear(dim, 6 * inner)
    model.norm_out.silu = nn.Identity()
    model.norm_out.linear = nn.Linear(dim, 2 * inner)
    return model


def transformer(config: Config) -> PAN2Transformer3DModel:
    """The DiT as the checkpoint stores it: the full form, or the curve form its config names (`CURVE`)."""
    settings = config.section("transformer").mapping()
    curve = [settings.pop(key, None) for key in CURVE]
    model = PAN2Transformer3DModel.from_config(settings)
    if curve[0] is not None:
        model = curved(model, int(curve[0]), int(curve[1]))
    return stored(model, "transformer", config)


class Graph:
    components: Components
    def __init__(self, config: Config) -> None:
        self.components = {
            "transformer": transformer(config),
            "text_encoder": stored(Qwen3_5TextModel(Qwen3_5TextConfig(**config.section("text_encoder").mapping())), "text_encoder", config),
            "vae": stored(AutoencoderKLWan.from_config(config.section("vae").mapping()), "vae", config),
        }
        self.scheduler_config = config.section("scheduler").mapping()


def stage(block: PAN2TextEncoderStep | PAN2VaeEncoderStep | PAN2VaeDecoderStep, components: Mapping[str, object], values: dict[str, object]) -> PipelineState:
    """A request-local stage with just its active weighted roots; no tensor deepcopy between stages."""
    pipe = cast(PAN2ModularPipeline, block.init_pipeline())
    pipe.register_to_config(enable_safety_checker=False)  # type: ignore[no-untyped-call]  # Diffusers' config registration is unannotated.
    pipe.update_components(**components)  # type: ignore[no-untyped-call]  # Diffusers' component injection is unannotated.
    state = PipelineState(values=values)
    block(pipe, state)
    return state


@sequence_parallel(degrees=(2, 4, 8))
class Reasoner(VisionLM):
    """Replicated auxiliary beside the sequence-sharded simulator."""


@sequence_parallel(degrees=(2, 4, 8))
class Pan2(Model[Graph]):
    serving = Serving(attention_bits={"transformer.transformer_blocks.": 8})
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
        state = stage(PAN2TextEncoderStep(), {"text_encoder": self.graph.components["text_encoder"], "tokenizer": self.tokenizer},  # type: ignore[no-untyped-call]  # Diffusers' block constructor is unannotated.
                      {"prompt": prompt, "negative_prompt": negative if guidance > 1 else None,
                       "num_frames": frames, "fps": float(FPS)})
        return cast(torch.Tensor, state.values["prompt_embeds"]), cast(torch.Tensor | None, state.values.get("negative_prompt_embeds"))

    @uses_components("vae")
    def encode_image(self, image: Image.Image, width: int, height: int) -> torch.Tensor:
        vae = self.graph.components["vae"]
        try:
            state = stage(PAN2VaeEncoderStep(), {"vae": vae}, {"image": image, "width": width, "height": height})  # type: ignore[no-untyped-call]  # Diffusers' block constructor is unannotated.
            return cast(torch.Tensor, state.values["image_latents"])
        finally:
            vae.clear_cache()  # type: ignore[no-untyped-call]  # Diffusers' cache reset is unannotated.

    @uses_components("transformer")
    def denoise(self, embeds: tuple[torch.Tensor, torch.Tensor | None], image: torch.Tensor | None, *,
                width: int, height: int, frames: int, steps: int, guidance: float, seed: int,
                check: Callable[[], None], progress: Callable[[int], None]) -> torch.Tensor:
        blueprint = PAN2CoreDenoiseStep() if image is None else PAN2Image2VideoCoreDenoiseStep()  # type: ignore[no-untyped-call]  # Diffusers' block constructors are unannotated.
        pipe = cast(PAN2ModularPipeline, blueprint.init_pipeline())
        # init_pipeline copies its block graph. Attach progress and cancellation to the
        # graph that actually runs, not the blueprint it was copied from.
        block = cast(PAN2CoreDenoiseStep | PAN2Image2VideoCoreDenoiseStep, pipe.blocks)
        pipe.update_components(  # type: ignore[no-untyped-call]  # Diffusers' component injection is unannotated.
            transformer=self.graph.components["transformer"],
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
            guider=ClassifierFreeGuidance(guidance_scale=guidance),
        )
        loop = block.sub_blocks["denoise"]
        original = loop.loop_step

        def step(components: PAN2ModularPipeline, state: BlockState, *, i: int, t: torch.Tensor) -> tuple[PAN2ModularPipeline, BlockState]:
            check()
            result = original(components, state, i=i, t=t)
            progress(i)
            return cast(tuple[PAN2ModularPipeline, BlockState], result)

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
                block(pipe, state)
            return cast(torch.Tensor, state.values["latents"])
        finally:
            pipe.transformer._reset_stateful_cache()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> VideoPixels:
        vae = self.graph.components["vae"]
        try:
            state = stage(PAN2VaeDecoderStep(), {"vae": vae}, {"latents": latents, "output_type": "np"})  # type: ignore[no-untyped-call]  # Diffusers' block constructor is unannotated.
            videos = cast(np.ndarray[tuple[int, int, int, int, int], np.dtype[np.float32]], state.values["videos"])
            return cast(VideoPixels, np.rint(videos[0].clip(0, 1) * 255).astype(np.uint8))
        finally:
            vae.clear_cache()  # type: ignore[no-untyped-call]  # Diffusers' cache reset is unannotated.
