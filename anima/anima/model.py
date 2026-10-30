"""Anima's model: Diffusers' maintained modular pipeline, its fused plan and its progress."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple

import torch
from cozy_runtime import fusion, rng
from cozy_runtime.author import Loader, Model, Telemetry, uses_components
from cozy_runtime.author.kernels import Plan
from diffusers import (
    AnimaAutoBlocks,
    AnimaModularPipeline,
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.first_block_cache import (
    _FBC_BLOCK_HOOK,
    _FBC_LEADER_BLOCK_HOOK,
    FirstBlockCacheConfig,
    apply_first_block_cache,
)
from diffusers.hooks.hooks import HookRegistry
from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock
from diffusers.modular_pipelines import ModularPipelineBlocks
from diffusers.modular_pipelines.anima.denoise import AnimaLoopDenoiser
from transformers import AutoModel, PreTrainedTokenizerFast, Qwen3Config
from transformers import initialization as transformer_init
from transformers.modeling_utils import local_torch_dtype

from . import optimization

_ROOT = Path(__file__).resolve().parent


class _Phase(NamedTuple):
    """One render phase: the stage name a person reads, and its overall-progress span."""

    stage: str
    start: float
    stop: float

    @property
    def bounds(self) -> tuple[float, float]:
        return self.start, self.stop


#: The render ladder. `cozy run ls` renders "<stage> <percent>" verbatim, so these names
#: are read by a person; the spans are contiguous and cover the whole request. Only
#: `denoise` is measured per step — it is the only phase long enough to need it, and at the
#: default 1536 class it is 150 of the roughly 160 seconds.
_ENCODE_PROMPT = _Phase("encoding prompt", 0.00, 0.06)
_CONDITION = _Phase("conditioning", 0.06, 0.10)
_DENOISE = _Phase("denoise", 0.10, 0.90)
_DECODE = _Phase("decoding", 0.90, 0.98)
_SAVE = _Phase("saving image", 0.98, 1.00)


def _tokenizer(path: Path) -> Any:
    config = json.loads((path / "tokenizer_config.json").read_text())
    config.pop("tokenizer_class", None)
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), **config
    )


class AnimaPipeline:
    def __init__(self, config: Any) -> None:
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            transformer = CosmosTransformer3DModel.from_config(mapping["transformer"]).to(
                torch.bfloat16
            )
            # Built in bf16, not cast to it: each rotary inv_freq stays f32, as upstream's.
            text_encoder: Any = AutoModel.from_config(
                Qwen3Config(**mapping["text_encoder"]), dtype=torch.bfloat16
            )
            with local_torch_dtype(torch.bfloat16):
                text_conditioner = AnimaTextConditioner.from_config(mapping["text_conditioner"])
            vae = AutoencoderKLQwenImage.from_config(mapping["vae"]).to(torch.bfloat16)

        for component in (transformer, text_encoder, text_conditioner, vae):
            component.eval()
        self.scheduler_config = mapping["scheduler"]
        self.tokenizer = _tokenizer(_ROOT / "tokenizer")
        self.t5_tokenizer = _tokenizer(_ROOT / "t5_tokenizer")
        self.components: dict[str, Any] = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "text_conditioner": text_conditioner,
            "vae": vae,
        }


def build_pipeline(config: Any) -> AnimaPipeline:
    return AnimaPipeline(config)


class AnimaModel(Model[AnimaPipeline], encoded_leaves="accept"):
    """Touches its components through their forward passes only, so an encoded lane may
    replace DiT linears with native leaves (the fp8 lane's whole point on sm89). The filled
    transformer runs the fused CUDA plan wherever this process supports it."""

    pipe: AnimaPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(AnimaPipeline, factory=build_pipeline)

    def execution_plan(self, plan: Plan) -> str:
        refusals = optimization.apply(self.pipe)
        parts = dict(zip(("anima.fused", "anima.frame-decode"), refusals, strict=True))
        for name, refusal in parts.items():
            # In the run's receipts (`plan_reports`), so a refusal is never silent.
            outcome: dict[str, object] = {"applied": not refusal}
            if refusal:
                outcome["reason"] = refusal
            plan.report(name, outcome)
            print(f"[anima] {name}: {refusal or 'applied'}", file=sys.stderr)
        vae = self.pipe.components["vae"]
        installed = (
            *(name for name, refusal in parts.items() if not refusal),
            fusion.install_norms(plan, optimization.VAE_NORM, vae),
            fusion.install_lone_frame(plan, optimization.VAE_FRAME, vae.decoder, vae.post_quant_conv),
        )
        return "+".join(identity for identity in installed if identity)

    @uses_components("text_encoder", "text_conditioner", "transformer", "vae")
    def render(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        cfg_interval: tuple[float, float],
        first_block_cache: float,
        seed: int,
        phases: _Phases,
        batched: bool,
    ) -> Any:
        device = torch.device(str(self.device))
        generator = rng.generator(seed)
        fused = optimization.installed(self.pipe.components["transformer"])
        pipeline: Any = _text2image_pipeline(device, phases, batched, fused)
        pipeline.register_components(
            **self.pipe.components,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler_config),
            tokenizer=self.pipe.tokenizer,
            t5_tokenizer=self.pipe.t5_tokenizer,
        )
        pipeline.guider.guidance_scale = guidance
        # Diffusers 0.40 stores the interval on private attrs set by BaseGuidance.__init__;
        # outside [start, stop) num_conditions == 1 and the uncond forward never runs.
        pipeline.guider._start, pipeline.guider._stop = cfg_interval
        transformer = self.pipe.components["transformer"]
        restore_cache = _apply_first_block_cache(transformer, pipeline.guider, first_block_cache)
        phases.enter(_ENCODE_PROMPT)
        try:
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                output="images",
                output_type="pt",
            )
        finally:
            restore_cache()
        return result


def _text2image_pipeline(device: Any, phases: _Phases, batched: bool, fused: bool) -> Any:
    """The text2image pipeline, instrumented on the blocks it will actually run, its CFG
    conditions in one forward when `batched`, its padding mask at latent size when `fused`.

    `ModularPipeline.blocks` is a property returning a DEEPCOPY, so a hook installed through
    it is thrown away and the meter never moves — the whole of the frozen `conditioning 0%`
    defect (se-026). `blocks=` hands the pipeline the object it keeps, and the identity is
    the fix.
    """
    class Announce(ModularPipelineBlocks):
        """A weightless block whose only effect is to advance the meter."""

        model_name = "anima"

        @property
        def description(self) -> str:
            return "Announces Anima's conditioning phase to Runtime telemetry."

        def __call__(self, components: Any, state: Any) -> tuple[Any, Any]:
            phases.enter(_CONDITION)
            return components, state

    class RuntimeAnimaPipeline(AnimaModularPipeline):
        @property
        def _execution_device(self) -> Any:
            return device

    blocks = AnimaAutoBlocks().get_workflow("text2image")
    order = list(blocks.sub_blocks)
    if not {"denoise.denoise", "denoise.text_conditioning", "denoise.prepare_latents"} <= set(
        order
    ):
        raise RuntimeError(f"Diffusers Anima text2image blocks changed: {order}")
    loop = blocks.sub_blocks["denoise.denoise"]
    loop.progress_bar = _progress_bar(phases)
    if batched:
        loop.sub_blocks["denoiser"] = _BatchedDenoiser()
    if fused:
        blocks.sub_blocks["denoise.prepare_latents"] = optimization.FusedLatents()
    blocks.sub_blocks.insert("conditioning", Announce(), order.index("denoise.text_conditioning"))
    return RuntimeAnimaPipeline(blocks=blocks)


class _BatchedDenoiser(AnimaLoopDenoiser):
    """Diffusers' Anima denoiser with the guider's conditions as one batch: one transformer
    forward per step instead of one per condition. In a small pool every forward streams the
    whole transformer through the card, so this halves a CFG step's copies (C4 of the
    small-pool analysis). Inside the CFG interval cond and uncond run together; outside it
    the guider asks for cond alone, and Diffusers' own step runs."""

    def __init__(self) -> None:
        super().__init__({"encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds")})

    @torch.no_grad()
    def __call__(self, components: Any, block_state: Any, i: int, t: torch.Tensor) -> Any:
        guider, transformer = components.guider, components.transformer
        guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        if guider.num_conditions == 1:
            return super().__call__(components, block_state, i, t)
        batches = guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)
        for _ in batches:
            guider.prepare_models(transformer)
        conditions = {
            key: torch.cat([getattr(batch, key) for batch in batches]).to(block_state.dtype)
            for key in self._guider_input_fields
        }
        count = len(batches)
        # One timestep for every condition: the modulation is computed once and broadcast.
        predictions = transformer(
            hidden_states=torch.cat([block_state.latent_model_input] * count),
            timestep=block_state.timestep,
            padding_mask=block_state.padding_mask,
            return_dict=False,
            **conditions,
        )[0]
        for batch, prediction in zip(batches, predictions.chunk(count), strict=True):
            batch.noise_pred = prediction
            guider.cleanup_models(transformer)
        block_state.noise_pred = guider(batches)[0]
        return components, block_state


class _Phases:
    """The one open `tel.stage` bracket, advanced from inside Diffusers' opaque call.

    `pipeline(...)` runs text encoding, conditioning, denoise and decode behind a single
    call, so the brackets are opened and closed by the blocks themselves rather than by
    `with` statements around them. Without a `tel` it is silent.
    """

    def __init__(self, tel: Telemetry | None = None) -> None:
        self.tel = tel
        self.open: AbstractContextManager[None] | None = None

    def enter(self, phase: _Phase) -> None:
        self.close()
        if self.tel is None:
            return
        self.open = self.tel.stage(phase.stage, overall_range=phase.bounds)
        self.open.__enter__()

    def steps(self, total: int) -> Callable[[int], None]:
        if self.tel is None:
            return lambda position: None
        return self.tel.step_callback(  # type: ignore[no-any-return]
            total, stage=_DENOISE.stage, overall_range=_DENOISE.bounds
        )

    def close(self) -> None:
        self.__exit__(None, None, None)

    def __enter__(self) -> _Phases:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        bracket, self.open = self.open, None
        if bracket is not None:
            bracket.__exit__(kind, exc, traceback)


class _DenoiseProgress:
    """Diffusers' denoise-loop progress bar projected onto Runtime telemetry."""

    def __init__(self, total: int, phases: _Phases) -> None:
        self.total = total
        self.phases = phases
        self.position = 0
        self.step: Callable[[int], None] | None = None

    def __enter__(self) -> _DenoiseProgress:
        self.phases.enter(_DENOISE)
        self.step = self.phases.steps(self.total)
        return self

    def update(self, count: int = 1) -> None:
        if self.step is None:
            raise RuntimeError("Anima denoise progress updated outside its loop")
        for _ in range(count):
            self.step(self.position)
            self.position += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.phases.enter(_DECODE)


def _progress_bar(phases: _Phases) -> Callable[..., _DenoiseProgress]:
    def progress_bar(iterable: object = None, total: int | None = None) -> _DenoiseProgress:
        if iterable is not None or total is None or total < 1:
            raise RuntimeError("Anima denoise progress requires one positive total")
        return _DenoiseProgress(total, phases)

    return progress_bar


def _apply_first_block_cache(transformer: Any, guider: Any, threshold: float) -> Callable[[], None]:
    """Attach FBCache hooks for one request; returns the restore. 0.0 attaches nothing.

    Diffusers ships `apply_first_block_cache` but its registry does not know
    `CosmosTransformerBlock` — the one-line registration below is the whole shim. With it on,
    CFG runs as two batch-1 forwards, and the cond and uncond forwards would otherwise
    share one residual cache and compare cond against uncond; wrapping the transformer
    forward in a per-condition `cache_context` keeps the two streams separate. Hooks are
    request-scoped (the HiDiffusion precedent in sdxl): removed in the restore so the
    resident module leaves exactly as it entered.
    """
    if threshold <= 0.0:
        return lambda: None

    try:
        TransformerBlockRegistry.get(CosmosTransformerBlock)
    except ValueError:
        TransformerBlockRegistry.register(
            CosmosTransformerBlock,
            TransformerBlockMetadata(return_hidden_states_index=0),
        )
    apply_first_block_cache(transformer, FirstBlockCacheConfig(threshold=threshold))
    original_forward = transformer.forward
    registry = HookRegistry.check_if_exists_or_initialize(transformer)

    def forward(*args: Any, **kwargs: Any) -> Any:
        # CosmosTransformer3DModel predates diffusers' CacheMixin, so the context is set
        # on the hook registry directly — the same two calls `cache_context` makes.
        registry._set_context("cond" if guider.is_conditional else "uncond")
        try:
            return original_forward(*args, **kwargs)
        finally:
            registry._set_context(None)

    transformer.forward = forward

    def restore() -> None:
        transformer.forward = original_forward
        for block in transformer.transformer_blocks:
            registry = HookRegistry.check_if_exists_or_initialize(block)
            for name in (_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK):
                if name in registry.hooks:
                    registry.remove_hook(name, recurse=False)

    return restore
