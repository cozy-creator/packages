"""Anima's model: Diffusers' modular pipeline, its fused plan, its denoise loop and its progress."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple, Protocol, TypedDict, cast

import torch
from cozy_runtime import rng
from cozy_runtime.accel import fusion, hold_dtype, stepcache
from cozy_runtime.author import Config, Loader, Model, Telemetry, uses_components
from cozy_runtime.accel.kernels import Plan
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditioner
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from diffusers.modular_pipelines.anima.modular_blocks_anima import AnimaAutoBlocks
from diffusers.modular_pipelines.anima.modular_pipeline import AnimaModularPipeline
from diffusers.modular_pipelines.modular_pipeline import BlockState, ModularPipeline, ModularPipelineBlocks, PipelineState
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.modular_pipelines.anima.denoise import AnimaDenoiseStep
from transformers import AutoModel, PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
from transformers import initialization as transformer_init
from transformers.modeling_utils import local_torch_dtype

from . import optimization

_ROOT = Path(__file__).resolve().parent
#: The DiT's block stack for `stepcache` (FBCache): each Cosmos block returns its hidden states.
STACK = stepcache.Stack(CosmosTransformer3DModel, blocks="transformer_blocks", carried=(0,))


#: The render ladder. `cozy run ls` renders "<stage> <percent>" verbatim, so these names
#: are read by a person. Only `denoise` is measured per step — it is the only phase long enough
#: to need it, and at the default 1536 class it is 150 of the roughly 160 seconds.
_ENCODE_PROMPT = "encoding prompt"
_CONDITION = "conditioning"
_DENOISE = "denoise"
_DECODE = "decoding"
_SAVE = "saving image"


def _tokenizer(path: Path) -> PreTrainedTokenizerFast:
    config = json.loads((path / "tokenizer_config.json").read_text())
    config.pop("tokenizer_class", None)
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # Transformers' tokenizer __init__ is unannotated
        tokenizer_file=str(path / "tokenizer.json"), **config
    )


class AnimaPipeline:
    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            transformer = _to(construct(CosmosTransformer3DModel, mapping["transformer"]), torch.bfloat16)
            # Built in bf16, not cast to it: each rotary inv_freq stays f32, as upstream's.
            text_encoder: Qwen3Model = AutoModel.from_config(  # type: ignore[no-untyped-call]  # Transformers' auto classes are unannotated
                Qwen3Config(**mapping["text_encoder"]), dtype=torch.bfloat16
            )
            with local_torch_dtype(torch.bfloat16):
                text_conditioner = construct(AnimaTextConditioner, mapping["text_conditioner"])
            vae = _to(construct(AutoencoderKLQwenImage, mapping["vae"]), torch.bfloat16)

        for component in (transformer, text_encoder, text_conditioner, vae):
            component.eval()
        self.scheduler_config: dict[str, object] = mapping["scheduler"]
        self.tokenizer = _tokenizer(_ROOT / "tokenizer")
        self.t5_tokenizer = _tokenizer(_ROOT / "t5_tokenizer")
        self.components: Components = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "text_conditioner": text_conditioner,
            "vae": vae,
        }


class Components(TypedDict):
    transformer: CosmosTransformer3DModel
    text_encoder: Qwen3Model
    text_conditioner: AnimaTextConditioner
    vae: AutoencoderKLQwenImage


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def _to[M: torch.nn.Module](module: M, dtype: torch.dtype) -> M:
    """`module.to(dtype)`, typed: Diffusers' `to` is `functools.wraps`'d, which the checkers bind without `self`."""
    return module.to(dtype)


def build_pipeline(config: Config) -> AnimaPipeline:
    return AnimaPipeline(config)


class AnimaModel(Model[AnimaPipeline]):
    """Touches its components through their forward passes only, so an encoded lane may
    replace DiT linears with native leaves (the fp8 lane's whole point on sm89). The filled
    transformer runs the fused CUDA plan wherever this process supports it."""

    pipe: AnimaPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(build_pipeline)

    def execution_plan(self, plan: Plan) -> str:
        refusals = optimization.apply(self.pipe)
        parts = dict(zip(("anima.fused", "anima.frame-decode"), refusals, strict=True))
        for name, refusal in parts.items():
            # In the run's receipts (`plan_reports`), so a refusal is never silent.
            outcome: dict[str, object] = {"applied": not refusal}
            if refusal:
                outcome["reason"] = refusal
            plan.report(name, outcome)
        vae = self.pipe.components["vae"]
        installed = (
            *(name for name, refusal in parts.items() if not refusal),
            stepcache.install(plan, STACK, self.pipe.components["transformer"]),
            fusion.install_norms(plan, optimization.VAE_NORM, vae),
            fusion.install_lone_frame(plan, optimization.VAE_FRAME, vae.decoder, vae.post_quant_conv),
        )
        return "+".join(identity for identity in installed if identity)

    def hold_vae(self, tel: Telemetry) -> None:
        """Decode in `optimization.VAE_DTYPE` on CUDA; the CPU keeps the stored bf16. The Runtime
        casts the stored bytes when it next fills them; called before any VAE scope opens."""
        vae = self.pipe.components["vae"]
        dtype = optimization.VAE_DTYPE
        if torch.device(str(self.device)).type != "cuda" or next(vae.parameters()).dtype == dtype:
            return
        hold_dtype(vae, dtype)
        tel.log("vae representation", dtype="float16")

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
    ) -> torch.Tensor:
        device = torch.device(str(self.device))
        generator = rng.generator(seed)
        fused = optimization.installed(self.pipe.components["transformer"])
        cache = stepcache.FirstBlockCache(first_block_cache) if first_block_cache > 0.0 else None
        pipeline = _text2image_pipeline(device, phases, batched, fused, cache)
        pipeline.register_components(  # type: ignore[no-untyped-call]  # Diffusers' own, unannotated
            **self.pipe.components,
            scheduler=construct(FlowMatchEulerDiscreteScheduler, self.pipe.scheduler_config),
            tokenizer=self.pipe.tokenizer,
            t5_tokenizer=self.pipe.t5_tokenizer,
        )
        pipeline.guider.guidance_scale = guidance
        # Diffusers 0.40 stores the interval on private attrs set by BaseGuidance.__init__;
        # outside [start, stop) num_conditions == 1 and the uncond forward never runs.
        pipeline.guider._start, pipeline.guider._stop = cfg_interval
        phases.enter(_ENCODE_PROMPT)
        images: torch.Tensor = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
            output="images",
            output_type="pt",
        )
        return images


def _text2image_pipeline(
    device: torch.device, phases: _Phases, batched: bool, fused: bool, cache: stepcache.Policy | None
) -> AnimaModularPipeline:
    """The text2image pipeline, instrumented on the blocks it will actually run, its denoise loop
    `_Denoise`, its padding mask at latent size when `fused`.

    `ModularPipeline.blocks` is a property returning a DEEPCOPY, so a hook installed through
    it is thrown away and the meter never moves — the whole of the frozen `conditioning 0%`
    defect (se-026). `blocks=` hands the pipeline the object it keeps, and the identity is
    the fix.
    """
    class Announce(ModularPipelineBlocks):
        """A weightless block whose only effect is to advance the meter."""

        model_name = "anima"  # type: ignore[assignment]  # Diffusers declares None

        @property
        def description(self) -> str:
            return "Announces Anima's conditioning phase to Runtime telemetry."

        def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
            phases.enter(_CONDITION)
            return components, state

    class RuntimeAnimaPipeline(AnimaModularPipeline):
        @property
        def _execution_device(self) -> torch.device:
            return device

    blocks = AnimaAutoBlocks().get_workflow("text2image")  # type: ignore[no-untyped-call]  # Diffusers' blocks, unannotated
    order = list(blocks.sub_blocks)
    if not {"denoise.denoise", "denoise.text_conditioning", "denoise.prepare_latents"} <= set(
        order
    ):
        raise RuntimeError(f"Diffusers Anima text2image blocks changed: {order}")
    blocks.sub_blocks["denoise.denoise"] = _Denoise(phases, batched, cache)
    if fused:
        blocks.sub_blocks["denoise.prepare_latents"] = optimization.FusedLatents()  # type: ignore[no-untyped-call]  # its base's __init__
    blocks.sub_blocks.insert("conditioning", Announce(), order.index("denoise.text_conditioning"))  # type: ignore[no-untyped-call]  # its base's __init__
    return RuntimeAnimaPipeline(blocks=blocks)


class _DenoiseState(Protocol):
    """The denoise loop's `BlockState`: the fields it reads and writes."""

    dtype: torch.dtype
    timesteps: torch.Tensor
    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    padding_mask: torch.Tensor
    num_inference_steps: int


class _Denoise(AnimaDenoiseStep):
    """Diffusers' Anima denoise loop: each step is `AnimaLoopBeforeDenoiser`, `AnimaLoopDenoiser` under
    the CFG guider as `render` sets it (its scale and interval), and `AnimaLoopAfterDenoiser`'s Euler
    step. With `batched`, a guided step's two conditions run as one forward: in a small pool every
    forward streams the whole transformer through the card, so this halves a CFG step's copies (C4 of
    the small-pool analysis). `cache` is FBCache as a `stepcache` policy: each forward of a step (the
    condition, then the unconditional one) keeps its own state, as Diffusers' per-condition cache
    contexts did."""

    def __init__(self, phases: _Phases, batched: bool, cache: stepcache.Policy | None) -> None:
        super().__init__()  # type: ignore[no-untyped-call]  # Diffusers' blocks, unannotated
        self.phases, self.batched, self.cache = phases, batched, cache

    @torch.no_grad()
    def __call__(
        self, components: AnimaModularPipeline, state: PipelineState
    ) -> tuple[AnimaModularPipeline, PipelineState]:
        block_state = cast(_DenoiseState, self.get_block_state(state))  # Diffusers annotates its BlockState as dict
        guider, transformer, scheduler = components.guider, components.transformer, components.scheduler
        dtype, timesteps, latents = block_state.dtype, block_state.timesteps, block_state.latents
        steps = len(timesteps)
        positive = block_state.prompt_embeds.to(dtype)
        negative: torch.Tensor | None = getattr(block_state, "negative_prompt_embeds", None)
        if negative is not None:
            negative = torch.cat([positive, negative.to(dtype)]) if self.batched else negative.to(dtype)

        def predict(hidden: torch.Tensor, timestep: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
            return transformer(  # type: ignore[no-any-return]
                hidden_states=hidden, timestep=timestep, padding_mask=block_state.padding_mask,
                encoder_hidden_states=text, return_dict=False,
            )[0]

        self.phases.enter(_DENOISE)
        on_step = self.phases.steps(steps)
        cached = nullcontext() if self.cache is None else stepcache.running(self.cache, steps, latents.device)
        with cached as run:
            for i, t in enumerate(timesteps):
                if run is not None:
                    run.step(i)
                guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
                hidden = latents.to(dtype)
                timestep = t.expand(1).to(dtype) / scheduler.config.num_train_timesteps
                if guider.num_conditions == 1:
                    prediction = predict(hidden, timestep, positive)
                else:
                    if self.batched:
                        conditional, unconditional = predict(torch.cat([hidden] * 2), timestep, negative).chunk(2)  # type: ignore[arg-type]  # two conditions: a negative
                    else:
                        conditional = predict(hidden, timestep, positive)
                        unconditional = predict(hidden, timestep, negative)  # type: ignore[arg-type]  # two conditions: a negative
                    prediction = unconditional + guider.guidance_scale * (conditional - unconditional)
                latents = scheduler.step(prediction, t, latents, return_dict=False)[0]
                on_step(i)
        self.phases.enter(_DECODE)
        block_state.latents = latents
        self.set_block_state(state, cast(BlockState, block_state))
        return components, state


class _Phases:
    """The one open `tel.stage` bracket, advanced from inside Diffusers' opaque call.

    `pipeline(...)` runs text encoding, conditioning, denoise and decode behind a single
    call, so the brackets are opened and closed by the blocks themselves rather than by
    `with` statements around them. `progress` is the handler's `ctx.progress`. Without `tel`
    there are no stages, and without `progress` no steps.
    """

    def __init__(
        self, tel: Telemetry | None = None, progress: Callable[[int, int], None] | None = None
    ) -> None:
        self.tel, self.progress = tel, progress
        self.open: AbstractContextManager[None] | None = None

    def enter(self, stage: str) -> None:
        self.close()
        if self.tel is None:
            return
        self.open = self.tel.stage(stage)
        self.open.__enter__()

    def steps(self, total: int) -> Callable[[int], None]:
        progress = self.progress
        if progress is None:
            return lambda position: None
        return lambda step: progress(step, total)

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
