"""MiniMax H3 models over Runtime's public Model contract."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Literal

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoTransformerBlock

from cozy_runtime import fusion
from cozy_runtime.author import (
    AdapterCompatibility,
    DerivedCache,
    Loader,
    Model,
    Telemetry,
    concurrently,
    placeable,
    sequence_parallel,
    uses_components,
)
from cozy_runtime.author.kernels import Plan

from . import glue, producer
from .official import (
    OfficialH3Pipeline,
    OfficialH3TurboLoRA,
    ReferenceMemo,
    ScheduleFacts,
    Task,
    _validate_sol_dense_steps,
    build_h3_pipeline,
    build_h3_turbo_base,
    build_h3_turbo_lora,
    model_plans,
)
from .turbo import ATTENTION_KWARG, OVERLAY_KWARG, TURBO_BANK

#: Reference latents one generation keeps (`ReferenceMemo`), least recently used first out; the
#: Runtime sheds them sooner when memory is short.
REFERENCE_MEMO_ENTRIES = 256


def _layer_scales(block: Any, arguments: Mapping[str, Any]) -> fusion.Modulation:
    """A video decoder block's layer scales: the gates of one row every token reads."""
    row = block.scale1.new_zeros(1, dtype=torch.long)
    return fusion.Modulation(row, gate_msa=block.scale1[None], gate_mlp=block.scale2[None])


#: The video VAE's ViT decoder block as Runtime fuses it: fp16 autocast over an fp32 residual
#: stream, torch RMSNorms, weightless q/k norms, split-half RoPE on 48 of 64 channels in fp16,
#: Diffusers' value-first SwiGLU, `h + branch * scale` layer scales.
VAE_BLOCK = fusion.DiTBlock(
    MiniMaxH3VideoTransformerBlock,
    width=2048,
    head_dim=64,
    modulation=_layer_scales,
    norms=("norm1", "norm2"),
    norm="rms",
    attention="attn",
    arguments=("rotary_emb",),
    mlp=fusion.GatedMLP("ff", gate="net.0.proj", down="net.2", value_first=True),
    rope=fusion.Rope("rotary_emb", interleaved=False, rotary=48, rounding="product"),
    scale=False,
    rows="fp16",
    rms_rounding="once",
    qk_affine=False,
)


def condition_references(model: H3Model, task: Task, state: Any, *, tel: Telemetry) -> None:
    """`condition_text` and `condition_ref2va_media` of one `ref2va` request, overlapped where
    the group keeps the text encoder on a follower (`author.concurrently`): rank 0 encodes
    the references while the follower runs the conditioner.

    Both read only the prompt and the references and write disjoint outputs. The text block
    reads a copy taken before the media block may append a continuation's layout, and its
    outputs land in `state` once both have finished, as the block writes them. So the state
    equals the sequential one wherever the two ran.
    """
    text = model.pipe.text_state(state)
    started = time.perf_counter()
    concurrently(
        partial(model.condition_text, task, text),
        partial(model.condition_ref2va_media, task, state, tel=tel),
    )
    model.pipe.adopt_text(task, state, text)
    tel.log("h3 conditioning", wall_ms=round((time.perf_counter() - started) * 1000, 1))


@placeable("text_encoder")
@sequence_parallel(degrees=(2, 4, 7, 8))
class H3Model(
    Model[OfficialH3Pipeline],
    encoded_leaves="accept",
    # The DiTs' attention holds at 4 bits (fp4 where the GPU has it); the VAEs and the text
    # encoder keep exact attention.
    attention_bits={"transformer": 4, "transformer_ref": 4},
):
    """H3 is head-shardable at every divisor of its 56 attention heads up to eight GPUs.

    The upstream `_cp_plan` shards the packed sequence itself with `tensor_split`, so any
    length splits and every block's GEMMs, norms, RoPE, AdaLN rows and SwiGLU run on the
    local rows; only attention exchanges, per head. `EncodedLinear` quantizes per TOKEN with
    rowwise `_scaled_mm` scales, so a row's numbers do not depend on which rank holds it; a
    per-tensor activation scale would be derived from the local shard and is refused by the
    runtime rather than served. Sampling and both VAEs stay on the leader. The 51.5 GB text
    conditioner is entered only through module calls, so a group whose leader cannot hold
    it beside the DiT keeps it resident on a follower instead (`placeable`).

    Degrees 2 and 4 reproduce the degree-1 video and audio velocities to 1.2e-7 and 2.4e-7
    max absolute error on this pipeline's own AdaLN-pruned DiT (2026-09-08), which is
    float32 round-off. 7 and 8 match one rank on CPU Gloo ranks (`test_parallel_calls`) and
    have not yet run on GPUs (h3a-019).
    """

    __adapter_compatibility__ = (
        AdapterCompatibility("lora", "", ("transformer", "transformer_ref"), -math.inf, math.inf),
    )

    pipe: OfficialH3Pipeline

    def load(self, loader: Loader) -> None:
        plans = model_plans(loader.assets)
        self.pipe = loader.construct(OfficialH3Pipeline, factory=partial(build_h3_pipeline, plans=plans))
        self._declare_caches(loader)

    def _declare_caches(self, loader: Loader) -> None:
        """The generation's reference latents (`ReferenceMemo`): memory only, bounded."""
        self.reference_latents = loader.cache("reference_latents", max_entries=REFERENCE_MEMO_ENTRIES)

    def unload(self, loader: Loader) -> None:
        return None

    def execution_plan(self, plan: Plan) -> str:
        """The fused glue on every filled H3 DiT, the fused video VAE decoder blocks, and the CP
        producer's rotary hook."""
        for component in self.pipe.components.values():
            producer.retain_rotary(component)
        plans = (
            glue.install(self, self.pipe, plan),
            fusion.install(plan, VAE_BLOCK, self.pipe.components["vae"]),
        )
        return "+".join(identity for identity in plans if identity)

    @uses_components("text_encoder")
    def condition_text(self, task: Task, state: Any) -> None:
        self.pipe.condition_text(task, state, self.device)

    @uses_components("audio_vae")
    def decode_audio(self, task: Task, state: Any) -> tuple[Any, int]:
        return self.pipe.decode_audio(task, state), int(state.sampling_rate)

    @uses_components("vae")
    def decode_video(
        self,
        task: Task,
        state: Any,
        *,
        on_chunk: Callable[[Any], None],
    ) -> int:
        """Hand every decoded temporal chunk to `on_chunk` inside the VAE's component scope;
        a generator would run its body after the scope had closed."""
        return self.pipe.decode_video_chunks(task, state, on_chunk)

    @uses_components("vae")
    def condition_fl2va_media(self, task: Task, state: Any) -> None:
        self.pipe.condition_media(task, state)

    @uses_components("vae", "audio_vae")
    def condition_ref2va_media(self, task: Task, state: Any, *, tel: Telemetry) -> None:
        cache: DerivedCache | None = getattr(self, "reference_latents", None)
        memo = None if cache is None else ReferenceMemo(cache, self.checkpoint_ref)
        reused, encoded = self.pipe.condition_media(task, state, memo=memo)
        tel.log("h3 reference latents", reused=reused, encoded=encoded)

    @uses_components("transformer")
    def sample_fl2va(self, state: Any, *, on_step: Any, cancel: Any) -> ScheduleFacts:
        return self.pipe.denoise("fl2va", state, on_step=on_step, cancel=cancel)

    @uses_components("transformer_ref")
    def sample_ref2va(self, state: Any, *, on_step: Any, cancel: Any) -> ScheduleFacts:
        return self.pipe.denoise("ref2va", state, on_step=on_step, cancel=cancel)


class H3TurboBase(H3Model, encoded_leaves="accept"):
    """The five-root base checkpoint with construction-time turbo consumers.

    ``sol_dense_steps`` counts initial transformer evaluations, not scheduler
    sigma points or CFG branches. The default 4 keeps evaluations 1-4 dense and
    allows sparse video attention in 5-8 when Sol is explicitly selected. Callers
    can still request 3, 8 or 10; no value selects Sol, changes its protected prefix,
    or removes the always-dense refiner and first two transformer blocks.
    """

    def load(self, loader: Loader) -> None:
        plans = model_plans(loader.assets)
        self.pipe = loader.construct(OfficialH3Pipeline, factory=partial(build_h3_turbo_base, plans=plans))
        self._declare_caches(loader)

    @uses_components("transformer")
    def sample_fl2va_turbo(
        self,
        state: Any,
        *,
        turbo_lora: H3TurboLoRA,
        on_step: Any,
        cancel: Any,
        sol_dense_steps: int = 4,
    ) -> ScheduleFacts:
        return turbo_lora.sample_fl2va(
            self,
            state,
            on_step=on_step,
            cancel=cancel,
            sol_dense_steps=sol_dense_steps,
        )

    @uses_components("transformer_ref")
    def sample_ref2va_turbo(
        self,
        state: Any,
        *,
        turbo_lora: H3TurboLoRA,
        on_step: Any,
        cancel: Any,
        sol_dense_steps: int = 4,
    ) -> ScheduleFacts:
        return turbo_lora.sample_ref2va(
            self,
            state,
            on_step=on_step,
            cancel=cancel,
            sol_dense_steps=sol_dense_steps,
        )


@sequence_parallel(degrees=(2, 4, 7, 8))
class H3TurboLoRA(Model[OfficialH3TurboLoRA], encoded_leaves="accept"):
    """Independent PDD weights, replicated alongside the base on every CP rank."""

    pipe: OfficialH3TurboLoRA

    def load(self, loader: Loader) -> None:
        plans = model_plans(loader.assets, turbo=True)
        self.pipe = loader.construct(OfficialH3TurboLoRA, factory=partial(build_h3_turbo_lora, plans=plans))

    def unload(self, loader: Loader) -> None:
        return None

    @uses_components("fl2va_turbo")
    def sample_fl2va(
        self,
        base: H3Model,
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        sol_dense_steps: int = 4,
    ) -> ScheduleFacts:
        return self._sample(
            base,
            "fl2va",
            state,
            on_step=on_step,
            cancel=cancel,
            sol_dense_steps=sol_dense_steps,
        )

    @uses_components("ref2va_turbo")
    def sample_ref2va(
        self,
        base: H3Model,
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        sol_dense_steps: int = 4,
    ) -> ScheduleFacts:
        return self._sample(
            base,
            "ref2va",
            state,
            on_step=on_step,
            cancel=cancel,
            sol_dense_steps=sol_dense_steps,
        )

    def _sample(
        self,
        base: H3Model,
        trunk: Literal["fl2va", "ref2va"],
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        sol_dense_steps: int,
    ) -> ScheduleFacts:
        _validate_sol_dense_steps(sol_dense_steps)
        task: Task = "fl2va_turbo" if trunk == "fl2va" else "ref2va_turbo"
        overlay = self.pipe.overlay(base.pipe, trunk)
        original = state.get("attention_kwargs")
        state.set(
            "attention_kwargs",
            {
                **(original or {}),
                ATTENTION_KWARG: TURBO_BANK,
                OVERLAY_KWARG: overlay,
            },
        )
        try:
            return base.pipe.denoise(
                task, state, on_step=on_step, cancel=cancel, sol_dense_steps=sol_dense_steps
            )
        finally:
            state.set("attention_kwargs", original)
