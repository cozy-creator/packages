"""H3's shared-QKV Kitchen Sol producers: the attention call of a sparse step.

On a sparse step the producer owns the H3 attention module's whole call: one rowwise fp8
quantization of the hidden states shared by the Q, K and V projections, fed in 4096-row
chunks to Kitchen's `sol_attn_chunked` (per-head RMSNorm and split-half RoPE fused), with the
previous step's sparse statistics reused per site within one request. Dense steps and dense
paths call the site's own processor, served by the producer's Runtime base backend.

The approved SM120 recipe (`kitchen-sol-producer-fp4-fixed-means-shared-qkv`) serves dense
calls with Sage3 fixed-means NVFP4 and also shards over Ulysses: its projection runs on the
local token shard, the exchange gathers whole sequences per head subset, and the rotary
tables are the full ones a hook keeps before Diffusers splits them (`retain_rotary`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from diffusers.hooks.hooks import HookRegistry, ModelHook
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
)
from torch.nn.modules import module as torch_module

from cozy_runtime import lora
from cozy_runtime.author import (
    AttentionLayout,
    AttentionProducer,
    ProducerCall,
    active_attention_layout,
    register_attention,
)
from cozy_runtime.author.kernels import ROWWISE_FP8, PreQuantized

from .glue import FusedMiniMaxH3AttnProcessor, H3AttnProcessor

FAST = "kitchen-sol-producer-fast-shared-qkv"
FP4_LOWMEM = "kitchen-sol-producer-fp4-lowmem-shared-qkv"
FIXED_MEANS = "kitchen-sol-producer-fp4-fixed-means-shared-qkv"
CHUNK = 4096
_ROTARY_HOOK = "cozy_cp_producer_rotary"

Statistics = tuple[int, tuple[int, ...], torch.Tensor | None, torch.Tensor | None]


@dataclass
class RequestState:
    statistics: dict[tuple[object, ...], Statistics] = field(default_factory=dict)
    # One rotary presentation is shared by all blocks of the current step.
    rope: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    parallel_rotary: dict[object, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    replay_inputs: dict[tuple[object, ...], Statistics] = field(default_factory=dict)

    def close(self) -> None:
        self.statistics.clear()
        self.parallel_rotary.clear()
        self.replay_inputs.clear()
        backend = sys.modules.get("comfy_kitchen.backends.cuda")
        cached = getattr(backend, "_ROPE_FAB_CACHE", {})
        if self.rope is not None:
            for key, (reference, _packed) in list(cached.items()):
                if reference() is self.rope[2]:
                    cached.pop(key, None)
        self.rope = None


def rows(value: torch.Tensor | PreQuantized, start: int, stop: int) -> torch.Tensor | PreQuantized:
    """View-only rows: preserve encoded payload/scales, never requantize an activation."""
    if isinstance(value, PreQuantized):
        if len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError("H3 producer requires batch1 activations")
        return PreQuantized(
            value.payload[start:stop], value.scale[start:stop], (1, stop - start, value.shape[-1])
        )
    return value[:, start:stop]


def shared_qkv_refusal(attn: MiniMaxH3Attention) -> str:
    """Only known tensor-preserving hooks may share the positional operand's codes: the turbo
    overlay's LoRA hook reads the original tensor operand and adds into the output.

    WeightResidency wraps forward rather than registering a pre-hook. Calling the original
    modules below keeps its acquire/bind/bake/release boundary intact.
    """
    if torch_module._global_forward_pre_hooks or torch_module._global_forward_hooks:
        return "global-projection-hook"
    for projection in (attn.to_q, attn.to_k, attn.to_v):
        if getattr(projection, "provider", None) != ROWWISE_FP8:
            return "non-rowwise-leaf"
        if projection._forward_pre_hooks:
            return "projection-pre-hook"
        if any(not isinstance(hook, lora.Hook) for hook in projection._forward_hooks.values()):
            return "unknown-projection-hook"
    return ""


def qkv_chunks(
    attn: MiniMaxH3Attention,
    hidden: torch.Tensor | PreQuantized,
    tokens: int,
    quantize: Callable[[Any], PreQuantized],
    *,
    shared: bool = False,
    counts: dict[str, int] | None = None,
) -> Iterator[torch.Tensor]:
    refusal = shared_qkv_refusal(attn) if shared else "disabled"
    for start in range(0, tokens, CHUNK):
        chunk = rows(hidden, start, min(tokens, start + CHUNK))
        operand = None
        outcome = refusal
        if shared and isinstance(chunk, PreQuantized):
            outcome = "already-quantized"
        elif shared and not refusal:
            operand = quantize(chunk)
            outcome = "shared"
        if shared and counts is not None:
            counts[outcome] = counts.get(outcome, 0) + 1
        # Original module calls retain encoded leaves, resident weights and Turbo hooks.
        yield torch.cat(
            [
                projection(chunk) if operand is None else projection(chunk, prequantized=operand)
                for projection in (attn.to_q, attn.to_k, attn.to_v)
            ],
            dim=-1,
        ).squeeze(0)


def rope_presentation(
    rotary: tuple[torch.Tensor, torch.Tensor], tokens: int, state: RequestState
) -> torch.Tensor:
    # Kept while the step's rotary tables are the same tensors, which the transformer computes
    # once per layout (`adaln_pruned.StepInvariants`), so every step of a phase reuses one
    # presentation, and Kitchen's table cache keyed on it. The held tensors cannot be reused
    # under another identity. A value key would read the device every call; a shape key would
    # serve one document's angles to another of the same length (a continuation's shifted
    # positions).
    cos, sin = rotary
    held = state.rope
    if held is not None and held[0] is cos and held[1] is sin and held[2].shape[0] == tokens:
        return held[2]
    if cos.shape != sin.shape or cos.shape[-1] % 8:
        raise ValueError("H3 producer requires matching split-half rotary tables")
    width = cos.shape[-1]
    c, s = cos.reshape(-1, width)[:tokens], sin.reshape(-1, width)[:tokens]
    if c.shape[0] != tokens:
        raise ValueError("H3 producer rotary tables do not cover the live document")
    half = width // 2
    freqs = torch.stack((c[:, :half], -s[:, :half], s[:, half:], c[:, half:]), dim=-1)
    freqs = freqs.reshape(tokens, 1, half, 2, 2).float()
    state.rope = cos, sin, freqs
    return freqs


def previous_statistics(
    state: RequestState,
    key: tuple[object, ...],
    step: int,
    geometry: tuple[int, ...],
    *,
    replay_safe: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    # Only the CP recipe changes replay semantics. A later projection or another GPU may fail
    # after this site succeeded; retry its original inputs.
    held = state.replay_inputs.get(key) if replay_safe else None
    if held is not None and held[:2] == (step, geometry):
        return held[2], held[3]
    prior = state.statistics.pop(key, None)
    inputs = prior[2:] if prior is not None and prior[:2] == (step - 1, geometry) else (None, None)
    if replay_safe:
        state.replay_inputs[key] = (step, geometry, *inputs)
    return inputs


def sparse_step(layout: AttentionLayout, path: str) -> bool:
    return layout.step >= layout.dense_until_step and not any(
        path == p or path.startswith(p + ".") for p in layout.dense_paths
    )


def produce(
    name: str,
    call: ProducerCall,
    hidden_states: torch.Tensor | PreQuantized,
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One H3 attention module call at a site `name` is selected for."""
    attn, layout, state = call.module, call.layout, call.state
    parallel, sharded = call.group is not None, PRODUCERS[name].context_parallel
    if parallel and not sharded:
        raise ValueError("This H3 chunked producer requires one GPU")
    if attention_mask is not None:
        raise ValueError("H3 producer serves unmasked noncausal attention only")
    # A recipe that shards over Ulysses keys its statistics and replays per module instance.
    key = (call.component, call.path, id(attn)) if sharded else (call.component, call.path)
    if not sparse_step(layout, call.path):
        state.statistics.pop(key, None)
        state.replay_inputs.pop(key, None)
        return call.original(hidden_states, rotary_emb, attention_mask)
    if rotary_emb is None or hidden_states.shape[0] != 1:
        raise ValueError("H3 producer requires batch1 and rotary tables")
    tokens = layout.live_tokens
    if not parallel and hidden_states.shape[1] != tokens:
        raise ValueError("single-GPU producer expects an unpadded live H3 document")
    if attn.norm_q.eps != attn.norm_k.eps:
        raise ValueError("Kitchen producer requires matching Q/K RMSNorm epsilon")
    shared_counts: dict[str, int] = {}
    if parallel:
        output, reuse = _parallel_produce(call, hidden_states, name, key, shared_counts)
    else:
        freqs = rope_presentation(rotary_emb, tokens, state)
        output, reuse = _produce(
            call,
            key,
            name,
            attn.heads,
            (tokens, attn.heads, layout.protected_prefix),
            freqs,
            lambda: qkv_chunks(
                attn, hidden_states, tokens, call.quantize, shared=True, counts=shared_counts
            ),
        )
    statistics = "previous-step" if reuse else "current-call"
    shared = ",".join(f"{kind}:{count}" for kind, count in sorted(shared_counts.items()))
    call.served(
        f"comfy_kitchen.sol_attn_chunked(int8, stats={statistics}, vmargin=1.1, "
        f"prefix=int8, qkv={shared})"
    )
    return attn.to_out[1](attn.to_out[0](output.flatten(2, 3)))


def _produce(
    call: ProducerCall,
    key: tuple[object, ...],
    name: str,
    heads: int,
    geometry: tuple[int, ...],
    freqs: torch.Tensor,
    chunks: Callable[[], Iterator[torch.Tensor]],
) -> tuple[torch.Tensor, bool]:
    attn, layout, state = call.module, call.layout, call.state
    prior_mean, prior_scale = previous_statistics(
        state, key, layout.step, geometry, replay_safe=PRODUCERS[name].context_parallel
    )
    output, mean, scale = call.kernel.sol_attn_chunked(
        chunks,
        layout.live_tokens,
        heads,
        freqs,
        (attn.norm_q.weight, attn.norm_k.weight),
        kmean=prior_mean,
        vscale=prior_scale,
        tau=1.0,
        scale=128**-0.5,
        rope_eps=attn.norm_q.eps,
        sink_blocks=[0, -(-layout.protected_prefix // 64)],
        sink_q=[0, -(-layout.protected_prefix // 64)],
        token_aug=0,
    )
    state.statistics[key] = (layout.step, geometry, mean, scale)
    return output, prior_mean is not None


def _project_qkv(
    attn: MiniMaxH3Attention,
    hidden: torch.Tensor | PreQuantized,
    tokens: int,
    quantize: Callable[[Any], PreQuantized],
    counts: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a local shard with one shared quantization and no QKV packing copy.

    Original module calls retain residency and Turbo hooks. The LoRA hook already owns its
    fixed-row arithmetic. Other projection semantics keep the chunk path.
    """
    if len(hidden.shape) != 3 or hidden.shape[0] != 1 or hidden.shape[1] != tokens or tokens < 1:
        raise ValueError("H3 CP projection requires the complete nonempty local token shard")
    if not isinstance(hidden, PreQuantized) and not shared_qkv_refusal(attn):
        operand = quantize(hidden)
        if counts is not None:
            counts["shared"] = counts.get("shared", 0) + 1
        query, key, value = (
            projection(hidden, prequantized=operand).reshape(1, tokens, attn.heads, 128)
            for projection in (attn.to_q, attn.to_k, attn.to_v)
        )
        return query, key, value
    packed = None
    offset = 0
    for chunk in qkv_chunks(attn, hidden, tokens, quantize, shared=True, counts=counts):
        if packed is None:
            packed = chunk.new_empty((tokens, chunk.shape[-1]))
        packed[offset : offset + chunk.shape[0]].copy_(chunk)
        offset += chunk.shape[0]
    if packed is None or offset != tokens:
        raise ValueError("H3 CP projection did not cover the local token shard")
    query, key, value = packed.reshape(1, tokens, 3, attn.heads, 128).unbind(2)
    return query, key, value


def _parallel_produce(
    call: ProducerCall,
    hidden: torch.Tensor | PreQuantized,
    name: str,
    key: tuple[object, ...],
    counts: dict[str, int],
) -> tuple[torch.Tensor, bool]:
    attn, layout, state = call.module, call.layout, call.state
    degree, rank = dist.get_world_size(call.group), dist.get_rank(call.group)
    tokens = layout.live_tokens
    expected = tokens // degree + int(rank < tokens % degree)
    if hidden.shape[1] != expected or attn.heads % degree:
        raise ValueError("H3 producer requires the actual unequal token shard and divisible heads")
    rotary = state.parallel_rotary.get(getattr(attn, "_cozy_producer_rotary_key", None))
    if rotary is None or rotary[0].shape[0] != tokens:
        raise ValueError("H3 CP producer requires this component's full pre-split rotary tables")
    freqs = rope_presentation(rotary, tokens, state)
    query, key_tensor, value = _project_qkv(attn, hidden, expected, call.quantize, counts)
    heads = attn.heads // degree
    reuse = False

    def compute(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        nonlocal reuse

        def chunks() -> Iterator[torch.Tensor]:
            for start in range(0, tokens, CHUNK):
                yield torch.cat(
                    [x[:, start : start + CHUNK].flatten(2, 3) for x in (q, k, v)], dim=-1
                ).squeeze(0)

        output, reuse = _produce(
            call,
            key,
            name,
            heads,
            (tokens, heads, layout.protected_prefix, degree, rank),
            freqs,
            chunks,
        )
        return output

    return call.exchange(query, key_tensor, value, compute), reuse


def validate(module: object) -> None:
    if not isinstance(module, MiniMaxH3Attention) or module.fused_projections:
        raise ValueError("Kitchen producer requires unfused MiniMaxH3Attention projection modules")
    if module.head_dim != 128 or module.to_q.out_features != module.heads * 128:
        raise ValueError("Kitchen producer requires H3 head dimension128")
    original = getattr(module.processor, "original", module.processor)
    if type(original) not in (MiniMaxH3AttnProcessor, H3AttnProcessor, FusedMiniMaxH3AttnProcessor):
        raise ValueError("Kitchen producer does not replace an unknown H3 processor")
    for projection in (module.to_q, module.to_k, module.to_v):
        # Weight-plane hooks are Runtime custody; author forward hooks must support replay and
        # chunks. The turbo overlay's LoRA hook adds a row-local update.
        for hook in projection._forward_hooks.values():
            if not isinstance(hook, lora.Hook):
                raise ValueError("Kitchen producer cannot replay an unknown projection hook")


def _register(name: str, base: str, *, context_parallel: bool = False) -> AttentionProducer:
    return register_attention(
        AttentionProducer(
            name,
            base,
            partial(produce, name),
            validate,
            requires=("sol_attn_chunked",),
            state=RequestState,
            context_parallel=context_parallel,
        )
    )


PRODUCERS = {
    producer.name: producer
    for producer in (
        _register(FIXED_MEANS, "kitchen-sol-fp4-fixed-means", context_parallel=True),
        _register(FP4_LOWMEM, "kitchen-sol-fp4-lowmem"),
        _register(FAST, "kitchen-sol"),
    )
}
_BACKENDS = {"_cozy_" + name.replace("-", "_"): name for name in PRODUCERS}


class _RetainRotary(ModelHook):  # type: ignore[misc]  # untyped Diffusers hook API
    """Keeps a sharded component's full rotary tables for the CP recipe's sparse calls, before
    Diffusers' split hook (registered after this one) shards them. Returns them unchanged."""

    def __init__(self, component: Any, modules: tuple[Any, ...], key: object) -> None:
        super().__init__()
        self.component, self.modules, self.key = component, modules, key

    def post_forward(self, module: object, output: object) -> object:
        if getattr(self.component, "_parallel_config", None) is None:
            return output  # one GPU: the producer reads the rotary its call receives
        state, layout = PRODUCERS[FIXED_MEANS].request_state(), active_attention_layout()
        # Kept at every step: the transformer computes the rotary once per layout, on a dense
        # step, and the sparse steps that read it reuse that computation.
        if state is None or layout is None:
            return output
        backends = (getattr(m.processor, "_attention_backend", None) for m in self.modules)
        names = (_BACKENDS.get(str(getattr(backend, "value", backend))) for backend in backends)
        if any(name is not None and PRODUCERS[name].context_parallel for name in names):
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or output[0].shape[0] != layout.live_tokens
            ):
                raise ValueError("H3 full rotary presentation changed before context splitting")
            state.parallel_rotary[self.key] = output
        return output


def retain_rotary(component: object) -> None:
    """Install `_RetainRotary` on an H3 transformer's rotary module, once. Part of the
    execution plan, so it precedes the Ulysses split hooks Runtime installs later."""
    if not isinstance(component, MiniMaxH3Transformer3DModel):
        return
    registry = HookRegistry.check_if_exists_or_initialize(component.rope)
    if registry.get_hook(_ROTARY_HOOK) is not None:
        return
    key = object()
    modules = tuple(block.attn for block in component.transformer_blocks)
    for attention in modules:
        attention._cozy_producer_rotary_key = key
    registry.register_hook(_RetainRotary(component, modules, key), _ROTARY_HOOK)
