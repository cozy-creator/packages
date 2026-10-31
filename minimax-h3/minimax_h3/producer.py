"""H3's shared-QKV Kitchen Sol producers: the attention call of a sparse step.

On a sparse step the producer owns the H3 attention module's whole call: the Q, K and V
projections of each 4096-row chunk (their leaves quantize the rows once between them), fed to
Kitchen's `sol_attn_chunked` (per-head RMSNorm and split-half RoPE
fused), with the previous step's key means reused per site within one request and the value
scales of the call's own values: a scale from the previous step saturates Kitchen's int8 values
wherever they grew (every site of H3 Turbo's steps 5-7, up to 44% of a block's channels). V is
projected once, before Kitchen's pass, for its scale, and that pass reads it back
(`project_values`). Dense steps, dense paths and exact paths call the site's own processor,
served by the producer's Runtime base backend.

The approved SM120 recipe (`kitchen-sol-producer-fp4-fixed-means-shared-qkv`) serves dense
calls with Sage3 fixed-means NVFP4 and also shards over Ulysses: its projection runs on the
local token shard, the exchange gathers whole sequences per head subset, and the rotary
tables are the full ones a hook keeps before Diffusers splits them (`retain_rotary`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
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
    Cut,
    ProducerCall,
    ProducerSite,
    register_attention,
    split,
)
from cozy_runtime.author.kernels import ROWWISE_FP8, PreQuantized, quantize_once
from cozy_runtime.rows import Rows
from cozy_runtime.rows import rows as columns

from .glue import KEY_MEAN, FusedMiniMaxH3AttnProcessor, H3AttnProcessor, KeyMean

FAST = "kitchen-sol-producer-fast-shared-qkv"
FP4_LOWMEM = "kitchen-sol-producer-fp4-lowmem-shared-qkv"
FIXED_MEANS = "kitchen-sol-producer-fp4-fixed-means-shared-qkv"
CHUNK = 4096
#: A channel's absmax in one reduction pass (`|x|` is never written): the max-abs norm.
INF = float("inf")
#: Kitchen's value-scale margin (`sol_attn_chunked`'s `vscale_next`): absmax / 127 x 1.1.
VMARGIN = 1.1
_ROTARY_HOOK = "cozy_cp_producer_rotary"

Statistics = tuple[int, tuple[int, ...], torch.Tensor | None, torch.Tensor | None]


@dataclass
class RequestState:
    statistics: dict[tuple[object, ...], Statistics] = field(default_factory=dict)
    # One rotary presentation is shared by all blocks of the current step.
    rope: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    parallel_rotary: dict[object, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    replay_inputs: dict[tuple[object, ...], Statistics] = field(default_factory=dict)
    # The attention modules whose `Carry` slot a call of the current denoise loop has written.
    carried: set[int] = field(default_factory=set)

    def close(self) -> None:
        self.statistics.clear()
        self.carried.clear()
        self.parallel_rotary.clear()
        self.replay_inputs.clear()
        backend = sys.modules.get("comfy_kitchen.backends.cuda")
        cached = getattr(backend, "_ROPE_FAB_CACHE", {})
        if self.rope is not None:
            for key, (reference, _packed) in list(cached.items()):
                if reference() is self.rope[2]:
                    cached.pop(key, None)
        self.rope = None


@dataclass(frozen=True, slots=True)
class Carry:
    """One `denoise_loop` step's Kitchen key means: a carried F32 [H, 128] slot per site, by
    attention module. A rolling step's calls read the previous step's mean from their slot; every
    sparse call leaves its own there, and the last dense step's calls seed it (`seeding`; a step
    loop seeds `RequestState.statistics` the same way), so the first sparse step reads one too
    instead of measuring its keys in a second producer pass. Eager
    reuse decides on `layout.step` in Python, which a recorded step would freeze; the step's phase
    decides instead (`rolling`, `seeding`)."""

    slots: Mapping[int, torch.Tensor]
    rolling: bool
    seeding: bool = False


_CARRY: ContextVar[Carry | None] = ContextVar("minimax_h3_kitchen_carry", default=None)


@contextmanager
def carrying(carry: Carry) -> Iterator[None]:
    token = _CARRY.set(carry)
    try:
        yield
    finally:
        _CARRY.reset(token)


def present(
    sites: Sequence[ProducerSite], rotary: tuple[torch.Tensor, torch.Tensor], tokens: int
) -> None:
    """Before a `denoise_loop`: the rotary presentation these sites' sparse calls read, and
    Kitchen's packed table of it, made eagerly so no recorded step allocates them from its pool.
    `RequestState.close` forgets the table."""
    states = {
        id(state): state
        for site in sites
        if isinstance(state := site.producer.request_state(), RequestState)
    }
    if not states:
        return
    for state in states.values():
        state.carried.clear()  # this loop's slots start unwritten
    first, *others = states.values()
    presentation = rope_presentation(rotary, tokens, first)
    for state in others:
        state.rope = (rotary[0], rotary[1], presentation)
    build = getattr(sys.modules.get("comfy_kitchen.backends.cuda"), "_packed_rope_fab", None)
    if callable(build):
        build(presentation, tokens, presentation.shape[-3] * 2)


def rows(value: torch.Tensor | PreQuantized, start: int, stop: int) -> torch.Tensor | PreQuantized:
    """View-only rows: preserve encoded payload/scales, never requantize an activation."""
    if isinstance(value, PreQuantized):
        if len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError("H3 producer requires batch1 activations")
        return PreQuantized(
            value.payload[start:stop], value.scale[start:stop], (1, stop - start, value.shape[-1])
        )
    return value[:, start:stop]


def _whole_shard(attn: MiniMaxH3Attention) -> bool:
    """Whether a CP shard's Q, K and V project whole, with no packing copy: rowwise fp8 leaves
    under no hooks but the turbo overlay's LoRA (it reads the original tensor operand and adds
    into the output). Other leaves keep the chunks' layout (`qkv_chunks`).

    WeightResidency wraps forward rather than registering a pre-hook. Calling the original
    modules below keeps its acquire/bind/bake/release boundary intact.
    """
    if torch_module._global_forward_pre_hooks or torch_module._global_forward_hooks:
        return False
    return all(
        getattr(projection, "provider", None) == ROWWISE_FP8
        and not projection._forward_pre_hooks
        and all(isinstance(hook, lora.Hook) for hook in projection._forward_hooks.values())
        for projection in (attn.to_q, attn.to_k, attn.to_v)
    )


def qkv_chunks(
    attn: MiniMaxH3Attention,
    hidden: torch.Tensor | PreQuantized,
    tokens: int,
    values: torch.Tensor | None = None,
) -> Iterator[torch.Tensor]:
    """Kitchen's `[rows, 3 x H x 128]` chunks: each chunk's Q, K and V projections (their leaves
    quantize the rows once between them). Where `values` holds this call's V (`project_values`),
    V is read from it."""
    for start in range(0, tokens, CHUNK):
        stop = min(tokens, start + CHUNK)
        chunk = rows(hidden, start, stop)
        # Original module calls retain encoded leaves, resident weights and Turbo hooks.
        parts = [attn.to_q(chunk), attn.to_k(chunk)]
        parts.append(attn.to_v(chunk) if values is None else values[:, start:stop])
        yield torch.cat(parts, dim=-1).squeeze(0)


def project_values(
    attn: MiniMaxH3Attention, hidden: torch.Tensor | PreQuantized, tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This call's V, projected by chunk as `qkv_chunks` would, and its per-(head, channel) absmax
    over the live document, F32 [H, 128]: Kitchen's value scale is the call's own, so V is
    projected here, before Kitchen's pass, which reads it back instead of projecting it again.
    It is one block the size of Kitchen's output, freed when the pass ends, just before Kitchen
    allocates that output, which reuses it: the call's peak does not grow (chunks would leave
    holes the output cannot use, and a 4090 then paged more than it saved)."""
    values: torch.Tensor | None = None
    found: torch.Tensor | None = None
    for start in range(0, tokens, CHUNK):
        stop = min(tokens, start + CHUNK)
        value = attn.to_v(rows(hidden, start, stop))
        if values is None:  # one block, the size of Kitchen's output: freed before it, reused by it
            values = value.new_empty((1, tokens, value.shape[-1]))
        values[:, start:stop].copy_(value)
        peak = torch.linalg.vector_norm(value.reshape(-1, value.shape[-1]), ord=INF, dim=0)
        found = peak if found is None else torch.maximum(found, peak)
    if values is None or found is None:
        raise ValueError("H3 producer has no live rows to scale")
    return values, found.float().reshape(attn.heads, 128)


def value_scale(absmax: torch.Tensor) -> torch.Tensor:
    """Kitchen's value scale for these values: no channel saturates its int8 range."""
    return (absmax.float() / 127.0 * VMARGIN).clamp_min(1e-8).contiguous()


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


def producing(layout: AttentionLayout, path: str) -> bool:
    """A site this producer serves on sparse steps: on neither a dense path nor an exact one. An
    exact path's dense rows (its protected prefix) are BF16 on the Runtime route only."""
    return not any(
        path == p or path.startswith(p + ".") for p in (*layout.dense_paths, *layout.exact_paths)
    )


def sparse_step(layout: AttentionLayout, path: str) -> bool:
    """A call this producer serves: a sparse step at a site it produces for."""
    return layout.step >= layout.dense_until_step and producing(layout, path)


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
        carry, seed = _CARRY.get(), None
        if not parallel and producing(layout, call.path):
            if carry is not None:  # `denoise_loop`: the phase says which step seeds
                seed = KeyMean(carry.slots[id(attn)]) if carry.seeding else None
            elif layout.step == layout.dense_until_step - 1:  # a step loop: the last dense step
                device = (hidden_states.payload if isinstance(hidden_states, PreQuantized)
                          else hidden_states).device  # fmt: skip
                seed = KeyMean(torch.empty(attn.heads, 128, dtype=torch.float32, device=device))
        token = KEY_MEAN.set(seed)
        try:
            output = call.original(hidden_states, rotary_emb, attention_mask)
        finally:
            KEY_MEAN.reset(token)
        if seed is not None and seed.heads >= attn.heads:
            if carry is not None:
                state.carried.add(id(attn))
            else:  # what `previous_statistics` reads at the first sparse step; its values stay its own
                geometry = (layout.live_tokens, attn.heads, layout.protected_prefix)
                state.statistics[key] = (layout.step, geometry, seed.slot, None)
        return output
    if rotary_emb is None or hidden_states.shape[0] != 1:
        raise ValueError("H3 producer requires batch1 and rotary tables")
    tokens = layout.live_tokens
    if not parallel and hidden_states.shape[1] != tokens:
        raise ValueError("single-GPU producer expects an unpadded live H3 document")
    if attn.norm_q.eps != attn.norm_k.eps:
        raise ValueError("Kitchen producer requires matching Q/K RMSNorm epsilon")
    if parallel:
        output, reuse = _parallel_produce(call, hidden_states, name, key)
    else:
        freqs = rope_presentation(rotary_emb, tokens, state)
        means = _Means(call, key, name, (tokens, attn.heads, layout.protected_prefix))
        heads = _Heads.of(call, means, hidden_states, freqs)
        if heads is None:
            output, mean, _ = _produce(
                call,
                attn.heads,
                freqs,
                lambda values: qkv_chunks(attn, hidden_states, tokens, values),
                lambda: project_values(attn, hidden_states, tokens),
                means.prior,
            )
            means.keep(slice(0, attn.heads), mean)
        else:
            # Kitchen's call holds the whole document's K/V codes and its whole output: an exact
            # split by query heads, which the Runtime sizes beside the DiT's weights.
            output = split(heads, heads, Cut(2))
        means.close()
        reuse = means.prior is not None
    statistics = "kmean=previous-step,vscale=current-call" if reuse else "current-call"
    call.served(
        f"comfy_kitchen.sol_attn_chunked(int8, stats={statistics}, vmargin=1.1, "
        f"prefix=int8, qkv=one-operand{',v=once' if reuse and not parallel else ''})"
    )
    out = attn.to_out[1](split(attn.to_out[0], output.flatten(2, 3), Cut(1)))
    del output
    return out


class _Means:
    """A one-GPU call's Kitchen key means, F32 [heads, 128]: the previous step's (`prior`, or None
    where the call measures its keys) and this call's, which each head window writes its rows of
    (`keep`). `close` leaves them where the next step reads them, the module's `Carry` slot in
    `denoise_loop`, else `RequestState.statistics` under the call's key, after the whole call:
    a split the Runtime retries in smaller windows reads the same prior."""

    __slots__ = ("call", "carry", "geometry", "key", "mean", "prior")

    def __init__(
        self, call: ProducerCall, key: tuple[object, ...], name: str, geometry: tuple[int, ...]
    ) -> None:
        attn, state = call.module, call.state
        self.call, self.key, self.geometry, self.carry = call, key, geometry, _CARRY.get()
        self.mean: torch.Tensor | None = None
        if self.carry is None:
            self.prior, _ = previous_statistics(
                state, key, call.layout.step, geometry, replay_safe=PRODUCERS[name].context_parallel
            )
        else:  # a rolling step reads the slot the previous step's call wrote
            rolling = self.carry.rolling and id(attn) in state.carried
            self.prior = self.carry.slots[id(attn)] if rolling else None

    def keep(self, window: slice, mean: torch.Tensor) -> None:
        if self.mean is None:
            self.mean = mean.new_empty((self.call.module.heads, *mean.shape[1:]))
        self.mean[window].copy_(mean)

    def close(self) -> None:
        attn, state = self.call.module, self.call.state
        if self.carry is None:
            state.statistics[self.key] = (self.call.layout.step, self.geometry, self.mean, None)
        else:
            self.carry.slots[id(attn)].copy_(self.mean)
            state.carried.add(id(attn))


class _Heads:
    """One sparse call as `split` cuts it by query heads: its shape is the call's [batch, tokens,
    heads, 128] output, a window of it a slice of heads, and calling it on a window runs Kitchen
    over those heads alone (`_produce`), their Q, K and V projected from their weight rows
    (`cozy_runtime.rows`). Exact: Kitchen's norms, RoPE, key means, value scales and routing are
    per head and its sink ranges per token. Called on itself (`split` running whole), all heads."""

    __slots__ = ("call", "device", "dtype", "freqs", "hidden", "means", "projections", "shape")

    def __init__(
        self,
        call: ProducerCall,
        means: _Means,
        hidden: torch.Tensor | PreQuantized,
        freqs: torch.Tensor,
        projections: tuple[Rows, Rows, Rows],
    ) -> None:
        self.call, self.means, self.hidden, self.freqs = call, means, hidden, freqs
        self.projections = projections
        attn = call.module
        self.shape = (1, call.layout.live_tokens, attn.heads, 128)
        quantized = isinstance(hidden, PreQuantized)
        self.dtype = torch.bfloat16 if quantized else hidden.dtype
        self.device = hidden.payload.device if quantized else hidden.device

    @classmethod
    def of(
        cls,
        call: ProducerCall,
        means: _Means,
        hidden: torch.Tensor | PreQuantized,
        freqs: torch.Tensor,
    ) -> _Heads | None:
        """The call cut by heads, or None where a projection does not cut by rows."""
        attn = call.module
        found = tuple(columns(m) for m in (attn.to_q, attn.to_k, attn.to_v))
        if any(r is None for r in found):
            return None
        return cls(call, means, hidden, freqs, found)  # type: ignore[arg-type]

    def dim(self) -> int:
        return len(self.shape)

    def element_size(self) -> int:
        return int(self.dtype.itemsize)

    def __getitem__(self, index: tuple[slice, ...]) -> slice:
        return index[2]

    def __call__(self, heads: _Heads | slice) -> torch.Tensor:
        attn, layout = self.call.module, self.call.layout
        window = slice(0, attn.heads) if heads is self else heads
        assert isinstance(window, slice)
        start, stop = window.start * 128, window.stop * 128
        tokens = layout.live_tokens
        q, k, v = self.projections

        def chunks(values: torch.Tensor | None) -> Iterator[torch.Tensor]:
            for at in range(0, tokens, CHUNK):
                end = min(tokens, at + CHUNK)
                chunk = rows(self.hidden, at, end)
                with quantize_once(chunk):
                    parts = [q(chunk, start, stop), k(chunk, start, stop)]
                    parts.append(v(chunk, start, stop) if values is None else values[:, at:end])
                yield torch.cat(parts, dim=-1).squeeze(0)

        def values() -> tuple[torch.Tensor, torch.Tensor]:
            # One block, as `project_values`: freed before Kitchen's output, which reuses it.
            projected: torch.Tensor | None = None
            found: torch.Tensor | None = None
            for at in range(0, tokens, CHUNK):
                end = min(tokens, at + CHUNK)
                value = v(rows(self.hidden, at, end), start, stop)
                if projected is None:
                    projected = value.new_empty((1, tokens, value.shape[-1]))
                projected[:, at:end].copy_(value)
                peak = torch.linalg.vector_norm(value.reshape(-1, value.shape[-1]), ord=INF, dim=0)
                found = peak if found is None else torch.maximum(found, peak)
            if projected is None or found is None:
                raise ValueError("H3 producer has no live rows to scale")
            return projected, found.float().reshape(window.stop - window.start, 128)

        count, prior = window.stop - window.start, self.means.prior
        output, mean, _ = _produce(
            self.call,
            count,
            self.freqs,
            chunks,
            values,
            None if prior is None else prior[window],
        )
        self.means.keep(window, mean)
        return output.reshape(1, tokens, count, 128)


def _produce(
    call: ProducerCall,
    heads: int,
    freqs: torch.Tensor,
    chunks: Callable[[torch.Tensor | None], Iterator[torch.Tensor]],
    values: Callable[[], tuple[torch.Tensor | None, torch.Tensor]],
    prior_mean: torch.Tensor | None,
    prior_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kitchen's call over `chunks(projected)`: its output, key mean and value scale. Where the
    previous step's key mean carries, `values()` gives the call's own V absmax for its value
    scale, and the V it projected for it (or None), which the one pass Kitchen then makes reads
    back; the pass's end drops it."""
    attn, layout = call.module, call.layout
    held: list[torch.Tensor | None] = []
    if prior_mean is not None:  # the key mean carries; the values are this call's own
        projected, absmax = values()
        prior_scale = value_scale(absmax)
        held.append(projected)
        del projected
    return call.kernel.sol_attn_chunked(  # type: ignore[no-any-return]
        # Kitchen passes once with both statistics, twice (measuring first) without.
        lambda: chunks(held.pop() if held else None),
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


def _project_qkv(
    attn: MiniMaxH3Attention, hidden: torch.Tensor | PreQuantized, tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A local shard's Q, K and V from one quantization: whole where `_whole_shard`, else by
    chunks packed into one buffer. Original module calls retain residency and Turbo hooks."""
    if len(hidden.shape) != 3 or hidden.shape[0] != 1 or hidden.shape[1] != tokens or tokens < 1:
        raise ValueError("H3 CP projection requires the complete nonempty local token shard")
    if not isinstance(hidden, PreQuantized) and _whole_shard(attn):
        query, key, value = (
            projection(hidden).reshape(1, tokens, attn.heads, 128)
            for projection in (attn.to_q, attn.to_k, attn.to_v)
        )
        return query, key, value
    packed = None
    offset = 0
    for chunk in qkv_chunks(attn, hidden, tokens):
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
    query, key_tensor, value = _project_qkv(attn, hidden, expected)
    reuse, groups = False, 0

    def compute(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Where a GPU's room is short, the exchange hands this call its heads in groups, one
        # `compute` each (`internal.parallel.ulysses.exchange`): each group's statistics are its own.
        nonlocal reuse, groups
        heads, group = q.shape[2], groups
        groups += 1

        def chunks(_: object) -> Iterator[torch.Tensor]:
            for start in range(0, tokens, CHUNK):
                yield torch.cat(
                    [x[:, start : start + CHUNK].flatten(2, 3) for x in (q, k, v)], dim=-1
                ).squeeze(0)

        statistics = (*key, group)
        geometry = (tokens, heads, layout.protected_prefix, degree, rank, group)
        prior_mean, prior_scale = previous_statistics(
            state, statistics, layout.step, geometry, replay_safe=PRODUCERS[name].context_parallel
        )
        output, mean, scale = _produce(
            call,
            heads,
            freqs,
            chunks,
            lambda: (None, torch.linalg.vector_norm(v[0], ord=INF, dim=0).float()),
            prior_mean,
            prior_scale,
        )
        state.statistics[statistics] = (layout.step, geometry, mean, scale)
        reuse = prior_mean is not None
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


class _RetainRotary(ModelHook):  # type: ignore[misc]  # untyped Diffusers hook API
    """Keeps a sharded component's full rotary tables for the CP recipe's sparse calls, before
    Diffusers' split hook (registered after this one) shards them. Returns them unchanged.
    Kept whenever the component runs context-parallel, whatever serves now: the transformer
    computes the rotary once per layout, on a dense step, and the recipe can land after it
    (a kernel opening during the request), so a step-0 fallback must not leave it without
    them. The producer checks their length."""

    def __init__(self, component: Any, key: object) -> None:
        super().__init__()
        self.component, self.key = component, key

    def post_forward(self, module: object, output: object) -> object:
        if getattr(self.component, "_parallel_config", None) is None:
            return output  # one GPU: the producer reads the rotary its call receives
        state = PRODUCERS[FIXED_MEANS].request_state()
        if state is not None and isinstance(output, tuple) and len(output) == 2:
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
    for block in component.transformer_blocks:
        block.attn._cozy_producer_rotary_key = key
    registry.register_hook(_RetainRotary(component, key), _ROTARY_HOOK)
