"""The MiniMax-H3 family's fused DiT glue (h3a-015), shared by minimax-h3, minimax-h3-image and
p-h3-image: every block of Diffusers' `MiniMaxH3Transformer3DModel` runs the Runtime's row kernels
(`cozy_runtime.accel.glue`, `GLUE`) between its GEMMs. `install` substitutes the fused forwards on
the filled transformers as the model's execution plan. Shapes are the ones the eager composition
already produces, so nothing moves a parameter, a buffer or a state-dict key (re-censused after
the swap).

What stays here is H3's: its widths and rounding (`GLUE`), its block topology, the attention cut
by query heads and the block's tail by tokens (the Runtime sizes both to the memory the DiT's
weights leave), and the post-RoPE key mean the sparse producer's first step reads (`KEY_MEAN`).

Numerics are the EAGER composition's: every intermediate eager rounds to bf16 is rounded too, and
the fp8 epilogue is the rowwise leaf's own quantizer on the same registers. Measured on sm89
against torch 2.13 eager: the modulate, gate-add, SwiGLU and RoPE chains are bit-identical; the
RMSNorm differs from torch's fused kernel by one bf16 ulp on ~4e-6 of elements (fp32 reduction
order of the row variance). An fp8 epilogue is chosen per site from what the fill installed: a
site whose every consumer is an `fp8.rowwise` encoded leaf hands it a `PreQuantized` operand; any
other consumer gets bf16.

Not installable (eager serves, the code is reported): no CUDA sm80+, no Triton, kernels absent or
failed, an unmatched Diffusers topology. A build that does not load, or a failure after
substitution began, refuses the construction: there is no eager left.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch
from diffusers.models.activations import SwiGLU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention, MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
    _apply_rotary_emb,
)

from cozy_runtime.accel import glue as fused
from cozy_runtime.accel import plain_forward
from cozy_runtime.author import Cut, split
from cozy_runtime.accel.kernels import (
    int8_rows,
    KernelsUnavailable,
    Launch,
    Plan,
    PreQuantized,
    quantize_once,
    ROWWISE_FP8,
    TritonEntry,
    TritonKernels,
)
from cozy_runtime.accel.rows import Rows, rows

try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    if TYPE_CHECKING:
        raise
    TRITON_ABSENT = f"triton does not import ({_absent})"  # type: ignore[unreachable]  # checkers import triton
else:
    TRITON_ABSENT = ""


@dataclass(slots=True)
class KeyMean:
    """Where an attention call leaves its post-RoPE key mean, F32 [heads, 128] over its tokens
    (`producer`: the last dense step seeds the first sparse step's Kitchen statistics), and how
    many heads it has written."""

    slot: torch.Tensor
    heads: int = 0


KEY_MEAN: ContextVar[KeyMean | None] = ContextVar("minimax_h3_key_mean", default=None)


class Owned[T]:
    """An argument whose only reference the callee takes (`take`): a caller's frame keeps
    every argument it passes until the call returns, so a tensor it passes plainly outlives its
    last use inside."""

    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value: T | None = value


def take(value: Hidden | Owned[Hidden]) -> Hidden:
    """`value`, or the tensor an `Owned` passed, which then holds it no more."""
    if not isinstance(value, Owned):
        return value
    taken, value.value = value.value, None
    return taken  # type: ignore[return-value]  # an Owned is taken once: its one callee empties it
#: The widths compiled: a transformer with others is not installable.
HIDDEN, FFN, HEAD_DIM, ROTARY = 5376, 14336, 128, 96
#: The DiT's row kernels: `torch.nn.RMSNorm`s modulated by `1 + scale` and `shift`, and RoPE on 96
#: of 128 channels with the tables cast to bf16 first, as eager computes them.
GLUE = fused.Glue(
    "minimax-h3-glue", width=HIDDEN, ffn=FFN, head_dim=HEAD_DIM, rotary=ROTARY, shift=True,
    rms_rounding="once", rope_rounding="product",
)  # fmt: skip
#: A tail window's peak, in its rows of hidden (`_Tail`): its normed rows, the FFN's [value | gate]
#: projection and SwiGLU product, its quantized operands and its down projection. Whole, 13.3 GB
#: at 15 s against 1.17 GB of hidden (run 25405).
_TAIL_HOLDS = 12
#: A head window's peak, in windows beside the whole output: Q, K and V, the attention's own K/V
#: copies and its output. 6.2 measured on a capped 4090 (run 29255).
HEAD_HOLDS = 7
_KEY_TOKENS = 64
_KEY_WARPS = 4
#: The codes eager serves past; any other is breakage and refuses.
NOT_INSTALLABLE = frozenset(
    {
        "fusion_kernels_absent",
        "fusion_kernels_failed",
        "fusion_contract_unsupported",
        "fusion_block_shape",
        "fusion_device_incapable",
    }
)



class GlueRefusal(RuntimeError):
    """The fused plan cannot install as matched; `code` says why."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


# ------- kernel source (Triton hashes its line: a moved one rebuilds its set everywhere)
if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]
    def _key_sum_kernel(  # type: ignore[no-untyped-def]
        key_ptr,
        out_ptr,
        tokens,
        stride_t,
        HEAD_DIM: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # One head's F32 key sum over its tokens, tile after tile in token order: the same additions
        # whatever heads the call holds. torch's sum regroups them by its output count, so a head
        # window's key mean differed from the whole call's (xs-089 diagnostic run 23636).
        head = tl.program_id(0)
        chan = tl.arange(0, HEAD_DIM)
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        for start in range(0, tokens, BLOCK_T):
            row = start + tl.arange(0, BLOCK_T)
            offset = row[:, None].to(tl.int64) * stride_t + head * HEAD_DIM + chan[None, :]
            tile = tl.load(key_ptr + offset, mask=(row < tokens)[:, None], other=0.0)
            acc += tl.sum(tile.to(tl.float32), axis=0)
        tl.store(out_ptr + head * HEAD_DIM + chan, acc)


#: An activation, or a rowwise leaf's quantized operand standing for it (the fp8 epilogue).
type Hidden = torch.Tensor | PreQuantized


class Pipeline(Protocol):
    """What `install` reads of a pipeline: its components by name."""

    @property
    def components(self) -> Mapping[str, object]: ...


def _eps(norm: torch.nn.RMSNorm) -> float:
    """`norm`'s epsilon: Diffusers builds every H3 norm with its config's."""
    return cast(float, norm.eps)


def _blocks(transformer: MiniMaxH3Transformer3DModel) -> Iterable[MiniMaxH3TransformerBlock]:
    """Its `transformer_blocks`, which Diffusers builds of `MiniMaxH3TransformerBlock`s."""
    return cast("Iterable[MiniMaxH3TransformerBlock]", transformer.transformer_blocks)


def _key_entries() -> tuple[TritonEntry, ...]:
    return (
        TritonEntry(
            _key_sum_kernel,
            {
                "key_ptr": "*bf16",
                "out_ptr": "*fp32",
                "tokens": "i32",
                "stride_t": "i32",
                "HEAD_DIM": "constexpr",
                "BLOCK_T": "constexpr",
            },
            {"HEAD_DIM": HEAD_DIM, "BLOCK_T": _KEY_TOKENS},
            ("key_ptr", "out_ptr", "stride_t"),
            _KEY_WARPS,
        ),
    )


#: The key sums (`key_sums`), on every sm80+ card.
KEYS = TritonKernels("minimax-h3-key-sums", () if TRITON_ABSENT else _key_entries())
CONTRACT = "h3-glue-call-and-topology/3"


def key_sums(launch: Launch, key: torch.Tensor) -> torch.Tensor:
    """F32 [heads, head_dim] sums over tokens of `key`'s [1, tokens, heads, head_dim]."""
    if key.dtype is not torch.bfloat16 or not key.is_cuda:
        raise GlueRefusal("fusion_block_shape", f"key sums take bf16 CUDA keys, not {key.dtype}")
    _, tokens, heads, head_dim = (int(d) for d in key.shape)
    flat = key[0].reshape(tokens, heads * head_dim)
    flat = flat if flat.stride(-1) == 1 else flat.contiguous()
    out = torch.empty(heads, head_dim, dtype=torch.float32, device=key.device)
    launch(_key_sum_kernel, (heads, 1, 1), flat, out, tokens, flat.stride(0), head_dim, _KEY_TOKENS)
    return out


def _swiglu(launch: Launch, both: torch.Tensor, *, fp8: bool = False) -> torch.Tensor | PreQuantized:
    """`value * silu(gate)` over Diffusers' `SwiGLU` projection, its [value | gate] halves."""
    half = int(both.shape[-1]) // 2
    return fused.swiglu(launch, GLUE, both[..., half:], both[..., :half], fp8=fp8)


# ------------------------------------------------------------------ fused modules


class FusedSwiGLU(SwiGLU):
    _fusion_fp8: bool = False
    _glue: Launch

    # An fp8 down projection takes the quantized operand in place of the bf16 rows.
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor | PreQuantized:  # type: ignore[override]
        return _swiglu(self._glue, self.proj(hidden_states), fp8=self._fusion_fp8)


class H3AttnProcessor(MiniMaxH3AttnProcessor):
    """Diffusers' H3 attention, declared as an exact split by query heads: each window computes
    its own rows of the projections (`cozy_runtime.rows`), per-head norms and RoPE, and the out
    projection runs by tokens. The Runtime sizes the windows against the memory the DiT's
    weights leave: whole on a card that holds both, in windows on a smaller, shared or capped
    one. A projection that does not cut by rows, a mask or a sequence-parallel exchange runs
    whole."""

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: Hidden | Owned[Hidden],
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = take(hidden_states)
        heads = _Heads.of(self, attn, hidden_states, rotary_emb, attention_mask)
        if heads is None:
            attended: torch.Tensor = self.whole(attn, hidden_states, rotary_emb, attention_mask)
            whole_out: torch.Tensor = attn.to_out[1](attn.to_out[0](attended.flatten(2, 3)))
            return whole_out
        # The windows read the q, k and v leaves' int8 operand where all three take one; the bf16
        # rows then have no reader and go here, before any window runs (var-030).
        heads = heads.quantized()
        del hidden_states
        attended = split(heads, heads, Cut(2), holds=HEAD_HOLDS)
        del heads  # an owned input goes before the out projection's output comes
        out: torch.Tensor = attn.to_out[1](split(attn.to_out[0], attended.flatten(2, 3), Cut(1), holds=2))
        del attended
        return out

    def whole(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: Hidden,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query, key, value = self.qkv(attn, hidden_states, rotary_emb)
        seed_key_mean(getattr(self, "_keys", None), key, slice(None))
        return self.attend(query, key, value, attention_mask)

    def attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The attention of `query`'s heads, in its dtype."""
        dtype = query.dtype
        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        ).to(dtype)

    def qkv(
        self, attn: MiniMaxH3Attention, hidden_states: Hidden, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key = attn.to_q(hidden_states), attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query, key, value = (
            t.unflatten(-1, (attn.heads, -1)) for t in (query, key, value)
        )
        return (*self.rope(attn, query, key, rotary_emb), value)

    def rope(
        self, attn: MiniMaxH3Attention, query: torch.Tensor, key: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key = attn.norm_q(query), attn.norm_k(key)
        if rotary_emb is not None:
            query, key = _apply_rotary_emb(query, *rotary_emb), _apply_rotary_emb(key, *rotary_emb)
        return query, key


class FusedMiniMaxH3AttnProcessor(H3AttnProcessor):
    _glue: Launch
    _keys: Launch

    def qkv(
        self, attn: MiniMaxH3Attention, hidden_states: Hidden, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query: torch.Tensor = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key: torch.Tensor = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value: torch.Tensor = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        return (*self.rope(attn, query, key, rotary_emb), value)

    def rope(
        self, attn: MiniMaxH3Attention, query: torch.Tensor, key: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rotary_emb is None:
            return attn.norm_q(query), attn.norm_k(key)
        cos, sin = rotary_emb
        query = fused.head_norm_rope(
            self._glue, GLUE, query, attn.norm_q.weight, _eps(attn.norm_q), cos, sin
        )
        key = fused.head_norm_rope(self._glue, GLUE, key, attn.norm_k.weight, _eps(attn.norm_k), cos, sin)
        return query, key


def seed_key_mean(launch: Launch | None, key: torch.Tensor, heads: slice[int | None]) -> None:
    """`KEY_MEAN`'s heads from this call's [1, tokens, heads, 128] post-RoPE keys, summed in F32
    without an F32 copy: by each head alone where the glue kernels serve (`key_sums`), so a head
    window seeds the bits the whole call does."""
    target = KEY_MEAN.get()
    if target is not None:
        found = key[0].sum(0, dtype=torch.float32) if launch is None else key_sums(launch, key)
        found /= key.shape[1]
        target.slot[heads].copy_(found)
        target.heads += found.shape[0]


class _Heads:
    """One attention call as `split` cuts it by query heads: its shape is the call's
    [batch, tokens, heads, head_dim] output, a window of it a slice of heads, and calling it on a
    window gives those heads' attention. Called on itself (`split` running whole), all heads."""

    __slots__ = ("attn", "device", "dtype", "hidden", "processor", "projections", "rotary", "shape")

    def __init__(
        self,
        processor: H3AttnProcessor,
        attn: MiniMaxH3Attention,
        hidden: Hidden,
        rotary: tuple[torch.Tensor, torch.Tensor] | None,
        projections: tuple[Rows, Rows, Rows],
    ) -> None:
        self.processor, self.attn, self.hidden, self.rotary = processor, attn, hidden, rotary
        self.projections = projections
        self.shape = (*hidden.shape[:-1], attn.heads, attn.head_dim)
        # A row-quantized input (the fused norm's fp8 epilogue) stands for its bf16 rows.
        self.dtype = torch.bfloat16 if isinstance(hidden, PreQuantized) else hidden.dtype
        self.device = hidden.payload.device if isinstance(hidden, PreQuantized) else hidden.device

    @classmethod
    def of(
        cls,
        processor: H3AttnProcessor,
        attn: MiniMaxH3Attention,
        hidden: Hidden,
        rotary: tuple[torch.Tensor, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> _Heads | None:
        """The call cut by heads, or None where it runs them whole."""
        if attn.fused_projections or mask is not None or processor._parallel_config is not None:
            return None
        q, k, v = (rows(m) for m in (attn.to_q, attn.to_k, attn.to_v))
        if q is None or k is None or v is None:
            return None
        return cls(processor, attn, hidden, rotary, (q, k, v))

    def quantized(self) -> _Heads:
        """These heads reading the int8 operand their q, k and v leaves would each make from the
        bf16 rows (the leaves' own quantizer: `int8_rows` with no prologue), where all three take
        one; themselves otherwise. Same bytes: the leaves read that operand either way."""
        if not isinstance(self.hidden, torch.Tensor):
            return self
        attn = self.attn
        operand = int8_rows(self.hidden, (attn.to_q, attn.to_k, attn.to_v))
        if operand is None:
            return self
        return type(self)(self.processor, attn, operand, self.rotary, self.projections)

    def dim(self) -> int:
        return len(self.shape)

    def element_size(self) -> int:
        return int(self.dtype.itemsize)

    def __getitem__(self, index: tuple[slice[int | None], ...]) -> slice[int, int, None]:
        return cast("slice[int, int, None]", index[2])  # `Cut(2)` hands whole head ranges

    def __call__(self, heads: _Heads | slice[int, int, None]) -> torch.Tensor:
        window = slice(0, self.attn.heads) if heads is self else heads
        assert isinstance(window, slice)
        d = self.attn.head_dim
        start, stop = window.start * d, window.stop * d
        # Match the native leaf's cases that disable implicit operand sharing: inference
        # tensors have no mutation counter, and a recording leaf has no sibling identity.
        # Versioned eager tensors keep their existing reuse across head windows. An explicit
        # scope ends before RoPE/attention and never holds an operand across windows.
        shared: torch.Tensor | None = None
        if isinstance(self.hidden, torch.Tensor) and (
            self.hidden.is_inference()
            or (self.hidden.is_cuda and torch.cuda.is_current_stream_capturing())
        ):
            shared = self.hidden
        with quantize_once(shared):
            query, key, value = (
                project(self.hidden, start, stop).unflatten(-1, (-1, d))
                for project in self.projections
            )
        query, key = self.processor.rope(self.attn, query, key, self.rotary)
        seed_key_mean(getattr(self.processor, "_keys", None), key, window)
        return self.processor.attend(query, key, value)


class FusedMiniMaxH3TransformerBlock(MiniMaxH3TransformerBlock):
    _fusion_attn_fp8: bool = False
    _fusion_ff_fp8: bool = False
    _glue: Launch

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        # The attention owns its normed input (no name here holds it), so it goes at its last
        # use, before the out projection's output (1.2 GB each at 15 s); the caller holds this
        # block's input throughout.
        attn_output: torch.Tensor = self.attn(
            Owned(
                fused.norm_modulate(
                    self._glue,
                    GLUE,
                    hidden_states,
                    self.norm1.weight,
                    scale_msa,
                    shift_msa,
                    adaln_indices,
                    _eps(self.norm1),
                    fp8=self._fusion_attn_fp8,
                )
            ),
            rotary_emb,
            attention_mask,
        )
        tail = _Tail(self, hidden_states, adaln_indices, gate_msa, scale_mlp, shift_mlp, gate_mlp)
        # The tail writes its rows in place: a view that does not flatten would take a copy.
        out: torch.Tensor = split(tail.run, tail, Cut(1), out=attn_output.contiguous(), holds=_TAIL_HOLDS)
        return out


class _Tail:
    """A block after its attention, as `split` cuts it by tokens: `hidden = residual + gate_msa *
    attention`, the FFN of its modulated norm, and `hidden + gate_mlp * ffn`, each window written
    into the attention output's own rows (their last use). Beside the block's input only that
    output lives whole; a window's normed rows and FFN are its own (xs-141: three [tokens, 5376]
    row sets, 3.3 GB at 15 s, were whole at once). Every op is per row, so windows are exact."""

    __slots__ = ("block", "device", "dtype", "gates", "index", "residual", "shape")

    def __init__(
        self, block: FusedMiniMaxH3TransformerBlock, residual: torch.Tensor, index: torch.Tensor, *gates: torch.Tensor
    ) -> None:
        self.block, self.residual, self.index, self.gates = block, residual, index, gates
        self.shape, self.dtype, self.device = residual.shape, residual.dtype, residual.device

    def dim(self) -> int:
        return len(self.shape)

    def element_size(self) -> int:
        return int(self.dtype.itemsize)

    def __getitem__(self, index: tuple[slice[int | None], ...]) -> slice[int | None]:
        return index[1]

    def run(self, rows: _Tail | slice[int | None], out: torch.Tensor) -> None:
        """The tail over token `rows`, written into `out` (those rows of the attention output).
        Called on itself (`split` running whole), every row."""
        rows = slice(None) if rows is self else rows
        assert isinstance(rows, slice)
        block, (gate_msa, scale, shift, gate) = self.block, self.gates
        residual, index = self.residual[:, rows], self.index[rows]
        hidden, normed = fused.gate_add_norm_modulate(
            block._glue,
            GLUE,
            residual,
            gate_msa,
            out,
            index,
            block.norm2.weight,
            scale,
            shift,
            _eps(block.norm2),
            fp8=block._fusion_ff_fp8,
            into=out,
        )
        ffn = _FFN(block.ff)(normed)
        del normed
        fused.gate_add(block._glue, GLUE, hidden, gate, ffn, index, into=out)


class _FFN:
    """The FFN over a window of tokens (`_Tail`), its identity dropout skipped."""

    __slots__ = ("ff",)

    def __init__(self, ff: FeedForward) -> None:
        self.ff = ff

    def __call__(self, normed: Hidden) -> torch.Tensor:
        act, down = self.ff.net[0], self.ff.net[2]
        if not isinstance(act, FusedSwiGLU) or act._fusion_fp8:
            out: torch.Tensor = down(act(normed))
            return out
        # An int8 down projection quantizes `value * silu(gate)` from the projection's
        # [value | gate] halves itself (the same law as `swiglu`): the product is never written.
        both: torch.Tensor = act.proj(normed)
        rows = int8_rows(both, (down,), activation="swiglu")
        product: torch.Tensor = down(_swiglu(act._glue, both) if rows is None else rows)
        return product


# ------------------------------------------------------------------------ install


@dataclass(frozen=True, slots=True)
class Applied:
    applied: bool
    reason: str
    substitutions: tuple[tuple[str, str, str], ...]
    """`(module path, was, now)` per site, in walk order."""
    fp8_sites: int
    warm_ms: float
    code: str = ""
    """Why the plan did not apply, typed; empty when it did."""

    def document(self) -> dict[str, object]:
        census: dict[str, int] = {}
        for _, was, now in self.substitutions:
            key = f"{was.rsplit(':', 1)[-1]}->{now.rsplit(':', 1)[-1]}"
            census[key] = census.get(key, 0) + 1
        return {
            "identity": identity() if self.applied else "",
            "applied": self.applied,
            "substitutions": census,
            "sites": len(self.substitutions),
            "fp8_sites": self.fp8_sites,
            "code": self.code,
            "reason": self.reason,
            "warm_ms": round(self.warm_ms, 1),
        }


def _type_name(value: object) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def identity() -> str:
    """The plan's identity: its kernels' sources and its call/topology contract."""
    document = {"kernels": [GLUE.bf16.identity, GLUE.fp8.identity, KEYS.identity], "plan": CONTRACT}
    return (
        "sha256:"
        + hashlib.sha256(
            b"minimax-h3.glue\0"
            + json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _require_call(  # type: ignore[explicit-any]  # any Diffusers callable: its signature is what is checked
    function: Callable[..., object], parameters: tuple[str, ...]
) -> None:
    """Validate the inputs this plan substitutes; annotations are not an ABI.

    Positional inputs must keep their names and order. Additional optional inputs are
    harmless; additional required inputs or reordered inputs cannot use this forward.
    """
    try:
        signature = inspect.signature(function)
        bound = signature.bind(*(object() for _ in parameters))
        if tuple(bound.arguments) != parameters:
            raise TypeError(f"expected positional inputs {parameters}")
    except (TypeError, ValueError) as exc:
        raise GlueRefusal("fusion_contract_unsupported", f"{function.__qualname__}: {exc}") from exc


def _require_calls() -> None:
    _require_call(
        MiniMaxH3TransformerBlock.forward,
        ("self", "hidden_states", "temb", "adaln_indices", "rotary_emb", "attention_mask"),
    )
    _require_call(
        MiniMaxH3AttnProcessor.__call__,
        ("self", "attn", "hidden_states", "rotary_emb", "attention_mask"),
    )
    _require_call(_apply_rotary_emb, ("hidden_states", "cos", "sin"))
    _require_call(SwiGLU.forward, ("self", "hidden_states"))
    _require_call(FeedForward.forward, ("self", "hidden_states"))


def record(model: object, plan: Plan, name: str, document: Mapping[str, object]) -> None:
    """One part of the execution plan's outcome, named by its kernel set (`GLUE.bf16.name`): on the
    model (`_cozy_fusion_report[name]`) and in the construction's
    receipts (`measurements.execution.plan_reports`), so `run show` names a refusal. An install
    deferred to a request boundary reports again, through the same plan, when it lands; one whose
    build fails never reaches the package, and Runtime reports that under the same kernel-set name."""
    if hasattr(model, "__dict__"):
        vars(model).setdefault("_cozy_fusion_report", {})[name] = dict(document)
    plan.report(name, document)


def _record(model: object, plan: Plan, result: Applied) -> Applied:
    record(model, plan, GLUE.bf16.name, result.document())
    return result


def _rowwise_leaf(module: torch.nn.Module) -> bool:
    # Semantic hooks must receive the original Tensor, not a quantized carrier.
    return bool(
        getattr(module, "provider", None) == ROWWISE_FP8
        and not module._forward_pre_hooks
        and not module._forward_hooks
    )


def _pointer(value: torch.Tensor) -> int:
    try:
        return int(value.data_ptr())
    except RuntimeError:  # the memory's stand-in for an unbound weight has no storage
        return 0


def _tensor_census(module: torch.nn.Module) -> tuple[object, ...]:
    return tuple(
        (name, tuple(value.shape), str(value.dtype), id(value), _pointer(value))
        for name, value in (
            *module.named_parameters(remove_duplicate=False),
            *module.named_buffers(remove_duplicate=False),
        )
    )


def _shape(condition: bool, detail: str) -> None:
    if not condition:
        raise GlueRefusal("fusion_block_shape", detail)


def _widths(transformer: MiniMaxH3Transformer3DModel, root: str) -> None:
    config = transformer.config
    widths = (
        int(config["hidden_size"]),
        int(config["ffn_dim"]),
        int(config["attention_head_dim"]),
        6 * int(config["rope_freq_dim"]),
    )
    _shape(
        widths == (HIDDEN, FFN, HEAD_DIM, ROTARY),
        f"{root}: widths {widths} are not the ones the glue compiles",
    )


def _validate_transformer(
    transformer: MiniMaxH3Transformer3DModel, root: str, forwards: tuple[object, ...]
) -> None:
    _shape(
        type(transformer).forward in (MiniMaxH3Transformer3DModel.forward, *forwards)
        and plain_forward(transformer),
        f"{root}: {_type_name(transformer)} overrides the Diffusers MiniMax-H3 forward",
    )
    for index, block in enumerate(_blocks(transformer)):
        path = f"{root}.transformer_blocks.{index}"
        attn, ff = block.attn, block.ff
        _shape(
            type(block) is MiniMaxH3TransformerBlock
            and plain_forward(block)
            and type(_processor(attn)) in (MiniMaxH3AttnProcessor, H3AttnProcessor)
            and not attn.fused_projections
            and type(ff) is FeedForward
            and plain_forward(ff)
            and len(ff.net) == 3
            and type(ff.net[0]) is SwiGLU
            and plain_forward(ff.net[0])
            and type(ff.net[0].activation) is torch.nn.SiLU
            and plain_forward(ff.net[0].activation)
            and type(ff.net[1]) is torch.nn.Dropout
            and plain_forward(ff.net[1])
            and ff.net[1].p == 0.0
            and not any(
                m._forward_hooks or m._forward_pre_hooks
                for m in (
                    attn,
                    ff,
                    ff.net,
                    ff.net[0],
                    ff.net[1],
                    ff.net[0].activation,
                    block.norm1,
                    block.norm2,
                    attn.norm_q,
                    attn.norm_k,
                )
            )
            and all(
                type(m) is torch.nn.RMSNorm and plain_forward(m)
                for m in (block.norm1, block.norm2, attn.norm_q, attn.norm_k)
            )
            and block.norm1.weight is not None
            and block.norm2.weight is not None
            and attn.norm_q.weight is not None
            and attn.norm_k.weight is not None,
            f"{path} is not the MiniMax-H3 block shape the fused plan was matched to ({CONTRACT})",
        )


def _install_transformer(
    transformer: MiniMaxH3Transformer3DModel, root: str, glue: Launch, keys: Launch
) -> tuple[list[tuple[str, str, str]], int]:
    substitutions: list[tuple[str, str, str]] = []
    fp8_sites = 0
    for index, block in enumerate(_blocks(transformer)):
        path = f"{root}.transformer_blocks.{index}"
        attn, ff = block.attn, block.ff
        attn_fp8 = all(_rowwise_leaf(m) for m in (attn.to_q, attn.to_k, attn.to_v))
        ff_fp8 = _rowwise_leaf(cast(SwiGLU, ff.net[0]).proj)  # validated: a SwiGLU (`_validate_transformer`)
        out_fp8 = _rowwise_leaf(ff.net[2])
        fp8_sites += attn_fp8 + ff_fp8 + out_fp8

        current = cast(H3AttnProcessor, attn.processor)  # validated, then `_release_qkv`'s
        processor = FusedMiniMaxH3AttnProcessor()
        processor._attention_backend = current._attention_backend
        processor._parallel_config = current._parallel_config
        processor._glue = glue
        processor._keys = keys
        substitutions.append(
            (f"{path}.attn.processor", _type_name(attn.processor), _type_name(processor))
        )
        attn.set_processor(processor)  # type: ignore[arg-type]  # Diffusers' union names its own processors

        substitutions.append((f"{path}.ff.net.0", _type_name(ff.net[0]), _type_name(FusedSwiGLU)))
        ff.net[0].__class__ = FusedSwiGLU
        swiglu_module = cast(FusedSwiGLU, ff.net[0])  # its class, swapped above
        swiglu_module._fusion_fp8 = out_fp8
        swiglu_module._glue = glue

        substitutions.append((path, _type_name(block), _type_name(FusedMiniMaxH3TransformerBlock)))
        block.__class__ = FusedMiniMaxH3TransformerBlock
        fused = cast(FusedMiniMaxH3TransformerBlock, block)  # its class, swapped above
        fused._fusion_attn_fp8 = attn_fp8
        fused._fusion_ff_fp8 = ff_fp8
        fused._glue = glue
    return substitutions, fp8_sites


def _processor(attn: MiniMaxH3Attention) -> object:
    """`attn`'s processor, whatever its class: Diffusers' union names only its own."""
    return attn.processor


def _release_qkv(attn: MiniMaxH3Attention) -> None:
    current = _processor(attn)
    if type(current) is MiniMaxH3AttnProcessor:
        processor = H3AttnProcessor()
        processor._attention_backend = current._attention_backend
        processor._parallel_config = current._parallel_config
        attn.set_processor(processor)  # type: ignore[arg-type]  # Diffusers' union names its own processors


def _transformers(pipe: Pipeline) -> Iterator[tuple[str, MiniMaxH3Transformer3DModel]]:
    """Every MiniMax-H3 transformer the pipeline holds, by component name."""
    seen: set[int] = set()
    for name, member in pipe.components.items():
        if isinstance(member, MiniMaxH3Transformer3DModel) and id(member) not in seen:
            seen.add(id(member))
            yield str(name), member


def _warm_keys(keys: Launch, device: torch.device, heads: int) -> float:
    """Launch the key sums once, on two rows; nothing compiles. The milliseconds it took to enqueue."""
    started = time.perf_counter()
    key_sums(keys, torch.ones(1, 2, heads, HEAD_DIM, device=device, dtype=torch.bfloat16))
    return (time.perf_counter() - started) * 1000


def _eager(model: object, plan: Plan, code: str, reason: str) -> str:
    _record(model, plan, Applied(False, reason, (), 0, 0.0, code))
    return ""


def install(model: object, pipe: Pipeline, plan: Plan, *, forwards: tuple[object, ...] = ()) -> str:
    """Install the fused glue on every H3 transformer `pipe` holds; the plan's identity, or ""
    when eager serves (a `NOT_INSTALLABLE` code, reported). Waits for the kernels' build.
    `forwards`: the package's own transformer forwards the blocks may run under, beside
    Diffusers'. Fused or eager, every block's attention runs `H3AttnProcessor`."""
    for _, transformer in _transformers(pipe):
        for block in _blocks(transformer):
            _release_qkv(block.attn)
    device = torch.device(plan.device)
    if device.type != "cuda":
        return _eager(
            model, plan, "fusion_device_incapable", f"fused glue serves CUDA, not {device}"
        )
    major, _ = torch.cuda.get_device_capability(device)
    if major < 8:
        return _eager(
            model,
            plan,
            "fusion_device_incapable",
            f"fused glue needs bf16 tensor cores (sm80+); this device is sm{major}x",
        )
    if TRITON_ABSENT or GLUE.absent:
        return _eager(model, plan, "fusion_kernels_absent", TRITON_ABSENT or GLUE.absent)
    try:
        _require_calls()
        transformers = tuple(_transformers(pipe))
        for name, transformer in transformers:
            _widths(transformer, name)
            _validate_transformer(transformer, name, forwards)
        if not transformers:
            return _eager(model, plan, "fusion_no_transformer", "no MiniMax-H3 transformer to fuse")
        glue = plan.load(GLUE.kernels(device))
        keys = plan.load(KEYS)
        heads = int(transformers[0][1].config["num_attention_heads"])
        wrong = fused.checked(glue, GLUE, device, heads=heads, fp8=GLUE.kernels(device) is GLUE.fp8)
        if wrong:  # no fused block has served yet: the eager model does, and the run says why
            return _eager(model, plan, "fusion_numeric", wrong)
    except KernelsUnavailable as exc:
        return _eager(model, plan, f"fusion_{exc.code}", str(exc))
    except GlueRefusal as exc:
        if exc.code not in NOT_INSTALLABLE:
            raise
        # NOTHING HAS BEEN SUBSTITUTED: the eager model serves.
        return _eager(model, plan, exc.code, str(exc))
    substitutions: list[tuple[str, str, str]] = []
    fp8_sites = 0
    warm_ms = 0.0
    for name, transformer in transformers:
        before = _tensor_census(transformer)
        installed, sites = _install_transformer(transformer, name, glue, keys)
        if _tensor_census(transformer) != before:
            raise GlueRefusal(
                "fusion_tensor_schema_changed",
                f"{name}: the substitution moved a parameter or buffer",
            )
        substitutions += installed
        fp8_sites += sites
        # PAST THE POINT OF NO RETURN: the forwards are substituted, so a failure refuses.
        heads = int(transformer.config["num_attention_heads"])
        warm_ms += fused.warm(glue, GLUE, device, heads=heads, fp8=sites > 0)
        warm_ms += _warm_keys(keys, device, heads)
    _record(model, plan, Applied(True, "", tuple(substitutions), fp8_sites, warm_ms))
    return identity()
