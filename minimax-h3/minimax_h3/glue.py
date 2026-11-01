"""H3's fused glue (h3a-015): the elementwise chains a DiT block runs between its GEMMs.

Each kernel is one pass over rows of a `[rows, width]` bf16 activation; `install` substitutes
the fused forwards on the filled H3 transformers as this model's execution plan. Shapes are
the ones the eager composition already produces, so nothing moves a parameter, a buffer or a
state-dict key (re-censused after the swap).

Numerics are the EAGER composition's, not "better": every intermediate eager rounds to bf16 is
rounded here too (`_rb`, in the integer domain so the compiler cannot fold the round-trip
away), and the fp8 epilogue is the rowwise leaf's own quantizer run on the same registers.
Measured on sm89 against torch 2.13 eager: the modulate, gate-add, SwiGLU and RoPE chains are
bit-identical; the RMSNorm differs from torch's fused kernel by one bf16 ulp on ~4e-6 of
elements (fp32 reduction order of the row variance), and those elements are the whole distance
between fused and eager.

The `@triton.jit` functions are the kernel SOURCES (`GLUE`): the Runtime compiles them on the
machine that runs them when the executor starts and `plan.load` waits for that build; a request
never compiles. An fp8 epilogue is chosen per site from what the fill installed: a site whose
every consumer is an `fp8.rowwise` encoded leaf hands it a `PreQuantized` operand; any other
consumer gets bf16.

Not installable (eager serves, the code is reported): no CUDA sm80+, kernels absent or failed,
an unmatched Diffusers topology. A build that does not load, or a failure after substitution
began, refuses the construction: there is no eager left.

Adapted from NVlabs/Sana `sol-engine` `models/minimax_h3/{GB200,GB10}/fusions.py`
(Apache-2.0): the per-row program shape, the table-row gather inside the kernel, the
partner-channel RoPE and the integer-domain bf16 rounding are theirs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl
from diffusers.models.activations import SwiGLU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
    _apply_rotary_emb,
)
from triton.language.extra import libdevice

from cozy_runtime.author import CapabilityError, Cut, plain_forward, split
from cozy_runtime.author.kernels import (
    ROWWISE_FP8,
    KernelsUnavailable,
    Launch,
    Plan,
    PreQuantized,
    TritonEntry,
    TritonKernels,
    int8_rows,
)
from cozy_runtime.rows import Rows, rows

from .adaln_pruned import AdaLNPrunedMiniMaxH3Transformer

CONTRACT = "h3-glue-call-and-topology/2"


@dataclass(slots=True)
class KeyMean:
    """Where an attention call leaves its post-RoPE key mean, F32 [heads, 128] over its tokens
    (`producer`: the last dense step seeds the first sparse step's Kitchen statistics), and how
    many heads it has written."""

    slot: torch.Tensor
    heads: int = 0


KEY_MEAN: ContextVar[KeyMean | None] = ContextVar("minimax_h3_key_mean", default=None)


class Owned:
    """An argument whose only reference the callee takes (`take`): a caller's frame keeps
    every argument it passes until the call returns, so a tensor it passes plainly outlives its
    last use inside."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


def take(value: Any) -> Any:
    """`value`, or the tensor an `Owned` passed, which then holds it no more."""
    if not isinstance(value, Owned):
        return value
    taken, value.value = value.value, None
    return taken
#: The widths compiled: a transformer with others is not installable.
HIDDEN, FFN, HEAD_DIM, ROTARY = 5376, 14336, 128, 96
_ROW_WARPS = 16
_ROPE_HEADS = 8
_ROPE_WARPS = 4
_KEY_TOKENS = 64
_KEY_WARPS = 4
_SWIGLU_BLOCK = 1024
_SWIGLU_WARPS = 8
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



class GlueRefusal(CapabilityError):
    """The fused plan cannot install as matched; `code` says why."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail, code=code)


# ------------------------------------------------------------------- kernel sources


@triton.jit
def _rb(x):  # type: ignore[no-untyped-def]
    # fp32 -> bf16 -> fp32 round-to-nearest-even in the integer domain: the compiler
    # folds `x.to(bf16).to(f32)` to identity and would keep fp32 precision eager lacks.
    bits = x.to(tl.int32, bitcast=True)
    bits = bits + 0x7FFF + ((bits >> 16) & 1)
    return (bits & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _e4m3(value, scale):  # type: ignore[no-untyped-def]
    # Exactly `leaves._triton_rowwise_kernel`'s cast: IEEE divide, clamp, RTNE.
    return tl.clamp(tl.div_rn(value, scale), -448.0, 448.0).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )


@triton.jit  # type: ignore[untyped-decorator]
def _norm_mod_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    branch_ptr,
    gate_ptr,
    hidden_ptr,
    out_ptr,
    scale_out_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    index_ptr,
    n_cols,
    n_index,
    stride_x,
    stride_table,
    eps,
    RESIDUAL: tl.constexpr,
    FP8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offset = row * stride_x + cols
    # One table row per SEQUENCE row; a batch axis wraps (eager broadcasts the index).
    table = tl.load(index_ptr + row % n_index).to(tl.int64) * stride_table + cols
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    if RESIDUAL:
        branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table, mask=mask, other=0.0).to(tl.float32)
        x = _rb(x + _rb(gate * branch))
        tl.store(hidden_ptr + offset, x.to(tl.bfloat16), mask=mask)
    variance = tl.sum(x * x, axis=0) / n_cols
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = _rb(x * tl.math.rsqrt(variance + eps) * weight)
    scale = tl.load(scale_ptr + table, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + table, mask=mask, other=0.0).to(tl.float32)
    out = _rb(_rb(normed * _rb(1.0 + scale)) + shift)
    if FP8:
        amax = tl.max(tl.abs(out), axis=0)
        row_scale = tl.maximum(amax / 448.0, 1e-12)
        tl.store(scale_out_ptr + row, row_scale)
        tl.store(out_ptr + offset, _e4m3(out, row_scale), mask=mask)
    else:
        tl.store(out_ptr + offset, out.to(tl.bfloat16), mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _gate_add_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    branch_ptr,
    gate_ptr,
    out_ptr,
    index_ptr,
    n_cols,
    n_index,
    stride_x,
    stride_table,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offset = row * stride_x + cols
    table = tl.load(index_ptr + row % n_index).to(tl.int64) * stride_table + cols
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + table, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offset, _rb(x + _rb(gate * branch)).to(tl.bfloat16), mask=mask)


@triton.jit
def _silu_product(x_ptr, base, cols, mask, half):  # type: ignore[no-untyped-def]
    value = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(x_ptr + base + half + cols, mask=mask, other=0.0).to(tl.float32)
    # torch's silu: `x / (1 + exp(-x))` in fp32 with IEEE divide and an exact exp.
    activated = _rb(tl.div_rn(gate, 1.0 + libdevice.exp(-gate)))
    return _rb(value * activated)


@triton.jit  # type: ignore[untyped-decorator]
def _swiglu_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    out_ptr,
    scale_out_ptr,
    half,
    stride_x,
    stride_out,
    FP8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One row per program, chunked: a 14336-wide product held twice would spill. The
    # fp8 epilogue needs the row amax first, so it recomputes the chunk on a second
    # sweep from a row that is by then L2-resident.
    row = tl.program_id(0).to(tl.int64)
    base = row * stride_x
    offsets = tl.arange(0, BLOCK)
    if FP8:
        amax = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(0, half, BLOCK):
            cols = start + offsets
            amax = tl.maximum(amax, tl.abs(_silu_product(x_ptr, base, cols, cols < half, half)))
        row_scale = tl.maximum(tl.max(amax, axis=0) / 448.0, 1e-12)
        tl.store(scale_out_ptr + row, row_scale)
        for start in range(0, half, BLOCK):
            cols = start + offsets
            product = _silu_product(x_ptr, base, cols, cols < half, half)
            tl.store(out_ptr + row * stride_out + cols, _e4m3(product, row_scale), mask=cols < half)
    else:
        for start in range(0, half, BLOCK):
            cols = start + offsets
            product = _silu_product(x_ptr, base, cols, cols < half, half)
            tl.store(out_ptr + row * stride_out + cols, product.to(tl.bfloat16), mask=cols < half)


@triton.jit  # type: ignore[untyped-decorator]
def _rope_norm_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    out_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    seq,
    heads,
    eps,
    HEAD_DIM: tl.constexpr,
    ROTARY: tl.constexpr,
    HEADS: tl.constexpr,
):
    # One program per (row, HEADS heads): the RMS is per head, the rotation pairs
    # channel j with j +/- ROTARY/2 inside the rotary span, and cos/sin are per
    # sequence position (a batch axis wraps).
    row = tl.program_id(0).to(tl.int64)
    head0 = tl.program_id(1) * HEADS
    token = row % seq
    head = head0 + tl.arange(0, HEADS)[:, None]
    chan = tl.arange(0, HEAD_DIM)[None, :]
    mask = head < heads
    base = (row * heads + head) * HEAD_DIM
    x = tl.load(x_ptr + base + chan, mask=mask, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1)[:, None] / HEAD_DIM + eps)
    weight = tl.load(weight_ptr + chan).to(tl.float32)
    normed = _rb(x * inv * weight)
    half = ROTARY // 2
    rotating = chan < ROTARY
    low = chan < half
    partner = tl.where(low, chan + half, chan - half)
    paired = tl.load(x_ptr + base + partner, mask=mask & rotating, other=0.0).to(tl.float32)
    paired_weight = tl.load(weight_ptr + partner, mask=rotating, other=0.0).to(tl.float32)
    rotated = _rb(paired * inv * paired_weight)
    rotated = tl.where(low, -rotated, rotated)
    # Eager casts the fp32 tables to bf16 BEFORE multiplying.
    cos = _rb(tl.load(cos_ptr + token * ROTARY + chan, mask=rotating, other=1.0))
    sin = _rb(tl.load(sin_ptr + token * ROTARY + chan, mask=rotating, other=0.0))
    turned = _rb(_rb(normed * cos) + _rb(rotated * sin))
    out = tl.where(rotating, turned, normed)
    tl.store(out_ptr + base + chan, out.to(tl.bfloat16), mask=mask)


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


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _entries(fp8s: tuple[int, ...]) -> Iterator[TritonEntry]:
    """Every entry the H3 widths need, with the epilogues `fp8s` names (0: bf16, 1: fp8). Row
    counts never enter one: they size grids."""
    block = _next_pow2(HIDDEN)
    for residual in (0, 1):
        for fp8 in fp8s:
            yield TritonEntry(
                _norm_mod_kernel,
                {
                    "x_ptr": "*bf16",
                    "branch_ptr": "*bf16",
                    "gate_ptr": "*bf16",
                    "hidden_ptr": "*bf16",
                    "out_ptr": "*fp8e4nv" if fp8 else "*bf16",
                    "scale_out_ptr": "*fp32" if fp8 else "*bf16",
                    "weight_ptr": "*bf16",
                    "scale_ptr": "*bf16",
                    "shift_ptr": "*bf16",
                    "index_ptr": "*i64",
                    "n_cols": "i32",
                    "n_index": "i32",
                    "stride_x": "i32",
                    "stride_table": "i32",
                    "eps": "fp32",
                    "RESIDUAL": "constexpr",
                    "FP8": "constexpr",
                    "BLOCK": "constexpr",
                },
                {"RESIDUAL": residual, "FP8": fp8, "BLOCK": block},
                (
                    "x_ptr",
                    "branch_ptr",
                    "gate_ptr",
                    "hidden_ptr",
                    "out_ptr",
                    "scale_out_ptr",
                    "weight_ptr",
                    "scale_ptr",
                    "shift_ptr",
                    "index_ptr",
                    "n_cols",
                    "stride_x",
                    "stride_table",
                ),
                _ROW_WARPS,
            )
    yield TritonEntry(
        _gate_add_kernel,
        {
            "x_ptr": "*bf16",
            "branch_ptr": "*bf16",
            "gate_ptr": "*bf16",
            "out_ptr": "*bf16",
            "index_ptr": "*i64",
            "n_cols": "i32",
            "n_index": "i32",
            "stride_x": "i32",
            "stride_table": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": block},
        (
            "x_ptr",
            "branch_ptr",
            "gate_ptr",
            "out_ptr",
            "index_ptr",
            "n_cols",
            "stride_x",
            "stride_table",
        ),
        _ROW_WARPS,
    )
    for fp8 in fp8s:
        yield TritonEntry(
            _swiglu_kernel,
            {
                "x_ptr": "*bf16",
                "out_ptr": "*fp8e4nv" if fp8 else "*bf16",
                "scale_out_ptr": "*fp32" if fp8 else "*bf16",
                "half": "i32",
                "stride_x": "i32",
                "stride_out": "i32",
                "FP8": "constexpr",
                "BLOCK": "constexpr",
            },
            {"FP8": fp8, "BLOCK": _SWIGLU_BLOCK},
            ("x_ptr", "out_ptr", "scale_out_ptr", "half", "stride_x", "stride_out"),
            _SWIGLU_WARPS,
        )
    yield TritonEntry(
        _rope_norm_kernel,
        {
            "x_ptr": "*bf16",
            "out_ptr": "*bf16",
            "weight_ptr": "*bf16",
            "cos_ptr": "*fp32",
            "sin_ptr": "*fp32",
            "seq": "i32",
            "heads": "i32",
            "eps": "fp32",
            "HEAD_DIM": "constexpr",
            "ROTARY": "constexpr",
            "HEADS": "constexpr",
        },
        {"HEAD_DIM": HEAD_DIM, "ROTARY": ROTARY, "HEADS": _ROPE_HEADS},
        ("x_ptr", "out_ptr", "weight_ptr", "cos_ptr", "sin_ptr"),
        _ROPE_WARPS,
    )
    yield TritonEntry(
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
    )


#: Every sm80+ card's set: the bf16 epilogues.
GLUE = TritonKernels("minimax-h3-glue", _entries((0,)))
#: With the fp8 epilogues, for cards that have fp8 (sm89+). Triton has no e4m3 below: on an
#: sm86 the fp8 entries do not build, and one failed entry refuses its whole set, so an RTX 3070
#: served the eager blocks, which do not fit 15 s in 8 GB.
GLUE_FP8 = TritonKernels("minimax-h3-glue-fp8", _entries((0, 1)))


def kernels(device: Any) -> TritonKernels:
    """The set `device` builds."""
    return GLUE_FP8 if torch.cuda.get_device_capability(device) >= (8, 9) else GLUE


# ---------------------------------------------------------------------- wrappers


def _rows(x: Any, width: int) -> Any:
    flat = x.reshape(-1, width)
    return flat if flat.stride(-1) == 1 else flat.contiguous()


def _table(t: Any) -> Any:
    # `chunk(6, dim=-1)` views have a 6*hidden row stride and a unit column stride; that is
    # addressable as `row * stride(0) + col`, so no copy — anything else is made contiguous.
    return t if t.stride(-1) == 1 else t.contiguous()


def _require(x: Any) -> None:
    if x.dtype is not torch.bfloat16 or not x.is_cuda:
        raise GlueRefusal(
            "fusion_block_shape",
            f"fused glue serves bfloat16 CUDA activations, got {x.dtype} on {x.device}",
        )


def _launch_norm_mod(
    launch: Launch,
    x: Any,
    branch: Any,
    gate: Any,
    weight: Any,
    scale: Any,
    shift: Any,
    index: Any,
    eps: float,
    fp8: bool,
    hidden_out: Any | None = None,
) -> tuple[Any, Any, Any]:
    _require(x)
    width = int(x.shape[-1])
    rows = _rows(x, width)
    n = int(rows.shape[0])
    if branch is None:
        hidden = rows
    else:  # a row is read whole before it is written: `hidden_out` may be `branch`'s own rows
        hidden = torch.empty_like(rows) if hidden_out is None else _rows(hidden_out, width)
    branch_rows = _rows(branch, width) if branch is not None else rows
    gate = _table(gate) if gate is not None else scale
    scale, shift = _table(scale), _table(shift)
    if fp8:
        out = torch.empty(n, width, dtype=torch.float8_e4m3fn, device=x.device)
        row_scale = torch.empty(n, 1, dtype=torch.float32, device=x.device)
    else:
        out = torch.empty_like(rows)
        row_scale = out
    launch(
        _norm_mod_kernel,
        (n, 1, 1),
        rows,
        branch_rows,
        gate,
        hidden,
        out,
        row_scale,
        weight,
        scale,
        shift,
        index,
        width,
        index.numel(),
        rows.stride(0),
        scale.stride(0),
        float(eps),
        int(branch is not None),
        int(fp8),
        _next_pow2(width),
    )
    return hidden, out, row_scale


def rmsnorm_modulate(
    launch: Launch,
    x: Any,
    weight: Any,
    scale: Any,
    shift: Any,
    index: Any,
    eps: float,
    *,
    fp8: bool = False,
) -> Any:
    """`rmsnorm(x) * (1 + scale[index]) + shift[index]`, one read and one write.

    With `fp8`, the write is the rowwise leaf's operand (E4M3 payload + per-row scale)
    instead of the bf16 tensor the leaf would have read back to quantize.
    """
    _, out, row_scale = _launch_norm_mod(
        launch, x, None, None, weight, scale, shift, index, eps, fp8
    )
    if fp8:
        return PreQuantized(out, row_scale, tuple(x.shape))
    return out.view_as(x)


def gate_add_rmsnorm_modulate(
    launch: Launch,
    residual: Any,
    gate: Any,
    branch: Any,
    index: Any,
    weight: Any,
    scale: Any,
    shift: Any,
    eps: float,
    *,
    fp8: bool = False,
    hidden_out: Any | None = None,
) -> tuple[Any, Any]:
    """`hidden = residual + gate[index] * branch`, then `rmsnorm_modulate(hidden)`.

    Returns `(hidden, normed)`: the new residual stream and the next GEMM's input, from
    one pass over the registers that already hold the row. `hidden` is written into
    `hidden_out` where given (`branch` itself may be it).
    """
    hidden, out, row_scale = _launch_norm_mod(
        launch, residual, branch, gate, weight, scale, shift, index, eps, fp8, hidden_out
    )
    normed = PreQuantized(out, row_scale, tuple(residual.shape)) if fp8 else out.view_as(residual)
    return hidden.view_as(residual), normed


def gate_add(
    launch: Launch, residual: Any, gate: Any, branch: Any, index: Any, out: Any | None = None
) -> Any:
    """`residual + gate[index] * branch`, one pass (eager: a gather, a mul and an add), into
    `out` where given (`residual` itself may be it: a row is read whole before it is written)."""
    _require(residual)
    width = int(residual.shape[-1])
    rows, branch_rows = _rows(residual, width), _rows(branch, width)
    gate = _table(gate)
    out = torch.empty_like(rows) if out is None else _rows(out, width)
    block = _next_pow2(width)
    launch(
        _gate_add_kernel,
        (int(rows.shape[0]), 1, 1),
        rows,
        branch_rows,
        gate,
        out,
        index,
        width,
        index.numel(),
        rows.stride(0),
        gate.stride(0),
        block,
    )
    return out.view_as(residual)


def swiglu(launch: Launch, x: Any, *, fp8: bool = False) -> Any:
    """`value * silu(gate)` over a `[..., 2F]` projection: one read of 2F, one write of F."""
    _require(x)
    width = int(x.shape[-1])
    half = width // 2
    rows = _rows(x, width)
    n = int(rows.shape[0])
    if fp8:
        out = torch.empty(n, half, dtype=torch.float8_e4m3fn, device=x.device)
        row_scale = torch.empty(n, 1, dtype=torch.float32, device=x.device)
    else:
        out = torch.empty(n, half, dtype=x.dtype, device=x.device)
        row_scale = out
    launch(
        _swiglu_kernel,
        (n, 1, 1),
        rows,
        out,
        row_scale,
        half,
        rows.stride(0),
        half,
        int(fp8),
        _SWIGLU_BLOCK,
    )
    if fp8:
        return PreQuantized(out, row_scale, (*x.shape[:-1], half))
    return out.view(*x.shape[:-1], half)


def rmsnorm_rope(launch: Launch, x: Any, weight: Any, cos: Any, sin: Any, eps: float) -> Any:
    """Per-head RMSNorm then partial rotary on `(batch, seq, heads, head_dim)`, in place: each
    program reads its heads' rows whole before it writes them, so `x` (a projection's own
    output) never exists twice (2.3 GB of Q and K at 15 s).

    `cos`/`sin` are `(seq, rotary_dim)`; channels past `rotary_dim` pass through.
    """
    _require(x)
    batch, seq, heads, head_dim = (int(d) for d in x.shape)
    rotary = int(cos.shape[-1])
    flat = x.reshape(batch * seq, heads * head_dim)
    flat = flat if flat.stride(-1) == 1 else flat.contiguous()
    launch(
        _rope_norm_kernel,
        (batch * seq, -(-heads // _ROPE_HEADS), 1),
        flat,
        flat,
        weight,
        cos.contiguous(),
        sin.contiguous(),
        seq,
        heads,
        float(eps),
        head_dim,
        rotary,
        _ROPE_HEADS,
    )
    return flat.view(batch, seq, heads, head_dim)


def key_sums(launch: Launch, key: Any) -> Any:
    """F32 [heads, head_dim] sums over tokens of `key`'s [1, tokens, heads, head_dim]."""
    _require(key)
    _, tokens, heads, head_dim = (int(d) for d in key.shape)
    flat = key[0].reshape(tokens, heads * head_dim)
    flat = flat if flat.stride(-1) == 1 else flat.contiguous()
    out = torch.empty(heads, head_dim, dtype=torch.float32, device=key.device)
    launch(_key_sum_kernel, (heads, 1, 1), flat, out, tokens, flat.stride(0), head_dim, _KEY_TOKENS)
    return out


def warm(launch: Launch, device: Any, *, heads: int, fp8: bool) -> float:
    """Launch every entry the served widths need, on two rows; nothing compiles. Returns the
    milliseconds the first launches took to enqueue."""
    started = time.perf_counter()
    rows, table = 2, 3
    x = torch.randn(1, rows, HIDDEN, device=device, dtype=torch.bfloat16)
    weight = torch.ones(HIDDEN, device=device, dtype=torch.bfloat16)
    tables = torch.randn(table, 6 * HIDDEN, device=device, dtype=torch.bfloat16).chunk(6, dim=-1)
    index = torch.arange(rows, device=device) % table
    for quant in {False, fp8}:
        rmsnorm_modulate(launch, x, weight, tables[1], tables[0], index, 1e-5, fp8=quant)
        gate_add_rmsnorm_modulate(
            launch, x, tables[2], x, index, weight, tables[4], tables[3], 1e-5, fp8=quant
        )
        ffn = torch.randn(1, rows, 2 * FFN, device=device, dtype=torch.bfloat16)
        swiglu(launch, ffn, fp8=quant)
    gate_add(launch, x, tables[5], x, index)
    q = torch.randn(1, rows, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    angles = torch.randn(rows, ROTARY, device=device)
    head_weight = torch.ones(HEAD_DIM, device=device, dtype=torch.bfloat16)
    rmsnorm_rope(launch, q, head_weight, angles, angles, 1e-5)
    key_sums(launch, q)
    return (time.perf_counter() - started) * 1000


# ------------------------------------------------------------------ fused modules


class FusedSwiGLU(SwiGLU):  # type: ignore[misc]
    _fusion_fp8: bool = False
    _glue: Launch

    def forward(self, hidden_states: Any) -> Any:
        return swiglu(self._glue, self.proj(hidden_states), fp8=self._fusion_fp8)


class H3AttnProcessor(MiniMaxH3AttnProcessor):  # type: ignore[misc]
    """Diffusers' H3 attention, declared as an exact split by query heads: each window computes
    its own rows of the projections (`cozy_runtime.rows`), per-head norms and RoPE, and the out
    projection runs by tokens. The Runtime sizes the windows against the memory the DiT's
    weights leave: whole on a card that holds both, in windows on a smaller, shared or capped
    one. A projection that does not cut by rows, a mask or a sequence-parallel exchange runs
    whole."""

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        rotary_emb: tuple[Any, Any] | None = None,
        attention_mask: Any | None = None,
    ) -> Any:
        hidden_states = take(hidden_states)
        heads = _Heads.of(self, attn, hidden_states, rotary_emb, attention_mask)
        if heads is None:
            hidden_states = self.whole(attn, hidden_states, rotary_emb, attention_mask)
            return attn.to_out[1](attn.to_out[0](hidden_states.flatten(2, 3)))
        attended = split(heads, heads, Cut(2))
        del heads, hidden_states  # an owned input goes before the out projection's output comes
        out = attn.to_out[1](split(attn.to_out[0], attended.flatten(2, 3), Cut(1)))
        del attended
        return out

    def whole(
        self,
        attn: Any,
        hidden_states: Any,
        rotary_emb: tuple[Any, Any] | None,
        attention_mask: Any | None,
    ) -> Any:
        query, key, value = self.qkv(attn, hidden_states, rotary_emb)
        seed_key_mean(getattr(self, "_glue", None), key, slice(None))
        return self.attend(query, key, value, attention_mask)

    def attend(self, query: Any, key: Any, value: Any, attention_mask: Any | None = None) -> Any:
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

    def qkv(self, attn: Any, hidden_states: Any, rotary_emb: tuple[Any, Any] | None) -> Any:
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key = attn.to_q(hidden_states), attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query, key, value = (t.unflatten(-1, (attn.heads, -1)) for t in (query, key, value))
        return (*self.rope(attn, query, key, rotary_emb), value)

    def rope(self, attn: Any, query: Any, key: Any, rotary_emb: tuple[Any, Any] | None) -> Any:
        query, key = attn.norm_q(query), attn.norm_k(key)
        if rotary_emb is not None:
            query, key = _apply_rotary_emb(query, *rotary_emb), _apply_rotary_emb(key, *rotary_emb)
        return query, key


class FusedMiniMaxH3AttnProcessor(H3AttnProcessor):
    _glue: Launch

    def qkv(self, attn: Any, hidden_states: Any, rotary_emb: tuple[Any, Any] | None) -> Any:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        return (*self.rope(attn, query, key, rotary_emb), value)

    def rope(self, attn: Any, query: Any, key: Any, rotary_emb: tuple[Any, Any] | None) -> Any:
        if rotary_emb is None:
            return attn.norm_q(query), attn.norm_k(key)
        cos, sin = rotary_emb
        query = rmsnorm_rope(self._glue, query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
        key = rmsnorm_rope(self._glue, key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
        return query, key


def seed_key_mean(launch: Launch | None, key: Any, heads: slice) -> None:
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
        attn: Any,
        hidden: Any,
        rotary: tuple[Any, Any] | None,
        projections: tuple[Rows, Rows, Rows],
    ) -> None:
        self.processor, self.attn, self.hidden, self.rotary = processor, attn, hidden, rotary
        self.projections = projections
        self.shape = (*hidden.shape[:-1], attn.heads, attn.head_dim)
        # A row-quantized input (the fused norm's fp8 epilogue) stands for its bf16 rows.
        quantized = isinstance(hidden, PreQuantized)
        self.dtype = torch.bfloat16 if quantized else hidden.dtype
        self.device = hidden.payload.device if quantized else hidden.device

    @classmethod
    def of(
        cls,
        processor: H3AttnProcessor,
        attn: Any,
        hidden: Any,
        rotary: tuple[Any, Any] | None,
        mask: Any | None,
    ) -> _Heads | None:
        """The call cut by heads, or None where it runs them whole."""
        if attn.fused_projections or mask is not None or processor._parallel_config is not None:
            return None
        found = tuple(rows(m) for m in (attn.to_q, attn.to_k, attn.to_v))
        if any(r is None for r in found):
            return None
        return cls(processor, attn, hidden, rotary, found)  # type: ignore[arg-type]

    def dim(self) -> int:
        return len(self.shape)

    def element_size(self) -> int:
        return int(self.dtype.itemsize)

    def __getitem__(self, index: tuple[slice, ...]) -> slice:
        return index[2]

    def __call__(self, heads: _Heads | slice) -> Any:
        window = slice(0, self.attn.heads) if heads is self else heads
        assert isinstance(window, slice)
        d = self.attn.head_dim
        start, stop = window.start * d, window.stop * d
        query, key, value = (
            project(self.hidden, start, stop).unflatten(-1, (-1, d)) for project in self.projections
        )
        query, key = self.processor.rope(self.attn, query, key, self.rotary)
        seed_key_mean(getattr(self.processor, "_glue", None), key, window)
        return self.processor.attend(query, key, value)


class FusedMiniMaxH3TransformerBlock(MiniMaxH3TransformerBlock):  # type: ignore[misc]
    _fusion_attn_fp8: bool = False
    _fusion_ff_fp8: bool = False
    _glue: Launch

    def forward(
        self,
        hidden_states: Any,
        temb: Any,
        adaln_indices: Any,
        rotary_emb: tuple[Any, Any],
        attention_mask: Any | None = None,
    ) -> Any:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        # The attention owns its normed input (no name here holds it), so it goes at its last
        # use, before the out projection's output (1.2 GB each at 15 s); the caller holds this
        # block's input throughout.
        attn_output = self.attn(
            Owned(
                rmsnorm_modulate(
                    self._glue,
                    hidden_states,
                    self.norm1.weight,
                    scale_msa,
                    shift_msa,
                    adaln_indices,
                    self.norm1.eps,
                    fp8=self._fusion_attn_fp8,
                )
            ),
            rotary_emb,
            attention_mask,
        )
        tail = _Tail(self, hidden_states, adaln_indices, gate_msa, scale_mlp, shift_mlp, gate_mlp)
        # The tail writes its rows in place: a view that does not flatten would take a copy.
        return split(tail.run, tail, Cut(1), out=attn_output.contiguous())


class _Tail:
    """A block after its attention, as `split` cuts it by tokens: `hidden = residual + gate_msa *
    attention`, the FFN of its modulated norm, and `hidden + gate_mlp * ffn`, each window written
    into the attention output's own rows (their last use). Beside the block's input only that
    output lives whole; a window's normed rows and FFN are its own (xs-141: three [tokens, 5376]
    row sets, 3.3 GB at 15 s, were whole at once). Every op is per row, so windows are exact."""

    __slots__ = ("block", "device", "dtype", "gates", "index", "residual", "shape")

    def __init__(self, block: Any, residual: Any, index: Any, *gates: Any) -> None:
        self.block, self.residual, self.index, self.gates = block, residual, index, gates
        self.shape, self.dtype, self.device = residual.shape, residual.dtype, residual.device

    def dim(self) -> int:
        return len(self.shape)

    def element_size(self) -> int:
        return int(self.dtype.itemsize)

    def __getitem__(self, index: tuple[slice, ...]) -> slice:
        return index[1]

    def run(self, rows: _Tail | slice, out: Any) -> None:
        """The tail over token `rows`, written into `out` (those rows of the attention output).
        Called on itself (`split` running whole), every row."""
        rows = slice(None) if rows is self else rows
        assert isinstance(rows, slice)
        block, (gate_msa, scale, shift, gate) = self.block, self.gates
        residual, index = self.residual[:, rows], self.index[rows]
        hidden, normed = gate_add_rmsnorm_modulate(
            block._glue,
            residual,
            gate_msa,
            out,
            index,
            block.norm2.weight,
            scale,
            shift,
            block.norm2.eps,
            fp8=block._fusion_ff_fp8,
            hidden_out=out,
        )
        ffn = _FFN(block.ff)(normed)
        del normed
        gate_add(block._glue, hidden, gate, ffn, index, out=out)


class _FFN:
    """The FFN over a window of tokens (`_Tail`), its identity dropout skipped."""

    __slots__ = ("ff",)

    def __init__(self, ff: Any) -> None:
        self.ff = ff

    def __call__(self, normed: Any) -> Any:
        act, down = self.ff.net[0], self.ff.net[2]
        if not isinstance(act, FusedSwiGLU) or act._fusion_fp8:
            return down(act(normed))
        # An int8 down projection quantizes `value * silu(gate)` from the projection's
        # [value | gate] halves itself (the same law as `swiglu`): the product is never written.
        both = act.proj(normed)
        rows = int8_rows(both, (down,), activation="swiglu")
        return down(swiglu(act._glue, both) if rows is None else rows)


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
    document = {"kernels": [GLUE.identity, GLUE_FP8.identity], "plan": CONTRACT}
    return (
        "sha256:"
        + hashlib.sha256(
            b"minimax-h3.glue\0"
            + json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _require_call(function: Callable[..., object], parameters: tuple[str, ...]) -> None:
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
    """One part of the execution plan's outcome, named by its kernel set (`GLUE.name`): on the
    model (`_cozy_fusion_report[name]`), in the executor's log, and in the construction's
    receipts where the Runtime's plan takes reports (0.31: `measurements.execution.plan_reports`),
    so `run show` names a refusal. An install deferred to a request boundary reports again,
    through the same plan, when it lands; one whose build fails never reaches the package, and
    Runtime reports that under the same kernel-set name. A plan without `report` (before 0.31)
    keeps the log line."""
    if hasattr(model, "__dict__"):
        vars(model).setdefault("_cozy_fusion_report", {})[name] = dict(document)
    print(f"[minimax-h3] {name} fusion: {json.dumps(document, sort_keys=True)}", file=sys.stderr)
    report = getattr(plan, "report", None)
    if report is not None:
        report(name, document)


def _record(model: object, plan: Plan, result: Applied) -> Applied:
    record(model, plan, GLUE.name, result.document())
    return result


def _rowwise_leaf(module: Any) -> bool:
    # Semantic hooks must receive the original Tensor, not a quantized carrier.
    return (
        getattr(module, "provider", None) == ROWWISE_FP8
        and not module._forward_pre_hooks
        and not module._forward_hooks
    )


def _pointer(value: Any) -> int:
    try:
        return int(value.data_ptr())
    except RuntimeError:  # the memory's stand-in for an unbound weight has no storage
        return 0


def _tensor_census(module: Any) -> tuple[object, ...]:
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


def _widths(transformer: Any, root: str) -> None:
    config = transformer.config
    widths = (
        int(config.hidden_size),
        int(config.ffn_dim),
        int(config.attention_head_dim),
        6 * int(config.rope_freq_dim),
    )
    _shape(
        widths == (HIDDEN, FFN, HEAD_DIM, ROTARY),
        f"{root}: widths {widths} are not the ones the glue compiles",
    )


def _validate_transformer(transformer: Any, root: str) -> None:
    forwards = (MiniMaxH3Transformer3DModel.forward, AdaLNPrunedMiniMaxH3Transformer.forward)
    _shape(
        type(transformer).forward in forwards and plain_forward(transformer),
        f"{root}: {_type_name(transformer)} overrides the Diffusers MiniMax-H3 forward",
    )
    for index, block in enumerate(transformer.transformer_blocks):
        path = f"{root}.transformer_blocks.{index}"
        attn, ff = block.attn, block.ff
        _shape(
            type(block) is MiniMaxH3TransformerBlock
            and plain_forward(block)
            and type(attn.processor) in (MiniMaxH3AttnProcessor, H3AttnProcessor)
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
    transformer: Any, root: str, glue: Launch
) -> tuple[list[tuple[str, str, str]], int]:
    substitutions: list[tuple[str, str, str]] = []
    fp8_sites = 0
    for index, block in enumerate(transformer.transformer_blocks):
        path = f"{root}.transformer_blocks.{index}"
        attn, ff = block.attn, block.ff
        attn_fp8 = all(_rowwise_leaf(m) for m in (attn.to_q, attn.to_k, attn.to_v))
        ff_fp8 = _rowwise_leaf(ff.net[0].proj)
        out_fp8 = _rowwise_leaf(ff.net[2])
        fp8_sites += attn_fp8 + ff_fp8 + out_fp8

        processor = FusedMiniMaxH3AttnProcessor()
        processor._attention_backend = attn.processor._attention_backend
        processor._parallel_config = attn.processor._parallel_config
        processor._glue = glue
        substitutions.append(
            (f"{path}.attn.processor", _type_name(attn.processor), _type_name(processor))
        )
        attn.set_processor(processor)

        substitutions.append((f"{path}.ff.net.0", _type_name(ff.net[0]), _type_name(FusedSwiGLU)))
        ff.net[0].__class__ = FusedSwiGLU
        ff.net[0]._fusion_fp8 = out_fp8
        ff.net[0]._glue = glue

        substitutions.append((path, _type_name(block), _type_name(FusedMiniMaxH3TransformerBlock)))
        block.__class__ = FusedMiniMaxH3TransformerBlock
        block._fusion_attn_fp8 = attn_fp8
        block._fusion_ff_fp8 = ff_fp8
        block._glue = glue
    return substitutions, fp8_sites


def _release_qkv(attn: Any) -> None:
    current = attn.processor
    if type(current) is MiniMaxH3AttnProcessor:
        processor = H3AttnProcessor()
        processor._attention_backend = current._attention_backend
        processor._parallel_config = current._parallel_config
        attn.set_processor(processor)


def _transformers(pipe: Any) -> Iterator[tuple[str, Any]]:
    """Every MiniMax-H3 transformer the pipeline holds, by component name."""
    seen: set[int] = set()
    for name, member in pipe.components.items():
        if isinstance(member, MiniMaxH3Transformer3DModel) and id(member) not in seen:
            seen.add(id(member))
            yield str(name), member


def _eager(model: object, plan: Plan, code: str, reason: str) -> str:
    _record(model, plan, Applied(False, reason, (), 0, 0.0, code))
    return ""


def install(model: object, pipe: Any, plan: Plan) -> str:
    """Install the fused glue on every H3 transformer `pipe` holds; the plan's identity, or ""
    when eager serves (a `NOT_INSTALLABLE` code, reported). Waits for the kernels' build.
    Fused or eager, every block's attention runs `H3AttnProcessor`."""
    for _, transformer in _transformers(pipe):
        for block in transformer.transformer_blocks:
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
    try:
        _require_calls()
        transformers = tuple(_transformers(pipe))
        for name, transformer in transformers:
            _widths(transformer, name)
            _validate_transformer(transformer, name)
        if not transformers:
            return _eager(model, plan, "fusion_no_transformer", "no MiniMax-H3 transformer to fuse")
        glue = plan.load(kernels(device))
    except KernelsUnavailable as exc:
        if exc.code == "kernels_corrupt":
            raise GlueRefusal("fusion_kernels_corrupt", str(exc)) from exc
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
        installed, sites = _install_transformer(transformer, name, glue)
        if _tensor_census(transformer) != before:
            raise GlueRefusal(
                "fusion_tensor_schema_changed",
                f"{name}: the substitution moved a parameter or buffer",
            )
        substitutions += installed
        fp8_sites += sites
        # PAST THE POINT OF NO RETURN: the forwards are substituted, so a failure refuses.
        heads = int(transformer.config.num_attention_heads)
        warm_ms += warm(glue, device, heads=heads, fp8=sites > 0)
    _record(model, plan, Applied(True, "", tuple(substitutions), fp8_sites, warm_ms))
    return identity()
