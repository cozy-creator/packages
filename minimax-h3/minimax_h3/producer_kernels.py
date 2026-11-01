"""Caller-owned statistics for the complete H3 sparse producer; no allocating wrappers."""

from __future__ import annotations

import triton
import triton.language as tl

from cozy_runtime.author.kernels import TritonEntry, TritonKernels

from .glue import _key_sum_kernel


@triton.jit  # type: ignore[untyped-decorator]
def _value_absmax_kernel(  # type: ignore[no-untyped-def]
    value_ptr,
    maximum_ptr,
    rows,
    channels,
    stride_t,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    value = tl.load(
        value_ptr + token[:, None] * stride_t + channel[None, :],
        mask=(token[:, None] < rows) & (channel[None, :] < channels),
        other=0.0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(value), axis=0)
    # Each channel's nonnegative maximum is exact and independent of tile ordering.
    # The caller zeroes the owned accumulator once before projecting this head window.
    tl.atomic_max(maximum_ptr + channel, maximum, mask=channel < channels)


VALUE_ABSMAX = TritonEntry(
    _value_absmax_kernel,
    {"value_ptr": "*bf16", "maximum_ptr": "*fp32", "rows": "i32", "channels": "i32",
     "stride_t": "i32", "BLOCK_T": "constexpr", "BLOCK_C": "constexpr"},
    {"BLOCK_T": 64, "BLOCK_C": 32},
    ("value_ptr", "maximum_ptr", "channels", "stride_t"),
    num_warps=4,
    fp_fusion=False,
)

# Reuse the existing bounded token sum on one head's FP32 workspace rows. A separate
# set is required: entry keys do not include pointer dtype. Old GLUE entries stay exact.
KEY_SUM = TritonEntry(
    _key_sum_kernel,
    {"key_ptr": "*fp32", "out_ptr": "*fp32", "tokens": "i32", "stride_t": "i32",
     "HEAD_DIM": "constexpr", "BLOCK_T": "constexpr"},
    {"HEAD_DIM": 128, "BLOCK_T": 64},
    ("key_ptr", "out_ptr", "stride_t"),
    num_warps=4,
    fp_fusion=False,
)

STATS = TritonKernels("minimax-h3-producer-stats", (VALUE_ABSMAX, KEY_SUM), origin=__name__)
