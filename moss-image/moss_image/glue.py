"""MOSS-Image's one package kernel: SiTU, the soft-capped gated activation between a DiT block's FFN
projections, `4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25)` over the up-gate projection's [hidden | gate]
halves.

Eager it is nine elementwise kernels over the [tokens, 16384] projection, 5.4 GB of traffic a block
at 4 MP; fused it reads the projection once and writes the [tokens, 8192] product once, in fp32 with
one rounding to bf16 (eager rounds every step). The Runtime has no gated activation of this form
(`cozy_runtime.fusion` fuses SwiGLU, `int8_rows` folds GELU), so it lives here. The kernel compiles on
the machine that runs it when the executor starts; until it is built, eager serves (`install`).
"""

from __future__ import annotations

from typing import Any

import torch
from cozy_runtime.author.kernels import KernelsUnavailable, Launch, Plan, TritonEntry, TritonKernels

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError as _absent:  # a platform without Triton (macOS, Windows): eager serves
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

#: Columns per program, and its warps: eight bf16 per thread, one 16-byte load from each half.
BLOCK, WARPS = 1024, 4

if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]
    def _situ_kernel(x_ptr, out_ptr, half, stride_x, stride_out, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < half
        hidden = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * stride_x + half + cols, mask=mask, other=0.0).to(tl.float32)
        soft_gate = 4.0 * libdevice.tanh(gate * 0.25) * tl.sigmoid(gate)
        capped = 25.0 * libdevice.tanh(hidden / 25.0)
        tl.store(out_ptr + row * stride_out + cols, (soft_gate * capped).to(tl.bfloat16), mask=mask)

    KERNELS: TritonKernels | None = TritonKernels(
        "moss-image-situ",
        [
            TritonEntry(
                _situ_kernel,
                {"x_ptr": "*bf16", "out_ptr": "*bf16", "half": "i32", "stride_x": "i32",
                 "stride_out": "i32", "BLOCK": "constexpr"},
                {"BLOCK": BLOCK},
                ("x_ptr", "out_ptr", "half", "stride_x", "stride_out"),
                WARPS,
            )
        ],
    )
else:
    KERNELS = None


def situ(launch: Launch, projection: torch.Tensor) -> torch.Tensor | None:
    """SiTU over a bf16 `[..., 2F]` up-gate projection on the GPU, `[..., F]`; None where the kernel
    does not take it (another dtype or layout), and the caller runs eager."""
    width = int(projection.shape[-1])
    half = width // 2
    if not projection.is_cuda or projection.dtype is not torch.bfloat16 or half % 16:
        return None
    rows = projection.reshape(-1, width)
    if rows.stride(-1) != 1 or rows.stride(0) % 16:
        return None
    out = torch.empty(rows.shape[0], half, dtype=projection.dtype, device=projection.device)
    launch(_situ_kernel, (rows.shape[0], triton.cdiv(half, BLOCK), 1), rows, out, half, rows.stride(0), half, BLOCK)
    return out.view(*projection.shape[:-1], half)


def install(plan: Plan, blocks: list[Any]) -> str:
    """Hand each DiT block the SiTU launcher once its kernel is built for `plan.device`: the set's
    identity, or "" where eager serves (no CUDA sm80+, no Triton, or the build failed; reported)."""
    device = torch.device(plan.device)
    if KERNELS is None or device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        reason = TRITON_ABSENT or f"the fused SiTU serves CUDA sm80+, not {device}"
        plan.report("moss-image-situ", {"applied": False, "reason": reason})
        return ""

    def activate(launch: Launch) -> None:
        for block in blocks:
            block.situ_launch = launch
        plan.report(KERNELS.name, {"applied": True, "blocks": len(blocks)})

    try:
        if plan.when_ready is None:
            activate(plan.load(KERNELS))
        elif not plan.when_ready(KERNELS, activate):
            plan.report(KERNELS.name, {"applied": False, "reason": "compiling; fused from the next request on"})
    except KernelsUnavailable as exc:
        if exc.code == "kernels_corrupt":
            raise
        plan.report(KERNELS.name, {"applied": False, "code": exc.code, "reason": str(exc)[:300]})
        return ""
    return KERNELS.identity
