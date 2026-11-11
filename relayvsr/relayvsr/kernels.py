"""RelayVSR's two local attentions, each exact: a Triton kernel per query tile that loads only its key
neighbourhood (upstream's, with the shapes as arguments so one build serves every video), and an
eager path of masked SDPA over row bands where the kernels are not installed.

- `window_attention` (Wan): every query of the newest keyframe sees a `window` of tokens in each
  cached keyframe, centred on it and shifted inward at the borders (`spatial_attention.py`).
- `local_attention` (Flash): a query sees, in each key plane, the half-open `window` centred on it,
  cut at the borders, with grouped K/V heads (`_local_attention_triton.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from cozy_runtime import rng
from cozy_runtime.accel.kernels import KernelsUnavailable, Launch, Plan, TritonEntry, TritonKernels

try:
    import triton
    import triton.language as tl

    from cozy_runtime.accel.kernels import jit
except ImportError as _absent:  # pragma: no cover - no Triton on this platform
    if TYPE_CHECKING:
        raise
    TRITON_ABSENT = f"triton does not import ({_absent})"  # type: ignore[unreachable]
else:
    TRITON_ABSENT = ""

Window = tuple[int, int]
#: Query tile rows and columns, key block, and the key columns a tile loads (a power of two).
BH, BW, BN, COLS = 4, 16, 64, 64
#: One launcher per CUDA device index, once its build is installed.
_LAUNCH: dict[int, Launch] = {}


def _band(height: int, width: int) -> int:
    """Query rows per eager SDPA call: about 2k queries."""
    return max(1, 2048 // width)


def window_eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, height: int, width: int, window: Window) -> torch.Tensor:
    """q (b, n, h, d); k, v (b, frames * n, h, d)."""
    b, n, heads, d = q.shape
    frames, (wh, ww) = k.shape[1] // n, window
    out = torch.empty_like(q)
    kx_all = torch.arange(width, device=q.device)
    k5, v5 = (t.view(b, frames, height, width, heads, d) for t in (k, v))
    for y0 in range(0, height, _band(height, width)):
        y1 = min(height, y0 + _band(height, width))
        t0 = min(max(y0 - wh // 2, 0), height - wh)
        t1 = min(max(y1 - 1 - wh // 2, 0), height - wh) + wh
        qy = torch.arange(y0, y1, device=q.device).repeat_interleave(width)
        qx = kx_all.repeat(y1 - y0)
        top = (qy - wh // 2).clamp(0, height - wh)
        left = (qx - ww // 2).clamp(0, width - ww)
        ky = torch.arange(t0, t1, device=q.device).repeat_interleave(width)
        kx = kx_all.repeat(t1 - t0)
        mask = (ky >= top[:, None]) & (ky < top[:, None] + wh) & (kx >= left[:, None]) & (kx < left[:, None] + ww)
        kb = k5[:, :, t0:t1].reshape(b, -1, heads, d)
        vb = v5[:, :, t0:t1].reshape(b, -1, heads, d)
        y = F.scaled_dot_product_attention(
            q[:, y0 * width : y1 * width].transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2), attn_mask=mask.repeat(1, frames)
        )
        out[:, y0 * width : y1 * width] = y.transpose(1, 2)
    return out


def local_eager(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, height: int, width: int, windows: tuple[Window, ...]
) -> torch.Tensor:
    """q (b, hq, n, d); k, v (b, hk, planes * n, d), one window per plane."""
    b, hq, n, d = q.shape
    hk = k.shape[1]
    out = torch.empty_like(q)
    kx_all = torch.arange(width, device=q.device)
    k5, v5 = (t.view(b, hk, len(windows), height, width, d) for t in (k, v))
    for y0 in range(0, height, _band(height, width)):
        y1 = min(height, y0 + _band(height, width))
        qy = torch.arange(y0, y1, device=q.device).repeat_interleave(width)
        qx = kx_all.repeat(y1 - y0)
        keys, values, masks = [], [], []
        for plane, (wh, ww) in enumerate(windows):
            t0, t1 = max(0, y0 - wh // 2), min(height, y1 - 1 + wh // 2)
            ky = torch.arange(t0, t1, device=q.device).repeat_interleave(width)
            kx = kx_all.repeat(t1 - t0)
            dy, dx = ky[None] - qy[:, None], kx[None] - qx[:, None]
            masks.append((dy >= -(wh // 2)) & (dy < wh // 2) & (dx >= -(ww // 2)) & (dx < ww // 2))
            keys.append(k5[:, :, plane, t0:t1].reshape(b, hk, -1, d))
            values.append(v5[:, :, plane, t0:t1].reshape(b, hk, -1, d))
        out[:, :, y0 * width : y1 * width] = F.scaled_dot_product_attention(
            q[:, :, y0 * width : y1 * width], torch.cat(keys, dim=2), torch.cat(values, dim=2), attn_mask=torch.cat(masks, dim=1), enable_gqa=hq != hk
        )
    return out


_ENTRIES: tuple[TritonEntry, ...] = ()
if not TRITON_ABSENT:

    @jit
    def _window_kernel(
        Q, K, V, OUT,
        s_qb, s_qn, s_kb, s_kn, s_ob, s_on,
        heads, key_frames, height, width, wh, ww, rows, scale,
        DIM: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr, BN: tl.constexpr, COLS: tl.constexpr,
    ):  # fmt: skip
        BM: tl.constexpr = BH * BW
        row_tile, segment, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        batch, head = bh // heads, bh % heads
        qy = row_tile * BH + tl.arange(0, BM) // BW
        qx = segment * BW + tl.arange(0, BM) % BW
        qi = (qy * width + qx).to(tl.int64)
        valid_q = (qx < width) & (qy < height)
        d = tl.arange(0, DIM)
        top = tl.minimum(tl.maximum(qy - wh // 2, 0), height - wh)
        left = tl.minimum(tl.maximum(qx - ww // 2, 0), width - ww)
        base_top = tl.minimum(tl.maximum(row_tile * BH - wh // 2, 0), height - wh)
        base_left = tl.minimum(tl.maximum(segment * BW - ww // 2, 0), width - ww)
        q = tl.load(Q + batch.to(tl.int64) * s_qb + qi[:, None] * s_qn + head * DIM + d[None, :], valid_q[:, None], 0)
        acc = tl.full((BM, DIM), 0, tl.float32)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        for kt in range(key_frames):
            for offset in range(tl.cdiv(rows * COLS, BN)):
                p = offset * BN + tl.arange(0, BN)
                ky = base_top + p // COLS
                kx = base_left + p % COLS
                valid_k = (p < rows * COLS) & (kx < width) & (ky < height)
                ki = ((kt * height + ky) * width + kx).to(tl.int64)
                k = tl.load(K + batch.to(tl.int64) * s_kb + ki[None, :] * s_kn + head * DIM + d[:, None], valid_k[None, :], 0)
                scores = tl.dot(q, k).to(tl.float32) * scale
                allowed = (
                    valid_k[None, :]
                    & (ky[None, :] >= top[:, None])
                    & (ky[None, :] < top[:, None] + wh)
                    & (kx[None, :] >= left[:, None])
                    & (kx[None, :] < left[:, None] + ww)
                )
                scores = tl.where(allowed, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
                probabilities = tl.exp2(scores - safe_max[:, None])
                correction = tl.exp2(maximum - safe_max)
                v = tl.load(V + batch.to(tl.int64) * s_kb + ki[:, None] * s_kn + head * DIM + d[None, :], valid_k[:, None], 0)
                acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
                denominator = denominator * correction + tl.sum(probabilities, 1)
                maximum = new_max
        out = acc / denominator[:, None]
        tl.store(OUT + batch.to(tl.int64) * s_ob + qi[:, None] * s_on + head * DIM + d[None, :], out, valid_q[:, None])

    @jit
    def _local_kernel(
        Q, K, V, OUT,
        hq, hk, height, width, planes, cond_planes, cwh, cww, vwh, vww,
        D: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr, BN: tl.constexpr, COLS: tl.constexpr,
    ):  # fmt: skip
        BM: tl.constexpr = BH * BW
        tile_y, tile_x, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        batch, head = bh // hq, bh % hq
        kv_head = head // (hq // hk)
        n = (height * width).to(tl.int64)
        row = tl.arange(0, BM)
        qy, qx = tile_y * BH + row // BW, tile_x * BW + row % BW
        qi = (qy * width + qx).to(tl.int64)
        valid_q = (qy < height) & (qx < width)
        d = tl.arange(0, D)
        q_base = (batch * hq + head).to(tl.int64) * n
        k_base = (batch * hk + kv_head).to(tl.int64) * planes * n
        q = tl.load(Q + (q_base + qi[:, None]) * D + d[None, :], valid_q[:, None], 0)
        acc = tl.full((BM, D), 0, tl.float32)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        scale: tl.constexpr = D**-0.5 * 1.4426950408889634
        for plane in range(planes):
            cond = plane < cond_planes
            wh = tl.where(cond, cwh, vwh)
            ww = tl.where(cond, cww, vww)
            rows = wh + BH - 1
            top, left = tile_y * BH - wh // 2, tile_x * BW - ww // 2
            for offset in range(tl.cdiv(rows * COLS, BN)):
                p = offset * BN + tl.arange(0, BN)
                ky, kx = top + p // COLS, left + p % COLS
                valid_k = (p < rows * COLS) & (ky >= 0) & (ky < height) & (kx >= 0) & (kx < width)
                ki = plane * n + (ky * width + kx).to(tl.int64)
                k = tl.load(K + (k_base + ki[None, :]) * D + d[:, None], valid_k[None, :], 0)
                scores = tl.dot(q, k).to(tl.float32) * scale
                allowed = (
                    valid_k[None, :]
                    & (ky[None, :] >= qy[:, None] - wh // 2)
                    & (ky[None, :] < qy[:, None] + wh // 2)
                    & (kx[None, :] >= qx[:, None] - ww // 2)
                    & (kx[None, :] < qx[:, None] + ww // 2)
                )
                scores = tl.where(allowed, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
                probabilities = tl.exp2(scores - safe_max[:, None])
                correction = tl.exp2(maximum - safe_max)
                v = tl.load(V + (k_base + ki[:, None]) * D + d[None, :], valid_k[:, None], 0)
                acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
                denominator = denominator * correction + tl.sum(probabilities, 1)
                maximum = new_max
        tl.store(OUT + (q_base + qi[:, None]) * D + d[None, :], acc / denominator[:, None], valid_q[:, None])

    _SHAPE = {"BH": BH, "BW": BW, "BN": BN, "COLS": COLS}
    _ENTRIES = (
        TritonEntry(
            _window_kernel,
            {
                **dict.fromkeys(("Q", "K", "V", "OUT"), "*bf16"),
                **dict.fromkeys(("s_qb", "s_qn", "s_kb", "s_kn", "s_ob", "s_on"), "i64"),
                **dict.fromkeys(("heads", "key_frames", "height", "width", "wh", "ww", "rows"), "i32"),
                "scale": "fp32",
                **dict.fromkeys(("DIM", *_SHAPE), "constexpr"),
            },
            {"DIM": 128, **_SHAPE},
            ("Q", "K", "V", "OUT"),
            num_warps=4,
            num_stages=2,
        ),
        TritonEntry(
            _local_kernel,
            {
                **dict.fromkeys(("Q", "K", "V", "OUT"), "*bf16"),
                **dict.fromkeys(("hq", "hk", "height", "width", "planes", "cond_planes", "cwh", "cww", "vwh", "vww"), "i32"),
                **dict.fromkeys(("D", *_SHAPE), "constexpr"),
            },
            {"D": 64, **_SHAPE},
            ("Q", "K", "V", "OUT"),
            num_warps=4,
            num_stages=2,
        ),
    )

KERNELS = TritonKernels("relayvsr-attention", _ENTRIES)


def _launcher(t: torch.Tensor) -> Launch | None:
    if t.device.type != "cuda" or t.dtype != torch.bfloat16:
        return None
    return _LAUNCH.get(t.device.index or 0)


def window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, height: int, width: int, window: Window) -> torch.Tensor:
    """Wan's keyframe self-attention, q (b, n, heads, 128) against every cached keyframe's k, v
    (b, frames * n, heads, 128)."""
    launch = _launcher(q)
    if launch is None:
        return window_eager(q, k, v, height, width, window)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    b, n, heads, d = q.shape
    out = torch.empty_like(q)
    wh, ww = window
    grid = (triton.cdiv(height, BH), triton.cdiv(width, BW), b * heads)
    launch(
        _window_kernel, grid, q, k, v, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), out.stride(0), out.stride(1),
        heads, k.shape[1] // n, height, width, wh, ww, min(height, wh + BH - 1), d**-0.5 * 1.4426950408889634,
        d, BH, BW, BN, COLS,
    )  # fmt: skip
    return out


def local_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, height: int, width: int, windows: tuple[Window, ...]
) -> torch.Tensor:
    """Flash's attention, q (b, hq, n, 64) against key planes k, v (b, hk, planes * n, 64): the
    first plane's window, or the two keyframes' then the video frames' (at most two windows)."""
    launch = _launcher(q)
    if launch is None:
        return local_eager(q, k, v, height, width, windows)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    b, hq, _, d = q.shape
    cond = sum(1 for w in windows if w == windows[0])
    video = windows[-1]
    out = torch.empty_like(q)
    grid = (triton.cdiv(height, BH), triton.cdiv(width, BW), b * hq)
    launch(
        _local_kernel, grid, q, k, v, out,
        hq, k.shape[1], height, width, len(windows), cond, *windows[0], *video,
        d, BH, BW, BN, COLS,
    )  # fmt: skip
    return out


def _check(launch: Launch, device: torch.device) -> str:
    """Each kernel against its eager path on a small problem: what differs, or ""."""
    generator = rng.generator(0)

    def rand(*shape: int) -> torch.Tensor:
        return rng.noise(shape, generator, dtype=torch.bfloat16, device=device)

    _LAUNCH[device.index or 0] = launch
    try:
        h, w = 26, 44
        q, k, v = rand(1, h * w, 2, 128), rand(1, 3 * h * w, 2, 128), rand(1, 3 * h * w, 2, 128)
        error = (window_attention(q, k, v, h, w, (22, 40)).float() - window_eager(q, k, v, h, w, (22, 40)).float()).abs().max()
        if error > 0.05:
            return f"window attention differs from eager by {float(error):.3g}"
        h, w = 20, 36
        windows = ((8, 8), (8, 8), (4, 4), (4, 4))
        q, k, v = rand(1, 4, h * w, 64), rand(1, 2, 4 * h * w, 64), rand(1, 2, 4 * h * w, 64)
        error = (local_attention(q, k, v, h, w, windows).float() - local_eager(q, k, v, h, w, windows).float()).abs().max()
        if error > 0.05:
            return f"local attention differs from eager by {float(error):.3g}"
        return ""
    finally:
        del _LAUNCH[device.index or 0]


def install(plan: Plan) -> str:
    """Serve both attentions from the kernels on this GPU, once built and checked against eager;
    "" where eager serves. Either way the plan's report says which, and why."""
    device = torch.device(plan.device)
    reason = ""
    if device.type != "cuda":
        reason = f"the kernels serve CUDA, not {device}"
    elif torch.cuda.get_device_capability(device) < (8, 0):
        reason = "the kernels need bf16 tensor cores (sm80+)"
    elif TRITON_ABSENT:
        reason = TRITON_ABSENT
    launch: Launch | None = None
    if not reason:
        try:
            launch = plan.load(KERNELS)
            reason = _check(launch, device)
        except KernelsUnavailable as exc:
            reason = f"{exc.code}: {exc}"
    plan.report(KERNELS.name, {"applied": not reason, **({"reason": reason} if reason else {})})
    if reason or launch is None:
        return ""
    _LAUNCH[device.index or 0] = launch
    return KERNELS.identity

