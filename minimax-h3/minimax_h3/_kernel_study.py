"""KERNEL-ERROR STUDY (proof-only diagnostic, never merge): at chosen denoise steps every H3
attention call through Sol's policy boundary is measured after it served. Per head: cheap Q/K/V
outlier statistics, two logit statistics, and the error of each production kernel (Sage3 FP4 fixed
means, Kitchen INT8, Sage2, Kitchen's sparse INT8) against BF16 SDPA on the same Q/K/V, over
sampled query rows (prefix rows and the rest kept apart). Head-chunked; the served output is
returned untouched."""

from __future__ import annotations

import json
import math
import time

import torch
from torch.nn import functional as F

from cozy_runtime.internal import attention, kernel_ready
from cozy_runtime.internal import attention_sol as _sol
from cozy_runtime.internal.encoding.selection import measure_device

HEADS = 8
#: dense_until_step -> the steps measured: first and last dense step, first sparse step, last step.
PROBE = {4: (0, 3, 4, 7), 10: (0, 9, 10, 29)}
PREFIX_ROWS, REST_ROWS, LOGIT_ROWS = 2048, 6144, 64
#: Per tensor (q, k, v): amax, rms, max over channels of amax_c/rms_c, |mean|/rms (the smooth-K
#: mean's share), then the same after per-channel mean removal (c*), the mean 16-channel group
#: crest after it, and the max / rms per-token norm after it.
STATS = ("amax", "rms", "chan", "bias", "camax", "crms", "cchan", "g16", "nmax", "nrms")
LOGITS = ("entropy", "pmax")
KERNELS = ("sage3", "kint8", "sage2", "sparse")
CALLS: list[dict[str, object]] = []
NOTES: dict[str, str] = {}
TOTAL = {"seconds": 0.0, "failed": 0}
_SERVE = _sol._execute
_SAMPLES: dict[tuple[int, int, str], tuple[torch.Tensor, int, torch.Tensor]] = {}
_SAGE2: dict[str, object] = {}
_TINY = 1e-20


def _round(rows: list[list[float]]) -> list[list[float | None]]:
    """Four significant digits; a non-finite value (a kernel that did not serve) as null."""
    return [[float(f"{x:.4g}") if math.isfinite(x) else None for x in row] for row in rows]


def _sample(tokens: int, prefix: int, device: torch.device) -> tuple[torch.Tensor, int, torch.Tensor]:
    key = (tokens, prefix, str(device))
    if key not in _SAMPLES:
        g = torch.Generator().manual_seed(1027 + tokens + 7 * prefix)
        p = torch.randperm(prefix, generator=g)[:PREFIX_ROWS].sort().values
        r = (prefix + torch.randperm(tokens - prefix, generator=g)[:REST_ROWS]).sort().values
        rows = torch.cat((p, r))
        logit = rows[torch.randperm(rows.numel(), generator=g)[:LOGIT_ROWS]]
        _SAMPLES[key] = (rows.to(device), p.numel(), logit.to(device))
    return _SAMPLES[key]


def _stats(x: torch.Tensor) -> torch.Tensor:
    """(len(STATS), heads) for one head chunk (1, T, h, D)."""
    x = x[0].float()
    t, h, d = x.shape
    amax, mean, ms = x.abs().amax(0), x.mean(0), x.square().mean(0)
    x.sub_(mean)
    camax, cms = x.abs().amax(0), x.square().mean(0)
    g = x.view(t, h, d // 16, 16)
    g16 = (g.abs().amax(-1) / g.square().mean(-1).sqrt().clamp_min(_TINY)).mean((0, 2))
    norm = x.norm(dim=-1)
    return torch.stack(
        (
            amax.amax(1),
            ms.mean(1).sqrt(),
            (amax / ms.sqrt().clamp_min(_TINY)).amax(1),
            (mean.square().sum(1) / ms.sum(1).clamp_min(_TINY)).sqrt(),
            camax.amax(1),
            cms.mean(1).sqrt(),
            (camax / cms.sqrt().clamp_min(_TINY)).amax(1),
            g16,
            norm.amax(0),
            norm.square().mean(0).sqrt(),
        )
    )


def _logits(q: torch.Tensor, k: torch.Tensor, rows: torch.Tensor, scale: float) -> torch.Tensor:
    """(len(LOGITS), heads): mean softmax entropy (over log T) and max probability of sampled rows."""
    logits = torch.matmul(q[0, rows].float().transpose(0, 1), k[0].float().permute(1, 2, 0)) * scale
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(_TINY).log()).sum(-1).mean(-1) / math.log(k.shape[1])
    return torch.stack((entropy, p.amax(-1).mean(-1)))


def _sage2(device: torch.device) -> object:
    """Sage2 (`sageattention`) once it serves: opened off the request, never waited for while it builds."""
    if "rows" in _SAGE2 or "failed" in _SAGE2:
        return _SAGE2.get("rows")
    facts = _SAGE2.setdefault("facts", measure_device(torch, device.index or 0))
    slot = kernel_ready._SLOTS.get(attention._slot("sageattention", facts))  # type: ignore[arg-type]
    try:
        attention.pinned("sageattention", facts, wait=slot is not None and slot.state in ("landed", "ready"))  # type: ignore[arg-type]
    except attention.AttentionRefusal as exc:
        NOTES["sage2"] = f"{exc.code}: {exc}"[:300]
        if exc.code != "attention_kernel_compiling":
            _SAGE2["failed"] = True
        return None
    except Exception as exc:
        NOTES["sage2"] = f"{type(exc).__name__}: {exc}"[:300]
        _SAGE2["failed"] = True
        return None
    _SAGE2["rows"] = attention._dense_rows(attention.BY_NAME["sageattention"])
    NOTES["sage2"] = f"opened after {len(CALLS)} measured calls"
    return _SAGE2["rows"]


def _error(got: torch.Tensor, ref: torch.Tensor, prefix: int) -> torch.Tensor:
    """(2, heads): squared error summed over the sampled prefix rows and the rest."""
    d = (got - ref).square().sum((0, 3))
    return torch.stack((d[:prefix].sum(0), d[prefix:].sum(0)))


def _measure(q, k, v, scale, tokens, prefix, kernels):  # type: ignore[no-untyped-def]
    rows, n_prefix, logit_rows = _sample(tokens, prefix, q.device)
    s = q.shape[-1] ** -0.5 if scale is None else scale
    stats, errors = [], []
    for h in range(0, q.shape[2], HEADS):
        qs, ks, vs = (t[:, :, h : h + HEADS].contiguous() for t in (q, k, v))
        stats.append(torch.cat([*(_stats(t) for t in (qs, ks, vs)), _logits(qs, ks, logit_rows, s)]))
        ref = F.scaled_dot_product_attention(
            qs[:, rows].transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2), scale=scale
        ).transpose(1, 2).float()
        found = [_error(torch.zeros_like(ref), ref, n_prefix)]
        for name in KERNELS:
            serve = kernels.get(name)
            if serve is None:
                found.append(torch.full_like(found[0], float("nan")))
                continue
            try:
                out = serve(qs, ks, vs, scale, prefix) if name == "sparse" else serve(query=qs, key=ks, value=vs, scale=scale)
                found.append(_error(out[:, rows].float(), ref, n_prefix))
                del out
            except Exception as exc:  # a kernel that refuses this call: recorded, never fatal
                NOTES.setdefault(name, f"{type(exc).__name__}: {exc}"[:300])
                found.append(torch.full_like(found[0], float("nan")))
        errors.append(torch.cat(found))
        del ref
    return torch.cat(stats, 1).cpu().tolist(), torch.cat(errors, 1).cpu().tolist()


def _on(path: str, paths: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p + ".") for p in paths)


def _execute(query, key, value, scale, layout, site, **options):  # type: ignore[no-untyped-def]
    output = _SERVE(query, key, value, scale, layout, site, **options)
    if layout.step not in PROBE.get(layout.dense_until_step, ()) or torch.cuda.is_current_stream_capturing():
        return output
    started = time.perf_counter()
    try:
        with torch.no_grad():
            short = query.shape[1] < layout.live_tokens
            tokens = query.shape[1] if short else layout.live_tokens
            prefix = 0 if short else layout.protected_prefix
            q, k, v = (t[:1, :tokens] for t in (query, key, value))
            kernels = {
                "sage3": options.get("dense"),
                "kint8": options.get("prefix_dense"),
                "sage2": _sage2(query.device),
                "sparse": None if short else options.get("sparse"),
            }
            stats, errors = _measure(q, k, v, scale, tokens, prefix, kernels)
        CALLS.append(
            {
                "path": site.path,
                "step": layout.step,
                "dense_until": layout.dense_until_step,
                "tokens": tokens,
                "prefix": prefix,
                "exact": _on(site.path, layout.exact_paths),
                "dense_path": _on(site.path, layout.dense_paths),
                "stats": _round(stats),
                "errors": _round(errors),
            }
        )
    except Exception as exc:  # the study never fails the run it rides
        TOTAL["failed"] += 1
        NOTES.setdefault("probe", f"{type(exc).__name__}: {exc}"[:300])
    TOTAL["seconds"] += time.perf_counter() - started
    return output


def install() -> None:
    _sol._execute = _execute


def drain() -> tuple[bytes, str]:
    """This request's calls as JSON bytes, and a one-line summary; the next request starts empty."""
    schema = {
        "stats": [f"{t}.{s}" for t in "qkv" for s in STATS] + [f"logit.{s}" for s in LOGITS],
        "errors": ["ref.prefix", "ref.rest"] + [f"{k}.{part}" for k in KERNELS for part in ("prefix", "rest")],
        "errors_are": "squared error sums over sampled rows per head; rel L2 = sqrt(kernel / ref)",
    }
    summary = {"calls": len(CALLS), **TOTAL, "notes": dict(NOTES)}
    data = json.dumps({"schema": schema, **summary, "calls": CALLS}, separators=(",", ":")).encode()
    CALLS.clear()
    NOTES.pop("probe", None)
    TOTAL.update(seconds=0.0, failed=0)
    return data, json.dumps(summary)
