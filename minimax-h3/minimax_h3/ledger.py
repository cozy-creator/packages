"""Diagnostic only (xs-089, never released): the VRAM ledger and H2D sources of one H3 denoise.

Per DiT block: Torch allocated/reserved at entry, Torch's peak inside the block, driver free,
Varena's resident and preferred bytes. Per transformer evaluation: the request's store/baked
bytes and prefetch wait (Runtime counters), the NVML PCIe RX integrated at 20 ms, and for two
evaluations a CUPTI trace: memcpy bytes by kind and the kernels' device time by name.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import tempfile
import threading
import time
import zlib
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from cozy_runtime.internal import accel, residency

from . import fit


class _Nvml(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlInit_v2()
        self.handle = ctypes.c_void_p()
        self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle))
        self.rx = 0.0
        self.done = threading.Event()

    def run(self) -> None:
        kbps, last = ctypes.c_uint(), time.monotonic()
        while not self.done.is_set():
            ok = self.lib.nvmlDeviceGetPcieThroughput(self.handle, 1, ctypes.byref(kbps)) == 0
            now = time.monotonic()
            if ok:
                self.rx += kbps.value * 1024 * (now - last)
            last = now
            time.sleep(0.02)


def _native() -> dict[str, Any]:
    memory = residency._MEMORIES.get("cuda:0")
    if memory is None or memory.native is None:
        return {}
    return dict(memory.native.stats())


def _request() -> Any:
    return residency._REQUEST.get()


def _gb(n: float) -> float:
    return round(n / 1e9, 3)


def _profile_summary(path: str) -> dict[str, Any]:
    with open(path) as f:
        events = json.load(f).get("traceEvents", [])
    copies: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    kernels: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    busy = 0.0
    for e in events:
        cat = e.get("cat", "")
        if cat == "gpu_memcpy":
            row = copies[e.get("name", "?")]
            row[0] += 1
            row[1] += float(e.get("args", {}).get("bytes", 0))
            row[2] += float(e.get("dur", 0))
        elif cat == "kernel":
            row = kernels[e.get("name", "?")[:90]]
            row[0] += 1
            row[1] += float(e.get("dur", 0))
            busy += float(e.get("dur", 0))
    top = sorted(kernels.items(), key=lambda kv: -kv[1][1])[:25]
    return {
        "memcpy": {k: {"n": v[0], "gb": _gb(v[1]), "ms": round(v[2] / 1e3, 1)} for k, v in copies.items()},
        "kernel_ms": round(busy / 1e3, 1),
        "top": [[k, v[0], round(v[1] / 1e3, 1)] for k, v in top],
    }


#: A log field's value is clipped at 400 characters and a record holds 8 fields: each record
#: carries the step, its part, the part count and five 390-character pieces of the JSON.
_PIECE = 390
_PIECES = 5


def _emit(tel: Any, step: int, raw: str) -> None:
    # url-safe base64 without padding renders unquoted in the run log's key=value text
    text = base64.urlsafe_b64encode(zlib.compress(raw.encode(), 9)).decode().rstrip("=")
    pieces = [text[i : i + _PIECE] for i in range(0, len(text), _PIECE)]
    parts = [pieces[i : i + _PIECES] for i in range(0, len(pieces), _PIECES)]
    for index, part in enumerate(parts):
        fields = {f"p{j}": piece for j, piece in enumerate(part)}
        tel.log("h3-ledger", step=step, part=index, parts=len(parts), **fields)


@contextmanager
def watch(tel: Any, transformer: Any, evaluations: int, sparse_from: int) -> Iterator[None]:
    device = torch.device("cuda", 0)
    blocks = list(getattr(transformer, "transformer_blocks", []))
    profiled = {2, min(sparse_from + 1, evaluations - 1)}
    nvml = _Nvml()
    nvml.start()
    state: dict[str, Any] = {"step": -1, "rows": [], "t0": 0.0, "prof": None, "base": {}}
    handles = []

    def released() -> dict[str, list[int]]:
        return {k: list(v) for k, v in getattr(accel, "RELEASED", {}).items()}

    def counters() -> dict[str, float]:
        r = _request()
        state["released"] = released()
        return {
            "store": float(getattr(r, "moved", 0)),
            "baked": float(getattr(r, "baked_bytes", 0)),
            "wait": float(getattr(r, "prefetch_wait_ns", 0)),
            "rx": nvml.rx,
        }

    def step_start(module: Any, args: Any, kwargs: Any) -> None:
        state["step"] += 1
        state["rows"] = []
        torch.cuda.synchronize(device)
        state["t0"] = time.monotonic()
        state["base"] = counters()
        if state["step"] in profiled:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
            state["prof"] = prof

    def step_end(module: Any, args: Any, kwargs: Any, out: Any) -> None:
        torch.cuda.synchronize(device)
        before = state.get("released", {})
        now, base, c = time.monotonic(), state["base"], counters()
        after = state["released"]
        rows = state["rows"]
        native = _native()
        summary: dict[str, Any] = {
            "step": state["step"],
            "wanted_gb": state["resident_wanted_gb"],
            "s": round(now - state["t0"], 2),
            "store_gb": _gb(c["store"] - base["store"]),
            "baked_gb": _gb(c["baked"] - base["baked"]),
            "wait_s": round((c["wait"] - base["wait"]) / 1e9, 2),
            "nvml_rx_gb": _gb(c["rx"] - base["rx"]),
            "released": {k: [v[0] - before.get(k, [0, 0])[0], _gb(v[1] - before.get(k, [0, 0])[1])] for k, v in after.items() if v != before.get(k)},
            "alloc_retries": torch.cuda.memory_stats(device).get("num_alloc_retries"),
            "native": {k: _gb(native[k]) for k in ("resident_bytes", "preferred_bytes", "ready_bytes", "reclaimable_bytes", "streaming_limit_bytes") if k in native},
        }
        if rows:
            summary["torch_alloc_gb"] = [_gb(min(r[1] for r in rows)), _gb(max(r[1] for r in rows))]
            summary["torch_reserved_gb"] = [_gb(min(r[2] for r in rows)), _gb(max(r[2] for r in rows))]
            peak = max(rows, key=lambda r: r[6])
            summary["block_peak_gb"] = [_gb(peak[6]), peak[0]]
            summary["driver_free_gb"] = [_gb(min(r[3] for r in rows)), _gb(max(r[3] for r in rows))]
            summary["resident_gb"] = [_gb(min(r[4] for r in rows)), _gb(max(r[4] for r in rows))]
            summary["preferred_gb"] = [_gb(min(r[5] for r in rows)), _gb(max(r[5] for r in rows))]
        prof = state.pop("prof", None)
        if prof is not None:
            try:
                prof.__exit__(None, None, None)
                path = os.path.join(tempfile.gettempdir(), f"h3-ledger-{os.getpid()}-{state['step']}.json")
                prof.export_chrome_trace(path)
                summary["profile"] = _profile_summary(path)
            except Exception as exc:  # a diagnostic never fails the request
                summary["profile_error"] = repr(exc)[:300]
            stats = torch.cuda.memory_stats(device)
            summary["torch_stats"] = {
                k: stats.get(k)
                for k in (
                    "segment.all.current",
                    "segment.large_pool.current",
                    "oversize_segments.current",
                    "oversize_allocations.peak",
                    "inactive_split_bytes.all.current",
                    "active_bytes.all.peak",
                    "reserved_bytes.all.peak",
                    "num_alloc_retries",
                    "num_device_alloc",
                    "num_device_free",
                )
            }
            segments = sorted(
                (
                    (seg["total_size"], sum(b["size"] for b in seg["blocks"] if b["state"] == "active_allocated"))
                    for seg in torch.cuda.memory_snapshot()
                    if seg.get("device") == device.index
                ),
                reverse=True,
            )
            summary["segments_gb"] = [[_gb(t), _gb(a)] for t, a in segments[:40]]
            summary["segments_small_gb"] = _gb(sum(t for t, _ in segments[40:]))
            summary["blocks"] = [[r[0], _gb(r[1]), _gb(r[2]), _gb(r[3]), _gb(r[4]), _gb(r[5]), _gb(r[6])] for r in rows]
        _emit(tel, state["step"], json.dumps(summary, separators=(",", ":")))

    def block_pre(index: int) -> Any:
        def hook(module: Any, args: Any) -> None:
            torch.cuda.reset_peak_memory_stats(device)
            native = _native()
            free, _ = torch.cuda.mem_get_info(device)
            state["pending"] = (
                index,
                torch.cuda.memory_allocated(device),
                torch.cuda.memory_reserved(device),
                free,
                native.get("resident_bytes", 0),
                native.get("preferred_bytes", 0),
            )

        return hook

    def block_post(module: Any, args: Any, out: Any) -> None:
        pending = state.pop("pending", None)
        if pending is not None:
            state["rows"].append((*pending, torch.cuda.max_memory_allocated(device)))

    memory = residency._MEMORIES.get("cuda:0")
    if memory is not None:
        fit.RESIDENT = sum(
            sum(b.nbytes for b in m.groups()) for m in memory.models.values() if m.name.endswith("/transformer")
        )
    state["resident_wanted_gb"] = _gb(fit.RESIDENT)
    handles.append(transformer.register_forward_pre_hook(step_start, with_kwargs=True))
    handles.append(transformer.register_forward_hook(step_end, with_kwargs=True))
    for index, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(block_pre(index)))
        handles.append(block.register_forward_hook(block_post))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        nvml.done.set()
