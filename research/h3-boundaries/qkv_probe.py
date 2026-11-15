"""Screen the real H3 head-window Q/K/V projections, stopping before norm/RoPE/attention.

Run with a CUDA torch interpreter and compatible Runtime/Diffusers dependencies. If only
torch is in that interpreter, --dependency-site appends another environment's site-packages
AFTER torch imports (it does not install or modify either environment).

The baseline is the original _Heads.__call__ from packages e4b179bb; the candidate calls
the worktree's actual method. Both stop at the same projection boundary.
Weights are synthetic INT8 payloads, with real Runtime native leaves/Rows/GEMMs. This is
an operator screen, never full-model, attention, or output-quality qualification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import weakref
from unittest.mock import patch

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dependency-site", type=Path)
parser.add_argument("--tokens", type=int, default=1024)
parser.add_argument("--repetitions", type=int, default=20)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.dependency_site:
    sys.path.append(str(args.dependency_site.resolve()))
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "minimax-h3"))

from cozy_runtime.accel import kernels, lora  # noqa: E402 (after dependency-path setup)
from cozy_runtime.accel.rows import rows  # noqa: E402
from cozy_runtime.internal import int8_linear, prebuilt  # noqa: E402
from cozy_runtime.internal.encoding.leaves import _native_leaf_class  # noqa: E402
import minimax_h3.glue as glue  # noqa: E402

WIDTH, HEADS, DIM, WINDOW = 5376, 16, 128, 8
device = torch.device("cuda", 0)
torch.manual_seed(20261115)
torch.cuda.set_device(device)
int8_linear.prime(torch, 0)
assert int8_linear.native(torch, 0) is not None, "native INT8 must serve; no fallback screen"
native_quantize = prebuilt.Prebuilt.int8_quantize
quantized: list[weakref.ReferenceType[torch.Tensor]] = []


def count_quantize(self, *values, **kwargs):
    payload, scale = native_quantize(self, *values, **kwargs)
    quantized.append(weakref.ref(payload))
    return payload, scale


class ProjectionProbe(glue.H3AttnProcessor):
    """Keep the actual _Heads projection calls; observe their boundary instead of attention."""

    def __init__(self):
        self.outputs = []
        self.record = True
        self.check_release = False

    def rope(self, attn, query, key, rotary):
        assert not kernels._SHARED.get(), "scope escaped into RoPE"
        if self.check_release:
            assert all(ref() is None for ref in quantized), "operand retained past projections"
        return query, key

    def attend(self, query, key, value, attention_mask=None):
        if self.record:
            self.outputs.append((query, key, value))
        return query


class Factors:
    def __init__(self):
        self.down = torch.randn(4, WIDTH, device=device, dtype=torch.bfloat16) * 0.01
        self.up = torch.randn(HEADS * DIM, 4, device=device, dtype=torch.bfloat16) * 0.01

    def factors(self):
        return self.down, self.up, 0.5

    def __call__(self, x, out):
        lora.accumulate(x, out, *self.factors())


probe = ProjectionProbe()
with torch.device("meta"):
    attention = glue.MiniMaxH3Attention(WIDTH, HEADS, DIM, processor=probe)
leaf = _native_leaf_class(torch)
for name in ("to_q", "to_k", "to_v"):
    data = torch.randint(-127, 128, (HEADS * DIM, WIDTH), device=device, dtype=torch.int8)
    scales = torch.full((HEADS * DIM,), 0.001, device=device)
    setattr(attention, name, leaf(None, {"weight": data, "scale": scales}, None,
                                torch.bfloat16, kernels.INT8_CONVROT, "int8.convrot", WIDTH, HEADS * DIM))


def release_implicit():
    held = getattr(kernels._RECENT, "held", None)
    if held is not None:
        kernels._retire(held)
    kernels._RECENT.last = None


candidate_heads = glue._Heads.__call__


def baseline_heads(self, heads):
    """Verbatim _Heads.__call__ body at packages e4b179bb (excluding its annotation)."""
    window = slice(0, self.attn.heads) if heads is self else heads
    assert isinstance(window, slice)
    d = self.attn.head_dim
    start, stop = window.start * d, window.stop * d
    query, key, value = (
        project(self.hidden, start, stop).unflatten(-1, (-1, d))
        for project in self.projections
    )
    query, key = self.processor.rope(self.attn, query, key, self.rotary)
    glue.seed_key_mean(getattr(self.processor, "_glue", None), key, window)
    return self.processor.attend(query, key, value)


def call(hidden, scoped):
    projections = tuple(rows(getattr(attention, name)) for name in ("to_q", "to_k", "to_v"))
    assert all(project is not None for project in projections)
    heads = glue._Heads(probe, attention, hidden, None, projections)
    with patch.object(glue._Heads, "__call__", candidate_heads if scoped else baseline_heads):
        for start in range(0, HEADS, WINDOW):
            heads(slice(start, start + WINDOW))


def verify(hidden, label, expected):
    sides = []
    counts = []
    for scoped in (False, True):
        release_implicit()
        quantized.clear()
        probe.outputs.clear()
        probe.record = True
        probe.check_release = scoped and label != "no_grad"
        with patch.object(prebuilt.Prebuilt, "int8_quantize", count_quantize):
            call(hidden, scoped)
        torch.cuda.synchronize()
        counts.append(len(quantized))
        sides.append([[part.cpu() for part in window] for window in probe.outputs])
        probe.outputs.clear()
    assert counts == expected, (label, counts, expected)
    assert all(torch.equal(a, b) for wa, wb in zip(*sides, strict=True)
               for a, b in zip(wa, wb, strict=True)), label
    release_implicit()
    return {"case": label, "quantizer_calls_before_after": counts,
            "windows": HEADS // WINDOW, "qkv_outputs_bit_identical": True,
            "shared_operand_released_before_rope": label != "no_grad"}


cases = []
with torch.no_grad():
    hidden = torch.randn(1, args.tokens, WIDTH, device=device, dtype=torch.bfloat16)
    cases.append(verify(hidden, "no_grad", [1, 1]))
with torch.inference_mode():
    hidden = torch.randn(1, args.tokens, WIDTH, device=device, dtype=torch.bfloat16)
    cases.append(verify(hidden, "inference", [6, 2]))
    factors = [Factors() for _ in range(3)]
    handles = [getattr(attention, name).register_forward_hook(lora.Hook(lambda f=f: f))
               for name, f in zip(("to_q", "to_k", "to_v"), factors, strict=True)]
    cases.append(verify(hidden, "inference_armed_lora", [6, 2]))
    for handle in handles:
        handle.remove()
    packed = int8_linear.quantize(torch, hidden)
    assert packed is not None
    cases.append(verify(packed, "prequantized", [0, 0]))
    del packed
    handle = attention.to_q.register_forward_pre_hook(lambda module, values: None)
    assert glue._Heads.of(probe, attention, hidden, None, None) is None
    handle.remove()

    probe.record = False
    probe.check_release = False
    timings = {False: [], True: []}
    peaks = {}
    for scoped in (False, True):
        for _ in range(4):
            call(hidden, scoped)
    for iteration in range(args.repetitions):
        for scoped in ((False, True) if iteration % 2 == 0 else (True, False)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call(hidden, scoped)
            end.record()
            end.synchronize()
            timings[scoped].append(start.elapsed_time(end))
    for scoped in (False, True):
        release_implicit()
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call(hidden, scoped)
        torch.cuda.synchronize()
        peaks[scoped] = torch.cuda.max_memory_allocated() - before
    assert peaks[True] <= peaks[False], peaks

    # Capture both inference and versioned no-grad tensors: native leaves disable implicit
    # sharing while recording either kind. Count real native quantizer calls during capture.
    def capture_check(graph_input):
        probe.record = True
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                probe.outputs.clear()
                call(graph_input, True)
        torch.cuda.current_stream().wait_stream(stream)
        want = [[part.clone() for part in window] for window in probe.outputs]
        counts = []
        for scoped in (False, True):
            probe.outputs.clear()
            quantized.clear()
            release_implicit()
            graph = torch.cuda.CUDAGraph()
            with patch.object(prebuilt.Prebuilt, "int8_quantize", count_quantize), torch.cuda.graph(graph):
                call(graph_input, scoped)
            counts.append(len(quantized))
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.equal(a, b) for wa, wb in zip(want, probe.outputs, strict=True)
                       for a, b in zip(wa, wb, strict=True))
        assert counts == [6, 2], counts
        return {"quantizer_calls_before_after": counts, "replay_bit_identical": True}

    graph_checks = {"inference": capture_check(hidden)}
    with torch.inference_mode(False), torch.no_grad():
        versioned = hidden.clone()
        assert not versioned.is_inference()
        graph_checks["no_grad"] = capture_check(versioned)
    del versioned

    traces = {}
    probe.record = False
    probe.outputs.clear()
    for scoped in (False, True):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            call(hidden, scoped)
            torch.cuda.synchronize()
        events = [event.name for event in profile.events()
                  if event.device_type == torch.autograd.DeviceType.CUDA and "quantize_kernel" in event.name]
        traces[str(scoped)] = events
    assert [len(traces["False"]), len(traces["True"])] == [6, 2], traces

payload_bytes = args.tokens * (WIDTH + 4)
result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
          "capability": torch.cuda.get_device_capability(), "tokens": args.tokens,
          "hidden_width": WIDTH, "heads": HEADS, "heads_per_window": WINDOW,
          "precision": "BF16 input/output, INT8 ConvRot256 payload, whole-row FP32 scale",
          "seed": 20261115, "cases": cases, "unknown_pre_hook_declines_rows": True,
          "graph_capture_replay": graph_checks,
          "median_ms_before_after": [statistics.median(timings[s]) for s in (False, True)],
          "samples_ms_before_after": [timings[s] for s in (False, True)],
          "peak_allocation_bytes_before_after": [peaks[s] for s in (False, True)],
          "quantizer_kernel_events_before_after": traces,
          "materialized_quantized_bytes_before_after": [6 * payload_bytes, 2 * payload_bytes],
          "avoided_logical_bf16_input_bytes": 4 * args.tokens * WIDTH * 2,
          "glue_sha256": hashlib.sha256(Path(glue.__file__).read_bytes()).hexdigest(),
          "native_gemm_sha256": hashlib.sha256(Path(kernels._kernels._gemm.__file__).read_bytes()).hexdigest(),
          "boundary": "synthetic QKV projections only; excludes norm/RoPE/attention/full model"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items()
                  if key not in ("samples_ms_before_after", "quantizer_kernel_events_before_after")}, indent=2))
