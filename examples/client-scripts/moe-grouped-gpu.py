# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.32", "msgspec>=0.19", "torch==2.14.0"]
# ///
"""cozy_runtime.moe's grouped expert GEMV on a GPU, against each expert's own Linears (se-085).

    cozy run ./examples/client-scripts/moe-grouped-gpu.py --on <machine>

Qwen3.6-35B-A3B's expert geometry (2048 -> 2x512 -> 2048) and Hunyuan's gate order, random weights:
- bf16: each pair's gate-and-up projection within one bf16 step of cuBLAS's; the whole path close;
- int8.convrot: each pair's projection bit for bit `int8_linear.eager`'s (exact int32 sums);
- a CUDA graph of the grouped path, replayed with new tokens and routing, equals the eager call;
- a decode-shaped layer (1 token, 8 of 256 experts): microseconds per layer, grouped and sorted.
"""

from __future__ import annotations

import json
import time

import msgspec
import torch
from torch import nn

from cozy_runtime import moe
from cozy_runtime.author import App, Telemetry
from cozy_runtime.author.kernels import INT8_CONVROT
from cozy_runtime.internal import int8_linear
from cozy_runtime.internal.encoding.formats import hadamard256

H, INNER, K = 2048, 512, 8

app = App()


class Nothing(msgspec.Struct):
    pass


class Passed(msgspec.Struct):
    results: str


def launch(kernel, grid, *args):  # type: ignore[no-untyped-def]
    kernel[grid[:2]](*args)


class Int8Leaf(nn.Module):
    """An int8.convrot leaf's operands and eager law (`int8_linear.eager`)."""

    provider = INT8_CONVROT

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        rotated = hadamard256(torch, weight.float()) / 16
        scale = (rotated.abs().amax(-1) / 127).clamp(min=1e-12)
        self.register_buffer(
            "data", torch.round(rotated / scale[:, None]).clamp(-127, 127).to(torch.int8)
        )
        self.register_buffer("scale", scale.float())
        self.bias = None
        self.out_features, self.in_features = weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_linear.eager(torch, x, self.data, self.scale, torch.bfloat16)


def layer(experts: int, gate_first: bool) -> moe.Experts:
    built = moe.Experts(
        [
            moe.RoutedExpert(H, INNER, gate_first=gate_first, device="cuda", dtype=torch.bfloat16)
            for _ in range(experts)
        ],
        layer=0,
    )
    for expert in built:
        for linear in (expert.gate_and_up_proj, expert.down_proj):
            linear.weight.data.normal_(0, 0.02)
    return built


def routing(tokens: int, experts: int) -> torch.Tensor:
    return torch.stack([torch.randperm(experts, device="cuda")[:K] for _ in range(tokens)])


def check(experts: moe.Experts, kind: str, tokens: int) -> dict[str, float]:
    hidden = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda")
    index = routing(tokens, len(experts))
    found = experts._operands(hidden.device)
    assert found is not None and found[0] == kind, found
    table = found[1]
    flat = index.reshape(-1)
    up = experts._project(launch, kind, table, 0, hidden, flat, 2 * INNER, K)
    worst = 0.0
    for p, e in enumerate(flat.tolist()):
        linear, x = experts[e].gate_and_up_proj, hidden[p // K : p // K + 1]
        want = linear(x)[0]
        if kind == "int8":
            assert torch.equal(up[p], want), (kind, p)
        else:
            step = want.float().abs().clamp(min=1e-30) * 2**-7
            assert ((up[p].float() - want.float()).abs() <= step).all(), (kind, p)
    whole, sorted_out = experts._run(launch, hidden, index), experts._sorted(hidden, index)
    assert whole is not None
    worst = (
        (whole.float() - sorted_out.float()).abs().max() / sorted_out.float().abs().max()
    ).item()
    assert worst < 2e-2, worst
    # One graph of the grouped path, replayed with other tokens and routing.
    static_hidden, static_index = hidden.clone(), index.clone()
    experts._run(launch, static_hidden, static_index)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = experts._run(launch, static_hidden, static_index)
    static_hidden.copy_(torch.randn_like(static_hidden))
    static_index.copy_(routing(tokens, len(experts)))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, experts._run(launch, static_hidden, static_index)), (
        "graph replay"
    )
    return {"whole_path_rel": round(worst, 5)}


def timed(call, repeats: int = 50) -> float:  # type: ignore[no-untyped-def]
    call()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()
    return round((time.perf_counter() - started) / repeats * 1e6, 1)


@app.entrypoint
def main(payload: Nothing, tel: Telemetry) -> Passed:
    torch.manual_seed(0)
    results: dict[str, object] = {"gpu": torch.cuda.get_device_name()}

    def passed(name: str, value: object) -> None:  # logged as it passes: the run log keeps it
        results[name] = value
        tel.log(f"PASS {name}: {json.dumps(value)}")

    for gate_first in (True, False):
        experts = layer(16, gate_first)
        passed(f"bf16 gate_first={gate_first}", check(experts, "bf16", 5))
        for expert in experts:
            expert.gate_and_up_proj = Int8Leaf(expert.gate_and_up_proj.weight.data)
            expert.down_proj = Int8Leaf(expert.down_proj.weight.data)
        passed(f"int8 gate_first={gate_first}", check(experts, "int8", 5))
    decode = layer(256, True)
    hidden, index = torch.randn(1, H, dtype=torch.bfloat16, device="cuda"), routing(1, 256)
    grouped = timed(lambda: decode._run(launch, hidden, index))
    passed(
        "decode layer us (bf16): grouped, sorted",
        (grouped, timed(lambda: decode._sorted(hidden, index))),
    )
    tel.log(f"ALL PASSED on {results['gpu']}")
    return Passed(json.dumps(results))
