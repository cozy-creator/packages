"""cozy_runtime.moe's grouped expert GEMV, run by Triton's interpreter on CPU:

    TRITON_INTERPRET=1 python scripts/moe-grouped-experts-proof.py

Each (token, expert) pair's gate-and-up projection equals its reference arithmetic: int8.convrot
bit for bit (int32 sums), bf16 within one bf16 step (f32 sums in another order). The interpreter
truncates a bf16 store where a GPU rounds, so the references truncate too. The whole expert path
(two launches, the gate between) matches each expert's own Linears, for Qwen's gate order and
Hunyuan's; an expert without an address (unbound, not contiguous) takes the sorted path."""

import inspect
from types import SimpleNamespace

import triton.runtime.interpreter as interpreter

# Other Runtime kernel sets read `params` off the JIT function at import; the interpreter's has none.
interpreter.InterpretedFunction.params = property(
    lambda self: [
        SimpleNamespace(name=n, is_constexpr="constexpr" in str(p.annotation))
        for n, p in inspect.signature(self.fn).parameters.items()
    ]
)
import torch  # noqa: E402 (after the interpreter shim)
from torch import nn  # noqa: E402 (after the interpreter shim)

from cozy_runtime.accel import moe  # noqa: E402 (after the interpreter shim)
from cozy_runtime.accel.kernels import INT8_CONVROT  # noqa: E402 (after the interpreter shim)
from cozy_runtime.internal import int8_linear  # noqa: E402 (after the interpreter shim)
from cozy_runtime.internal.encoding.formats import hadamard256  # noqa: E402 (after the interpreter shim)

torch.manual_seed(0)
H, INNER, E, K, T = 512, 256, 16, 4, 5


def launch(kernel, grid, *args):
    kernel[grid[:2]](*args)


def truncated(t32):
    return (t32.contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


class Int8Leaf(nn.Module):
    """An int8.convrot leaf's operands and eager law (`int8_linear.eager`)."""

    provider = INT8_CONVROT

    def __init__(self, weight):
        super().__init__()
        rotated = hadamard256(torch, weight.float()) / 16
        scale = (rotated.abs().amax(-1) / 127).clamp(min=1e-12)
        self.register_buffer(
            "data", torch.round(rotated / scale[:, None]).clamp(-127, 127).to(torch.int8)
        )
        self.register_buffer("scale", scale.float())
        self.bias = None
        self.out_features, self.in_features = weight.shape

    def forward(self, x):
        return int8_linear.eager(torch, x, self.data, self.scale, torch.bfloat16)


for gate_first in (True, False):
    experts = moe.Experts(
        [moe.RoutedExpert(H, INNER, gate_first=gate_first, dtype=torch.bfloat16) for _ in range(E)],
        layer=0,
    )
    for e in experts:
        for lin in (e.gate_and_up_proj, e.down_proj):
            lin.weight.data.normal_(0, 0.05)
    hidden = torch.randn(T, H, dtype=torch.bfloat16)
    index = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    flat = index.reshape(-1)
    for kind in ("bf16", "int8"):
        if kind == "int8":
            for e in experts:
                e.gate_and_up_proj = Int8Leaf(e.gate_and_up_proj.weight.data)
                e.down_proj = Int8Leaf(e.down_proj.weight.data)
        found = experts._operands(torch.device("cpu"))
        assert isinstance(found, tuple) and found[0] == kind, found
        table = found[1]
        up = experts._gemv(launch, kind, table, flat.to(torch.int64), 0, hidden, 2 * INNER, K)
        for p, e in enumerate(flat.tolist()):
            linear, x = experts[e].gate_and_up_proj, hidden[p // K : p // K + 1]
            if kind == "bf16":
                want = truncated(x.float() @ linear.weight.float().T)[0]
            else:
                q, s = int8_linear.quantize_rows(torch, x)
                acc = (q.to(torch.int32) @ linear.data.to(torch.int32).T).float()
                want = truncated(acc.mul_(s).mul_(linear.scale.reshape(1, -1)))[0]
            if kind == "int8":  # exact in int32: the same bits
                assert torch.equal(up[p], want), (kind, p)
            else:  # f32 sums in another order: within one bf16 step
                step = want.float().abs().clamp(min=1e-30) * 2**-7
                assert ((up[p].float() - want.float()).abs() <= step).all(), (kind, p)
        whole = experts._run(launch, kind, table, hidden, index)
        sorted_out = experts._sorted(hidden, index)
        rel = (
            (whole.float() - sorted_out.float()).abs().max() / sorted_out.float().abs().max()
        ).item()
        print(
            f"{kind} gate_first={gate_first}: gate-and-up per pair as its reference; whole path rel |Δ| {rel:.1e}"
        )
        assert rel < 3e-2, rel
experts[3].down_proj.data = experts[3].down_proj.data.t()  # not contiguous: no address
assert isinstance(experts._operands(torch.device("cpu")), str)
print("ok")
