"""The exact MiniMax-H3 modulation calculation, with no artifact policy."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .plans import TimestepPlan

TensorReader = Callable[[str, torch.dtype, tuple[int, ...]], torch.Tensor]
TableWriter = Callable[[str, torch.Tensor], None]
Progress = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class LowRankAdapter:
    """`y += scale · up(down(x))` on every block modulation projection.

    The adapted row is computed in the adapter's own inference order (`LoRALinear.forward`:
    bf16 base output plus the bf16 low-rank update), so a table row is exactly what the
    dynamic module plus its adapter would emit at that timestep. `norm_out` and the time
    embedder carry no adapter.
    """

    rank: int
    scale: float
    read: TensorReader


@dataclass(frozen=True, slots=True)
class CurveAdapter:
    """A LoRA on ComfyUI's AdaLN curve (`<block>.adaln_curve`: A `[r, 8]`, B `[18·hidden, r]` and
    its `[1025, 8]` curve `table`): `y += B·A·lerp(table, t)`, a function of the timestep alone.

    Open-H3-Regenerate-2K's `CurveLinear` order: the curve and the update in float32, cast to the
    projection's bf16 output and added to it.
    """

    rank: int
    read: TensorReader
    #: The curve table's timestep grid points.
    points: int = 1025


@dataclass(frozen=True, slots=True)
class H3Topology:
    hidden_size: int
    num_layers: int
    freq_dim: int
    time_embed_hidden_dim: int
    time_embed_dim: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> H3Topology:
        names = (
            "hidden_size",
            "num_layers",
            "freq_dim",
            "time_embed_hidden_dim",
            "time_embed_dim",
        )
        try:
            values = tuple(int(config[name]) for name in names)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "MiniMax-H3 config is missing its exact timestep/AdaLN dimensions"
            ) from exc
        topology = cls(*values)
        if topology != cls(5376, 50, 256, 5376, 2688):
            raise ValueError(f"unsupported MiniMax-H3 modulation topology {topology!r}")
        return topology


def removed_keys(topology: H3Topology) -> tuple[str, ...]:
    keys = [
        "time_embedder.linear_1.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_2.bias",
        "time_embedder.linear_2.weight",
    ]
    for index in range(topology.num_layers):
        keys.extend(
            (
                f"transformer_blocks.{index}.adaln_proj.linear.bias",
                f"transformer_blocks.{index}.adaln_proj.linear.weight",
            )
        )
    keys.extend(("norm_out.linear.bias", "norm_out.linear.weight"))
    return tuple(sorted(keys))


def table_shapes(topology: H3Topology, plan: TimestepPlan) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        f"transformer_blocks.{index}.adaln_proj.table": (
            len(plan.block_rows),
            6,
            topology.hidden_size,
        )
        for index in range(topology.num_layers)
    }
    shapes["norm_out.table"] = (len(plan.timesteps), 2, topology.hidden_size)
    return shapes


def table_bytes(topology: H3Topology, plan: TimestepPlan) -> int:
    """The exact BF16 bytes one task's tables occupy under this plan."""
    return sum(2 * math.prod(shape) for shape in table_shapes(topology, plan).values())


def source_shapes(
    topology: H3Topology,
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    hidden = topology.hidden_size
    time_hidden = topology.time_embed_hidden_dim
    time_dim = topology.time_embed_dim
    shapes: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
        "time_embedder.linear_1.weight": (
            torch.float32,
            (time_hidden, topology.freq_dim),
        ),
        "time_embedder.linear_1.bias": (torch.float32, (time_hidden,)),
        "time_embedder.linear_2.weight": (torch.float32, (time_dim, time_hidden)),
        "time_embedder.linear_2.bias": (torch.float32, (time_dim,)),
        "norm_out.linear.weight": (torch.bfloat16, (2 * hidden, time_dim)),
        "norm_out.linear.bias": (torch.bfloat16, (2 * hidden,)),
    }
    for index in range(topology.num_layers):
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        shapes[f"{prefix}.weight"] = (torch.bfloat16, (18 * hidden, time_dim))
        shapes[f"{prefix}.bias"] = (torch.bfloat16, (18 * hidden,))
    return shapes


def adapter_shapes(
    topology: H3Topology, rank: int
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    shapes: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
    for index in range(topology.num_layers):
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        shapes[f"{prefix}.lora_down"] = (torch.bfloat16, (rank, topology.time_embed_dim))
        shapes[f"{prefix}.lora_up"] = (torch.bfloat16, (18 * topology.hidden_size, rank))
    return shapes


def _read(
    reader: TensorReader,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    value = reader(name, dtype, shape)
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(
            f"source reader returned {name} as {value.dtype} {tuple(value.shape)}, "
            f"expected {dtype} {shape}"
        )
    return value.to(device=device)


def curve_coordinates(table: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """`lerp(table, t)` on the table's uniform grid over [0, 1], as `CurveContext.pre`."""
    t = timesteps.float()
    if t.min() < 0 or t.max() > 1:
        raise ValueError("AdaLN curve timesteps lie in [0, 1]")
    position = t * (len(table) - 1)
    low = position.floor().long().clamp(max=len(table) - 2)
    return torch.lerp(table[low], table[low + 1], (position - low).unsqueeze(-1))


def _timestep_features(timesteps: torch.Tensor, channels: int) -> torch.Tensor:
    """Diffusers 0.40 ``Timesteps(channels, True, 0)`` without that package."""
    if timesteps.ndim != 1 or channels <= 0 or channels % 2:
        raise ValueError("H3 timestep features require a 1-D input and even channels")
    half = channels // 2
    exponent = -math.log(10_000) * torch.arange(
        0, half, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half
    phase = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


@torch.inference_mode()
def precompute_tables(
    *,
    plan: TimestepPlan,
    topology: H3Topology,
    read: TensorReader,
    write: TableWriter,
    progress: Progress,
    device: torch.device,
    completed: frozenset[str] = frozenset(),
    adapter: LowRankAdapter | None = None,
    curve: CurveAdapter | None = None,
) -> None:
    """Precompute one task checkpoint's tables in Diffusers 0.40 operation order.

    Only one block projection is resident at a time. The source's unrelated 112+ GB never
    enters the process; TensorFS carries it by ObjectRef in the derived snapshot.
    """
    expected = table_shapes(topology, plan)
    adapted = adapter_shapes(topology, adapter.rank) if adapter is not None else {}
    if not completed <= expected.keys():
        raise ValueError("completed table set contains undeclared keys")
    if completed == expected.keys():
        progress(len(expected), len(expected))
        return
    shapes = source_shapes(topology)
    timestep = torch.tensor(plan.timesteps, dtype=torch.float32, device=device)
    time_features = _timestep_features(timestep, topology.freq_dim)
    w1 = _read(
        read,
        "time_embedder.linear_1.weight",
        *shapes["time_embedder.linear_1.weight"],
        device,
    )
    b1 = _read(
        read,
        "time_embedder.linear_1.bias",
        *shapes["time_embedder.linear_1.bias"],
        device,
    )
    w2 = _read(
        read,
        "time_embedder.linear_2.weight",
        *shapes["time_embedder.linear_2.weight"],
        device,
    )
    b2 = _read(
        read,
        "time_embedder.linear_2.bias",
        *shapes["time_embedder.linear_2.bias"],
        device,
    )
    temb = F.linear(F.silu(F.linear(time_features, w1, b1)), w2, b2)
    del time_features, w1, b1, w2, b2

    sparse_rows = torch.tensor(
        [row * 3 + modality for row, modality in plan.block_rows],
        dtype=torch.int64,
        device=device,
    )
    activated = F.silu(temb).to(torch.bfloat16)
    for index in range(topology.num_layers):
        name = f"transformer_blocks.{index}.adaln_proj.table"
        if name in completed:
            progress(index + 1, topology.num_layers + 1)
            continue
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        weight = _read(read, f"{prefix}.weight", *shapes[f"{prefix}.weight"], device)
        bias = _read(read, f"{prefix}.bias", *shapes[f"{prefix}.bias"], device)
        dense = F.linear(activated, weight, bias)
        if adapter is not None:
            down_name, up_name = f"{prefix}.lora_down", f"{prefix}.lora_up"
            down = _read(adapter.read, down_name, *adapted[down_name], device)
            up = _read(adapter.read, up_name, *adapted[up_name], device)
            dense = dense + adapter.scale * F.linear(F.linear(activated, down), up)
            del down, up
        if curve is not None:
            site = f"transformer_blocks.{index}.adaln_curve"
            width = 18 * topology.hidden_size
            down = _read(curve.read, f"{site}.lora_A.weight", torch.float32, (curve.rank, 8), device)
            up = _read(curve.read, f"{site}.lora_B.weight", torch.float32, (width, curve.rank), device)
            table = _read(curve.read, f"{site}.table", torch.float32, (curve.points, 8), device)
            coordinates = curve_coordinates(table, timestep)
            dense = dense + F.linear(F.linear(coordinates, down), up).to(dense.dtype)
            del down, up, table
        dense = dense.reshape(-1, 6, topology.hidden_size)
        write(
            f"transformer_blocks.{index}.adaln_proj.table",
            dense.index_select(0, sparse_rows),
        )
        del weight, bias, dense
        progress(index + 1, topology.num_layers + 1)

    if "norm_out.table" in completed:
        progress(topology.num_layers + 1, topology.num_layers + 1)
        return
    weight = _read(read, "norm_out.linear.weight", *shapes["norm_out.linear.weight"], device)
    bias = _read(read, "norm_out.linear.bias", *shapes["norm_out.linear.bias"], device)
    final = F.linear(activated, weight, bias).reshape(len(plan.timesteps), 2, topology.hidden_size)
    write("norm_out.table", final)
    progress(topology.num_layers + 1, topology.num_layers + 1)
