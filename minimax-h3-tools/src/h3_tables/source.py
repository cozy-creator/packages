"""A granted H3 checkpoint's rows, configs and bytes; tensors written to an output."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.resources import files
from typing import Any

import msgspec
import torch
from cozy_runtime.author import ConformanceError, Context, UnsupportedInput, Weights, canonical_json
from cozy_runtime.weights_store import Checkpoint, Output, TensorMetadata

from .kernel import CurveAdapter, H3Topology, LowRankAdapter, precompute_tables, table_shapes
from .model_config import DIT_COMPONENT
from .plans import Task, TimestepPlan, parse_plan
from .table_layout import TableLayout

TIMESTEP_PLAN = "timestep-plan.json"

_FULL_SPECS_DIGEST = "sha256:3af7354b5080f4c117922157971261fd079f12790f9584615e87b4fdd95ae3e2"
_FULL_CONFIG_DIGEST = "sha256:4150e2b9009aad13cf5a18b9878337346b2bfa7662b3f8037a0286cdaa806382"
_TEXT_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)
DTYPE = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


#: A header tensor: TensorFS dtype, shape and encoding (None when plain).
Row = TensorMetadata
Rows = Mapping[str, Mapping[str, Row]]


def dtype(row: Row) -> str:
    """The Runtime's spelling of a row's dtype: `bf16`, `f16`, `f32`."""
    return row.dtype.lower()


def plain(row: Row, expected: str, shape: tuple[int, ...]) -> bool:
    return row.encoding is None and dtype(row) == expected and tuple(row.shape) == shape


def open_source(ctx: Context, weights: Weights[Any]) -> Checkpoint:
    return ctx.tensorfs_source(weights)


def rows(source: Checkpoint) -> Rows:
    """`[component][key]` in header (construction) order."""
    return {name: component.tensors for name, component in source.components.items()}


def config(source: Checkpoint) -> bytes:
    """The checkpoint's `{component: config}` document."""
    return canonical_json.encode({name: c.config for name, c in source.components.items()})


def timestep_plan(source: Checkpoint, task: Task, *, component: str | None = None) -> TimestepPlan:
    """The source model owns its schedule; package resources are never a fallback."""
    name = component or DIT_COMPONENT[task]
    held = source.components.get(name)
    if held is None or TIMESTEP_PLAN not in held.assets:
        raise UnsupportedInput(
            f"H3 source {name} lacks its required {TIMESTEP_PLAN} asset; use a migrated model",
            code="adaln_plan",
        )
    try:
        plan = parse_plan(held.asset(TIMESTEP_PLAN), task=task)
        stamp = held.config.get("cozy_h3")
        if isinstance(stamp, dict) and stamp.get("modulation") == "adaln-pruned":
            layout = TableLayout.parse(stamp.get("table_keys"))
            available = {(layout.timesteps[row], tag) for row, tag in layout.block_keys}
            required = {(plan.timesteps[row], tag) for row, tag in plan.block_rows}
            if not required <= available:
                raise ValueError("the stored table rows do not cover the source plan")
    except (ValueError, TypeError, OverflowError, ConformanceError) as exc:
        raise UnsupportedInput(
            f"H3 source {name} has an invalid timestep plan or table layout: {exc}",
            code="adaln_plan",
        ) from exc
    return plan


def copy_assets(
    out: Output, source: Checkpoint, component: str, *, target: str | None = None,
    omit: tuple[str, ...] = (),
) -> None:
    """Preserve a component's assets when deriving a projection or renamed component."""
    if out.done is None:
        writer = out.component(target or component)
        held = source.components[component]
        for name in held.assets:
            if name not in omit:
                writer.asset(name, held.asset(name))


def write_config(out: Output, document: bytes) -> None:
    """Each component's config of a `{component: config}` document, before its tensors."""
    for component, section in canonical_json.decode(document).items():
        out.config(component, section)


def read(source: Checkpoint, component: str, key: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    """One plain tensor's stored bytes on the CPU."""
    value = torch.empty(shape, dtype=dtype)
    source.components[component].tensors.read_into(key, memoryview(value.view(torch.uint8).numpy()))
    return value


def put(out: Output, component: str, key: str, value: torch.Tensor) -> int:
    """One plain tensor's bytes into `out`; the bytes written."""
    data = value.detach().to("cpu").contiguous()
    raw = memoryview(data.view(torch.uint8).numpy()).cast("B")
    out.put(component, key, _NAMES[data.dtype], tuple(data.shape), raw)
    return len(raw)


def write_tables(
    ctx: Context,
    *,
    plan: TimestepPlan,
    topology: H3Topology,
    source: Checkpoint,
    component: str,
    outputs: Sequence[Output],
    progress: Callable[[int, int], None],
    target: str | None = None,
    adapter: LowRankAdapter | None = None,
    curve: CurveAdapter | None = None,
) -> tuple[int, int, int]:
    """One DiT's AdaLN tables from `source`'s `component` modulation weights, into `target`
    (default `component`) of each of `outputs` that lacks them; (bytes read, bytes written,
    tables reused)."""
    into = target or component
    tables = table_shapes(topology, plan)
    completed = frozenset(k for k in tables if all(out.has(into, k) for out in outputs))
    read_bytes = written = 0

    def reader(key: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal read_bytes
        ctx.raise_if_cancelled()
        value = read(source, component, key, dtype, shape)
        read_bytes += value.numel() * value.element_size()
        return value

    def writer(key: str, value: torch.Tensor) -> None:
        nonlocal written
        ctx.raise_if_cancelled()
        for out in outputs:
            if not out.has(into, key):
                written += put(out, into, key, value)

    precompute_tables(
        plan=plan, topology=topology, read=reader, write=writer, progress=progress,
        device=torch.device(str(ctx.device)), completed=completed, adapter=adapter, curve=curve,
    )  # fmt: skip
    return read_bytes, written, len(completed)


class _FullSpec(msgspec.Struct, forbid_unknown_fields=True):
    dtype: str
    shape: tuple[int, ...]


class _FullSpecs(msgspec.Struct, forbid_unknown_fields=True):
    config_digest: str
    specs: tuple[tuple[str, _FullSpec], ...]


def official_full_specs(
    config: dict[str, Any],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Load the ordered 638-row serving destination contract without importing Diffusers.

    The existing resource is an ordered pair array, so canonical JSON cannot sort
    away the real factory's state_dict traversal. h3-conform checks every row,
    shape and dtype against the same OfficialH3Pipeline used by inference.
    """
    if canonical_json.digest(config) != _FULL_CONFIG_DIGEST:
        raise ValueError("H3 model config changed without a matching full-spec contract")
    raw = files(__package__).joinpath("assets", "dit-full-specs.json").read_bytes()
    if (
        canonical_json.normalize(raw) != raw
        or canonical_json.digest_bytes(raw) != _FULL_SPECS_DIGEST
    ):
        raise ValueError("H3 full-spec resource is not its exact canonical package bytes")
    document = msgspec.convert(canonical_json.decode(raw), type=_FullSpecs)
    if document.config_digest != _FULL_CONFIG_DIGEST or len(document.specs) != 638:
        raise ValueError("H3 full-spec resource changed its config binding or 638-row census")
    result = {name: (value.dtype, value.shape) for name, value in document.specs}
    if len(result) != len(document.specs):
        raise ValueError("H3 full-spec resource repeats a tensor destination")
    if any(
        dtype not in {"bf16", "f32"} or not shape or any(n <= 0 for n in shape)
        for dtype, shape in result.values()
    ):
        raise ValueError("H3 full-spec resource contains an invalid dtype or shape")
    return result


def text_source_only_keys() -> tuple[str, ...]:
    """Official Qwen rows excluded by the reviewed 50-layer pre-norm conditioner."""
    rows = [
        f"model.language_model.layers.{layer}.{suffix}"
        for layer in range(50, 64)
        for suffix in _TEXT_LAYER_SUFFIXES
    ]
    rows.extend(("lm_head.weight", "model.language_model.norm.weight"))
    if len(rows) != 156:
        raise AssertionError("the text-conditioner source-only census changed")
    return tuple(sorted(rows))
