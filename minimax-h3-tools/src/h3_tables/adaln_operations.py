"""Independent H3 generating-weight projections, table banks and their attachment."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import msgspec
import torch
from cozy_runtime.author import (
    Context,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    Weights,
    canonical_json,
    invocable,
)
from cozy_runtime.weights_store import Checkpoint, Derivation

from ._memo import ADALN
from .kernel import H3Topology, removed_keys, source_shapes, table_shapes
from .model_config import (
    COMPONENTS,
    DIT_COMPONENT,
    dual_adaln_pruned_config,
    dual_full_config,
    parse_full_config,
)
from .order import current_order, full_order
from .plans import Task, TimestepPlan
from .source import (
    TIMESTEP_PLAN, Rows, config, copy_assets, open_source, plain, rows,
    timestep_plan, write_config, write_tables,
)
from .table_layout import TableLayout

_BINDING = "generating_projection_digest"
#: A projection's or bank's own description, an asset of its DiT component.
METADATA = "adaln.json"
_NUMERICS = "h3-fp32-time-bf16-adaln/1"
_TASKS: tuple[Task, ...] = ("fl2va", "ref2va")


class Selection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    projection: ModelArtifact | None
    ready: bool
    plan_digest: str
    steps: list[int]


def _asset(name: str) -> bytes:
    return files(__package__).joinpath("assets", name).read_bytes()


def _plan(source: Checkpoint, task: Task) -> TimestepPlan:
    return timestep_plan(source, task)


def _sections() -> dict[str, dict[str, Any]]:
    return parse_full_config(_asset("model-config.json"))


def _topology(task: Task) -> H3Topology:
    return H3Topology.from_config(_sections()[DIT_COMPONENT[task]])


def _projection_config(task: Task, topology: H3Topology | None = None) -> bytes:
    topology = topology or _topology(task)
    return canonical_json.encode(
        {
            "schema": "h3-adaln-weights/1",
            "task": task,
            "topology": {
                name: getattr(topology, name)
                for name in (
                    "hidden_size",
                    "num_layers",
                    "freq_dim",
                    "time_embed_hidden_dim",
                    "time_embed_dim",
                )
            },
        }
    )


def _body_kind(present: Rows) -> bool:
    """Return whether the exact assembled source is already pruned."""
    current = current_order(_asset("whole-order.json")).rows
    actual = {(component, key) for component, keys in present.items() for key in keys}
    if actual == set(current):
        return True
    if actual == set(full_order(_sections(), current)):
        return False
    raise UnsupportedInput(
        "AdaLN preparation requires one assembled full or pruned H3 model", code="adaln_source"
    )


def _bindings(document: dict[str, Any]) -> dict[Task, str]:
    result: dict[Task, str] = {}
    for task, component in DIT_COMPONENT.items():
        row = document.get(component)
        stamp = row.get("cozy_h3") if isinstance(row, dict) else None
        digest = stamp.get(_BINDING) if isinstance(stamp, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 71
            or not digest.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in digest[7:])
        ):
            raise UnsupportedInput(
                "H3 generating-weight binding is not an exact digest", code="adaln_binding"
            )
        result[task] = digest
    return result


def _validate_body_config(raw: bytes, pruned: bool) -> dict[str, Any]:
    """The body's `{component: config}` document, checked against its supported constructor."""
    if not pruned:
        if raw != dual_full_config(_sections()):
            raise UnsupportedInput(
                "H3 full-body config differs from its supported constructor", code="adaln_config"
            )
        return {}
    value = canonical_json.decode(raw)
    expected = canonical_json.decode(_asset("model-config.json"))
    if not isinstance(value, dict) or set(value) != set(COMPONENTS):
        raise UnsupportedInput(
            "H3 pruned config has an unexpected component set", code="adaln_config"
        )
    document: dict[str, Any] = canonical_json.decode(raw)
    for task, component in DIT_COMPONENT.items():
        row = value.get(component)
        stamp = row.get("cozy_h3") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or not isinstance(stamp, dict)
            or set(stamp)
            not in (
                {"task", "modulation", "table_keys"},
                {"task", "modulation", "table_keys", _BINDING},
            )
            or stamp["task"] != task
            or stamp["modulation"] != "adaln-pruned"
            or not isinstance(stamp["table_keys"], dict)
        ):
            raise UnsupportedInput(
                "H3 pruned config has no task/table metadata", code="adaln_config"
            )
        TableLayout.parse(stamp["table_keys"])
        # A previous supported bank may be replaced while keeping constructor facts.
        del row["cozy_h3"]
    if value != expected:
        raise UnsupportedInput("H3 pruned body changed its constructor facts", code="adaln_config")
    return document


def _validate_generators(present: Rows, task: Task, topology: H3Topology | None = None) -> None:
    rows_of = present.get(DIT_COMPONENT[task], {})
    for key, (dtype, shape) in source_shapes(topology or _topology(task)).items():
        row = rows_of.get(key)
        if row is None or not plain(row, "f32" if dtype == torch.float32 else "bf16", shape):
            raise UnsupportedInput(
                f"H3 generating weight {key} must preserve its exact plain dtype and shape",
                code="adaln_generating_weights",
            )


def _metadata(source: Checkpoint, task: Task) -> bytes | None:
    held = source.components.get(DIT_COMPONENT[task])
    return held.asset(METADATA) if held is not None and METADATA in held.assets else None


def _project(ctx: Context, source: Checkpoint, task: Task, output: str) -> ModelArtifact:
    """One DiT's generating weights alone, copied with their objects."""
    _validate_generators(rows(source), task)
    component = DIT_COMPONENT[task]
    order = tuple((component, key) for key in removed_keys(_topology(task)))
    with ctx.output(output, Derivation(base=None, sources=(source,), order=order)) as out:
        if out.done is None:
            copy_assets(out, source, component, omit=(METADATA,))
            out.component(component).asset(METADATA, _projection_config(task))
        for row in order:
            if not out.has(*row):
                out.copy(*row, source)
        return ctx.adopt_model(out)


@invocable(memo_dependencies=ADALN)
async def select_adaln_weights(ctx: Context, *, source: Weights[Any], task: Task) -> Selection:
    """One DiT's AdaLN generating weights, selected from a full or pruned checkpoint."""
    ctx.raise_if_cancelled()
    opened = open_source(ctx, source)
    present = rows(opened)
    plan = _plan(opened, task)
    pruned = _body_kind(present)
    document = _validate_body_config(config(opened), pruned)
    if not pruned:
        return Selection(_project(ctx, opened, task, "model"), False, plan.digest, list(plan.steps))
    _bindings(document)
    layout = document[DIT_COMPONENT[task]]["cozy_h3"]["table_keys"]
    if layout != plan.table_keys:
        raise UnsupportedInput(
            "retabling a previous schedule requires the full generating_model",
            code="adaln_generating_weights",
        )
    _validate_table_rows(present, task, plan, bank=False)
    order = current_order(_asset("whole-order.json")).rows
    with ctx.output("model", Derivation(base=opened, order=order)) as out:
        return Selection(ctx.adopt_model(out), True, plan.digest, list(plan.steps))


def _bank_config(task: Task, projection: str, plan: TimestepPlan) -> bytes:
    return canonical_json.encode(
        {
            "schema": "h3-adaln-bank/1",
            "task": task,
            "projection": projection,
            "table_keys": plan.table_keys,
            "dtype": "bf16",
            "numerics": _NUMERICS,
        }
    )


@invocable(memo_dependencies=ADALN)
async def compute_adaln_tables(
    ctx: Context,
    *,
    source: Weights[Any],
    task: Task,
    plan_digest: str,
    tel: Telemetry,
) -> ModelArtifact:
    """One DiT's AdaLN timestep tables, computed from its generating weights."""
    opened = open_source(ctx, source)
    plan = _plan(opened, task)
    if plan_digest != plan.digest:
        raise UnsupportedInput(
            "AdaLN plan differs from the source projection plan", code="adaln_plan"
        )
    present = rows(opened)
    _validate_generators(present, task)
    component = DIT_COMPONENT[task]
    topology = _topology(task)
    if {(c, key) for c, keys in present.items() for key in keys} != {
        (component, key) for key in removed_keys(topology)
    } or _metadata(opened, task) != _projection_config(task):
        raise UnsupportedInput(
            "AdaLN bank input must be one minimal generating-weight projection",
            code="adaln_projection",
        )
    order = tuple((component, key) for key in table_shapes(topology, plan))
    if ctx.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    with ctx.output("model", Derivation(base=None, sources=(opened,), order=order)) as out:
        if out.done is None:
            copy_assets(out, opened, component, omit=(METADATA,))
            out.component(component).asset(METADATA, _bank_config(task, source.checkpoint_ref, plan))
        read, _, reused = write_tables(
            ctx, plan=plan, topology=topology, source=opened, component=component,
            outputs=(out,),
            progress=lambda done, total: tel.progress(done / total, stage=f"timestep-table-{task}"),
        )  # fmt: skip
        artifact = ctx.adopt_model(out)
    tel.metric("h3.adaln.source_bytes", float(read), unit="bytes")
    tel.metric("h3.adaln.reused_tables", float(reused), unit="tables")
    return artifact


def _validate_table_rows(present: Rows, task: Task, plan: TimestepPlan, *, bank: bool) -> None:
    component = DIT_COMPONENT[task]
    tables = table_shapes(_topology(task), plan)
    found = {
        key: row
        for key, row in present.get(component, {}).items()
        if bank or key in tables
    }
    if bank and set(present) != {component}:
        raise UnsupportedInput(
            "AdaLN bank includes another task or unrelated tensors", code="adaln_bank"
        )
    if set(found) != set(tables) or not all(
        plain(row, "bf16", tables[key]) for key, row in found.items()
    ):
        raise UnsupportedInput(
            "AdaLN bank has changed table geometry or encoding", code="adaln_bank"
        )


def _bank_projection(bank: Checkpoint, task: Task, plan: TimestepPlan) -> str:
    metadata = _metadata(bank, task)
    value = None if metadata is None else canonical_json.decode(metadata)
    if not isinstance(value, dict) or not isinstance(value.get("projection"), str):
        raise UnsupportedInput("AdaLN bank has no generating-weight binding", code="adaln_binding")
    projection = str(value["projection"])
    if value != canonical_json.decode(_bank_config(task, projection, plan)):
        raise UnsupportedInput(
            "AdaLN bank task, schedule or numerical contract differs", code="adaln_binding"
        )
    _validate_table_rows(rows(bank), task, plan, bank=True)
    return projection


def _apply_adaln(
    ctx: Context,
    *,
    source: Weights[Any],
    fl2va: Weights[Any],
    ref2va: Weights[Any],
    require_pruned: bool,
) -> ModelArtifact:
    ctx.raise_if_cancelled()
    opened = open_source(ctx, source)
    present = rows(opened)
    pruned = _body_kind(present)
    if pruned != require_pruned:
        raise UnsupportedInput(
            "AdaLN attachment source has a different pruning state", code="adaln_source"
        )
    document = _validate_body_config(config(opened), pruned)
    for task in _TASKS:
        _plan(opened, task)
    bindings: dict[Task, str] = (
        _bindings(document)
        if pruned
        else {task: _project(ctx, opened, task, f"{task}-weights").checkpoint for task in _TASKS}
    )
    banks: dict[Task, Checkpoint] = {"fl2va": open_source(ctx, fl2va), "ref2va": open_source(ctx, ref2va)}
    plans = {task: _plan(bank, task) for task, bank in banks.items()}
    bank_of = {DIT_COMPONENT[task]: bank for task, bank in banks.items()}
    tables: set[tuple[str, str]] = set()
    for task in _TASKS:
        if _bank_projection(banks[task], task, plans[task]) != bindings[task]:
            raise UnsupportedInput(
                "AdaLN bank was computed from different generating weights", code="adaln_binding"
            )
        tables |= {(DIT_COMPONENT[task], key) for key in table_shapes(_topology(task), plans[task])}
    configured = canonical_json.decode(
        dual_adaln_pruned_config(_sections(), plans["fl2va"], plans["ref2va"])
    )
    for task in _TASKS:
        configured[DIT_COMPONENT[task]]["cozy_h3"][_BINDING] = bindings[task]
    order = current_order(_asset("whole-order.json")).rows
    derivation = Derivation(
        base=opened if pruned else None, sources=(*([] if pruned else [opened]), *banks.values()),
        order=order,
    )  # fmt: skip
    with ctx.output("model", derivation) as out:
        write_config(out, canonical_json.encode(configured))
        if not pruned:
            out.inherit(opened)
        if out.done is None:
            for task, component in DIT_COMPONENT.items():
                out.component(component).asset(TIMESTEP_PLAN, plans[task].canonical_bytes)
        for row in order:
            if out.has(*row):
                continue
            if row in tables:
                out.copy(*row, bank_of[row[0]])
            elif not pruned:
                out.copy(*row, opened)
        return ctx.adopt_model(out)


@invocable(memo_dependencies=ADALN)
async def apply_adaln(
    ctx: Context,
    *,
    source: Weights[Any],
    fl2va: Weights[Any],
    ref2va: Weights[Any],
) -> ModelArtifact:
    """A full checkpoint AdaLN-pruned: both DiTs' modulation rows replaced by their tables."""
    return _apply_adaln(ctx, source=source, fl2va=fl2va, ref2va=ref2va, require_pruned=False)


@invocable(memo_dependencies=ADALN)
async def retable_adaln(
    ctx: Context,
    *,
    source: Weights[Any],
    fl2va: Weights[Any],
    ref2va: Weights[Any],
) -> ModelArtifact:
    """An AdaLN-pruned checkpoint with both DiTs' tables replaced."""
    return _apply_adaln(ctx, source=source, fl2va=fl2va, ref2va=ref2va, require_pruned=True)
