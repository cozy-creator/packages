"""Typed functions producing the exact MiniMax H3 checkpoint variants and their AdaLN tables."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from typing import Any, Literal, get_args

import msgspec
import torch
from cozy_runtime.author import (
    App,
    Context,
    DerivationHook,
    Model,
    ModelArtifact,
    Recipe,
    Telemetry,
    UnsupportedInput,
    Weights,
    WeightsOutput,
    canonical_json,
    invocable,
)
from cozy_runtime.derive.quantization import (
    ArtifactQuantizationRequest,
    quantize_component_into,
)
from cozy_runtime.weights_store import Checkpoint, Derivation, Output

from . import adaln_operations as _adaln_operations
from . import lanes as _lanes
from ._memo import LANES as LANES_MEMO
from .kernel import H3Topology, removed_keys, source_shapes, table_bytes, table_shapes
from .lanes import (
    COMPONENT_MAX_NEW_BYTES,
    LANES,
    MAX_DIT_BYTES,
    NORMALISED_COMPONENTS,
    Selection,
    lane_max_new_bytes,
    lane_treatments,
    write_cast,
)
from .model_config import (
    COMPONENTS,
    DIT_COMPONENT,
    dual_adaln_pruned_config,
    dual_full_config,
    parse_full_config,
)
from .order import current_order, full_order
from .plans import Task, TimestepPlan
from .quantization import dit_plan
from .source import (
    DTYPE,
    TIMESTEP_PLAN,
    copy_assets,
    Rows,
    official_full_specs,
    open_source,
    plain,
    rows,
    timestep_plan,
    write_config,
    write_tables,
)

app = App()


def _asset(name: str) -> bytes:
    return files(__package__).joinpath("assets", name).read_bytes()


def _production_plan(source: Checkpoint, task: Task) -> TimestepPlan:
    return timestep_plan(source, task)


def _check_table_budget(plans: Mapping[Task, TimestepPlan], declared: int) -> None:
    """Refuse before native writes if either source task plan exceeds the ceiling."""
    sections = parse_full_config(_asset("model-config.json"))
    measured = max(
        table_bytes(H3Topology.from_config(sections[component]), plans[task])
        for task, component in DIT_COMPONENT.items()
    )
    if measured > declared:
        raise UnsupportedInput(
            f"the source plans need {measured} table bytes, above budget {declared}",
            code="adaln_plan",
        )


# An author-declared per-task ceiling, not a second copy of table geometry.
MAX_TABLE_BYTES = 1 << 30
MAX_FULL_BYTES = 64 << 10
MAX_PRUNED_BYTES = 2 * MAX_TABLE_BYTES + (128 << 10)
MAX_QUANTIZED_BYTES = 2 * MAX_DIT_BYTES + MAX_PRUNED_BYTES


#: The declared output slots, spelled as literals because cr-114's static reader folds the
#: decorator from SOURCE over a closed vocabulary; `_check_lane_outputs` proves they ARE the
#: catalogue, ceiling for ceiling. Every lane carries the video VAE normalisation.
MAX_VIDEO_VAE_BYTES = 12 << 30
MAX_TEXT_ENCODER_BYTES = 56 << 30
MAX_QUANTIZED_LANE_BYTES = MAX_QUANTIZED_BYTES + MAX_VIDEO_VAE_BYTES + MAX_TEXT_ENCODER_BYTES
LANE_OUTPUTS = (
    WeightsOutput("bf16-full", max_new_bytes=MAX_FULL_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("bf16", max_new_bytes=MAX_PRUNED_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("fp8", max_new_bytes=MAX_QUANTIZED_LANE_BYTES),
)
LaneName = Literal["bf16-full", "bf16", "fp8"]


def _check_lane_outputs() -> None:
    for component, declared_bytes in (
        ("vae", MAX_VIDEO_VAE_BYTES),
        ("text_encoder", MAX_TEXT_ENCODER_BYTES),
    ):
        if COMPONENT_MAX_NEW_BYTES[component] != declared_bytes:
            raise ValueError(
                f"the declared {component} bound {declared_bytes} is not the catalogue's "
                f"{COMPONENT_MAX_NEW_BYTES[component]}"
            )
    declared = {output.name: output.max_new_bytes for output in LANE_OUTPUTS}
    if set(declared) != set(LANES) or set(get_args(LaneName)) != set(LANES):
        raise ValueError(
            f"declared outputs {sorted(declared)} and request literal "
            f"{sorted(get_args(LaneName))} are not the catalogue {sorted(LANES)}"
        )
    for name, lane in LANES.items():
        needed = lane_max_new_bytes(lane, full_bytes=MAX_FULL_BYTES, pruned_bytes=MAX_PRUNED_BYTES)
        if declared[name] != needed:
            raise ValueError(
                f"lane {name!r} declares {declared[name]} new bytes, but its treatments "
                f"need {needed}"
            )


_check_lane_outputs()


class TreatmentStats(msgspec.Struct):
    """What one treatment of one component recorded on this attempt. The cast and the
    encoding keep separate round-trip numbers: different carriers, different bounds."""

    cast_keys: int = 0
    cast_worst_relative_frobenius: float | None = None
    encoded_keys: int = 0
    saturated_elements: int = 0
    worst_relative_frobenius: float | None = None
    reused_keys: int = 0
    source_bytes_read: int = 0
    new_bytes_written: int = 0


class RetableResult(msgspec.Struct, frozen=True):
    steps: list[int]
    adaln_pruned: ModelArtifact
    tables: ModelArtifact


def _require_cuda(ctx: Context) -> None:
    if ctx.device.type != "cuda":
        raise UnsupportedInput(
            "H3 AdaLN timestep tables need a CUDA worker; run on a GPU rental",
            code="h3_tables_need_cuda",
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _tables(
    ctx: Context,
    tel: Telemetry,
    source: Checkpoint,
    outputs: tuple[Output, ...],
) -> tuple[int, int, int]:
    """Both DiTs' tables from `source`'s modulation weights into `outputs`."""
    sections = parse_full_config(_asset("model-config.json"))
    read = written = reused = 0
    for task, component in DIT_COMPONENT.items():
        stage = f"timestep-table-{task}"

        def progress(done: int, total: int, stage: str = stage) -> None:
            tel.progress(done / total, stage=stage)

        with tel.stage(stage):
            got = write_tables(
                ctx,
                plan=_production_plan(source, task),
                topology=H3Topology.from_config(sections[component]),
                source=source,
                component=component,
                outputs=outputs,
                progress=progress,
            )
        read, written, reused = read + got[0], written + got[1], reused + got[2]
    return read, written, reused


def _treat(
    source: Checkpoint,
    out: Output,
    ctx: Context,
    tel: Telemetry,
    request: ArtifactQuantizationRequest,
    selection: Selection,
) -> TreatmentStats:
    """One component's resolved treatment: the cast, then the encoding (disjoint key sets)."""
    casted = write_cast(source, out, ctx, tel, selection=selection)
    stats = TreatmentStats(
        cast_keys=casted.converted_keys,
        cast_worst_relative_frobenius=casted.worst_relative_frobenius,
        reused_keys=casted.reused_keys,
        source_bytes_read=casted.source_bytes_read,
        new_bytes_written=casted.new_bytes_written,
    )
    encoding = selection.treatment.encode
    if selection.plan is None or encoding is None:
        return stats
    cast = selection.treatment.cast
    encoded = quantize_component_into(
        source, out, ctx, request, tel, encoding=encoding, plan=selection.plan,
        component=selection.component, logical_dtype=lambda t: cast or t.logical_dtype,
    )  # fmt: skip
    return msgspec.structs.replace(
        stats,
        encoded_keys=encoded.encoded_keys,
        saturated_elements=encoded.saturated_elements,
        worst_relative_frobenius=encoded.worst_relative_frobenius,
        reused_keys=stats.reused_keys + encoded.reused_keys,
        source_bytes_read=stats.source_bytes_read + encoded.source_bytes_read,
        new_bytes_written=stats.new_bytes_written + encoded.new_bytes_written,
    )


def _log_treatment(tel: Telemetry, slot: str, selection: Selection, stats: TreatmentStats) -> None:
    tel.log(
        "weight fidelity",
        level="info",
        output_slot=slot,
        component=selection.component,
        treatment=selection.treatment.describe(),
        encoded_keys=stats.encoded_keys,
        reused_keys=stats.reused_keys,
        saturated_elements=stats.saturated_elements,
        worst_relative_frobenius=stats.worst_relative_frobenius,
        cast_worst_relative_frobenius=stats.cast_worst_relative_frobenius,
        new_bytes=stats.new_bytes_written,
    )


def lane_rows(
    name: str, present: Rows, plans: Mapping[Task, TimestepPlan]
) -> tuple[tuple[tuple[str, str], ...], dict[str, dict[str, Any]], frozenset[tuple[str, str]]]:
    """A lane's output order, the source rows it carries per component, and the rows it
    computes (tables). Refuses a source that is not one full-precision H3 checkpoint."""
    if set(present) != set(COMPONENTS):
        raise UnsupportedInput(
            f"source components are {sorted(present)}, not H3's five", code="h3_lane_source"
        )
    sections = parse_full_config(_asset("model-config.json"))
    for component in DIT_COMPONENT.values():
        for key, (dtype, shape) in official_full_specs(sections[component]).items():
            row = present[component].get(key)
            if row is None or not plain(row, dtype, shape):
                raise UnsupportedInput(
                    f"{component}.{key} is not the full-precision plain {dtype} {list(shape)} "
                    "row; produce lanes from the converted release or a bf16-full lane",
                    code="h3_lane_source",
                )
    current = current_order(_asset("whole-order.json")).rows
    pruned = LANES[name].modulation == "adaln-pruned"
    order = current if pruned else full_order(sections, current)
    computed = frozenset(
        (component, key)
        for task, component in DIT_COMPONENT.items()
        for key in table_shapes(H3Topology.from_config(sections[component]), plans[task])
    ) if pruned else frozenset()  # fmt: skip
    carried: dict[str, dict[str, Any]] = {component: {} for component in COMPONENTS}
    for component, key in order:
        if (component, key) in computed:
            continue
        row = present[component].get(key)
        if row is None:
            raise UnsupportedInput(f"source lacks {component}.{key}", code="h3_lane_source")
        carried[component][key] = row
    return order, carried, computed


def _produce_lane(
    ctx: Context,
    tel: Telemetry,
    weights: Weights[Any],
    name: LaneName,
    max_relative_frobenius: float | None = None,
) -> ModelArtifact:
    """One reviewed lane from a full-precision H3 checkpoint (the converted release or a
    `bf16-full` lane). Every row it does not rewrite is copied, keeping the source's objects;
    an interrupted attempt resumes what it wrote, a committed one replays."""
    source = open_source(ctx, weights)
    plans = {task: _production_plan(source, task) for task in DIT_COMPONENT}
    _check_table_budget(plans, MAX_TABLE_BYTES)
    order, carried, computed = lane_rows(name, rows(source), plans)
    lane = LANES[name]
    selections = [
        _lanes.select(
            component,
            treatment,
            source,
            carried[component],
            dit_plan=dit_plan(),
            allow_inert=component in NORMALISED_COMPONENTS,
        )
        for component, treatment in lane_treatments(lane).items()
    ]
    sections = parse_full_config(_asset("model-config.json"))
    config = (
        dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"])
        if computed
        else dual_full_config(sections)
    )
    if computed:
        _require_cuda(ctx)
    replaced = {(s.component, key) for s in selections for key in s.replaced}
    request = ArtifactQuantizationRequest(max_relative_frobenius=max_relative_frobenius)
    reused = written = read = 0
    with ctx.output(name, Derivation(base=None, sources=(source,), order=order)) as out:
        write_config(out, config)
        out.inherit(source)
        with tel.stage(f"{name}-copy"):
            for done, row in enumerate(order, 1):
                if row not in computed and row not in replaced and not out.has(*row):
                    out.copy(*row, source)
                tel.progress(done / len(order), stage=f"{name}-copy")
        if computed:
            read, written, reused = _tables(ctx, tel, source, (out,))
        for selection in selections:
            with tel.stage(f"{name}-{selection.component}"):
                stats = _treat(source, out, ctx, tel, request, selection)
            _log_treatment(tel, name, selection, stats)
            read, written = read + stats.source_bytes_read, written + stats.new_bytes_written
            reused += stats.reused_keys
        artifact = ctx.adopt_model(out)
    tel.metric("h3.source_bytes", float(read), unit="bytes")
    tel.metric("h3.new_bytes", float(written), unit="bytes")
    tel.metric("h3.reused_tensors", float(reused))
    return artifact


# One function per lane: Runtime requires a receipt for EVERY declared weights output.
@invocable(memo_version="h3-lane/1", memo_dependencies=LANES_MEMO)
async def bf16_full(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The reviewed full-precision lane: source-only rows dropped, video VAE normalised."""
    return _produce_lane(ctx, tel, source, "bf16-full")


@invocable(memo_version="h3-lane/1", memo_dependencies=LANES_MEMO)
async def bf16_pruned(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """AdaLN-pruned BF16: 106 modulation rows per DiT replaced by timestep tables."""
    return _produce_lane(ctx, tel, source, "bf16")


@invocable(memo_version="h3-lane/1", memo_dependencies=LANES_MEMO)
async def fp8_pruned(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """AdaLN-pruned with both DiTs row-wise FP8."""
    return _produce_lane(ctx, tel, source, "fp8", max_relative_frobenius)


app.entrypoint(
    bf16_full,
    name="bf16-full",
    weights=(WeightsOutput("bf16-full", max_new_bytes=MAX_FULL_BYTES + MAX_VIDEO_VAE_BYTES),),
    accelerator=False,
)
# The pruned lanes compute AdaLN tables on the GPU.
app.entrypoint(
    bf16_pruned,
    name="bf16-pruned",
    weights=(WeightsOutput("bf16", max_new_bytes=MAX_PRUNED_BYTES + MAX_VIDEO_VAE_BYTES),),
)
app.entrypoint(
    fp8_pruned,
    name="fp8-pruned",
    weights=(WeightsOutput("fp8", max_new_bytes=MAX_QUANTIZED_LANE_BYTES),),
)


#: The quantized lanes' recipe (`FP8`, `INT8`, `NVFP4`, `W4A8` below) over an
#: AdaLN-pruned checkpoint. Both DiTs encode through the reviewed plan and the conditioner's 350
#: decoder-layer linears through the structural selection minus `TEXT_ENCODER_KEEP`. A recipe
#: component the source still holds plain is encoded; one already carrying this scheme is
#: inherited; one carrying another refuses. Everything else carries over from the source. nvfp4
#: takes the DiTs minus `NVFP4_DIT_KEEP` and its conditioner is int8.convrot (the nvfp4 leaf is
#: W4A4 at prompt length, and the conditioner runs once a request); `NVFP4_DIT_INT8` serve int8.convrot.
#: w4a8 (xs-062) takes nvfp4's kept keys with every other DiT linear in W4A8 (its per-token
#: activation scales have none of nvfp4's subnormal-scale hazard), the conditioner int8.convrot.
#: int8 also takes the video VAE decoder's block linears minus `VAE_INT8_KEEP`.
QUANTIZER_RECIPES: Mapping[str, Mapping[str, _lanes.Treatment]] = {
    "fp8.rowwise": {
        component: _lanes.Treatment(encode="fp8.rowwise", keep=treatment.keep)
        for component, treatment in LANES["fp8"].components.items()
    },
    "int8.convrot": {
        **{
            component: _lanes.Treatment(encode="int8.convrot", keep=treatment.keep)
            for component, treatment in LANES["fp8"].components.items()
        },
        "vae": _lanes.Treatment(encode="int8.convrot", keep=_lanes.VAE_INT8_KEEP),
    },
    "nvfp4": {
        **{dit: _lanes.Treatment(
               encode="nvfp4", keep=_lanes.NVFP4_DIT_KEEP,
               encode_as=tuple((key, "int8.convrot") for key in _lanes.NVFP4_DIT_INT8),
           ) for dit in sorted(_lanes.DIT_COMPONENTS)},
        "text_encoder": _lanes.Treatment(encode="int8.convrot", keep=_lanes.TEXT_ENCODER_KEEP),
    },
    "w4a8.convrot": {
        **{dit: _lanes.Treatment(encode="w4a8.convrot", keep=_lanes.NVFP4_DIT_KEEP)
           for dit in sorted(_lanes.DIT_COMPONENTS)},
        "text_encoder": _lanes.Treatment(encode="int8.convrot", keep=_lanes.TEXT_ENCODER_KEEP),
    },
}


def quantizer_selections(source: Checkpoint, encoding: _lanes.Encoding) -> list[Selection]:
    """The recipe's components this source still holds plain, resolved for `encoding`."""
    present = rows(source)
    if set(present) != set(COMPONENTS):
        raise UnsupportedInput(
            f"source components are {sorted(present)}, not H3's five", code="h3_quantize_source"
        )
    sections = parse_full_config(_asset("model-config.json"))
    for component in DIT_COMPONENT.values():
        if set(removed_keys(H3Topology.from_config(sections[component]))) & set(present[component]):
            raise UnsupportedInput(
                f"{component} still carries its AdaLN modulation rows; quantize an AdaLN-pruned "
                "lane (bf16 or fp8)",
                code="h3_quantize_unpruned",
            )
    selections = []
    for component, treatment in QUANTIZER_RECIPES[encoding].items():
        schemes = {row.encoding.scheme for row in present[component].values() if row.encoding}
        if treatment.encode in schemes:
            continue
        if schemes:
            raise UnsupportedInput(
                f"{component} already carries another encoding ({sorted(schemes)}); quantize "
                f"a lane whose {component} is plain or already {treatment.encode}",
                code="h3_quantize_mixed",
            )
        selections.append(
            _lanes.select(component, treatment, source, present[component], dit_plan=dit_plan())
        )
    if not selections:
        raise UnsupportedInput(
            f"every component of H3's recipe already carries {encoding}",
            code="h3_quantize_nothing",
        )
    return selections


def _quantize_lane(
    ctx: Context,
    tel: Telemetry,
    weights: Weights[Any],
    lane: str,
    encoding: _lanes.Encoding,
    max_relative_frobenius: float | None,
) -> ModelArtifact:
    source = open_source(ctx, weights)
    for task in DIT_COMPONENT:
        _production_plan(source, task)
    selections = quantizer_selections(source, encoding)
    order = [(component, key) for component, keys in rows(source).items() for key in keys]
    request = ArtifactQuantizationRequest(max_relative_frobenius=max_relative_frobenius)
    with ctx.output(lane, Derivation(base=source, order=order)) as out:
        for selection in selections:
            with tel.stage(f"{lane}-{selection.component}"):
                stats = _treat(source, out, ctx, tel, request, selection)
            _log_treatment(tel, lane, selection, stats)
        tel.log(
            "h3 quantize",
            lane=lane,
            encoded=",".join(selection.component for selection in selections),
            inherited=",".join(
                sorted(set(QUANTIZER_RECIPES[encoding]) - {s.component for s in selections})
            ),
        )
        return ctx.adopt_model(out)


#: Each quantized lane's encoding.
QUANTIZER_ENCODINGS: Mapping[str, _lanes.Encoding] = {
    "fp8": "fp8.rowwise",
    "int8": "int8.convrot",
    "nvfp4": "nvfp4",
    "w4a8": "w4a8.convrot",
}


def quantize_lane(ctx: Context, tel: Telemetry, source: Weights[Any], output: str) -> ModelArtifact:
    """H3's recipe for lane `output` (`QUANTIZER_RECIPES`): fp8 row-wise, int8 W8A8 on the int8
    tensor cores, nvfp4 W4A4 on Blackwell's FP4 tensor cores, w4a8 W4A8 on the int8 tensor cores
    (Ampere, Ada, Hopper)."""
    return _quantize_lane(ctx, tel, source, output, QUANTIZER_ENCODINGS[output], None)


#: `cozy model quantize <H3 checkpoint> --format fp8|int8|nvfp4|w4a8 --package <this>`: code, not a
#: plan, since what a lane encodes depends on what the source still holds plain.
FP8 = Recipe(lane="fp8", model=Model, plan=DerivationHook(quantize_lane))
INT8 = Recipe(lane="int8", model=Model, plan=DerivationHook(quantize_lane))
NVFP4 = Recipe(lane="nvfp4", model=Model, plan=DerivationHook(quantize_lane))
W4A8 = Recipe(lane="w4a8", model=Model, plan=DerivationHook(quantize_lane))


#: The function that writes each lane: the pruned lanes' functions keep their operation names.
LANE_FUNCTIONS = {"bf16-full": "bf16-full", "bf16": "bf16-pruned", "fp8": "fp8-pruned"}


def _check_lane_functions() -> None:
    """Each lane's function declares exactly the catalogue's output for that lane."""
    declared = {output.name: output for output in LANE_OUTPUTS}
    for name in LANES:
        registered = app.get(LANE_FUNCTIONS[name]).weights_outputs
        if registered != (declared[name],):
            raise ValueError(f"lane function {name!r} declares {registered}, not {declared[name]}")


_check_lane_functions()


def retable_rows(
    pruned: Rows, full: Rows, plans: Mapping[Task, TimestepPlan]
) -> dict[str, tuple[str, ...]]:
    """Each DiT's table keys, once `pruned` is an AdaLN-pruned H3 carrying table rows only and
    `full` carries the exact modulation weights they are computed from."""
    if set(pruned) != set(COMPONENTS):
        raise UnsupportedInput(
            f"retable source components are {sorted(pruned)}", code="h3_retable_source"
        )
    sections = parse_full_config(_asset("model-config.json"))
    tables: dict[str, tuple[str, ...]] = {}
    for task, component in DIT_COMPONENT.items():
        topology = H3Topology.from_config(sections[component])
        keys = set(pruned[component])
        tables[component] = tuple(table_shapes(topology, plans[task]))
        if not set(tables[component]) <= keys or set(removed_keys(topology)) & keys:
            raise UnsupportedInput(
                f"{component} is not an AdaLN-pruned source carrying table rows only",
                code="h3_retable_source",
            )
        for key, (dtype, shape) in source_shapes(topology).items():
            row = full.get(component, {}).get(key)
            if row is None or tuple(row.shape) != shape or DTYPE.get(row.dtype.lower()) != dtype:
                raise UnsupportedInput(
                    f"full source lacks modulation weight {component}/{key}",
                    code="h3_retable_source",
                )
    return tables


@invocable(memo_version="h3-retable/1", memo_dependencies=LANES_MEMO)
async def retable(
    ctx: Context,
    *,
    pruned: Weights[Any],
    full: Weights[Any],
    tel: Telemetry,
) -> RetableResult:
    """Recompute one AdaLN-pruned checkpoint's tables for the current plans.

    Every non-table tensor of `pruned` (BF16, FP8 or int8 alike) carries over; only the
    modulation rows are read from `full` and recomputed, so a plan that adds schedules costs
    table bytes, never a requantization. The same rows also commit as a two-DiT table bank.
    """
    pruned_source, full_source = open_source(ctx, pruned), open_source(ctx, full)
    for task in DIT_COMPONENT:
        _production_plan(pruned_source, task)
    plans = {task: _production_plan(full_source, task) for task in DIT_COMPONENT}
    _check_table_budget(plans, MAX_TABLE_BYTES)
    tables = retable_rows(rows(pruned_source), rows(full_source), plans)
    sections = parse_full_config(_asset("model-config.json"))
    config = dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"])
    bank_config = canonical_json.encode(
        {c: section for c, section in canonical_json.decode(config).items() if c in tables}
    )
    order = current_order(_asset("whole-order.json")).rows
    bank_order = tuple((c, k) for c, k in order if k in tables.get(c, ()))
    retabled_derivation = Derivation(base=pruned_source, sources=(full_source,), order=order)
    bank_derivation = Derivation(base=None, sources=(full_source,), order=bank_order)
    with ctx.output("adaln-pruned", retabled_derivation) as retabled, ctx.output(
        "tables", bank_derivation
    ) as bank:
        write_config(retabled, config)
        write_config(bank, bank_config)
        for task, component in DIT_COMPONENT.items():
            copy_assets(bank, full_source, component)
            if retabled.done is None:
                retabled.component(component).asset(TIMESTEP_PLAN, plans[task].canonical_bytes)
        if not all(retabled.has(*row) and bank.has(*row) for row in bank_order):
            _require_cuda(ctx)
        read, written, _ = _tables(ctx, tel, full_source, (retabled, bank))
        result = RetableResult(
            list(plans["fl2va"].steps), ctx.adopt_model(retabled), ctx.adopt_model(bank)
        )
    tel.metric("h3.source_bytes", float(read), unit="bytes")
    tel.metric("h3.new_bytes", float(written), unit="bytes")
    return result


app.entrypoint(
    retable,
    name="retable",
    weights=(
        WeightsOutput("adaln-pruned", max_new_bytes=MAX_PRUNED_BYTES),
        WeightsOutput("tables", max_new_bytes=MAX_PRUNED_BYTES),
    ),
)


# Native output bounds include newly written construction and provenance configs.
app.entrypoint(
    _adaln_operations.select_adaln_weights,
    name="select-adaln-weights",
    weights=(WeightsOutput("model", 1 << 20),),
    accelerator=False,
)
app.entrypoint(
    _adaln_operations.compute_adaln_tables,
    name="compute-adaln-tables",
    weights=(WeightsOutput("model", MAX_TABLE_BYTES),),
)
app.entrypoint(
    _adaln_operations.apply_adaln,
    name="apply-adaln",
    weights=(
        WeightsOutput("model", 1 << 20),
        WeightsOutput("fl2va-weights", 1 << 20),
        WeightsOutput("ref2va-weights", 1 << 20),
    ),
    accelerator=False,
)

app.entrypoint(
    _adaln_operations.retable_adaln,
    name="retable-adaln",
    weights=(WeightsOutput("model", 1 << 20),),
    accelerator=False,
)


from .turbo import turbo_lora  # noqa: E402

app.entrypoint(turbo_lora, name="turbo-lora", weights=(WeightsOutput("pdd8", 1 << 29),))
