"""H3 quantization for client scripts, through the Runtime's shared operation."""

from __future__ import annotations

from importlib.resources import files

from cozy_runtime.author import ModelArtifact, ModelChoice, QuantizationPlan, call, canonical_json
from cozy_runtime.derive.operations import QuantizeRequest, quantize as runtime_quantize

from .lanes import Encoding
from .model_config import DIT_COMPONENT, parse_full_config
from .order import current_order, full_order
from .quantization import dit_plan


def quantization_plan(encoding: Encoding) -> QuantizationPlan:
    """Bind reviewed H3 geometry and canonical order without loading a source model."""
    plan = dit_plan()
    components = tuple(DIT_COMPONENT.values())
    selected = [
        [component, key, "bf16", list(shape)] for component in components for key, shape in plan
    ]
    assets = files(__package__).joinpath("assets")
    pruned = current_order(assets.joinpath("whole-order.json").read_bytes())
    sections = parse_full_config(assets.joinpath("model-config.json").read_bytes())
    full = full_order(sections, pruned.rows)
    return QuantizationPlan(
        encoding=encoding,
        components=components,
        keys=tuple(key for key, _ in plan),
        selected_schema_digest=canonical_json.digest(selected),
        source_order_digests=(
            pruned.digest,
            canonical_json.digest([[component, key] for component, key in full]),
        ),
        output_precision="preserve",
    )


async def quantize(
    *,
    source: ModelArtifact,
    encoding: Encoding,
    max_relative_frobenius: float | None = None,
) -> ModelArtifact:
    """Quantize both reviewed DiTs through the shared Runtime operation.

    The exact assembled full/pruned source order is required. Unselected tensors and
    configs retain their source precision; an encoded or malformed source refuses.
    """
    return await call(
        runtime_quantize,
        QuantizeRequest(
            plan=quantization_plan(encoding),
            max_relative_frobenius=max_relative_frobenius,
        ),
        models={"source": ModelChoice(ref=source)},
    )

