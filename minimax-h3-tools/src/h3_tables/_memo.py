"""Reviewed helper and resource closure; composition callers are deliberately absent."""
from cozy_runtime.author import MemoDistribution, MemoResource

STRUCTURE = (
    "h3_tables.source", "h3_tables.order", "h3_tables.model_config",
    "h3_tables.table_layout",
    "h3_tables.kernel", "h3_tables.plans", "cozy_runtime.weights_store",
    MemoResource("h3_tables", "assets/dit-full-specs.json"),
    MemoResource("h3_tables", "assets/model-config.json"),
    MemoResource("h3_tables", "assets/whole-order.json"),
    MemoDistribution("tensorfs"), MemoDistribution("msgspec"),
)
ADALN = (
    *STRUCTURE, "h3_tables.adaln_operations",
    MemoDistribution("torch"), MemoDistribution("numpy"),
)
TURBO = (
    *ADALN, "h3_tables.turbo",
)
REGENERATE = (
    *ADALN, "h3_tables.regenerate",
)
LANES = (
    *STRUCTURE, "h3_tables.job", "h3_tables.lanes", "h3_tables.quantization",
    "cozy_runtime.derive.quantization",
    "cozy_runtime.derive.microscale", "cozy_runtime.derive.safetensors_io",
    MemoResource("h3_tables", "assets/h3-dit-quantization-plan.json"),
    MemoDistribution("torch"), MemoDistribution("numpy"),
)
