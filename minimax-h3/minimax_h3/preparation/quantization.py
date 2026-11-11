"""The reviewed MiniMax-H3 DiT quantization plan: which linears each DiT encodes."""

from importlib.resources import files

from cozy_runtime.author import canonical_json

DIT_PLAN_ROWS = 313


def dit_plan() -> tuple[tuple[str, tuple[int, int]], ...]:
    """The 313 bf16 DiT weights the encoded lanes replace, in construction order, with shapes."""
    raw = files(__package__).joinpath("assets/h3-dit-quantization-plan.json").read_bytes()
    value = canonical_json.decode(raw)
    if not isinstance(value, list) or canonical_json.encode(value) != raw:
        raise ValueError("the H3 DiT plan is not its canonical [key, rows, cols] array")
    plan = tuple((str(key), (int(rows), int(cols))) for key, rows, cols in value)
    if len(plan) != DIT_PLAN_ROWS or len({key for key, _ in plan}) != DIT_PLAN_ROWS:
        raise ValueError(f"the H3 DiT plan must name {DIT_PLAN_ROWS} distinct weights")
    return plan
