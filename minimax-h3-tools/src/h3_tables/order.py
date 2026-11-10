"""The exact post-se018 H3 construction traversal."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from cozy_runtime.author import canonical_json

from .model_config import DIT_COMPONENT
from .source import official_full_specs

CURRENT_ORDER = (
    "sha256:fbc4fb5675efe3a11e1f9f101b9be6b834dde37f48c4218a1253217538cca582"
)
CURRENT_ROWS = 3858
CURRENT_COMPONENT_COUNTS = {
    "transformer": 583,
    "transformer_ref": 583,
    "text_encoder": 902,
    "vae": 703,
    "audio_vae": 1087,
}
MAX_BYTES = 8 << 20
MAX_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class ConstructionOrder:
    rows: tuple[tuple[str, str], ...]
    digest: str

    def identity_rows(self) -> list[list[str]]:
        return [[component, key] for component, key in self.rows]

    def select(self, component: str) -> ConstructionOrder:
        rows = tuple(row for row in self.rows if row[0] == component)
        if not rows:
            raise ValueError(f"construction order has no rows for {component!r}")
        return ConstructionOrder(rows, canonical_json.digest(self.identity(rows)))

    @staticmethod
    def identity(rows: tuple[tuple[str, str], ...]) -> list[list[str]]:
        return [[component, key] for component, key in rows]


def parse_construction_order(
    raw: bytes, name: str = "construction order"
) -> ConstructionOrder:
    if not 0 < len(raw) <= MAX_BYTES:
        raise ValueError(f"{name} must be 1..{MAX_BYTES} bytes, got {len(raw)}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not a JSON pair-array: {error}") from error
    if not isinstance(value, list) or not value or len(value) > MAX_ROWS:
        raise ValueError(f"{name} must contain 1..{MAX_ROWS} rows")
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(value):
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(isinstance(item, str) and item for item in row)
        ):
            raise ValueError(f"{name} row {index} must be [component, key]")
        pair = (row[0], row[1])
        if pair in seen:
            raise ValueError(f"{name} repeats row {index} {pair!r}")
        seen.add(pair)
        rows.append(pair)
    identity = ConstructionOrder.identity(tuple(rows))
    return ConstructionOrder(tuple(rows), canonical_json.digest(identity))


def current_order(raw: bytes) -> ConstructionOrder:
    order = parse_construction_order(raw)
    counts = dict(Counter(component for component, _ in order.rows))
    if (
        order.digest != CURRENT_ORDER
        or len(order.rows) != CURRENT_ROWS
        or counts != CURRENT_COMPONENT_COUNTS
    ):
        raise ValueError(
            f"construction order is {order.digest}/{len(order.rows)} rows/{counts}, "
            "not the exact post-se018 H3 construction"
        )
    return order


def full_order(
    sections: Mapping[str, Mapping[str, object]], current: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    fl, ref = (
        tuple((component, key) for key in official_full_specs(sections[component]))
        for component in DIT_COMPONENT.values()
    )
    shared = tuple(row for row in current if row[0] not in DIT_COMPONENT.values())
    if len(fl) != 638 or len(ref) != 638 or len(shared) != 2692:
        raise ValueError(
            f"full H3 order is {len(fl)}/{len(ref)}/{len(shared)}, expected 638/638/2692"
        )
    return (*fl, *ref, *shared)
