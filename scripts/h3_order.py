#!/usr/bin/env python3
"""Project TensorFS order rows from Runtime's exact construction contract."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

_SCHEMA = "cozy.runtime.model_construction_contract.v1"


def construction_order(
    document: Mapping[str, object], components: Sequence[str] = ()
) -> list[list[str]]:
    """Strip the Runtime-qualified prefix; preserve DestinationSet order exactly."""
    if document.get("schema") != _SCHEMA:
        raise ValueError("input is not a Runtime ModelConstructionContract")
    sets = document.get("destination_sets")
    if not isinstance(sets, list):
        raise TypeError("construction contract has no destination_sets list")
    selected = set(components)
    rows: list[list[str]] = []
    seen_components: set[str] = set()
    for value in sets:
        if not isinstance(value, dict):
            raise TypeError("construction contract contains a malformed DestinationSet")
        row = cast(dict[str, object], value)
        component = row.get("component")
        keys = row.get("keys")
        if not isinstance(component, str) or not isinstance(keys, list):
            raise TypeError("construction contract contains a malformed DestinationSet")
        if component in seen_components:
            raise ValueError(f"construction contract repeats component {component!r}")
        seen_components.add(component)
        if selected and component not in selected:
            continue
        prefix = component + "."
        for key in keys:
            if not isinstance(key, str) or not key.startswith(prefix):
                raise ValueError(
                    f"DestinationSet {component!r} contains non-local key {key!r}"
                )
            rows.append([component, key.removeprefix(prefix)])
    missing = selected - seen_components
    if missing:
        raise ValueError(f"construction contract has no components {sorted(missing)}")
    if not rows:
        raise ValueError("construction-order projection is empty")
    return rows


def encode_order(rows: list[list[str]]) -> bytes:
    return json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("usage: h3_order.py CONTRACT.json [COMPONENT ...]")
    document = json.loads(Path(args[0]).read_bytes())
    if not isinstance(document, dict):
        raise SystemExit("construction contract is not a JSON object")
    sys.stdout.buffer.write(encode_order(construction_order(document, args[1:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
