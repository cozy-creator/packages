"""Evidence only (h3-vae-decode red, cr-306): where a decode's time goes, per phase.

`phase(name)` times its body twice: until the host has queued it (`issue_ms`) and until the device has run it
(`wall_ms`, after a synchronize). A phase whose issue is most of its wall waits on the host; one whose wall is far
past its issue waits on the device. Each phase is also a `tel.stage` (`PREFIX` + name), so the run record has its wall."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import torch

TOTALS: dict[str, list[float]] = {}
ON = [False]
TEL: list[Any] = [None]
PREFIX = [""]


@contextmanager
def phase(name: str) -> Iterator[None]:
    if not ON[0]:
        yield
        return
    with ExitStack() as stack:
        if TEL[0] is not None:
            stack.enter_context(TEL[0].stage(PREFIX[0] + name))
        start = time.perf_counter()
        yield
        issued = time.perf_counter()
        torch.cuda.synchronize()
        done = time.perf_counter()
    row = TOTALS.setdefault(PREFIX[0] + name, [0.0, 0.0, 0])
    row[0] += (issued - start) * 1000
    row[1] += (done - start) * 1000
    row[2] += 1


def dump() -> dict[str, dict[str, float]]:
    rows = {k: {"issue_ms": round(v[0], 1), "wall_ms": round(v[1], 1), "count": int(v[2])} for k, v in TOTALS.items()}
    TOTALS.clear()
    return rows


def line(**fields: object) -> str:
    return json.dumps(fields)
