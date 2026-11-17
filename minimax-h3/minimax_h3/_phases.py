"""Evidence only (h3-vae-decode red, cr-306): where a decode's time goes, per phase.

`phase(name)` times its body twice: until the host has queued it (`issue_ms`) and until the device has run it
(`wall_ms`, after a synchronize). A phase whose issue is most of its wall waits on the host; one whose wall is far
past its issue waits on the device."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch

TOTALS: dict[str, list[float]] = {}
ON = [False]


@contextmanager
def phase(name: str) -> Iterator[None]:
    if not ON[0]:
        yield
        return
    start = time.perf_counter()
    yield
    issued = time.perf_counter()
    torch.cuda.synchronize()
    done = time.perf_counter()
    row = TOTALS.setdefault(name, [0.0, 0.0, 0])
    row[0] += (issued - start) * 1000
    row[1] += (done - start) * 1000
    row[2] += 1


def dump(tag: str, **extra: object) -> str:
    rows = {k: {"issue_ms": round(v[0], 1), "wall_ms": round(v[1], 1), "count": int(v[2])} for k, v in TOTALS.items()}
    line = "CR306PHASES " + json.dumps({"tag": tag, "phases": rows, **extra})
    print(line, file=sys.stderr, flush=True)
    TOTALS.clear()
    return line
