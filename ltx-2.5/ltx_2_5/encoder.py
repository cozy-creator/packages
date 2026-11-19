"""This package's abliterated encoder, written from upstream (`transplant`, as pan2's `curve` writes its lane source): an
edit moved onto a fine-tune as its difference. `target`'s `component` plus `edited - original` on each tensor `edited`
changes, rounded to bf16 (nearest even); every other tensor is `target`'s, referenced.

LTX-2.5's encoder is Lightricks' fine-tune of Gemma 4 12B, and igorls' Heretic abliteration of Gemma 4 12B differs from
Google's in 22 output projections, so the abliteration moves onto LTX's encoder this way (as diverofdark's int8 release
made it, its `build.py`). Every tensor of `edited` and `original` is compared whole; each changed one must be a rank-one
edit. The log names each written tensor with its sha256 and the first ConvRot block (256 values) of its first and last
rows as bf16 words, to check against an int8 ConvRot release of the same edit. The component's `task-vector.json` records
the three checkpoints, the changed tensors and the request's `sources`.

    cozy model install hf://igorls/gemma-4-12B-it-heretic-v1@192d14f658a5cb274ac280b2d6d5cecaf37c5c6f/model.safetensors.index.json \
      local/gemma-4-12b-it-heretic-v1 --on <machine>
    cozy model install hf://google/gemma-4-12B-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/model.safetensors \
      local/gemma-4-12b-it --on <machine>
    cozy run ./ltx-2.5/transplant component=text_encoder model.target=lightricks/ltx-2.5@0.2.0 \
      model.edited=local/gemma-4-12b-it-heretic-v1 model.original=local/gemma-4-12b-it --lane bf16 --on <machine>
    cozy model upload run:<run> igorls/gemma-4-12b-it-heretic-v1-ltx-2.5 --lane bf16 \
      --mirror https://huggingface.co/igorls/gemma-4-12B-it-heretic-v1 --on <machine>
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Protocol

import msgspec
import numpy as np
from cozy_runtime.author import Context, ModelArtifact, Telemetry, UnsupportedInput, Weights, WeightsOutput
from cozy_runtime.weights_store import Checkpoint, Derivation, TensorMetadata
from numpy.typing import NDArray

from . import app

CHUNK = 64 << 20
BLOCK = 256
#: Rows whose first ConvRot block of each written tensor the log holds as bf16 words.
SAMPLE = (0, -1)
SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}


class Tensors(Protocol):
    def __getitem__(self, key: str, /) -> TensorMetadata: ...
    def __iter__(self) -> Iterator[str]: ...
    def read_into(self, key: str, out: memoryview, *, offset: int = 0) -> None: ...


def nbytes(meta: TensorMetadata) -> int:
    return int(np.prod(meta.shape)) * SIZES[meta.dtype.upper()]


def read(tensors: Tensors, key: str) -> bytearray:
    size = nbytes(tensors[key])
    out = bytearray(size)
    for at in range(0, size, CHUNK):
        tensors.read_into(key, memoryview(out)[at:min(size, at + CHUNK)], offset=at)
    return out


def same(a: Tensors, b: Tensors, key: str) -> bool:
    size, x, y = nbytes(a[key]), bytearray(CHUNK), bytearray(CHUNK)
    for at in range(0, size, CHUNK):
        n = min(CHUNK, size - at)
        a.read_into(key, memoryview(x)[:n], offset=at)
        b.read_into(key, memoryview(y)[:n], offset=at)
        if x[:n] != y[:n]:
            return False
    return True


def floats(data: bytes | bytearray, meta: TensorMetadata) -> NDArray[np.float32]:
    shape, dtype = tuple(meta.shape), meta.dtype.upper()
    if dtype == "BF16":
        return (np.frombuffer(data, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(data, np.float32).reshape(shape)
    raise UnsupportedInput(f"a changed tensor is {dtype}; this writes bf16 from bf16 or f32", code="task_vector_dtype")


def bf16(values: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Round to nearest even, as torch's float32 -> bfloat16 cast."""
    u = np.ascontiguousarray(values, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def rank1(d: NDArray[np.float64]) -> float:
    """The share of the difference's energy along its first singular direction."""
    u: NDArray[np.float64] = np.random.default_rng(0).standard_normal((d.shape[0],))
    for _ in range(30):
        u = d @ (d.T @ u)
        u /= np.linalg.norm(u)
    return float(np.linalg.norm(d.T @ u) ** 2 / max((d * d).sum(), 1e-300))


def only(ck: Checkpoint, slot: str) -> tuple[str, Tensors]:
    held = [(name, c.tensors) for name, c in ck.components.items() if len(list(c.tensors))]
    if len(held) != 1:
        raise UnsupportedInput(f"{slot} holds tensors in {[n for n, _ in held]}: one component, please", code="task_vector_input")
    return held[0]


def resolve(target: Tensors, key: str) -> str:
    """`key` in `target`, or the one target key that differs from it by a leading prefix."""
    names = list(target)
    if key in names:
        return key
    found = [t for t in names if t.endswith("." + key) or key.endswith("." + t)]
    if len(found) != 1:
        raise UnsupportedInput(f"{key}: {len(found)} target tensors match it", code="task_vector_key")
    return found[0]


class Transplant(msgspec.Struct, kw_only=True):
    #: The component of `target` the edit moves onto; `edited` and `original` each hold one weighted component.
    component: str
    #: Where the three came from (upstream pages at their commits, licences), kept in `task-vector.json`.
    sources: str = ""


def transplant(ctx: Context, payload: Transplant, target: Weights, edited: Weights, original: Weights,
               tel: Telemetry) -> ModelArtifact:
    """`target` with `edited - original` added to `component`, its bf16 lane."""
    tk, ek, ok = ctx.tensorfs_source(target), ctx.tensorfs_source(edited), ctx.tensorfs_source(original)
    if payload.component not in tk.components:
        raise UnsupportedInput(f"target has no component {payload.component!r}", code="task_vector_input")
    held: Tensors = tk.components[payload.component].tensors
    (_, theirs), (_, base) = only(ek, "edited"), only(ok, "original")
    keys, base_keys = list(theirs), set(base)
    if set(keys) != base_keys:
        raise UnsupportedInput(f"edited and original differ in keys: {sorted(set(keys) ^ base_keys)[:6]}", code="task_vector_input")
    tel.log("checkpoints", target=tk.id, edited=ek.id, original=ok.id, tensors=len(keys))
    changed = []
    with tel.stage("comparing edited with original"):
        for n, key in enumerate(keys):
            a, b = theirs[key], base[key]
            if (a.dtype.upper(), tuple(a.shape)) != (b.dtype.upper(), tuple(b.shape)) or a.encoding or b.encoding:
                raise UnsupportedInput(f"{key}: edited {a.dtype}{tuple(a.shape)}, original {b.dtype}{tuple(b.shape)}",
                                       code="task_vector_input")
            if not same(theirs, base, key):
                changed.append(key)
            ctx.progress(n, len(keys))
    tel.log("changed", count=len(changed), keys=json.dumps(changed))
    if not changed:
        raise UnsupportedInput("edited is original: nothing to move", code="task_vector_input")
    record: list[dict[str, object]] = []
    with ctx.output("bf16", Derivation(base=tk)) as out, tel.stage("writing"):
        for n, key in enumerate(changed):
            mine = resolve(held, key)
            meta = held[mine]
            if meta.dtype.upper() != "BF16" or meta.encoding or tuple(meta.shape) != tuple(theirs[key].shape):
                raise UnsupportedInput(f"{mine}: target {meta.dtype}{tuple(meta.shape)}, edited {tuple(theirs[key].shape)}",
                                       code="task_vector_key")
            d = floats(read(theirs, key), theirs[key]) - floats(read(base, key), base[key])
            stock = read(held, mine)
            w = floats(stock, meta)
            new = bf16(w + d)
            share = rank1(d.astype(np.float64).reshape(d.shape[0], -1))
            if share < 0.9:
                raise UnsupportedInput(f"{key}'s difference is not a rank-one edit ({share:.3f})", code="task_vector_edit")
            data = new.tobytes()
            out.put(payload.component, mine, "BF16", tuple(meta.shape), data)
            entry: dict[str, object] = {
                "key": mine, "sha256": hashlib.sha256(data).hexdigest(), "rank1": round(share, 5),
                "relative": round(float(np.linalg.norm(d) / np.linalg.norm(w)), 6),
                "changed": round(float((new != np.frombuffer(stock, np.uint16).reshape(new.shape)).mean()), 5),
            }
            record.append(entry)
            rows = new.reshape(new.shape[0], -1)
            blocks = {f"row{r}": rows[r, :BLOCK].astype(">u2").tobytes().hex() for r in SAMPLE}
            tel.log("written", **entry, **blocks)  # type: ignore[arg-type]  # str, int and float fields
            ctx.progress(n, len(changed))
        out.component(payload.component).asset("task-vector.json", json.dumps({
            "target": tk.id, "edited": ek.id, "original": ok.id, "component": payload.component,
            "sources": payload.sources, "tensors": record,
        }, indent=1).encode())
        return ctx.adopt_model(out)


app.entrypoint(transplant, name="transplant", accelerator=False, weights=(WeightsOutput("bf16", max_new_bytes=3 << 30),))
