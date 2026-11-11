"""AtomicChat's abliterated models, as the stock checkpoint with the edited tensors in place.

AtomicChat publishes each edit only as a BF16 GGUF, and TensorFS's GGUF import has no name map for
its archs (qwen3vl, qwen35), so the imported commit keeps llama.cpp's names. The edit is
`W' = W - 0.75 r rᵀW` on the residual writers (its card), so every other tensor is the stock
model's. This writes a commit on the stock one (`stock`) whose edited tensors are the imported
GGUF's (`gguf`), referenced, never rewritten. Before it, every tensor the edit leaves is checked
byte-identical in the GGUF (first 64 KiB of each); after, each edited tensor's change from stock
is checked to be rank one, along one direction shared by every writer. Qwen3.5's linear-attention
`out_proj` is the one edited tensor llama.cpp stores in another layout (its value heads tiled), so it
is written back in stock's head order: each column block's nearest stock head.

    cozy model upload hf://AtomicChat/…/…-BF16.gguf <org/name> --lane gguf --mirror <page> --on <machine>
    cozy run ./ingest.py/assemble kind=te model.stock=qwen/qwen-image-2.1@0.1.0 \
      model.gguf=<org/name>@sha256:<gguf commit> --lane bf16 --on <machine>
    cozy model upload run:<run> <org/name> --lane bf16
"""

from __future__ import annotations

import json
from typing import Protocol

import msgspec
import numpy as np
from cozy_runtime.author import App, Context, ModelArtifact, Telemetry, Weights, WeightsOutput
from cozy_runtime.weights_store import Derivation, TensorMetadata
from numpy.typing import NDArray

app = App()

#: The stock component each kind edits.
COMPONENT = {"te": "text_encoder", "pup": "model"}
CHUNK = 64 << 20
LM = "model.language_model"


def plan(kind: str) -> tuple[dict[str, str], dict[str, str]]:
    """(edited, kept): stock key -> GGUF name, for the card's writers and for tensors it leaves."""
    edited, kept = {}, {"lm_head.weight": "output.weight"}
    if kind == "te":
        edited[f"{LM}.embed_tokens.weight"] = "token_embd.weight"
        for i in range(36):
            edited[f"{LM}.layers.{i}.self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"
            edited[f"{LM}.layers.{i}.mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"
            for a, b in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                         ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up")):
                kept[f"{LM}.layers.{i}.{a}.weight"] = f"blk.{i}.{b}.weight"
        return edited, kept
    kept[f"{LM}.embed_tokens.weight"] = "token_embd.weight"
    for i in range(32):
        full = i % 4 == 3
        writer = ("self_attn.o_proj", "attn_output") if full else ("linear_attn.out_proj", "ssm_out")
        edited[f"{LM}.layers.{i}.{writer[0]}.weight"] = f"blk.{i}.{writer[1]}.weight"
        edited[f"{LM}.layers.{i}.mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"
        pairs = [("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up")]
        pairs += [("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v")] if full else []
        for a, b in pairs:
            kept[f"{LM}.layers.{i}.{a}.weight"] = f"blk.{i}.{b}.weight"
    return edited, kept


class Tensors(Protocol):
    """What `read` takes of a component's TensorFS tensors."""

    def __getitem__(self, key: str, /) -> TensorMetadata: ...
    def read_into(self, key: str, out: memoryview, *, offset: int = 0) -> None: ...


def read(tensors: Tensors, key: str, n: int | None = None) -> bytes:
    size = int(np.prod(tensors[key].shape)) * 2
    n = size if n is None else min(n, size)
    out = bytearray(n)
    for at in range(0, n, CHUNK):
        tensors.read_into(key, memoryview(out)[at:min(n, at + CHUNK)], offset=at)
    return bytes(out)


#: Qwen3.5's linear attention: 32 value heads of 128 channels, whose order llama.cpp's qwen35 converter changes in
#: `out_proj`'s columns.
V_HEADS, V_DIM = 32, 128


def head_order(new: NDArray[np.float64], old: NDArray[np.float64]) -> list[int]:
    """The stock head of each GGUF column block of an edited `out_proj`: the nearest, since the edit moves each
    block by about 1% and any other head is a different matrix. Refuses a match that is not clear and one-to-one."""
    a = new.reshape(new.shape[0], V_HEADS, V_DIM).transpose(1, 0, 2).reshape(V_HEADS, -1)
    b = old.reshape(old.shape[0], V_HEADS, V_DIM).transpose(1, 0, 2).reshape(V_HEADS, -1)
    dist = ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T).clip(min=0) ** 0.5
    order = [int(i) for i in dist.argmin(1)]
    near = np.sort(dist, axis=1)
    if sorted(order) != list(range(V_HEADS)) or (near[:, 0] * 5 > near[:, 1]).any():
        raise ValueError(f"out_proj's column blocks match no clear head order: {order}")
    return order


def bf16(data: bytes, shape: tuple[int, ...]) -> NDArray[np.float32]:
    return (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


class Assemble(msgspec.Struct):
    #: `te` (Qwen-Image-2.1's text encoder) or `pup` (FLUX.3 Klein's prompt upsampler).
    kind: str


@app.entrypoint(weights=(WeightsOutput("bf16", max_new_bytes=2 << 30),), accelerator=False)
def assemble(ctx: Context, payload: Assemble, gguf: Weights, stock: Weights, tel: Telemetry) -> ModelArtifact:
    """The abliterated commit of `kind` on `stock`, its bf16 lane."""
    component = COMPONENT[payload.kind]
    ck, gg = ctx.tensorfs_source(stock), ctx.tensorfs_source(gguf)
    held = ck.components[component].tensors
    (named, imported), = gg.components.items()
    theirs = imported.tensors
    edited, kept = plan(payload.kind)
    tel.log("checkpoints", stock=ck.id, gguf=gg.id, gguf_component=named, gguf_tensors=len(theirs))
    for key, name in {**edited, **kept}.items():
        mine, other = held[key], theirs[name]
        if (mine.dtype.upper(), tuple(mine.shape)) != (other.dtype.upper(), tuple(other.shape)) or other.encoding:
            raise ValueError(f"{key} {mine.dtype}{tuple(mine.shape)} against {name} {other.dtype}{tuple(other.shape)}")
    differ = [key for key, name in kept.items() if read(held, key, 65536) != read(theirs, name, 65536)]
    tel.log("unedited tensors", checked=len(kept), identical=len(kept) - len(differ), differ=str(differ[:8]))
    if differ:
        raise ValueError(f"{len(differ)} tensors the edit leaves differ from stock: {differ[:4]}")
    report: dict[str, dict[str, float]] = {}
    directions, orders = [], set()
    with ctx.output("bf16", Derivation(base=ck, sources=(gg,))) as out:
        with tel.stage("copying edited tensors"):
            for n, (key, name) in enumerate(edited.items()):
                new, old, shape = read(theirs, name), read(held, key), tuple(held[key].shape)
                if name.endswith("ssm_out.weight"):
                    # Back to stock's head order: GGUF column block j is stock head order[j].
                    order = head_order(bf16(new, shape).astype(np.float64), bf16(old, shape).astype(np.float64))
                    tiled = np.frombuffer(new, np.uint16).reshape(shape[0], V_HEADS, V_DIM)
                    stock_order = np.empty_like(tiled)
                    stock_order[:, order] = tiled
                    new = stock_order.tobytes()
                    orders.add(tuple(order))
                    out.put(component, key, "BF16", shape, new)
                else:
                    out.copy(component, key, gg, from_component=named, from_key=name)
                entry: dict[str, float] = {"changed": float((np.frombuffer(new, np.uint16) != np.frombuffer(old, np.uint16)).mean())}
                if "embed" not in key and (n % 6 == 1 or name.endswith("ssm_out.weight")):
                    d = (bf16(new, shape) - bf16(old, shape)).astype(np.float64)
                    u = np.random.default_rng(0).standard_normal(int(d.shape[0]))
                    for _ in range(30):
                        u = d @ (d.T @ u)
                        u /= np.linalg.norm(u)
                    entry["rank1"] = float(np.linalg.norm(d.T @ u) ** 2 / (d * d).sum())
                    entry["relative"] = float(np.linalg.norm(d) / np.linalg.norm(bf16(old, shape)))
                    directions.append(u)
                report[key] = entry
                ctx.progress(n, len(edited))
        cosines = [abs(float(directions[0] @ u)) for u in directions[1:]]
        if min(e.get("rank1", 1.0) for e in report.values()) < 0.95 or min(cosines) < 0.999:
            raise ValueError(f"the edit is not one shared rank-1 direction: {report}")
        tel.log("edit", tensors=len(report), changed_min=round(min(e["changed"] for e in report.values()), 4),
                rank1=str([round(e["rank1"], 5) for e in report.values() if "rank1" in e]),
                relative=str([round(e["relative"], 5) for e in report.values() if "relative" in e]),
                direction_cosines=str([round(c, 5) for c in cosines]),
                head_orders=str([list(o) for o in orders]))
        tel.log("edit tensors", report=json.dumps(report)[:60000])
        return ctx.adopt_model(out)
