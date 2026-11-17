"""Abliteration of MiniMax-H3-Image-PE aimed at what it does to a sensitive request: it never refuses, it rewrites
the request into a sanitized caption (a clothed embrace for a sex scene). Heretic's method (p-e-w/heretic), scored on
that rewrite instead of on refusals.

- Directions (`Evidence`), one per layer output, each a difference of two means: `prompt`, sensitive vs neutral
  requests at the last prompt token (attempt 1's, which erased the content itself); `decision`, requests the stock
  PE sanitized vs sensitive ones it kept, same place; `response`, the stock PE's sanitized answers vs faithful
  answers to the same requests, over their tokens. `projected` takes each off the neutral mean first.
- Ablation: every residual writer (attention's `out_proj`/`o_proj`, the MLP's `down_proj`) loses `w · d dᵀ` of its
  output: `w` from a kernel over the layers per component (a peak at a layer, falling linearly to a floor at a
  distance, 0 beyond), `d` one interpolated direction or each layer's own.
- `abliteration_study` searches those parameters with the answers forced as text: how much likelier the faithful
  answers get than the sanitized ones, against the KL from the stock PE over its own answers to neutral requests.
  Its best, and any candidates given, then answer real requests sampled as the tool samples them.
- `abliteration` bakes a study's chosen ablation into a `bf16` lane: `W - w d dᵀ W` per writer, in float32, rounded
  once.
"""

from __future__ import annotations

import base64
import copy
import json
import math
import random
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

import msgspec
import torch
from cozy_runtime.author import (
    Context,
    Loader,
    ModelArtifact,
    Outputs,
    Telemetry,
    UnsupportedInput,
    Weights,
    WeightsOutput,
    uses_components,
)
from cozy_runtime.media import File, Files, ImageFile
from cozy_runtime.vlm import COMPONENT, Params, VisionLM
from cozy_runtime.weights_store import Derivation
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .application import app
from .expander import MAX_TOKENS, TEMPERATURE, TOP_P, H3ImagePE
from .lanes import _plain, _put
from .model import upright

Source = Literal["prompt", "decision", "response"]
#: The neutral mean each source's directions are projected off.
_BASIS: dict[str, str] = {"prompt": "prompt", "decision": "prompt", "response": "response"}
#: Stock top-k log-probabilities kept per reference position; the rest of the mass is one bucket.
TOP_K = 32
_LOGIT_ROWS = 512


class Kernel(msgspec.Struct, frozen=True, kw_only=True):
    """Heretic's weight over the layers: `peak` at layer `at`, falling linearly to `floor` `reach` layers away, 0
    beyond."""

    peak: float
    at: float
    floor: float
    reach: float

    def weight(self, layer: int) -> float:
        distance = abs(layer - self.at)
        if distance > self.reach:
            return 0.0
        return self.peak + distance / self.reach * (self.floor - self.peak)


class Ablation(msgspec.Struct, frozen=True, kw_only=True):
    source: Source
    #: Each direction off the neutral mean first (Heretic's projected abliteration).
    projected: bool
    #: One direction for every layer, interpolated at this layer output; None: each layer its own.
    index: float | None
    attention: Kernel
    mlp: Kernel


class Pair(msgspec.Struct, frozen=True):
    """A T2I request, the stock PE's answer that sanitized it and a faithful answer, each the whole turn as made
    (thought, `</think>`, the JSON)."""

    request: str
    sanitized: str
    faithful: str


class Reference(msgspec.Struct, frozen=True):
    """A neutral T2I request and the stock PE's answer to it: the KL is taken over this answer."""

    request: str
    answer: str


class Evidence(msgspec.Struct, frozen=True, kw_only=True):
    sensitive: list[str]
    neutral: list[str]
    #: Sensitive requests the stock PE sanitized, and ones it kept (`decision`); either may be empty.
    sanitized: list[str]
    kept: list[str]
    pairs: list[Pair]
    references: list[Reference]


class Request(msgspec.Struct, frozen=True, kw_only=True):
    id: str
    text: str
    #: Indexes into the run's pictures, `<Picture 1>` first: an edit. None: text to image.
    pictures: list[int] | None = None
    aspect_ratio: str | None = None


class Arm(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    ablation: Ablation | None


class Study(msgspec.Struct, frozen=True, kw_only=True):
    #: Requests every arm answers, sampled as the tool samples (thinking, temperature 1, top-p 0.95), at `seed`.
    requests: list[Request]
    seed: int = 0
    #: Text appended to the T2I PE's system prompt for every arm (a faithful-rewrite run); "" keeps MiniMax's.
    system_suffix: str = ""
    #: MiniMax's T2I system and task prompts, for a checkpoint that carries none (another Qwen3.8 writing faithful
    #: answers); None reads the checkpoint's.
    t2i_system: str | None = None
    t2i_task: str | None = None
    batch: int = 16
    edit_batch: int = 4
    #: Answer the requests with the stock PE too.
    stock: bool = True
    evidence: Evidence | None = None
    #: Search trials (the first `startup` of them random), then the `top` best within `kl_budget` answer too.
    trials: int = 0
    startup: int = 24
    top: int = 0
    kl_budget: float = 0.05
    #: Each nat per token of KL over the budget costs this much of the faithfulness shift.
    kl_penalty: float = 20.0
    candidates: list[Arm] | None = None
    #: Tokens per teacher-forced batch (rows times the longest row after the shared prefix).
    token_budget: int = 16384


class StudyResult(msgspec.Struct, frozen=True):
    report: File
    trials: int
    arms: list[str]


class Bake(msgspec.Struct, frozen=True, kw_only=True):
    """An ablation as `abliteration` writes it (a study's report holds one per arm)."""

    #: Unit directions, each base64 of `hidden` little-endian float32 values.
    vectors: list[str]
    #: (layer, its vector, its attention writer's weight, its MLP writer's weight); layers not here keep their bytes.
    rows: list[tuple[int, int, float, float]]


def _writers(layer: nn.Module) -> tuple[nn.Module, nn.Module]:
    attention = layer.linear_attn.out_proj if layer.block_type == "linear_attention" else layer.self_attn.o_proj
    return attention, layer.mlp.down_proj


def _writer_keys(layer: int, linear: bool) -> tuple[str, str]:
    prefix = f"model.language_model.layers.{layer}"
    return f"{prefix}.{'linear_attn.out_proj' if linear else 'self_attn.o_proj'}.weight", f"{prefix}.mlp.down_proj.weight"


class Directions:
    """Per source, each layer output's raw direction `(layers, hidden)` float32, and the neutral means."""

    def __init__(self, raw: dict[str, torch.Tensor], neutral: dict[str, torch.Tensor]) -> None:
        self.raw, self.neutral = raw, neutral

    def sources(self) -> list[str]:
        return sorted(self.raw)

    def per_layer(self, ablation: Ablation) -> torch.Tensor:
        """`(layers, hidden)` unit rows: the direction each layer's writers lose."""
        r = self.raw[ablation.source]
        if ablation.projected:
            n = F.normalize(self.neutral[_BASIS[ablation.source]], dim=-1)
            r = r - (r * n).sum(-1, keepdim=True) * n
        r = F.normalize(r, dim=-1)
        if ablation.index is None:
            return r
        low = min(math.floor(ablation.index), r.shape[0] - 1)
        high = min(low + 1, r.shape[0] - 1)
        return F.normalize(torch.lerp(r[low], r[high], ablation.index - low), dim=0).expand_as(r)

    def stats(self) -> dict[str, object]:
        """Each source's per-layer difference norm relative to the mean norm, and its cosine with the others."""
        out: dict[str, object] = {}
        for name, r in self.raw.items():
            row: dict[str, object] = {"norm": [round(float(v), 4) for v in r.norm(dim=-1)]}
            for other, o in self.raw.items():
                if other < name:
                    row[f"cos_{other}"] = [round(float(v), 3) for v in F.cosine_similarity(r, o, dim=-1)]
            out[name] = row
        return out


def weights(ablation: Ablation, layers: int) -> torch.Tensor:
    """`(layers, 2)`: each layer's attention and MLP writer weight."""
    return torch.tensor([[ablation.attention.weight(i), ablation.mlp.weight(i)] for i in range(layers)])


def bake(directions: Directions, ablation: Ablation) -> Bake:
    d, w = directions.per_layer(ablation), weights(ablation, directions.raw[ablation.source].shape[0])
    vectors: list[str] = []
    rows: list[tuple[int, int, float, float]] = []
    for layer in range(d.shape[0]):
        if not (w[layer] != 0).any():
            continue
        packed = base64.b64encode(d[layer].float().contiguous().numpy().astype("<f4").tobytes()).decode()
        if not vectors or vectors[-1] != packed:
            vectors.append(packed)
        rows.append((layer, len(vectors) - 1, float(w[layer, 0]), float(w[layer, 1])))
    return Bake(vectors=vectors, rows=rows)


class _Ablator:
    """Forward hooks for forced (eager) passes: every residual writer's output `y` becomes `y - w (y · d) d`, `w` and
    `d` set in place per trial."""

    def __init__(self, layers: Sequence[nn.Module], hidden: int, device: torch.device) -> None:
        self.weights = torch.zeros(len(layers), 2, device=device)
        self.directions = torch.zeros(len(layers), 2, hidden, device=device)
        self.handles = [
            writer.register_forward_hook(self._hook(i, c))
            for i, layer in enumerate(layers)
            for c, writer in enumerate(_writers(layer))
        ]

    def _hook(self, i: int, c: int) -> Callable[[nn.Module, object, torch.Tensor], torch.Tensor]:
        w, d = self.weights[i, c], self.directions[i, c]

        def hook(_module: nn.Module, _args: object, output: torch.Tensor) -> torch.Tensor:
            return output - (output @ d.to(output.dtype)).unsqueeze(-1) * (w * d).to(output.dtype)

        return hook

    def set(self, directions: Directions | None, ablation: Ablation | None) -> None:
        if directions is None or ablation is None:
            self.weights.zero_()
            return
        d = directions.per_layer(ablation).to(self.directions.device)
        self.directions.copy_(d[:, None, :].expand(-1, 2, -1))
        self.weights.copy_(weights(ablation, d.shape[0]))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


class _Baked:
    """An ablation baked into the writers' bf16 weights in place, as `abliteration` writes it (float32, rounded once),
    then taken out again: an arm decodes through Runtime's captured graphs, which refuse forward hooks."""

    def __init__(self, layers: Sequence[nn.Module], directions: Directions, ablation: Ablation) -> None:
        d, w = directions.per_layer(ablation), weights(ablation, len(layers))
        self.saved: list[tuple[nn.Parameter, torch.Tensor]] = []
        self.probe: tuple[nn.Parameter, float] | None = None
        try:
            for i, layer in enumerate(layers):
                for c, writer in enumerate(_writers(layer)):
                    weight = float(w[i, c])
                    if not weight:
                        continue
                    param = writer.weight
                    if param.dtype != torch.bfloat16 or param.device.type != "cuda":
                        raise UnsupportedInput(f"an ablated arm answers on a `bf16` lane on a GPU, not {param.dtype} on "
                                               f"{param.device}", code="study_lane")  # fmt: skip
                    self.saved.append((param, param.detach().to("cpu", copy=True)))
                    vec = d[i].to(param.device)
                    value = param.detach().float()
                    value -= weight * vec[:, None] * (vec @ value)[None, :]
                    param.data.copy_(value.to(torch.bfloat16))
                    del value
                    self.probe = (param, float(param.detach().float().sum()))
        except BaseException:
            self.restore()
            raise

    def held(self) -> bool:
        """Whether the baked weights are still in place (a paged-out weight comes back as stored)."""
        return self.probe is None or float(self.probe[0].detach().float().sum()) == self.probe[1]

    def restore(self) -> None:
        for param, original in self.saved:
            param.data.copy_(original.to(param.device))
        self.saved.clear()


class _Seq:
    """A sequence after the shared prefix: its ids, and `[start, end)`, the positions it is read at (averaged
    outputs) or the tokens it scores."""

    __slots__ = ("end", "ids", "start")

    def __init__(self, ids: list[int], start: int, end: int) -> None:
        self.ids, self.start, self.end = ids, start, end


def _rows(value: torch.Tensor, rows: int) -> torch.Tensor:
    if value.dim() and value.shape[0] == 1:
        return value.expand(rows, *value.shape[1:]).clone()
    return value.clone()


def _repeat(cache: Any, rows: int) -> Any:  # type: ignore[explicit-any]  # Transformers' cache
    """The prefix's cache once per row, every tensor its own: a forward from it writes into it."""
    made = copy.copy(cache)
    made.layers = []
    for layer in cache.layers:
        twin = copy.copy(layer)
        for name, value in vars(layer).items():
            if isinstance(value, torch.Tensor):
                setattr(twin, name, _rows(value, rows))
            elif isinstance(value, dict):
                setattr(twin, name, {k: _rows(v, rows) if isinstance(v, torch.Tensor) else v for k, v in value.items()})
        made.layers.append(twin)
    return made


class _Teacher:
    """Sequences forced through the language model after one shared prefix (the T2I PE's system and task prompt),
    each batch continuing from the prefix's cache, right-padded."""

    def __init__(self, pe: StudyPE, prefix: list[int], budget: int) -> None:
        model = pe.pipe.components[COMPONENT]
        self.text: nn.Module = model.model.language_model
        self.head: nn.Module = model.lm_head
        self.device = next(self.head.parameters()).device
        self.prefix = torch.tensor([prefix], device=self.device) if prefix else None
        self.pad = int(pe.processor.tokenizer.pad_token_id or 0)
        self.budget = budget
        self.cache: Any = None  # type: ignore[explicit-any]  # Transformers' cache

    def refresh(self) -> None:
        """The prefix's cache under the current weights and hooks."""
        self.cache = None
        if self.prefix is not None:
            self.cache = self.text(input_ids=self.prefix, use_cache=True).past_key_values

    def batches(self, seqs: Sequence[_Seq]) -> Iterator[list[int]]:
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i].ids))
        batch: list[int] = []
        for i in order:
            if batch and (len(batch) + 1) * len(seqs[i].ids) > self.budget:
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def hidden(self, rows: Sequence[_Seq]) -> torch.Tensor:
        """`(rows, longest, hidden)`: the final norm's output at each position after the prefix."""
        longest = max(len(r.ids) for r in rows)
        ids = torch.full((len(rows), longest), self.pad, dtype=torch.long)
        before = 0 if self.prefix is None else self.prefix.shape[1]
        mask = torch.zeros(len(rows), before + longest, dtype=torch.long)
        for b, row in enumerate(rows):
            ids[b, : len(row.ids)] = torch.tensor(row.ids)
            mask[b, : before + len(row.ids)] = 1
        cache = None if self.cache is None else _repeat(self.cache, len(rows))
        out = self.text(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), past_key_values=cache,
                        use_cache=cache is not None)  # fmt: skip
        return out.last_hidden_state

    def scored(self, hidden: torch.Tensor, rows: Sequence[_Seq]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each scored position's hidden state, target token and row: `(n, hidden)`, `(n,)`, `(n,)`."""
        states, targets, owners = [], [], []
        for b, row in enumerate(rows):
            states.append(hidden[b, row.start - 1 : row.end - 1])
            targets.append(torch.tensor(row.ids[row.start : row.end]))
            owners.append(torch.full((row.end - row.start,), b))
        return torch.cat(states), torch.cat(targets).to(self.device), torch.cat(owners).to(self.device)

    def log_probs(self, states: torch.Tensor) -> Iterator[tuple[int, torch.Tensor]]:
        for at in range(0, states.shape[0], _LOGIT_ROWS):
            logits = self.head(states[at : at + _LOGIT_ROWS]).float()
            yield at, logits - logits.logsumexp(-1, keepdim=True)


class StudyPE(H3ImagePE):
    """The PE with what a study reads: its language model's layers, forced sequences, and answers with their
    thoughts."""

    def load(self, loader: Loader) -> None:
        if "prompt/t2i/task_prompt.txt" in loader.assets.names():
            super().load(loader)
            return
        VisionLM.load(self, loader)  # a Qwen3.8 without MiniMax's prompts: the study's payload gives them
        self.t2i_system = self.t2i_task = self.edit_task = ""

    @uses_components(COMPONENT)
    def holding[T](self, fn: Callable[[], T]) -> T:
        """`fn`, the PE's weights held for it."""
        return fn()

    def t2i(self, request: str, answer: str | None = None) -> list[int]:
        """`request` under the T2I PE's prompts, the thought opened; then `answer` and the turn's end."""
        content = [{"type": "text", "text": self.t2i_task.format(request)}]
        chat = self.processor.apply_chat_template(
            [{"role": "system", "content": self.t2i_system}, {"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, enable_thinking=True,
        )  # fmt: skip
        text = chat if answer is None else f"{chat}{answer}<|im_end|>"
        ids: list[int] = self.processor.tokenizer.encode(text, add_special_tokens=False)
        return ids


def _shared(rows: Sequence[list[int]]) -> int:
    """The longest prefix every row shares, less a token, so every row keeps one."""
    first, n = rows[0], min(len(r) for r in rows)
    at = 0
    while at < n and all(r[at] == first[at] for r in rows):
        at += 1
    return max(0, at - 1)


class _Data:
    """A study's evidence tokenized once: prompt-only rows (last token read) and answered rows (answer read)."""

    def __init__(self, pe: StudyPE, evidence: Evidence) -> None:
        self.sets: dict[str, list[tuple[list[int], int]]] = {}

        def prompts(requests: list[str]) -> list[tuple[list[int], int]]:
            return [((ids := pe.t2i(r)), len(ids) - 1) for r in requests]

        def answered(request: str, answer: str) -> tuple[list[int], int]:
            ids, prompt = pe.t2i(request, answer), pe.t2i(request)
            return ids, _shared([ids, prompt]) + 1  # where the answer's tokens start, should they merge with the prompt's

        self.sets["sensitive"] = prompts(evidence.sensitive)
        self.sets["neutral"] = prompts(evidence.neutral)
        self.sets["sanitized"] = prompts(evidence.sanitized)
        self.sets["kept"] = prompts(evidence.kept)
        self.sets["unfaithful"] = [answered(p.request, p.sanitized) for p in evidence.pairs]
        self.sets["faithful"] = [answered(p.request, p.faithful) for p in evidence.pairs]
        self.sets["reference"] = [answered(r.request, r.answer) for r in evidence.references]
        everything = [ids for rows in self.sets.values() for ids, _ in rows]
        self.prefix = _shared(everything) if everything else 0

    def seqs(self, name: str, read: Literal["last", "answer"]) -> list[_Seq]:
        out = []
        for ids, start in self.sets[name]:
            tail = ids[self.prefix :]
            if read == "last":
                out.append(_Seq(tail, len(tail) - 1, len(tail)))
            else:
                out.append(_Seq(tail, start - self.prefix, len(tail)))
        return out


def _means(teacher: _Teacher, layers: Sequence[nn.Module], seqs: list[_Seq]) -> torch.Tensor:
    """`(layers, hidden)` float32: each layer's output averaged over each row's read positions, then over rows."""
    sums = torch.zeros(len(layers), layers[0].hidden_size, device=teacher.device, dtype=torch.float32)
    weight: list[torch.Tensor | None] = [None]

    def hook(i: int) -> Callable[[nn.Module, object, torch.Tensor], None]:
        def read(_module: nn.Module, _args: object, output: torch.Tensor) -> None:
            if weight[0] is not None:
                sums[i] += torch.einsum("bs,bsh->h", weight[0], output.float())

        return read

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    try:
        teacher.refresh()
        for batch in teacher.batches(seqs):
            rows = [seqs[i] for i in batch]
            w = torch.zeros(len(rows), max(len(r.ids) for r in rows), device=teacher.device)
            for b, row in enumerate(rows):
                w[b, row.start : row.end] = 1.0 / (row.end - row.start)
            weight[0] = w
            teacher.hidden(rows)
            weight[0] = None
    finally:
        for handle in handles:
            handle.remove()
    return (sums / len(seqs)).cpu()


def _directions(data: _Data, teacher: _Teacher, layers: Sequence[nn.Module], tel: Telemetry) -> Directions:
    raw: dict[str, torch.Tensor] = {}
    neutral: dict[str, torch.Tensor] = {}

    def mean(name: str, read: Literal["last", "answer"]) -> torch.Tensor:
        return _means(teacher, layers, data.seqs(name, read))

    if data.sets["neutral"]:
        neutral["prompt"] = mean("neutral", "last")
        if data.sets["sensitive"]:
            raw["prompt"] = mean("sensitive", "last") - neutral["prompt"]
    if data.sets["sanitized"] and data.sets["kept"]:
        raw["decision"] = mean("sanitized", "last") - mean("kept", "last")
    if data.sets["unfaithful"]:
        raw["response"] = mean("unfaithful", "answer") - mean("faithful", "answer")
    if data.sets["reference"]:
        neutral["response"] = mean("reference", "answer")
    if "response" not in neutral and "prompt" in neutral:
        neutral["response"] = neutral["prompt"]
    if "prompt" not in neutral and "response" in neutral:
        neutral["prompt"] = neutral["response"]
    tel.log("directions", sources=sorted(raw))
    return Directions(raw, neutral)


class _Measure(msgspec.Struct, frozen=True):
    faith: float
    prefer: float
    kl: float


class _Scorer:
    """One trial's reading: the pairs' log-likelihoods and the KL over the references, against the stock PE's."""

    def __init__(self, data: _Data, teacher: _Teacher) -> None:
        self.teacher = teacher
        self.unfaithful = data.seqs("unfaithful", "answer")
        self.faithful = data.seqs("faithful", "answer")
        self.references = data.seqs("reference", "answer")
        self.seqs = [*self.unfaithful, *self.faithful, *self.references]
        self.kinds = [0] * len(self.unfaithful) + [1] * len(self.faithful) + [2] * len(self.references)
        self.stock: torch.Tensor | None = None
        self.top: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _read(self) -> tuple[torch.Tensor, float, int]:
        """Each sequence's summed log-probability, and the summed KL and positions over the references."""
        teacher = self.teacher
        teacher.refresh()
        sums = torch.zeros(len(self.seqs), dtype=torch.float64)
        kl, positions = 0.0, 0
        for batch in teacher.batches(self.seqs):
            rows = [self.seqs[i] for i in batch]
            states, targets, owners = teacher.scored(teacher.hidden(rows), rows)
            totals = torch.zeros(len(rows), device=teacher.device, dtype=torch.float64)
            for at, logp in teacher.log_probs(states):
                n = logp.shape[0]
                own = owners[at : at + n]
                totals.index_add_(0, own, logp.gather(1, targets[at : at + n, None])[:, 0].double())
                for b in own.unique().tolist():
                    seq = batch[b]
                    if self.kinds[seq] != 2:
                        continue
                    lp = logp[own == b]
                    key = (seq, at)
                    if self.stock is None:
                        top = lp.topk(TOP_K, dim=-1)
                        self.top[key] = (top.indices, top.values)
                        continue
                    ids, p_log = self.top[key]
                    q_log = lp.gather(1, ids)
                    p, q = p_log.exp(), q_log.exp()
                    p_rest = (1 - p.sum(-1)).clamp_min(1e-9)
                    q_rest = (1 - q.sum(-1)).clamp_min(1e-9)
                    rows_kl = (p * (p_log - q_log)).sum(-1) + p_rest * (p_rest.log() - q_rest.log())
                    kl += float(rows_kl.sum())
                    positions += int(rows_kl.shape[0])
            sums[batch] = totals.cpu()
        return sums, kl, positions

    def baseline(self) -> None:
        self.stock = None
        self.top.clear()
        sums, _, _ = self._read()
        self.stock = sums

    def measure(self) -> _Measure:
        assert self.stock is not None
        sums, kl, positions = self._read()
        n = len(self.unfaithful)
        shifts, prefer = [], 0
        for i in range(n):
            s, f = self.unfaithful[i], self.faithful[i]
            ns, nf = s.end - s.start, f.end - f.start
            now = float(sums[n + i] - sums[i])
            before = float(self.stock[n + i] - self.stock[i])
            shifts.append((now - before) / ((ns + nf) / 2))
            prefer += int(float(sums[n + i]) / nf > float(sums[i]) / ns)
        return _Measure(sum(shifts) / max(1, n), prefer / max(1, n), kl / max(1, positions))


def _space(rng: random.Random, sources: list[str], last: int) -> Ablation:
    def kernel() -> Kernel:
        peak = rng.uniform(0.2, 1.5)
        return Kernel(peak=peak, at=rng.uniform(0.3 * last, last), floor=peak * rng.uniform(0, 1),
                      reach=rng.uniform(1.0, 0.6 * last))  # fmt: skip

    index = None if rng.random() < 0.5 else rng.uniform(0.3 * last, 0.9 * last)
    return Ablation(source=rng.choice(sources), projected=rng.random() < 0.5, index=index,  # type: ignore[arg-type]  # one of `sources`
                    attention=kernel(), mlp=kernel())  # fmt: skip


def _mutated(rng: random.Random, parent: Ablation, sources: list[str], last: int, scale: float) -> Ablation:
    def jitter(value: float, low: float, high: float) -> float:
        return min(high, max(low, value + rng.gauss(0, scale * (high - low))))

    def kernel(k: Kernel) -> Kernel:
        peak = jitter(k.peak, 0.0, 2.0)
        return Kernel(peak=peak, at=jitter(k.at, 0.0, last), floor=min(peak, jitter(k.floor, 0.0, 2.0)),
                      reach=jitter(k.reach, 1.0, last))  # fmt: skip

    index = parent.index
    if rng.random() < 0.15:
        index = None if index is not None else rng.uniform(0.3 * last, 0.9 * last)
    elif index is not None:
        index = jitter(index, 0.0, last)
    source = rng.choice(sources) if rng.random() < 0.15 else parent.source
    projected = (not parent.projected) if rng.random() < 0.15 else parent.projected
    return Ablation(source=source, projected=projected, index=index, attention=kernel(parent.attention),  # type: ignore[arg-type]  # one of `sources`
                    mlp=kernel(parent.mlp))  # fmt: skip


def _search(ctx: Context, payload: Study, data: _Data, teacher: _Teacher, layers: Sequence[nn.Module],
            directions: Directions, rng: random.Random, last: int, report: dict[str, object],
            tel: Telemetry) -> list[Arm]:  # fmt: skip
    """The stock reading, `payload.trials` ablations and the given candidates scored by forcing the evidence's
    answers (writer hooks, removed after), and the `top` best within the KL budget."""
    scorer = _Scorer(data, teacher)
    ablator = _Ablator(layers, layers[0].hidden_size, teacher.device)
    trials: list[dict[str, object]] = []
    measured: list[tuple[Ablation, _Measure, float]] = []
    try:
        if scorer.unfaithful or scorer.references:
            with tel.stage("stock reading"):
                ablator.set(None, None)
                scorer.baseline()
        if scorer.stock is None:
            return []
        sources = directions.sources()
        with tel.stage("search"):
            for t in range(payload.trials):
                ctx.raise_if_cancelled()
                if t < payload.startup or rng.random() < 0.15 or not measured:
                    ablation = _space(rng, sources, last)
                else:
                    best = sorted(measured, key=lambda m: -m[2])[:5]
                    parent = best[min(len(best) - 1, int(rng.expovariate(1.0)))][0]
                    ablation = _mutated(rng, parent, sources, last, 0.12 * (1 - t / payload.trials) + 0.03)
                began = time.monotonic()
                ablator.set(directions, ablation)
                m = scorer.measure()
                score = m.faith - payload.kl_penalty * max(0.0, m.kl - payload.kl_budget)
                measured.append((ablation, m, score))
                trials.append({"trial": t, "ablation": msgspec.to_builtins(ablation), **msgspec.to_builtins(m),
                               "score": round(score, 5), "seconds": round(time.monotonic() - began, 1)})  # fmt: skip
                tel.log("trial", t=t, faith=round(m.faith, 4), prefer=round(m.prefer, 3), kl=round(m.kl, 4),
                        score=round(score, 4))  # fmt: skip
                ctx.progress(t, payload.trials)
            for arm in payload.candidates or []:
                if arm.ablation is not None:
                    ablator.set(directions, arm.ablation)
                    m = scorer.measure()
                    trials.append({"trial": arm.name, "ablation": msgspec.to_builtins(arm.ablation), **msgspec.to_builtins(m)})
                    tel.log("candidate", name=arm.name, faith=round(m.faith, 4), prefer=round(m.prefer, 3), kl=round(m.kl, 4))
    finally:
        ablator.remove()
        report["trials"] = trials
    within = sorted((x for x in measured if x[1].kl <= payload.kl_budget), key=lambda x: -x[1].faith)
    return [Arm(name=f"top{i + 1}", ablation=x[0]) for i, x in enumerate(within[: payload.top])]


def _answer(pe: StudyPE, requests: Sequence[Request], pictures: list[Image.Image], payload: Study,
            ctx: Context) -> list[dict[str, object]]:  # fmt: skip
    """Every request answered as the tool samples, edits and T2I in their own batches."""
    out: dict[str, dict[str, object]] = {}
    groups = [[r for r in requests if not r.pictures], [r for r in requests if r.pictures]]
    for group, size in zip(groups, (payload.batch, payload.edit_batch), strict=True):
        for at in range(0, len(group), size):
            ctx.raise_if_cancelled()
            chunk = group[at : at + size]
            built = [pe.question(r.text, [pictures[i] for i in r.pictures or []], r.aspect_ratio) for r in chunk]
            system = built[0][1] + payload.system_suffix if not chunk[0].pictures else ""
            params = Params(system=system, thinking=True, reasoning_effort=None, stop=[], max_new_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE, top_p=TOP_P, presence_penalty=0.0, seed=payload.seed)  # fmt: skip
            began = time.monotonic()
            final = pe.engine.generate([q for q, _, _ in built], params, on_delta=lambda _delta: None)
            seconds = round(time.monotonic() - began, 1)
            for row, request in enumerate(chunk):
                out[request.id] = {
                    "id": request.id, "thought": final.thoughts[row], "answer": final.texts[row],
                    "finish": final.finish_reasons[row], "key": built[row][2], "batch_seconds": seconds,
                }  # fmt: skip
    return [out[r.id] for r in requests]


def abliteration_study(ctx: Context, payload: Study, expander: StudyPE, out: Outputs, tel: Telemetry, *,
                       pictures: Files[ImageFile]) -> StudyResult:  # fmt: skip
    """Directions from `evidence`, a search over ablations scored by forcing its answers, then each arm (stock, the
    search's best, `candidates`) answering `requests`; everything in one JSON report."""
    shown = [upright(p.read()) for p in pictures]
    expander.t2i_system = expander.t2i_system if payload.t2i_system is None else payload.t2i_system
    expander.t2i_task = expander.t2i_task if payload.t2i_task is None else payload.t2i_task
    if not expander.t2i_task:
        raise UnsupportedInput("this checkpoint carries no T2I PE prompts: give `t2i_system` and `t2i_task`", code="study_prompts")
    report: dict[str, object] = {"study": msgspec.to_builtins(msgspec.structs.replace(payload, evidence=None))}
    rng = random.Random(payload.seed)

    def run() -> None:
        model = expander.pipe.components[COMPONENT]
        layers = list(model.model.language_model.layers)
        last = len(layers) - 1
        with torch.inference_mode():
            directions = None
            chosen: list[Arm] = []
            if payload.evidence is not None:
                with tel.stage("tokenizing"):
                    data = _Data(expander, payload.evidence)
                teacher = _Teacher(expander, _prefix_ids(data), payload.token_budget)
                report["prefix_tokens"] = data.prefix
                with tel.stage("prefix check"):
                    report["prefix_check"] = _prefix_check(expander, data, payload.token_budget)
                with tel.stage("directions"):
                    directions = _directions(data, teacher, layers, tel)
                report["directions"] = directions.stats()
                chosen = _search(ctx, payload, data, teacher, layers, directions, rng, last, report, tel)
            arms = ([Arm(name="stock", ablation=None)] if payload.stock else []) + chosen + list(payload.candidates or [])
            answered = []
            with tel.stage("answering"):
                for arm in arms:
                    if arm.ablation is not None and directions is None:
                        raise UnsupportedInput("an ablated arm needs `evidence` for its directions", code="study_evidence")
                    baked = _Baked(layers, directions, arm.ablation) if arm.ablation is not None and directions else None
                    began = time.monotonic()
                    try:
                        answers = _answer(expander, payload.requests, shown, payload, ctx) if payload.requests else []
                        held = baked is None or baked.held()
                    finally:
                        if baked is not None:
                            baked.restore()
                    row: dict[str, object] = {"name": arm.name, "seconds": round(time.monotonic() - began, 1),
                                              "held": held, "answers": answers}  # fmt: skip
                    if arm.ablation is not None and directions is not None:
                        row["ablation"] = msgspec.to_builtins(arm.ablation)
                        row["bake"] = msgspec.to_builtins(bake(directions, arm.ablation))
                    answered.append(row)
                    tel.log("arm", name=arm.name, seconds=row["seconds"], held=held)
            report["arms"] = answered

    expander.holding(run)
    names = [str(a["name"]) for a in report.get("arms", [])]  # type: ignore[attr-defined]  # a list of rows
    trials_run = len(report.get("trials", []))  # type: ignore[arg-type]  # a list
    report_file = out.save_bytes(json.dumps(report).encode(), media_type="application/json")
    return StudyResult(report=report_file, trials=trials_run, arms=names)


def _prefix_ids(data: _Data) -> list[int]:
    for rows in data.sets.values():
        if rows:
            return rows[0][0][: data.prefix]
    return []


def _prefix_check(pe: StudyPE, data: _Data, budget: int) -> dict[str, float]:
    """One reference answer scored from the prefix's cache and from scratch: the two must agree."""
    rows = data.sets["reference"] or data.sets["faithful"]
    if not rows or not data.prefix:
        return {}
    ids, start = rows[0]
    cached = _Teacher(pe, ids[: data.prefix], budget)
    whole = _Teacher(pe, [], budget)
    totals = []
    for teacher, seq in ((cached, _Seq(ids[data.prefix :], start - data.prefix, len(ids) - data.prefix)),
                         (whole, _Seq(ids, start, len(ids)))):  # fmt: skip
        teacher.refresh()
        states, targets, _ = teacher.scored(teacher.hidden([seq]), [seq])
        total = 0.0
        for at, logp in teacher.log_probs(states):
            total += float(logp.gather(1, targets[at : at + logp.shape[0], None]).sum())
        totals.append(total)
    tokens = len(ids) - start
    return {"cached": totals[0], "whole": totals[1], "per_token_gap": abs(totals[0] - totals[1]) / max(1, tokens)}


app.entrypoint(abliteration_study, name="abliteration_study")


async def abliteration(ctx: Context, payload: Bake, *, source: Weights, tel: Telemetry) -> ModelArtifact:
    """The PE's `bf16` lane (`source`) with `payload`'s ablation baked in: each listed layer's attention and MLP
    writers `W - w d dᵀ W`, in float32 from the bf16 bytes, rounded once. Every other tensor is the source's."""
    src = ctx.tensorfs_source(source)
    tensors = src.components[COMPONENT].tensors
    if any(row.encoding is not None for row in tensors.values()):
        raise UnsupportedInput("abliterate a `bf16` lane (`expander_bf16`)", code="expander_source")
    vectors = []
    for packed in payload.vectors:
        d = torch.frombuffer(bytearray(base64.b64decode(packed)), dtype=torch.float32)
        vectors.append(d / d.norm())
    writes: list[tuple[str, float, torch.Tensor]] = []
    for layer, vector, attention, mlp in payload.rows:
        linear = _writer_keys(layer, True)[0] in tensors
        attention_key, mlp_key = _writer_keys(layer, linear)
        for key, weight in ((attention_key, attention), (mlp_key, mlp)):
            if key not in tensors:
                raise UnsupportedInput(f"this checkpoint has no {key}", code="expander_source")
            if weight:
                writes.append((key, weight, vectors[vector]))
    with ctx.output("bf16", Derivation(base=src, order=[(COMPONENT, k) for k in tensors])) as out:
        with tel.stage("orthogonalizing"):
            for done, (key, weight, d) in enumerate(writes, 1):
                ctx.raise_if_cancelled()
                if out.has(COMPONENT, key):
                    continue
                value = _plain(src, COMPONENT, key).float()
                if value.shape[0] != d.shape[0]:
                    raise UnsupportedInput(f"{key} writes {value.shape[0]} values, not {d.shape[0]}", code="expander_source")
                value -= weight * d[:, None] * (d @ value)[None, :]
                _put(out, COMPONENT, key, value.to(torch.bfloat16))
                del value
                ctx.progress(done - 1, len(writes))
        artifact = ctx.adopt_model(out)
    tel.log("abliterated", writers=len(writes), vectors=len(vectors))
    return artifact


app.entrypoint(abliteration, name="abliteration", accelerator=False,
               weights=(WeightsOutput("bf16", max_new_bytes=24 << 30),))  # fmt: skip
