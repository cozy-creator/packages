"""LTX-2.5's video modulation, one row per latent frame.

Image-to-video gives each video token its own timestep, `t * (1 - mask)`, and Diffusers' LTX2 embeds it per token:
`temb` is [batch, tokens, 9 x 4096] (8.86 GiB at 20 s 1080p, 129,076 tokens) and each block's `get_mod_params` adds
its table into a second copy as large. A latent frame's tokens share its timestep, so inside `per_frame` the video
time embedding runs once per frame (61 rows at 20 s) and stays per frame (`Frames`) through Diffusers' block and
output code, which run as written. Where a row meets its tokens (`x * (1 + scale) + shift`, `x + y * gate`) the op
runs on a [batch, frames, tokens per frame, width] view: the same elementwise arithmetic on the same values.

Diffusers' code reaches `Frames` only through operators and the few tensor methods it calls on `temb` (`view`,
`reshape`, `unbind`, `squeeze`, indexing the trailing dims); `Tensor * frames` returns NotImplemented, so Python asks
`Frames.__rmul__`. Anything else raises. Copying Diffusers' block and transformer forwards instead would duplicate
543 lines to change 12.
"""

from __future__ import annotations

import inspect
import math
import operator
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from diffusers.models.transformers.transformer_ltx2 import LTX2AdaLayerNormSingle, LTX2VideoTransformer3DModel
from torch import Tensor

type Op = Callable[[Tensor | float, Tensor | float], Tensor]
type Run = tuple[int, int, int, int]

#: The caller's statement for the transformer calls inside `per_frame`.
_PER_FRAME: ContextVar[bool] = ContextVar("ltx_2_5_per_frame", default=False)
#: The running transformer call's video tokens, where `_PER_FRAME` holds and its shapes allow it.
_CALL: ContextVar[Grid | None] = ContextVar("ltx_2_5_frame_grid", default=None)


@contextmanager
def per_frame() -> Iterator[None]:
    """Within it, the caller states that every video token's timestep is its latent frame's."""
    token = _PER_FRAME.set(True)
    try:
        yield
    finally:
        _PER_FRAME.reset(token)


@dataclass(frozen=True, slots=True)
class Grid:
    """This rank's video tokens: `tokens` of them from `start` of the whole video, latent frames of `size` tokens.
    `runs`: (first token, end token, first frame, end frame) from this rank's first token and frame; whole frames in
    a row make one run, a frame the rank's split cuts is a run alone."""

    batch: int
    tokens: int
    start: int
    size: int
    runs: tuple[Run, ...]

    @classmethod
    def of(cls, batch: int, tokens: int, start: int, size: int) -> Grid:
        first, last = start // size, (start + tokens - 1) // size
        runs: list[Run] = []
        for frame in range(first, last + 1):
            begin = max(start, frame * size) - start
            end = min(start + tokens, (frame + 1) * size) - start
            if runs and end - begin == size:
                a, b, f, g = runs[-1]
                if b - a == (g - f) * size:
                    runs[-1] = (a, end, f, frame - first + 1)
                    continue
            runs.append((begin, end, frame - first, frame - first + 1))
        return cls(batch, tokens, start, size, tuple(runs))

    def frames(self) -> int:
        return self.runs[-1][3]

    def firsts(self, timestep: Tensor) -> Tensor:
        """Each frame's timestep, [batch, frames], from this rank's [batch, tokens]: its first token's."""
        parts = [
            timestep[:, a:b].reshape(self.batch, g - f, (b - a) // (g - f))[:, :, 0] for a, b, f, g in self.runs
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


class Frames:
    """A per-token value [batch, tokens, *tail] as its frames' rows [batch, frames, *tail]."""

    __slots__ = ("grid", "rows")

    def __init__(self, grid: Grid, rows: Tensor) -> None:
        self.grid, self.rows = grid, rows

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.grid.batch, self.grid.tokens, *self.rows.shape[2:]))

    @property
    def dtype(self) -> torch.dtype:
        return self.rows.dtype

    @property
    def device(self) -> torch.device:
        return self.rows.device

    def size(self, dim: int) -> int:
        return self.shape[dim]

    def dim(self) -> int:
        return self.rows.dim()

    def view(self, *shape: int) -> Frames:
        return self.reshape(*shape)

    def reshape(self, *shape: int) -> Frames:
        """Its tail reshaped (`temb.reshape(batch, tokens, 9, -1)`); batch and tokens stay, tokens or -1."""
        batch, tokens, *tail = shape
        tail_of = math.prod(self.rows.shape[2:])
        if (
            batch != self.grid.batch
            or tokens not in (self.grid.tokens, -1)
            or (tokens == -1 and (-1 in tail or math.prod(tail) != tail_of))
        ):
            raise TypeError(f"per-frame rows {tuple(self.shape)} reshape only their tail, not to {shape}")
        return Frames(self.grid, self.rows.reshape(batch, self.rows.shape[1], *tail))

    def unbind(self, dim: int) -> tuple[Frames, ...]:
        return tuple(Frames(self.grid, rows) for rows in self.rows.unbind(self._tail(dim)))

    def squeeze(self, dim: int) -> Frames:
        return Frames(self.grid, self.rows.squeeze(self._tail(dim)))

    def __getitem__(self, index: tuple[slice[int | None] | int | None, ...]) -> Frames:
        """`[:, :, …]`: an index of its tail."""
        whole = slice(None)
        if len(index) < 2 or index[0] != whole or index[1] != whole:
            raise TypeError(f"per-frame rows index only their tail, not {index!r}")
        return Frames(self.grid, self.rows[index])

    def __add__(self, other: object) -> Frames | Tensor:
        return self._apply(operator.add, other, first=True)

    def __radd__(self, other: object) -> Frames | Tensor:
        return self._apply(operator.add, other, first=False)

    def __mul__(self, other: object) -> Frames | Tensor:
        return self._apply(operator.mul, other, first=True)

    def __rmul__(self, other: object) -> Frames | Tensor:
        return self._apply(operator.mul, other, first=False)

    def _kept(self, rows: Tensor) -> Frames:
        if rows.shape[:2] != self.rows.shape[:2]:
            raise TypeError(f"per-frame rows {tuple(self.shape)} would change batch or frames")
        return Frames(self.grid, rows)

    def _tail(self, dim: int) -> int:
        axis = dim % self.rows.dim()
        if axis < 2:
            raise TypeError(f"per-frame rows act on their tail, not dim {dim}")
        return axis

    def _apply(self, op: Op, other: object, *, first: bool) -> Frames | Tensor:
        """`self op other` (`first`) or `other op self`: on the rows where `other` is a number, these rows or a
        table without tokens; per frame where it is a [batch, tokens, …] tensor, a token one."""
        if isinstance(other, Frames):
            if other.grid != self.grid:
                raise TypeError("per-frame rows of two grids")
            return self._kept(op(self.rows, other.rows) if first else op(other.rows, self.rows))
        if isinstance(other, int | float):
            return self._kept(op(self.rows, other) if first else op(other, self.rows))
        if not isinstance(other, Tensor) or other.dim() > self.dim():
            raise TypeError(f"per-frame rows {tuple(self.shape)} meet {other!r:.80}")
        if other.dim() <= self.dim() - 2 or other.shape[-self.dim() + 1] == 1:
            return self._kept(op(self.rows, other) if first else op(other, self.rows))
        if other.dim() != self.dim() or other.shape[1] != self.grid.tokens:
            raise TypeError(f"per-frame rows {tuple(self.shape)} meet {tuple(other.shape)}")
        parts: list[Tensor] = []
        for a, b, f, g in self.grid.runs:
            frames = g - f
            tokens = other[:, a:b]
            by_frame = tokens.reshape(tokens.shape[0], frames, (b - a) // frames, *tokens.shape[2:])
            rows = self.rows[:, f:g].unsqueeze(2)
            made = op(rows, by_frame) if first else op(by_frame, rows)
            parts.append(made.reshape(made.shape[0], b - a, *made.shape[3:]))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


class FrameEmbedding(LTX2AdaLayerNormSingle):
    """Diffusers' LTX2 AdaLN timestep embedding, on one timestep a frame where the call's grid holds."""

    def forward(  # type: ignore[override]  # returns `Frames` where the call is per frame
        self,
        timestep: Tensor,
        added_cond_kwargs: dict[str, Tensor] | None = None,
        batch_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Frames, Frames]:
        grid = _CALL.get()
        if grid is None or timestep.numel() != grid.batch * grid.tokens:
            whole = super().forward(timestep, added_cond_kwargs, batch_size, hidden_dtype)
            return whole[0], whole[1]
        firsts = grid.firsts(timestep.view(grid.batch, grid.tokens)).flatten()
        made = super().forward(firsts, added_cond_kwargs, batch_size, hidden_dtype)
        rows = [m.view(grid.batch, grid.frames(), m.shape[-1]) for m in (made[0], made[1])]
        return Frames(grid, rows[0]), Frames(grid, rows[1])


class Transformer(LTX2VideoTransformer3DModel):
    """Diffusers' LTX2 transformer; inside `per_frame`, its video timestep embeds per frame (`FrameEmbedding`)."""

    def forward(self, *args: object, **kwargs: object) -> object:
        token = _CALL.set(_grid(self, kwargs) if _PER_FRAME.get() else None)
        try:
            return super().forward(*args, **kwargs)
        finally:
            _CALL.reset(token)


# Diffusers' context-parallel split finds a positional input by the forward's parameter names.
vars(Transformer.forward)["__signature__"] = inspect.signature(LTX2VideoTransformer3DModel.forward)


def install(model: LTX2VideoTransformer3DModel) -> None:
    model.__class__ = Transformer
    model.time_embed.__class__ = FrameEmbedding


def _grid(model: LTX2VideoTransformer3DModel, kwargs: Mapping[str, object]) -> Grid | None:
    """The call's video tokens by frame, from its keyword inputs; None where they do not say."""
    timestep, hidden = kwargs.get("timestep"), kwargs.get("hidden_states")
    frames, height, width = (kwargs.get(name) for name in ("num_frames", "height", "width"))
    if not (
        isinstance(timestep, Tensor)
        and isinstance(hidden, Tensor)
        and timestep.dim() == 2
        and isinstance(frames, int)
        and isinstance(height, int)
        and isinstance(width, int)
    ):
        return None
    patch, patch_t = int(model.config["patch_size"]), int(model.config["patch_size_t"])
    size = len(range(0, height, patch)) * len(range(0, width, patch))
    whole = len(range(0, frames, patch_t)) * size
    batch, tokens = timestep.shape
    start = _start(model, whole, tokens)
    if start is None or size < 1 or tuple(hidden.shape[:2]) != (batch, tokens):
        return None
    return Grid.of(batch, tokens, start, size)


def _start(model: LTX2VideoTransformer3DModel, whole: int, tokens: int) -> int | None:
    """This rank's first video token: 0 unsplit; split, Diffusers' context parallel's (`tensor_split` over its
    flattened mesh, the first `whole % ranks` shards one longer; `chunk` when even, the same)."""
    if tokens == whole:
        return 0
    parallel = getattr(model, "_parallel_config", None)
    config = getattr(parallel, "context_parallel_config", None)
    mesh = getattr(config, "_flattened_mesh", None)
    if mesh is None:
        return None
    ranks, rank = int(mesh.size()), int(torch.distributed.get_rank(mesh.get_group()))
    share, extra = divmod(whole, ranks)
    if tokens != share + (rank < extra):
        return None
    return rank * share + min(rank, extra)
