"""Whether one H3 attention call's working set fits the card now, and returning Torch's cache
where it does not: the one memory judgement the model makes, until the Runtime answers it."""

from __future__ import annotations

import functools
import math
from typing import Any

import torch
from cozy_runtime.author.kernels import PreQuantized

#: What a card holds beside one attention call: the CUDA context and the weights' streaming
#: window (two int8 DiT blocks, 0.8 GB). An approximation; the cut sizes itself on the device.
_BESIDE = 3 << 29


@functools.cache
def _capacity(index: int) -> int:
    return int(torch.cuda.mem_get_info(index)[1])


#: Query heads per window of a dense call beside a DiT that streams, coarsest first: at 15 s the
#: attention kernel runs a 14-head window as fast as the whole call (RTX 4090).
_WINDOWS = (28, 14)


def _room(hidden: Any) -> int | None:
    """The device's bytes beyond what is allocated now and `_BESIDE`; None off CUDA."""
    quantized = isinstance(hidden, PreQuantized)
    device = hidden.payload.device if quantized else hidden.device
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _capacity(index) - torch.cuda.memory_allocated(index) - _BESIDE


def _need(hidden: Any, attn: Any, heads: int) -> int:
    """One call's working set over `heads` query heads: Q, K and V, the RoPE copies of two and
    the output (bf16), and the input's int8 rows."""
    tokens, width = math.prod(hidden.shape[:-1]), int(hidden.shape[-1])
    return 6 * tokens * heads * attn.head_dim * 2 + tokens * width


def fits(hidden: Any, attn: Any) -> bool:
    """Whether one attention call's whole working set fits the device beside what is allocated
    now. Off CUDA, always."""
    room = _room(hidden)
    return room is None or _need(hidden, attn, attn.heads) <= room


def window_heads(hidden: Any, attn: Any, weights: int) -> int:
    """Query heads per window of a dense call that `fits`: all of them where the whole call also
    fits beside the DiT's `weights`, else the coarsest of `_WINDOWS` that does, else the finest.
    A whole call beside a DiT that streams takes the room its weights would keep resident: H3
    15 s on an RTX 4090 peaked at 11.3 GB and kept 9.4 of the DiT's 21.1 GB, reading 11.8 GB a
    step; in 14-head windows it peaks at 8.2 GB and keeps 14.2 GB, reading 6.9 GB."""
    room = _room(hidden)
    if room is None or _need(hidden, attn, attn.heads) <= room - weights:
        return int(attn.heads)
    return next((h for h in _WINDOWS if _need(hidden, attn, h) <= room - weights), _WINDOWS[-1])


def release_if_tight(hidden: Any, attn: Any) -> None:
    """Return Torch's free cached segments where the call does not fit: an odd-sized block left
    in the cache is split for the next differently sized allocation, and on an 8 GB card the
    holes it leaves then refuse a 1.09 GiB row set (RTX 3070, 15 s)."""
    if not fits(hidden, attn):
        release()


def release() -> None:
    """`torch.cuda.empty_cache()`, except while a native loop records the step: its pool is
    sized at recording, and Torch would free other graphs' pools inside the capture."""
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.empty_cache()
