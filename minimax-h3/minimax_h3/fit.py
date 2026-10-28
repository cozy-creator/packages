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


def fits(hidden: Any, attn: Any) -> bool:
    """Whether one attention call's whole working set fits the device beside what is allocated
    now: Q, K and V, the RoPE copies of two and the output (bf16), and the input's int8 rows.
    Off CUDA, always."""
    quantized = isinstance(hidden, PreQuantized)
    device = hidden.payload.device if quantized else hidden.device
    if device.type != "cuda":
        return True
    tokens, width = math.prod(hidden.shape[:-1]), int(hidden.shape[-1])
    need = 6 * tokens * attn.heads * attn.head_dim * 2 + tokens * width
    index = device.index if device.index is not None else torch.cuda.current_device()
    return need <= _capacity(index) - torch.cuda.memory_allocated(index) - _BESIDE


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
