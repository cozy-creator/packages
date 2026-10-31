# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""Window attention as one Runtime attention site: the Runtime picks its kernel."""

from __future__ import annotations

import torch
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn

#: (first row, windows, rows per window): consecutive windows of one length.
Runs = list[tuple[int, int, int]]


class WindowAttnProcessor:
    """Each run of equal-length windows is one batched call of the selected kernel."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self, attn: WindowAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, runs: Runs
    ) -> torch.Tensor:
        if len(runs) == 1 and runs[0][0] == 0 and runs[0][1] * runs[0][2] == q.shape[0]:
            _, count, length = runs[0]
            return dispatch_attention_fn(
                *(x.view(count, length, *x.shape[1:]) for x in (q, k, v)),
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            ).flatten(0, 1)
        out = torch.empty_like(q)
        for start, count, length in runs:
            rows = slice(start, start + count * length)
            out[rows] = dispatch_attention_fn(
                *(x[rows].view(count, length, *x.shape[1:]) for x in (q, k, v)),
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            ).flatten(0, 1)
        return out


class WindowAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WindowAttnProcessor
    _available_processors = [WindowAttnProcessor]
    #: Q, K and V reach the kernel in bfloat16, whatever the projections' lane.
    activation_dtype = torch.bfloat16

    def __init__(self, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.set_processor(WindowAttnProcessor())

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, runs: Runs) -> torch.Tensor:
        """(rows, heads, head_dim) windows, laid out as `runs` says, to their outputs."""
        dtype = self.activation_dtype
        return self.processor(self, q.to(dtype), k.to(dtype), v.to(dtype), runs)
