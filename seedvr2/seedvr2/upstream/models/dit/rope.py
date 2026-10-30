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

from functools import lru_cache
from math import pi
from typing import Tuple
import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn

from seedvr2.upstream.common.cache import Cache


class F32Rotary(RotaryEmbedding):
    """[Override] F32 rotary frequencies, made where they are first used and held off the
    module's tensors (a cast never rounds them; the parameters they derive from are meta at
    construction). They are rotary_embedding_torch's `pixel` ones, `linspace(1, max_freq / 2) * pi`."""

    @property
    def freqs(self):
        device = self.dummy.device
        if self.f32 is None or self.f32.device != device:
            count, top = self.f32_spec
            self.f32 = torch.linspace(1.0, top / 2, count, device=device, dtype=torch.float32) * pi
        return self.f32


class RotaryEmbeddingBase(nn.Module):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__()
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="pixel",
            max_freq=256,
        )
        # 1. Set model.requires_grad_(True) after model creation will make
        #    the `requires_grad=False` for rope freqs no longer hold.
        # 2. Even if we don't set requires_grad_(True) explicitly,
        #    FSDP is not memory efficient when handling fsdp_wrap
        #    with mixed requires_grad=True/False.
        # With above consideration, it is easier just remove the freqs
        # out of nn.Parameters when `learned_freq=False`
        freqs = self.rope.freqs
        del self.rope.freqs
        # [Override] F32 and derived, never the checkpoint's `freqs`: the phases reach 128*pi and
        # rotary_embedding_torch computes them in the frequencies' dtype (the lane's fp16 rounded
        # the high bands by up to 0.125 rad; upstream's checkpoint is F32).
        self.rope.__class__ = F32Rotary
        self.rope.f32, self.rope.f32_spec = None, (freqs.shape[0], 256)

    @lru_cache(maxsize=128)
    def get_axial_freqs(self, *dims):
        return self.rope.get_axial_freqs(*dims)


class RotaryEmbedding3d(RotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)

    def forward(
        self,
        q: torch.FloatTensor,  # b h l d
        k: torch.FloatTensor,  # b h l d
        size: Tuple[int, int, int],
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        T, H, W = size
        freqs = self.get_axial_freqs(T, H, W)
        q = rearrange(q, "b h (T H W) d -> b h T H W d", T=T, H=H, W=W)
        k = rearrange(k, "b h (T H W) d -> b h T H W d", T=T, H=H, W=W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        q = rearrange(q, "b h T H W d -> b h (T H W) d")
        k = rearrange(k, "b h T H W d -> b h (T H W) d")
        return q, k


class NaRotaryEmbedding3d(RotaryEmbedding3d):
    def forward(
        self,
        q: torch.FloatTensor,  # L h d
        k: torch.FloatTensor,  # L h d
        shape: torch.LongTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        freqs = cache("rope_freqs_3d", lambda: self.get_freqs(shape))
        q = rearrange(q, "L h d -> h L d")
        k = rearrange(k, "L h d -> h L d")
        q = apply_rotary_emb(freqs, q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs, k.float()).to(k.dtype)
        q = rearrange(q, "h L d -> L h d")
        k = rearrange(k, "h L d -> L h d")
        return q, k

    def get_freqs(
        self,
        shape: torch.LongTensor,
    ) -> torch.Tensor:
        freq_list = []
        for f, h, w in shape.tolist():
            freqs = self.get_axial_freqs(f, h, w)
            freq_list.append(freqs.view(-1, freqs.size(-1)))
        return torch.cat(freq_list, dim=0)
