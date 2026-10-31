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

from typing import Tuple, Union
import torch
from cozy_runtime import fusion
from cozy_runtime.author import Cut, split
from einops import rearrange

# from ..cache import Cache
from seedvr2.upstream.common.cache import Cache
from rotary_embedding_torch import apply_rotary_emb
from .. import na
from ..attention import WindowAttention
from ..blocks.mmdit_window_block import MMWindowAttention, MMWindowTransformerBlock
from ..mm import MMArg
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..rope import NaRotaryEmbedding3d
from ..window import get_window_op


class NaSwinAttention(MMWindowAttention):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            window=window,
            window_method=window_method,
            shared_qkv=shared_qkv,
        )
        self.rope = NaRotaryEmbedding3d(dim=head_dim // 2) if qk_rope else None
        self.attn = WindowAttention(heads, head_dim)
        self.window_op = get_window_op(window_method)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """[Override] Upstream's window attention a group of whole windows at a time (at most
        `GROUP` video tokens, from the shape alone): each window attends to its own tokens and the
        text, so a group's projections, norms, rotary and attention are the whole call's for those
        rows, and the text output is the same mean over every window. No tensor spans all video
        tokens but the input and the output.

        The layout orders the windows by length (cached for the forward), so a group is runs of
        equal windows: a run is one batch of `[window rows | text rows]`, built by one concat each
        for q, k and v, with no index plan, gather or host sync per group."""
        txt_qkv = rearrange(self.proj_qkv.txt(txt), "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        txt_q, txt_k = self.norm_q.txt(txt_q), self.norm_k.txt(txt_k)
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        def layout():
            rows = torch.arange(int(vid_shape.prod(-1).sum()), device=vid_shape.device)[:, None]
            rows, shapes, count = na.window(rows, vid_shape, make_window)
            lengths = shapes.prod(-1).tolist()
            order = sorted(range(len(lengths)), key=lengths.__getitem__)
            spans = rows[:, 0].split(lengths)
            return torch.cat([spans[w] for w in order]), shapes[order], [lengths[w] for w in order], int(count.sum())

        rows, window_shape, lengths, windows = cache_win("win_rows_by_length", layout)
        freqs = cache_win("rope_freqs_3d_by_length", lambda: self.rope.get_freqs(window_shape)) if self.rope else None
        head = QK_ROPE.get(self.norm_q.vid.weight.dtype) if freqs is not None else None
        tables = cache_win("rope_tables", lambda: head.tables((freqs.cos(), freqs.sin()))) if head else None
        txt_rows = txt_q.shape[0]
        out = None
        txt_sum = None
        first = 0
        for group in _groups(lengths, GROUP):
            last = first + sum(lengths[w] for w in group)
            at = rows[first:last]
            qkv = rearrange(self.proj_qkv.vid(vid.index_select(0, at)), "l (o h d) -> l o h d", o=3, d=self.head_dim)
            q, k, v = qkv.unbind(1)
            fused = None
            if tables is not None:
                part = (tables[0][first:last], tables[1][first:last], *tables[2:])
                fused = [head.apply(n, x, part) for n, x in ((self.norm_q.vid, q), (self.norm_k.vid, k))]
            if fused is not None and None not in fused:
                q, k = fused
            else:
                q, k = self.norm_q.vid(q), self.norm_k.vid(k)
                if freqs is not None:
                    q, k = (_rotary(freqs[first:last], x) for x in (q, k))
            o_vid, start = [], 0
            for count, length in _runs(lengths[w] for w in group):
                rows_ = slice(start, start + count * length)
                batch = [
                    torch.cat([x[rows_].view(count, length, *x.shape[1:]), t.expand(count, *t.shape)], dim=1)
                    for x, t in ((q, txt_q), (k, txt_k), (v, txt_v))
                ]
                o = self.attn(*(x.flatten(0, 1) for x in batch), [(0, count, length + txt_rows)])
                o = o.unflatten(0, (count, length + txt_rows)).type_as(q)
                o_vid.append(o[:, :length].flatten(0, 1))
                o_txt = o[:, length:].float().sum(0)
                txt_sum = o_txt if txt_sum is None else txt_sum + o_txt
                start += count * length
            o_vid = self.proj_out.vid(rearrange(torch.cat(o_vid) if len(o_vid) > 1 else o_vid[0], "l h d -> l (h d)"))
            out = o_vid.new_empty((vid.shape[0], o_vid.shape[1])) if out is None else out
            out.index_copy_(0, at, o_vid)
            first = last
        assert out is not None and txt_sum is not None
        txt_out = (txt_sum / windows).type_as(txt_q)
        return out, self.proj_out.txt(rearrange(txt_out, "l h d -> l (h d)"))


#: The video q/k per-head RMSNorm + RoPE in one Runtime kernel (`fusion.HeadNormRope`), by the norm
#: weights' dtype: the eager path normalizes and rotates in F32 and rounds once; 60 of 128 channels
#: rotate in interleaved pairs (three axes of 10 frequencies).
QK_ROPE = {
    dtype: fusion.HeadNormRope(
        f"seedvr2_qk_{name}", head_dim=128, weights=name, rms_rounding="once",
        rope=fusion.Rope("freqs", interleaved=True, rotary=60, rounding="fp32"),
    )
    for dtype, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32"))
}  # fmt: skip

#: Video tokens a window group holds at most (whole windows; a longer window is a group alone).
GROUP = 16384


def _groups(lengths, most):
    """Consecutive windows, as few groups as hold at most `most` tokens each."""
    group, size = [], 0
    for w, n in enumerate(lengths):
        if group and size + n > most:
            yield group
            group, size = [], 0
        group.append(w)
        size += n
    if group:
        yield group


def _runs(lengths):
    """Consecutive equal lengths as (count, length)."""
    runs = []
    for n in lengths:
        if runs and runs[-1][1] == n:
            runs[-1][0] += 1
        else:
            runs.append([1, n])
    return runs


def _rotary(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """`NaRotaryEmbedding3d`'s rotation of (L, h, d) rows by their frequencies, in F32."""
    return rearrange(apply_rotary_emb(freqs, rearrange(x, "L h d -> h L d").float()), "h L d -> L h d").to(x.dtype)


class NaMMSRTransformerBlock(MMWindowTransformerBlock):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        shared_qkv: bool,
        shared_mlp: bool,
        mlp_type: str,
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            emb_dim=emb_dim,
            heads=heads,
            head_dim=head_dim,
            expand_ratio=expand_ratio,
            norm=norm,
            norm_eps=norm_eps,
            ada=ada,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            shared_qkv=shared_qkv,
            shared_mlp=shared_mlp,
            mlp_type=mlp_type,
            **kwargs,
        )

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            shared_qkv=shared_qkv,
            **kwargs,
        )

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        hid_len = MMArg(
            cache("vid_len", lambda: vid_shape.prod(-1)),
            cache("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        # [Override] The video MLP in pieces of `GROUP` tokens: exact, token-local.
        vid_mlp = split(self.mlp.vid, vid_mlp, Cut(0), most=GROUP * vid_mlp.shape[-1])
        txt_mlp = self.mlp.txt(txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp, vid_shape, txt_shape
