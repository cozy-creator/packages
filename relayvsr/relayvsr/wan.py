"""RelayVSR's one-step Wan2.2-TI2V-5B (`refvsr_inference/models/wan`): from a 4x bilinear frame, one
keyframe's 48-channel latent, attending to the last few keyframes through a ring of unrotated K/V.
Inference only, the streaming path; the module and parameter names are the checkpoint's."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from cozy_runtime.author import State, Store, retained
from torch import nn

from . import kernels

#: `wan.safetensors`' `config` (format `refvsr-wan-v1`).
DIM, HEADS, HEAD_DIM, LAYERS, FFN, TEXT, FREQ, CHANNELS = 3072, 24, 128, 30, 14336, 4096, 256, 48
PATCH = (1, 2, 2)
#: Tokens a query sees around itself (rows, columns) and the keyframes it sees (itself included).
WINDOW, CONTEXT = (22, 40), 4
#: RoPE's temporal origin trails the newest keyframe by this many (`temporal_rope.training_frames`).
ROPE_FRAMES, ROPE_LENGTH = 8, 1024
#: The one step: t=1000 (sigma 1), so the latent is noise minus the predicted velocity.
TIMESTEP = 1000.0


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)


class Linears(nn.Module):
    """Diffusers' `TimestepEmbedding` (SiLU) or `PixArtAlphaTextProjection` (tanh GELU)."""

    def __init__(self, width: int, dim: int, act: nn.Module) -> None:
        super().__init__()
        self.linear_1, self.act, self.linear_2 = nn.Linear(width, dim), act, nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.linear_2(self.act(self.linear_1(x)))
        return y


class ConditionEmbedder(nn.Module):
    def __init__(self, dim: int, freq: int, text: int) -> None:
        super().__init__()
        self.freq = freq
        self.time_embedder = Linears(freq, dim, nn.SiLU())
        self.time_proj = nn.Linear(dim, 6 * dim)
        self.text_embedder = Linears(text, dim, nn.GELU(approximate="tanh"))

    def forward(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`temb` (1, 1, dim), `timestep_proj` (1, 1, 6, dim) and the projected text, at t=1000."""
        half = self.freq // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=prompt.device) / half
        angle = torch.tensor([TIMESTEP], device=prompt.device)[:, None] * torch.exp(exponent)[None]
        steps = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)[None]  # flip_sin_to_cos
        temb = self.time_embedder(steps.to(self.time_embedder.linear_1.weight.dtype)).type_as(prompt)
        projected = self.time_proj(F.silu(temb)).unflatten(2, (6, -1))
        return temb, projected, self.text_embedder(prompt)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])
        self.norm_q = nn.RMSNorm(dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(dim, eps=1e-6)

    def keys(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized, unrotated keys and values, (batch, tokens, heads, head_dim)."""
        return self.norm_k(self.to_k(x)).unflatten(2, (self.heads, -1)), self.to_v(x).unflatten(2, (self.heads, -1))

    def query(self, x: torch.Tensor) -> torch.Tensor:
        q: torch.Tensor = self.norm_q(self.to_q(x)).unflatten(2, (self.heads, -1))
        return q

    def out(self, x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.to_out[0](x.flatten(2, 3).type_as(like))
        return y


class GELU(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner)


class FeedForward(nn.Module):
    """Diffusers' `FeedForward` with a tanh GELU: `net.0.proj`, then `net.2`."""

    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([GELU(dim, inner), nn.Identity(), nn.Linear(inner, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(cast(GELU, self.net[0]).proj(x), approximate="tanh")
        y: torch.Tensor = self.net[2](h)
        return y


class Block(nn.Module):
    def __init__(self, dim: int, ffn: int, heads: int) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, 1e-6, elementwise_affine=False)
        self.attn1 = Attention(dim, heads)
        self.attn2 = Attention(dim, heads)
        self.norm2 = FP32LayerNorm(dim, 1e-6, elementwise_affine=True)
        self.ffn = FeedForward(dim, ffn)
        self.norm3 = FP32LayerNorm(dim, 1e-6, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, dim))


class RMSNorm2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(dim=1, keepdim=True) + 1e-6) * self.weight).to(x.dtype)


class LQProjection(nn.Module):
    """The 4x frame's tokens, added to the noise's: a 16x pixel unshuffle, a 1x1 and a strided 3x3
    convolution, then a linear onto the DiT's width."""

    def __init__(self, dim: int, hidden1: int, hidden2: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3 * 16 * 16, hidden1, kernel_size=1)
        self.norm1 = RMSNorm2d(hidden1)
        self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=3, stride=2, padding=1)
        self.norm2 = RMSNorm2d(hidden2)
        self.out_proj = nn.Linear(hidden2, dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """(b, 3, h, w) -> (b, h/32 * w/32, dim)."""
        x = F.pixel_unshuffle(frame, 16)
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        y: torch.Tensor = self.out_proj(x.flatten(2).transpose(1, 2))
        return y


def rope_tables(height: int, width: int, frames: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Wan's 3-D RoPE (44 temporal, 42 row and 42 column channels, each pair repeated) at `frames`,
    every row and column: F32 (1, len(frames) * height * width, 1, 128), as Diffusers' tables."""
    spans = (HEAD_DIM - 4 * (HEAD_DIM // 6), 2 * (HEAD_DIM // 6), 2 * (HEAD_DIM // 6))
    cos: list[torch.Tensor] = []
    sin: list[torch.Tensor] = []
    for span, positions in zip(spans, (frames, range(height), range(width)), strict=True):
        inverse = 1.0 / (10000.0 ** (torch.arange(0, span, 2, dtype=torch.float64) / span))
        angle = torch.outer(torch.tensor(list(positions), dtype=torch.float64), inverse)
        cos.append(angle.cos().repeat_interleave(2, dim=1).float())
        sin.append(angle.sin().repeat_interleave(2, dim=1).float())
    count, out = len(frames), []
    for table in (cos, sin):
        t, h, w = table
        grid = torch.cat(
            [
                t.view(count, 1, 1, -1).expand(count, height, width, -1),
                h.view(1, height, 1, -1).expand(count, height, width, -1),
                w.view(1, 1, width, -1).expand(count, height, width, -1),
            ],
            dim=-1,
        )
        out.append(grid.reshape(1, count * height * width, 1, -1).to(device))
    return out[0], out[1]


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Diffusers' Wan rotary: adjacent channel pairs, in F32, rounded to `x`'s dtype."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    c, s = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


@dataclass
class Ring:
    """The last `CONTEXT` keyframes' K/V of every block, held by the Runtime's memory (`retained`),
    in the slots upstream's ring holds them: keyframe `f` sits in slot `f % CONTEXT`."""

    #: per keyframe: its store and each block's (k, v), (batch, tokens, heads, head_dim)
    frames: dict[int, tuple[Store, list[State]]] = field(default_factory=dict)
    count: int = 0

    def slots(self) -> list[int]:
        """The keyframes held, in slot order, with the one being made."""
        f = self.count
        return [f - ((f - slot) % CONTEXT) for slot in range(min(f + 1, CONTEXT))]

    def kv(self, frame: int, block: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.frames[frame][1][block].get()
        return k, v

    def keep(self, kv: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """The new keyframe's K/V; the one leaving the window goes."""
        f = self.count
        store = retained(sum(k.nbytes + v.nbytes for k, v in kv))
        self.frames[f] = (store, [store.put(k, v) for k, v in kv])
        for old in [s for s in self.frames if s <= f - CONTEXT]:
            self.frames.pop(old)[0].close()
        self.count += 1

    def close(self) -> None:
        for store, _ in self.frames.values():
            store.close()
        self.frames.clear()


class WanVSR(nn.Module):
    """Component `wan`: Wan2.2-TI2V-5B with RelayVSR's LQ projection and its prompt."""

    prompt_embeds: torch.Tensor

    def __init__(
        self,
        dim: int = DIM,
        heads: int = HEADS,
        layers: int = LAYERS,
        ffn: int = FFN,
        text: int = TEXT,
        prompt: int = 512,
        lq: tuple[int, int] = (768, 3072),
    ) -> None:
        super().__init__()
        self.heads = heads
        self.patch_embedding = nn.Conv3d(CHANNELS, dim, kernel_size=PATCH, stride=PATCH)
        self.condition_embedder = ConditionEmbedder(dim, FREQ, text)
        self.blocks = nn.ModuleList([Block(dim, ffn, heads) for _ in range(layers)])
        self.norm_out = FP32LayerNorm(dim, 1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, CHANNELS * math.prod(PATCH))
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, dim))
        self.lq_proj = LQProjection(dim, *lq)
        self.register_buffer("prompt_embeds", torch.empty(1, prompt, text))

    def keyframe(self, upsampled: torch.Tensor, noise: torch.Tensor, ring: Ring) -> torch.Tensor:
        """One keyframe's latent (b, 48, h/16, w/16) from its 4x frame (b, 3, h, w) in [-1, 1] and
        F32 noise of the latent's shape; appends its K/V to `ring`, dropping the oldest."""
        b, _, lh, lw = noise.shape
        ph, pw = lh // PATCH[1], lw // PATCH[2]
        dtype = upsampled.dtype
        temb, projected, text = self.condition_embedder(self.prompt_embeds.to(dtype))
        x = self.patch_embedding(noise.to(dtype)[:, :, None]).flatten(2).transpose(1, 2)
        x = x + self.lq_proj(upsampled)

        f = ring.count
        slots = ring.slots()
        origin = max(0, f - ROPE_FRAMES + 1)
        q_rope = rope_tables(ph, pw, [f - origin], x.device)
        k_rope = rope_tables(ph, pw, [s - origin for s in slots], x.device)
        window = (min(WINDOW[0], ph), min(WINDOW[1], pw))
        local = window != (ph, pw)
        made: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(cast(list[Block], list(self.blocks))):
            shift, scale, gate, c_shift, c_scale, c_gate = (block.scale_shift_table[None] + projected.float()).unbind(2)
            h = (block.norm1(x.float()) * (1 + scale) + shift).type_as(x)
            attn = block.attn1
            key, value = attn.keys(h)
            made.append((key, value))
            held = [(key, value) if s == f else ring.kv(s, index) for s in slots]
            keys, values = [kv[0] for kv in held], [kv[1] for kv in held]
            q = rotate(attn.query(h), *q_rope).to(value.dtype)
            k = rotate(torch.cat(keys, dim=1), *k_rope).to(value.dtype)
            v = torch.cat(values, dim=1)
            if local:
                y = kernels.window_attention(q, k, v, ph, pw, window)
            else:
                y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            x = (x.float() + attn.out(y, q) * gate).type_as(x)

            cross = block.attn2
            h = block.norm2(x.float()).type_as(x)
            ck, cv = cross.keys(text)
            q = cross.query(h)
            y = F.scaled_dot_product_attention(q.transpose(1, 2), ck.transpose(1, 2), cv.transpose(1, 2)).transpose(1, 2)
            x = x + cross.out(y, q)

            h = (block.norm3(x.float()) * (1 + c_scale) + c_shift).type_as(x)
            x = (x.float() + block.ffn(h).float() * c_gate).type_as(x)

        ring.keep(made)
        del made

        shift, scale = (self.scale_shift_table[None] + temb[:, :, None]).unbind(2)
        x = (self.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        x = self.proj_out(x).reshape(b, ph, pw, PATCH[1], PATCH[2], CHANNELS)
        velocity: torch.Tensor = x.permute(0, 5, 1, 3, 2, 4).reshape(b, CHANNELS, lh, lw)
        return (noise - velocity.float()).to(dtype)
