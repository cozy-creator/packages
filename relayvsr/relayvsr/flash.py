"""RelayVSR's Flash decoder (`refvsr_inference/models/flash`, variant S, endpoint conditions): every
low-resolution frame becomes its 4x frame, bicubic plus a predicted residual, attending locally to
the two keyframe latents that bound its window and to the previous frame. Inference only; the
module and parameter names are the checkpoint's."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from . import kernels

#: `flash.safetensors`' `config` (format `refvsr-flash-v1`, variant S).
DIM, DEPTH, HEADS, KV_HEADS, ENCODER, MLP, HEAD_WIDTH, LATENT = 512, 10, 8, 2, (64, 128, 128), 8 / 3, 1024, 48
#: Each key plane's window (rows, columns): the start and end keyframes', then the video frames'.
CONDITION_WINDOW, VIDEO_WINDOW = (48, 48), (24, 24)
#: Frames of video history a frame sees, itself included (`window_size`).
HISTORY = 2
#: Temporal, row and column RoPE channels of a 64-channel head.
ROPE = (16, 24, 24)
SCALE = 4


class Attention(nn.Module):
    """Grouped-query attention over the two keyframes' planes and the video planes."""

    def __init__(self, dim: int, heads: int, kv_heads: int) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, dim // heads
        kv = kv_heads * self.head_dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, kv)
        self.value = nn.Linear(dim, kv)
        self.condition_key = nn.Linear(dim, kv)
        self.condition_value = nn.Linear(dim, kv)
        self.key_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.value_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.output = nn.Linear(dim, dim)

    def project(self, x: torch.Tensor, condition: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query (b, heads, n, d) and normalized, unrotated key and value (b, kv_heads, n, d)."""
        b, n, _ = x.shape
        key, value = (self.condition_key, self.condition_value) if condition else (self.key, self.value)
        q = self.query(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.key_norm(key(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2))
        v = self.value_norm(value(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2))
        return q, k, v


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        y: torch.Tensor = self.down(F.silu(gate) * up)
        return y


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, hidden: int) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim, eps=1e-5)
        self.attention = Attention(dim, heads, kv_heads)
        self.feed_forward_norm = nn.RMSNorm(dim, eps=1e-5)
        self.feed_forward = SwiGLU(dim, hidden)


class Backbone(nn.Module):
    def __init__(self, depth: int, dim: int, heads: int, kv_heads: int, hidden: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, kv_heads, hidden) for _ in range(depth)])

    def layers(self) -> list[Block]:
        return cast(list[Block], list(self.blocks))


class Encoder(nn.Module):
    """A frame's tokens at 1/4 resolution: three 3x3 convolutions (strides 2, 2, 1), a 1x1, RMSNorm."""

    def __init__(self, dim: int, channels: tuple[int, int, int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = 3
        for out, stride in zip(channels, (2, 2, 1), strict=True):
            layers += [nn.Conv2d(width, out, kernel_size=3, stride=stride, padding=1), nn.SiLU()]
            width = out
        layers.append(nn.Conv2d(width, dim, kernel_size=1))
        self.layers = nn.Sequential(*layers)
        self.norm = nn.RMSNorm(dim, eps=1e-5)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.norm(self.layers(frame).flatten(2).transpose(1, 2))
        return y


class ConditionEncoder(nn.Module):
    def __init__(self, dim: int, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, dim, kernel_size=1)
        self.norm = nn.RMSNorm(dim, eps=1e-5)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.norm(self.projection(latent).flatten(2).transpose(1, 2))
        return y


def rope(frames: list[int], height: int, width: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The 3-D RoPE tables (tokens, 64) of `frames`' every row and column, in `dtype`."""
    t, y, x = torch.meshgrid(
        torch.tensor(frames, device=device), torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    cos, sin = [], []
    for positions, span in zip((t.flatten(), y.flatten(), x.flatten()), ROPE, strict=True):
        inverse = 1.0 / (10000.0 ** (torch.arange(0, span, 2, dtype=torch.float32, device=device) / span))
        angle = positions.float().unsqueeze(-1) * inverse
        cos.append(torch.cos(angle).repeat_interleave(2, dim=-1))
        sin.append(torch.sin(angle).repeat_interleave(2, dim=-1))
    return torch.cat(cos, dim=-1).to(dtype), torch.cat(sin, dim=-1).to(dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos[..., 0::2] - odd * sin[..., 0::2]
    out[..., 1::2] = even * sin[..., 1::2] + odd * cos[..., 1::2]
    return out


@dataclass(frozen=True)
class Condition:
    """One keyframe's normalized, unrotated K/V per block, (b, kv_heads, tokens, d)."""

    kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]


#: The previous frame's K/V per block, empty at a window's first frame.
type History = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class FlashVSR(nn.Module):
    """Component `flash`: 31.9M parameters."""

    def __init__(self, dim: int = DIM, depth: int = DEPTH, heads: int = HEADS, kv_heads: int = KV_HEADS) -> None:
        super().__init__()
        hidden = math.ceil(dim * MLP / 64) * 64
        shuffle = 4 * SCALE
        self.lr_encoder = Encoder(dim, ENCODER)
        self.condition_encoder = ConditionEncoder(dim, LATENT)
        self.video_type_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.condition_type_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.backbone = Backbone(depth, dim, heads, kv_heads, hidden)
        self.spatial_head = nn.Sequential(nn.Linear(dim, HEAD_WIDTH), nn.SiLU())
        self.residual_projection = nn.Linear(HEAD_WIDTH, 3 * shuffle * shuffle)
        #: Training's direct-RGB head; inference never reads it.
        self.condition_rgb_projection = nn.Linear(HEAD_WIDTH, 3 * shuffle * shuffle)

    def condition(self, latent: torch.Tensor) -> Condition:
        """A keyframe latent (b, 48, h, w) primed through every block, each attending within the
        latent plane: the K/V a window's frames attend to."""
        _, _, h, w = latent.shape
        x = self.condition_encoder(latent) + self.condition_type_embedding
        cos, sin = rope([0], h, w, x.dtype, x.device)
        kv = []
        for block in self.backbone.layers():
            attn = block.attention
            q, k, v = attn.project(block.attention_norm(x), condition=True)
            kv.append((k, v))
            y = kernels.local_attention(rotate(q, cos, sin), rotate(k, cos, sin), v, h, w, (CONDITION_WINDOW,))
            x = x + attn.output(y.transpose(1, 2).reshape(x.shape))
            x = x + block.feed_forward(block.feed_forward_norm(x))
        return Condition(tuple(kv))

    def decode(self, frame: torch.Tensor, start: Condition, end: Condition, history: History) -> tuple[torch.Tensor, History]:
        """One frame (b, 3, h, w) in [-1, 1], h and w multiples of 4, to its 4x frame, and the
        history the next frame of its window attends to."""
        _, _, h, w = frame.shape
        th, tw = h // 4, w // 4
        x = self.lr_encoder(frame) + self.video_type_embedding
        frames = 1 if history else 0
        q_cos, q_sin = rope([0], th, tw, x.dtype, x.device)
        k_cos, k_sin = (
            torch.cat(parts, dim=0)
            for parts in zip(rope([0, 0], th, tw, x.dtype, x.device), rope(list(range(-frames, 1)), th, tw, x.dtype, x.device), strict=True)
        )
        windows = (CONDITION_WINDOW, CONDITION_WINDOW) + (VIDEO_WINDOW,) * (frames + 1)
        made = []
        for index, block in enumerate(self.backbone.layers()):
            attn = block.attention
            q, k, v = attn.project(block.attention_norm(x))
            if history:
                k, v = torch.cat((history[index][0], k), dim=2), torch.cat((history[index][1], v), dim=2)
            made.append((k[:, :, -th * tw :], v[:, :, -th * tw :]))
            (sk, sv), (ek, ev) = start.kv[index], end.kv[index]
            keys = rotate(torch.cat((sk, ek, k), dim=2), k_cos, k_sin)
            y = kernels.local_attention(rotate(q, q_cos, q_sin), keys, torch.cat((sv, ev, v), dim=2), th, tw, windows)
            x = x + attn.output(y.transpose(1, 2).reshape(x.shape))
            x = x + block.feed_forward(block.feed_forward_norm(x))
        pixels = self.residual_projection(self.spatial_head(x)).transpose(1, 2).reshape(x.shape[0], -1, th, tw)
        bicubic = F.interpolate(frame, scale_factor=SCALE, mode="bicubic", align_corners=False)
        return bicubic + F.pixel_shuffle(pixels, 4 * SCALE), tuple(made)
