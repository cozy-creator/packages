"""MiniMax-H3-Image's VAE (Diffusers 0.43 `AutoencoderKLMiniMaxH3Image`) as two components: `vae_encoder`, H3's causal
3D CNN encoder over one frame (MiniMax-H3's: float16 and fused, `VAE_ENCODER`), and `vae_decoder`, a 24-layer 2D ViT.
A request without pictures never fetches the encoder. Both tile as Diffusers does (256 px tiles, at least 64 px of
overlap, linearly blended); the decoder runs its tiles as one batch, block by block, its rows cut as the device holds
them (the Runtime's `split`). The decoder is batch-independent, so each tile's arithmetic is Diffusers'.

The 2D decoder is a port of Diffusers' `MiniMaxH3ImageViTDecoder2d` (Apache-2.0, Copyright 2026 The MiniMax and
HuggingFace Teams); its blocks are Diffusers 0.41's `MiniMaxH3VideoTransformerBlock`, which 0.43's image block copies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch
from cozy_runtime.accel import fusion
from cozy_runtime.author import Cut, split
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoDownsample3d,
    MiniMaxH3VideoEncoder3d,
    MiniMaxH3VideoGroupNorm,
    MiniMaxH3VideoResnetBlock3d,
    MiniMaxH3VideoTransformerBlock,
)
from diffusers.models.modeling_utils import get_parameter_dtype
from torch import nn

#: ImageNet's normalization, which the VAE's pixels are in.
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TILE, OVERLAP = 256, 64


def split_tiles(length: int, ratio: int) -> tuple[list[int], list[int], list[int]]:
    """Diffusers' `_split_tiles`: `TILE`-wide tiles over `length` pixels, every overlap at least `OVERLAP`, the slack
    spread over the overlaps in whole `ratio` steps."""
    if TILE >= length:
        return [0], [length], []
    count = math.ceil(length / TILE)
    while TILE * count - OVERLAP * (count - 1) - length < 0:
        count += 1
    overlaps = [OVERLAP] * (count - 1)
    for i in range((TILE * count - sum(overlaps) - length) // ratio):
        overlaps[i % (count - 1)] += ratio
    starts = [0]
    for i in range(count - 1):
        starts.append(starts[-1] + TILE - overlaps[i])
    return starts, [TILE] * count, overlaps


def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a, weight_b = (1 - positions / extent).view(shape), (positions / extent).view(shape)
    blended = a.narrow(dim, a.shape[dim] - extent, extent) * weight_a + b.narrow(dim, 0, extent) * weight_b
    if extent == b.shape[dim]:
        return blended
    return torch.cat([blended, b.narrow(dim, extent, b.shape[dim] - extent)], dim=dim)


def stitch(tiles: Sequence[Sequence[torch.Tensor]], height_overlaps: list[int], width_overlaps: list[int]) -> torch.Tensor:
    """Diffusers' `_stitch_tiles`: each tile blended into the one above and the one before, overlaps cut."""
    result_rows = []
    for i, row in enumerate(tiles):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
            if j > 0:
                tile = _blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
            if i < len(tiles) - 1:
                tile = tile[..., : -height_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -width_overlaps[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def _causal_padding(conv: nn.Module) -> fusion.Padding:
    """A causal conv's padding: `spatial_padding` pixels each side, reflected as the release's `spatial_padding_mode`
    says, and `temporal_padding` zero frames before."""
    layer = cast(MiniMaxH3VideoCausalConv3d, conv)  # `VAE_ENCODER` declares this class alone
    side = int(layer.spatial_padding)
    fills: dict[str, Literal["reflect", "zeros"]] = {"reflect": "reflect", "constant": "zeros", "zeros": "zeros"}
    fill = fills.get(layer.spatial_padding_mode)
    if fill is None:
        raise ValueError(f"no fused {layer.spatial_padding_mode!r} padding")
    return (side, side, side, side), int(layer.temporal_padding), fill


def _downsample_pad(block: nn.Module) -> tuple[int, int, int, int]:
    """A downsampler's reflected right column and bottom row before its 2x spatial stride."""
    stride = cast(MiniMaxH3VideoDownsample3d, block).spatial_stride  # `VAE_ENCODER` declares it
    return (0, 1, 0, 1) if stride == 2 else (0, 0, 0, 0)


#: H3's causal-conv encoder as Runtime runs it (cr-306), minimax-h3's `VAE_ENCODER`: fp16 NDHWC, each GroupNorm, SiLU
#: and pad one pass, each conv accumulating in fp16 with its block's residual in its epilogue, as ComfyUI's under
#: `--fast fp16_accumulation`. It serves the encoder's f16 rows (`lanes.prepare`); f32 rows run eager.
VAE_ENCODER = fusion.CausalConvEncoder(
    fusion.CausalConv(MiniMaxH3VideoCausalConv3d, padding=_causal_padding),
    norm=MiniMaxH3VideoGroupNorm,
    blocks=(
        fusion.ConvResidual(
            MiniMaxH3VideoResnetBlock3d, steps=(("norm1", "conv1"), ("norm2", "conv2")), shortcut="conv_shortcut"
        ),
        fusion.PaddedConv(MiniMaxH3VideoDownsample3d, conv="conv", pad=_downsample_pad),
    ),
)


#: The encoder's device peak beside its moments, in tiles of them (`split`'s `holds`), before a pass is measured: a
#: 256 px tile's 128-channel float16 activations, about four held at once (64 MiB), over its 48x16x16 moments (24 KiB).
ENCODE_HOLDS = 2700


class ImageEncoder(nn.Module):
    """`encoder` and `quant_conv` of the VAE: pixels to the posterior's moments, in the rows' dtype."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        latent = int(config.get("latent_channels", 24))  # type: ignore[call-overload]  # a config int
        self.ratio = math.prod(config["spatial_downsample_factors"])  # type: ignore[arg-type]  # a config list
        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=config.get("in_channels", 3), out_channels=2 * latent,  # type: ignore[arg-type]  # config values
            block_out_channels=tuple(config["block_out_channels"]), layers_per_block=config["layers_per_block"],  # type: ignore[arg-type, call-overload]
            spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),  # type: ignore[call-overload]
            temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),  # type: ignore[call-overload]
            norm_num_groups=config["norm_num_groups"], norm_eps=config["norm_eps"],  # type: ignore[arg-type]
            spatial_padding_mode=config["spatial_padding_mode"],  # type: ignore[arg-type]
        )  # fmt: skip
        self.quant_conv = nn.Conv3d(2 * latent, 2 * latent, kernel_size=1)

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x[:, :, None]))[:, :, -1]

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalized `(1, 3, H, W)` pixels to `(1, 48, H/16, W/16)` moments, tiled as Diffusers' `_encode`."""
        pixels = pixels.to(get_parameter_dtype(self.encoder))
        ys, y_lengths, y_overlaps = split_tiles(pixels.shape[-2], self.ratio)
        xs, x_lengths, x_overlaps = split_tiles(pixels.shape[-1], self.ratio)
        # The tiles go through the encoder as ONE batch, cut as the device holds them (cr-323): one by one, a 256 px tile
        # is too little work for the card and its launches wait on the host. Every tile is the same size and the encoder
        # is batch-independent (its GroupNorm normalizes each tile alone), so each tile's moments are the per-tile loop's.
        tiles = [pixels[..., y : y + h, x : x + w] for y, h in zip(ys, y_lengths, strict=True)
                 for x, w in zip(xs, x_lengths, strict=True)]  # fmt: skip
        batch = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
        moments = split(self._clip, batch, Cut(0), holds=ENCODE_HOLDS).split(pixels.shape[0], dim=0)
        rows = [list(moments[row * len(xs) : (row + 1) * len(xs)]) for row in range(len(ys))]
        return stitch(rows, [o // self.ratio for o in y_overlaps], [o // self.ratio for o in x_overlaps])


class _Rope2d(nn.Module):
    """The decoder's rotary embedding over `(h, w)`: each axis's angles, concatenated, then duplicated."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float) -> None:
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / theta ** torch.arange(0, 1, 4 / dim, dtype=torch.float32), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class ViTDecoder2d(nn.Module):
    """Every latent pixel one token, `num_register_tokens` registers and one zero token appended at position 0,
    full self-attention; each token's head paints its 16x16 block."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        heads, head_dim = int(config["decoder_num_attention_heads"]), int(config["decoder_attention_head_dim"])  # type: ignore[call-overload]
        dim, eps = heads * head_dim, float(config["decoder_norm_eps"])  # type: ignore[arg-type]
        self.patch_size = math.prod(config["spatial_downsample_factors"])  # type: ignore[arg-type]
        self.out_channels = int(config.get("out_channels", 3))  # type: ignore[call-overload]
        self.num_register_tokens = int(config["decoder_num_register_tokens"])  # type: ignore[call-overload]
        self.rope = _Rope2d(int(head_dim * float(config["decoder_rope_dim_ratio"])), float(config["decoder_rope_theta"]))  # type: ignore[arg-type]
        self.proj_in = nn.Linear(int(config.get("latent_channels", 24)), dim)  # type: ignore[call-overload]
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList([
            MiniMaxH3VideoTransformerBlock(dim=dim, heads=heads, dim_head=head_dim, ffn_mult=int(config["decoder_ffn_mult"]), eps=eps)  # type: ignore[call-overload]
            for _ in range(int(config["decoder_num_layers"]))  # type: ignore[call-overload]
        ])  # fmt: skip
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = nn.Linear(dim, self.out_channels * self.patch_size**2)


class ImageDecoder(nn.Module):
    """`post_quant_conv` and the 2D ViT `decoder`: latents to pixels."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        latent = int(config.get("latent_channels", 24))  # type: ignore[call-overload]
        self.post_quant_conv = nn.Conv2d(latent, latent, kernel_size=1)
        self.decoder = ViTDecoder2d(config)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Denormalized `(1, 24, h, w)` latents to ImageNet-normalized `(1, 3, 16h, 16w)` pixels, tiled as Diffusers'
        `_decode` (tiles laid in pixels, mapped to the latent grid)."""
        z = z.to(get_parameter_dtype(self.post_quant_conv))
        ratio = self.decoder.patch_size
        ys, y_lengths, y_overlaps = split_tiles(z.shape[-2] * ratio, ratio)
        xs, x_lengths, x_overlaps = split_tiles(z.shape[-1] * ratio, ratio)
        tiles = [z[..., y // ratio : (y + h) // ratio, x // ratio : (x + w) // ratio]
                 for y, h in zip(ys, y_lengths, strict=True) for x, w in zip(xs, x_lengths, strict=True)]  # fmt: skip
        shapes = {tuple(t.shape[-2:]) for t in tiles}
        if len(shapes) == 1:
            batch = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
            decoded = list(self._tiles(batch).split(z.shape[0], dim=0))
        else:  # a side under one tile: each tile alone
            decoded = [self._tiles(tile) for tile in tiles]
        columns = len(xs)
        rows = [decoded[row * columns : (row + 1) * columns] for row in range(len(ys))]
        return stitch(rows, y_overlaps, x_overlaps)

    def _tiles(self, batch: torch.Tensor) -> torch.Tensor:
        """Diffusers' decoder forward over a batch of same-shaped tiles, block by block: every tile passes a block
        before any tile enters the next, `split` cutting each block's rows to the room."""
        decoder = self.decoder
        x = self.post_quant_conv(batch)
        tiles, channels, height, width = x.shape
        hidden: torch.Tensor = decoder.proj_in(x.permute(0, 2, 3, 1).reshape(tiles, height * width, channels))
        patches = hidden.shape[1]
        registers = decoder.register_tokens.expand(tiles, -1, -1).to(hidden.dtype)
        hidden = torch.cat([hidden, registers, torch.zeros_like(hidden[:, :1, :])], dim=1)
        grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden.device) / size) - 1.0
                 for size in (height, width)]  # fmt: skip
        positions = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)[None]
        suffix = positions.new_zeros((1, decoder.num_register_tokens + 1, 2))
        rotary = decoder.rope(torch.cat([positions, suffix], dim=1))
        for block in decoder.transformer_blocks:
            hidden = split(_BlockRows(block, rotary), hidden, Cut(0))
        hidden = decoder.norm_out(hidden).to(get_parameter_dtype(decoder.proj_out))
        hidden = decoder.proj_out(hidden)[:, :patches, :]
        p = decoder.patch_size
        hidden = hidden.view(tiles, height, width, decoder.out_channels, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
        return hidden.reshape(tiles, decoder.out_channels, height * p, width * p)


class _BlockRows:
    """One decoder block over a window of tile rows, with the tiles' shared rotary table."""

    __slots__ = ("block", "rotary")

    def __init__(self, block: nn.Module, rotary: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.block = block
        self.rotary = rotary

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.block(hidden, self.rotary)
        return out


#: Diffusers' `AutoencoderKLMiniMaxH3Image._keep_in_fp32_modules`: what a bf16 load leaves float32, by module name.
FP32_MODULES = frozenset({"encoder", "quant_conv", "norm1", "norm2", "norm_out", "scale1", "scale2"})


def kept_fp32(key: str) -> bool:
    """Whether Diffusers' bf16 load keeps VAE tensor `key` (`decoder.transformer_blocks.0.norm1.weight`) float32."""
    return any(part in FP32_MODULES for part in key.split("."))
