"""Pruna's `AutoencoderKLPH3Image`: the encoder and 24-channel latent space of Diffusers'
`AutoencoderKLMiniMaxH3` (the classes Diffusers 0.41 ships), with a ViT decoder for one frame, each token a
16x16 pixel block. The decode's tiles go through `post_quant_conv` and the decoder as one batch, block by
block, its rows cut as the device holds them (the Runtime's `split`), and are stitched by Diffusers'
`_stitch_tiles` in its order. The decoder is batch-independent (its norms reduce over the last dimension,
its register tokens are per-tile replicas, its rotary comes from the tile geometry), so the arithmetic per
tile is Diffusers'.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from cozy_runtime import rng
from cozy_runtime.author import Cut, split
from cozy_runtime.author import device as placed_device
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    AutoencoderKLMiniMaxH3,
    MiniMaxH3VideoViTDecoder3d,
)
from diffusers.models.modeling_utils import get_parameter_dtype

PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
#: Rotary time of every token of the decoder's one frame (Pruna's `PH3ImageViTDecoder`).
FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0


class ImageDecoder(MiniMaxH3VideoViTDecoder3d):
    """The ViT decoder for one latent frame (`PH3ImageViTDecoder`)."""

    def __init__(self, **config: object) -> None:
        super().__init__(patch_size_t=1, **config)  # type: ignore[arg-type]  # the VAE's decoder keywords


class ImageVAE(AutoencoderKLMiniMaxH3):
    """`AutoencoderKLPH3Image`, its tiles decoded as one batch."""

    @classmethod
    def build(cls, config: Mapping[str, object]) -> ImageVAE:
        vae = cast(ImageVAE, cls.from_config(dict(config)))
        c = vae.config
        vae.decoder = ImageDecoder(
            in_channels=c["latent_channels"], out_channels=c["out_channels"], patch_size=vae.spatial_compression_ratio,
            num_layers=c["decoder_num_layers"], num_attention_heads=c["decoder_num_attention_heads"],
            attention_head_dim=c["decoder_attention_head_dim"], num_register_tokens=c["decoder_num_register_tokens"],
            ffn_mult=c["decoder_ffn_mult"], rope_theta=c["decoder_rope_theta"], rope_dim_ratio=c["decoder_rope_dim_ratio"],
            norm_eps=c["decoder_norm_eps"],
        )  # fmt: skip
        return vae

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """A picture is one frame: one clip."""
        return self._encode_clip(x)

    def decode_image(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized `(1, 24, 1, h, w)` latents to `(1, 3, H, W)` RGB in 0..1."""
        mean = torch.tensor(self.config["latents_mean"], device=latents.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.config["latents_std"], device=latents.device).view(1, -1, 1, 1, 1)
        z = (latents * std + mean).to(self.post_quant_conv.weight.dtype)
        pixels = self._decode_clip(z)[:, :, 0].float()
        mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(1, -1, 1, 1)
        deviation = torch.tensor(PIXEL_STD, device=pixels.device).view(1, -1, 1, 1)
        return (pixels * deviation + mean).clamp(0, 1)

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            decoded: torch.Tensor = self.decoder(self.post_quant_conv(z))
            return decoded
        ratio = self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            z.shape[-2] * ratio, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            z.shape[-1] * ratio, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        tiles = [
            z[..., y // ratio : (y + y_len) // ratio, x // ratio : (x + x_len) // ratio]
            for y, y_len in zip(y_indices, y_lengths, strict=True)
            for x, x_len in zip(x_indices, x_lengths, strict=True)
        ]
        batch = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
        decoded = self._decode_tiles(batch).split(z.shape[0], dim=0)
        columns = len(x_indices)
        rows = [decoded[row * columns : (row + 1) * columns] for row in range(len(y_indices))]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)  # type: ignore[arg-type]  # rows of tile tuples: it indexes them

    def _decode_tiles(self, batch: torch.Tensor) -> torch.Tensor:
        """Diffusers' ViT decoder over a batch of tiles, block by block: every tile passes a block before any
        tile enters the next, and `split` cuts each block's rows to the room. The tiles share one geometry, so
        one rotary table serves them all."""
        decoder = self.decoder
        x = self.post_quant_conv(batch)
        tiles, channels, frames, height, width = x.shape
        hidden: torch.Tensor = decoder.proj_in(x.permute(0, 2, 3, 4, 1).reshape(tiles, frames * height * width, channels))
        patches = hidden.shape[1]
        registers = decoder.register_tokens.expand(tiles, -1, -1).to(hidden.dtype)
        hidden = torch.cat([hidden, registers, torch.zeros_like(hidden[:, :1, :])], dim=1)
        grids = [
            torch.full((frames,), FRAME_POSITION, device=hidden.device),
            *(
                2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden.device) / size) - 1.0
                for size in (height, width)
            ),
        ]
        positions = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)[None]
        suffix = positions.new_zeros((1, decoder.num_register_tokens + 1, 3))
        rotary = decoder.rope(torch.cat([positions, suffix], dim=1))
        for block in decoder.transformer_blocks:
            hidden = split(_BlockRows(block, rotary), hidden, Cut(0))
        hidden = decoder.norm_out(hidden).to(get_parameter_dtype(decoder.proj_out))
        hidden = decoder.proj_out(hidden)[:, :patches, :]
        p, pt = decoder.patch_size, decoder.patch_size_t
        hidden = hidden.view(tiles, frames, height, width, decoder.out_channels, pt, p, p)
        hidden = hidden.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden.reshape(tiles, decoder.out_channels, frames * pt, height * p, width * p)


class _BlockRows:
    """One decoder block over a window of tile rows, with the tiles' shared rotary table."""

    __slots__ = ("block", "rotary")

    def __init__(self, block: torch.nn.Module, rotary: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.block = block
        self.rotary = rotary

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.block(hidden, self.rotary)
        return out


def encode_picture(vae: ImageVAE, pixels: torch.Tensor, seed: int) -> torch.Tensor:
    """A picture's `(H, W, 3)` uint8 pixels to latents `(1, 24, 1, h, w)` as Diffusers encodes a reference: over
    255, normalized by `PIXEL_MEAN`/`PIXEL_STD`, a posterior sample under `seed` rounded to float16."""
    device = torch.device(str(placed_device()))
    pixels = pixels.to(device).permute(2, 0, 1)[None, :, None]
    mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    normalized = (pixels.to(torch.float32).div(255.0) - mean) / std
    posterior = vae.encode(normalized, return_dict=False)[0]
    latents: torch.Tensor = posterior.sample(generator=rng.generator(seed))
    return latents.to(torch.float16).float()
