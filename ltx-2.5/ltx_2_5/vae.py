"""LTX's spatial tiling with Runtime-owned decoded tiles between decoder calls.

The blends and crop are Diffusers 0.43's AutoencoderKLLTX2Video.tiled_decode.
Blending each new tile before retaining it gives the same row-major arithmetic;
the decoder never reads another tile's decoded output.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from cozy_runtime.author import retained
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from diffusers.models.autoencoders.vae import DecoderOutput

GetTile = Callable[[], tuple[torch.Tensor, ...]]


class VideoVAE(AutoencoderKLLTX2Video):
    def tiled_decode(self, z: torch.Tensor, temb: torch.Tensor | None, causal: bool | None = None,
                     return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        batch, _, frames, height, width = z.shape
        scale = self.spatial_compression_ratio
        tile_h = self.tile_sample_min_height // scale
        tile_w = self.tile_sample_min_width // scale
        step_h = self.tile_sample_stride_height // scale
        step_w = self.tile_sample_stride_width // scale
        blend_h = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_w = self.tile_sample_min_width - self.tile_sample_stride_width
        out_frames = (frames - 1) * self.temporal_compression_ratio + 1
        area = sum(min(tile_h, height - i) * min(tile_w, width - j) * scale * scale
                   for i in range(0, height, step_h) for j in range(0, width, step_w))
        capacity = batch * int(self.config["out_channels"]) * out_frames * area * z.element_size()
        rows: list[list[GetTile]] = []
        with retained(capacity) as keep:
            for i in range(0, height, step_h):
                row: list[GetTile] = []
                for j in range(0, width, step_w):
                    tile = self.decoder(z[:, :, :, i:i + tile_h, j:j + tile_w], temb, causal=causal)
                    if rows:
                        tile = self.blend_v(rows[-1][len(row)]()[0], tile, blend_h)
                    if row:
                        tile = self.blend_h(row[-1]()[0], tile, blend_w)
                    row.append(keep.put(tile).get)
                    del tile
                rows.append(row)
            assembled = [torch.cat([get()[0][:, :, :, :self.tile_sample_stride_height, :self.tile_sample_stride_width]
                                    for get in row], dim=4) for row in rows]
            pixels = torch.cat(assembled, dim=3)[:, :, :, :height * scale, :width * scale]
        return DecoderOutput(sample=pixels) if return_dict else (pixels,)
