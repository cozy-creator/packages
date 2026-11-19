"""LTX's spatial tiling, each tile written out as it is finished.

The blends and crop are Diffusers 0.43's AutoencoderKLLTX2Video.tiled_decode, in its row-major order: a tile blends
with the last rows of the tile above it and the last columns of the tile before it, both already blended, and its
stride crop is then final. So only those overlap strips are kept between decoder calls (Runtime-owned, `retained`),
and the video is never whole on the card: `frames` writes each crop as RGB8 into host memory (a 20 s 1080p clip is
6 GB in bf16).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

import numpy as np
import torch
from cozy_runtime.author import retained
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from diffusers.models.autoencoders.vae import DecoderOutput

VideoPixels = np.ndarray[tuple[int, int, int, int], np.dtype[np.uint8]]
GetStrip = Callable[[], tuple[torch.Tensor, ...]]

#: Frames converted to RGB8 at once: a float32 copy of this many frames of a crop.
_FRAMES = 16


class VideoVAE(AutoencoderKLLTX2Video):
    def tiled_decode(self, z: torch.Tensor, temb: torch.Tensor | None, causal: bool | None = None,
                     return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        batch, _, frames, height, width = z.shape
        scale = self.spatial_compression_ratio
        shape = (batch, int(self.config["out_channels"]), (frames - 1) * self.temporal_compression_ratio + 1,
                 height * scale, width * scale)
        pixels = torch.empty(shape, dtype=z.dtype, device=z.device)
        for y, x, crop in self._tiles(z, temb, causal):
            rows, cols = min(crop.shape[3], shape[3] - y), min(crop.shape[4], shape[4] - x)
            pixels[:, :, :, y:y + rows, x:x + cols] = crop[:, :, :, :rows, :cols]
        return DecoderOutput(sample=pixels) if return_dict else (pixels,)

    def frames(self, z: torch.Tensor) -> VideoPixels:
        """The decoded video as RGB8 [frames, height, width, 3] in host memory, from one latent video: the bytes
        `decode` then a per-frame `(x + 1) * 127.5`, rounded and clamped, would give."""
        if self.use_framewise_decoding:
            raise ValueError("LTX-2.5 writes its decode by spatial tiles; framewise decoding is not one of them")
        _, _, frames, height, width = z.shape
        scale = self.spatial_compression_ratio
        count = (frames - 1) * self.temporal_compression_ratio + 1
        result = cast(VideoPixels, np.empty((count, height * scale, width * scale, int(self.config["out_channels"])),
                                            dtype=np.uint8))
        tiled = self.use_tiling and (width > self.tile_sample_min_width // scale
                                     or height > self.tile_sample_min_height // scale)
        whole = None if tiled else ((0, 0, self.decoder(z, None)),)
        for y, x, crop in whole or self._tiles(z, None, None):
            rows, cols = min(crop.shape[3], result.shape[1] - y), min(crop.shape[4], result.shape[2] - x)
            part = crop[0, :, :, :rows, :cols].permute(1, 2, 3, 0)
            for start in range(0, count, _FRAMES):
                chunk = part[start:start + _FRAMES].to(dtype=torch.float32, copy=True).add_(1).mul_(127.5)
                rgb = chunk.round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
                result[start:start + _FRAMES, y:y + rows, x:x + cols] = rgb
        return result

    def _tiles(self, z: torch.Tensor, temb: torch.Tensor | None,
               causal: bool | None) -> Iterator[tuple[int, int, torch.Tensor]]:
        """Each tile's finished stride crop, row-major, with its first pixel row and column."""
        batch, _, frames, height, width = z.shape
        scale = self.spatial_compression_ratio
        tile_h = self.tile_sample_min_height // scale
        tile_w = self.tile_sample_min_width // scale
        step_h = self.tile_sample_stride_height // scale
        step_w = self.tile_sample_stride_width // scale
        blend_h = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_w = self.tile_sample_min_width - self.tile_sample_stride_width
        out_frames = (frames - 1) * self.temporal_compression_ratio + 1
        tops, lefts = range(0, height, step_h), range(0, width, step_w)
        heights = [min(tile_h, height - i) * scale for i in tops]
        widths = [min(tile_w, width - j) * scale for j in lefts]
        # What a later tile reads: every row's bottom strip but the last row's, every tile's right strip but the
        # last column's.
        strips = (sum(min(blend_h, h) for h in heights[:-1]) * sum(widths)
                  + sum(heights) * sum(min(blend_w, w) for w in widths[:-1]))
        capacity = batch * int(self.config["out_channels"]) * out_frames * strips * z.element_size()
        with retained(capacity) as keep:
            above: list[GetStrip] = []
            for i in tops:
                below: list[GetStrip] = []
                left: GetStrip | None = None
                for column, j in enumerate(lefts):
                    tile = self.decoder(z[:, :, :, i:i + tile_h, j:j + tile_w], temb, causal=causal)
                    if above:
                        tile = self.blend_v(above[column]()[0], tile, blend_h)
                    if left is not None:
                        tile = self.blend_h(left()[0], tile, blend_w)
                    if i + step_h < height:
                        below.append(keep.put(tile[:, :, :, -blend_h:, :]).get)
                    left = keep.put(tile[:, :, :, :, -blend_w:]).get if j + step_w < width else None
                    yield i * scale, j * scale, tile[:, :, :, :self.tile_sample_stride_height,
                                                     :self.tile_sample_stride_width]
                    del tile
                above = below
