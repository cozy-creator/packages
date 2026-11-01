"""The official H3 video VAE with one temporal chunk's spatial tiles decoded as ONE batch.

Diffusers' `_decode_clip` lays a grid of equal 256 px tiles (64 px minimum overlap) over
the chunk and runs the ViT decoder once per tile: at 1344x768 that is a 4x7 grid, 28
batch-1 forwards per chunk and ~590 per 362-frame clip, each too small to fill the card
(19.4 s on an H100). Every tile is the same size — `_split_tiles` pushes the slack into
the overlaps, never into a shorter edge tile — and the decoder is batch-independent: a
ViT over `(B, S, C)` tokens whose norms reduce over the last dimension only, whose
register/cls tokens are per-batch replicas and whose RoPE comes from the tile geometry.
(`MiniMaxH3VideoGroupNorm`, the one batch-mixing module, is encoder-side.) So the tiles
of one chunk go through `post_quant_conv` and the decoder as one `(tiles, C, T, h, w)`
batch, or as few as the device holds (the Runtime's `split`), and are stitched by the
unchanged `_stitch_tiles` in the unchanged order.

The arithmetic per tile is the same; the BYTES are a property of the kernel a larger GEMM
selects, so they are measured, never assumed. Measured 2026-09-07 at the release tile
geometry: the batched decode differs from the sequential one by at most ONE ULP of the
compute dtype at the output's peak magnitude — fp32 on CPU (1.5e-8, 157 dB) and fp16
autocast on an RTX 4070 (1.2e-4, 85-91 dB at every batch from 2 to 28); the H100 verdict
is banked on the tracker issue (h3a-017). Tile geometry, autocast and precision are untouched —
h3a-006 owns the lossy decode variants.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from diffusers import AutoencoderKLMiniMaxH3
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_utils import get_parameter_dtype

from cozy_runtime.author import Cut, split, spread
from cozy_runtime.author import device as placed_device


class TileBatchedVideoVAE(AutoencoderKLMiniMaxH3):  # type: ignore[misc]
    """`AutoencoderKLMiniMaxH3` whose tiled clip decode batches the tiles and whose temporal
    decode hands out each finished chunk as it lands (h3a-017)."""

    def decode_chunks(self, z: torch.Tensor) -> Iterator[torch.Tensor]:
        """Upstream's `_decode` temporal loop as a generator: every yielded `(B, C, t, H, W)`
        piece is final — cross-faded with its predecessor and clear of the padding frames
        upstream cuts off the end — so a consumer may encode it while the next chunk
        decodes. `_decode` below is this same loop concatenated, so the two are one
        computation, not two.
        """
        z = z.to(next(self.decoder.parameters()).dtype)
        tokens_chunk_size = self.tokens_chunk_size
        temporal_ratio = self.temporal_compression_ratio
        num_tokens = z.shape[2] + self.config.token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        intra_tail = self.config.clip_length % temporal_ratio
        pad_frames = sum(
            intra_tail
            if intra_tail and (z.shape[2] + k) % tokens_chunk_size == 0
            else temporal_ratio
            for k in range(pad_tokens)
        )
        # Release a piece only while at least `pad_frames` frames stay held, so the trailing
        # padding is never handed out and no length formula has to predict the loop.
        held: list[torch.Tensor] = []
        held_frames = 0
        for piece in self._decode_pieces(z, pad_tokens):
            held.append(piece)
            held_frames += int(piece.shape[2])
            while held and held_frames - int(held[0].shape[2]) >= pad_frames:
                first = held.pop(0)
                held_frames -= int(first.shape[2])
                yield first
        if held and held_frames > pad_frames:
            rest = held[0] if len(held) == 1 else torch.cat(held, dim=2)
            yield rest[:, :, : held_frames - pad_frames]

    def _decode_pieces(self, z: torch.Tensor, pad_tokens: int) -> Iterator[torch.Tensor]:
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        chunk_num_frames = tokens_chunk_size * self.temporal_compression_ratio
        num_tokens = z.shape[2] + token_drop + pad_tokens
        num_chunks = num_tokens // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        overlap = None
        # Every clip decodes from its own latent window alone; only the cross-fade below
        # reads a neighbour. So a group decodes a round of clips on its ranks at once and
        # the fade runs here, in order, unchanged.
        windows = (
            (z[:, :, i * tokens_chunk_size : (i + 1) * tokens_chunk_size + self.token_overlap],)
            for i in range(num_chunks)
        )
        for clip in spread(self, "_decode_clip", windows):
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    yield chunk
                else:
                    overlap = chunk
        if overlap is not None:
            yield overlap

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(self.decode_chunks(z)), dim=2)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Upstream's temporal encode with its clips spread over the group: each
        `clip_length` clip encodes from its own frames alone, so a round runs on the ranks
        at once and the moments join here in order, as one rank computes them."""
        clip_length = self.config.clip_length
        if x.shape[2] == 1:
            return self._encode_clip(x)
        if x.shape[2] % clip_length:
            pad = x[:, :, -1:].repeat(1, 1, (-x.shape[2]) % clip_length, 1, 1)
            x = torch.cat([x, pad], dim=2)
        clips = ((x[:, :, i : i + clip_length],) for i in range(0, x.shape[2], clip_length))
        moments = torch.cat(list(spread(self, "_encode_clip", clips)), dim=2)
        token_drop = self.config.token_drop
        return moments[:, :, :-token_drop] if token_drop > 0 else moments

    def encode_frames(
        self,
        pixels: torch.Tensor,
        pixel_mean: tuple[float, float, float],
        pixel_std: tuple[float, float, float],
    ) -> DiagonalGaussianDistribution:
        """`encode` of uint8 `(1, 3, T, H, W)` pixels normalized as `encode_condition` does,
        each clip normalized on the rank that encodes it: a clip crosses to a follower as
        uint8, a quarter of its float32 bytes (run 1516: 3.5-3.9 s per continuation export).
        Normalization is elementwise and padding repeats the last frame, so every clip's
        floats are the ones `_encode` would slice from the normalized video."""
        clip_length = self.config.clip_length
        if pixels.shape[2] % clip_length:
            pad = pixels[:, :, -1:].repeat(1, 1, (-pixels.shape[2]) % clip_length, 1, 1)
            pixels = torch.cat([pixels, pad], dim=2)
        clips = (
            (pixels[:, :, i : i + clip_length], pixel_mean, pixel_std)
            for i in range(0, pixels.shape[2], clip_length)
        )
        moments = torch.cat(list(spread(self, "_encode_frames_clip", clips)), dim=2)
        token_drop = self.config.token_drop
        if token_drop > 0:
            moments = moments[:, :, :-token_drop]
        return DiagonalGaussianDistribution(moments)

    def _encode_frames_clip(
        self,
        clip: torch.Tensor,
        pixel_mean: tuple[float, float, float],
        pixel_std: tuple[float, float, float],
    ) -> torch.Tensor:
        dtype = get_parameter_dtype(self.encoder)  # as `encode` aligns its pixels
        return self._encode_clip(_normalized(clip, pixel_mean, pixel_std).to(dtype))

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

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
        decoded = split(self._decode_tiles, batch, Cut(0)).split(z.shape[0], dim=0)
        columns = len(x_indices)
        rows = [decoded[row * columns : (row + 1) * columns] for row in range(len(y_indices))]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _decode_tiles(self, batch: torch.Tensor) -> torch.Tensor:
        """Upstream's ViT decoder over a batch of tiles, block by block: every tile passes a
        block before any tile enters the next, and `split` cuts each block's rows to the room.
        Cutting the whole decoder instead re-read all 36 blocks once per piece: on a 24 GB
        RTX 4090 a chunk ran in 10 pieces at 3.2-3.5 s against 2.4 s with the weights resident
        (runs 11220, 11228); block by block it holds 2.36-2.46 s (11325, 11335). The tiles share
        one geometry, so one rotary table serves them all; every op is upstream's `forward`:
        uncut, the two orders are bit-identical on the release weights."""
        decoder = self.decoder
        x = self.post_quant_conv(batch)
        tiles, channels, frames, height, width = x.shape
        hidden = decoder.proj_in(
            x.permute(0, 2, 3, 4, 1).reshape(tiles, frames * height * width, channels)
        )
        patches = hidden.shape[1]
        registers = decoder.register_tokens.expand(tiles, -1, -1).to(hidden.dtype)
        hidden = torch.cat([hidden, registers, torch.zeros_like(hidden[:, :1, :])], dim=1)
        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden.device) / size) - 1.0
            for size in (frames, height, width)
        ]
        positions = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)[None]
        suffix = positions.new_zeros((1, decoder.num_register_tokens + 1, 3))
        rotary = decoder.rope(torch.cat([positions, suffix], dim=1))
        for block in decoder.transformer_blocks:
            hidden = _BlockRows(block, rotary).run(hidden)
        hidden = decoder.norm_out(hidden).to(get_parameter_dtype(decoder.proj_out))
        hidden = decoder.proj_out(hidden)[:, :patches, :]
        p, pt = decoder.patch_size, decoder.patch_size_t
        hidden = hidden.view(tiles, frames, height, width, decoder.out_channels, pt, p, p)
        hidden = hidden.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden.reshape(tiles, decoder.out_channels, frames * pt, height * p, width * p)

    def encode_condition(
        self,
        pixels: torch.Tensor,
        pixel_mean: tuple[float, float, float],
        pixel_std: tuple[float, float, float],
        seed: int,
    ) -> torch.Tensor:
        """`encode_condition` as this component's method, so `author.spread` can run it on
        another rank of the group."""
        return encode_condition(self, pixels, pixel_mean, pixel_std, seed)


class _BlockRows:
    """One decoder block over a window of tile rows, with the tiles' shared rotary table."""

    __slots__ = ("block", "rotary")

    def __init__(self, block: torch.nn.Module, rotary: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.block = block
        self.rotary = rotary

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.block(hidden, self.rotary)

    def run(self, hidden: torch.Tensor) -> torch.Tensor:
        """Declare this complete layer-scale block before its outer batch chooser."""
        from cozy_runtime.author.kernels import LayerScaleBlock

        intent = LayerScaleBlock(self.block, hidden, self.rotary, "scale1", "scale2")
        return split(self, hidden, Cut(0), intent=intent)


def encode_condition(
    vae: AutoencoderKLMiniMaxH3,
    pixels: torch.Tensor,
    pixel_mean: tuple[float, float, float],
    pixel_std: tuple[float, float, float],
    seed: int,
) -> torch.Tensor:
    """Diffusers' `encode_vae_condition` for one reference, up to its `.cpu()`: the sampled
    latents rounded to float16, on this process's device, so a group rank can hand them back over
    the collective. `pixels` are the reference's uint8 pixels as the official step reads
    them, `(H, W, 3)` for an image or `(T, H, W, 3)` for a video, laid out on the device
    exactly as that step lays them out."""
    device = torch.device(str(placed_device()))
    pixels = pixels.to(device)
    if pixels.ndim == 4 and isinstance(vae, TileBatchedVideoVAE):
        posterior = vae.encode_frames(pixels.permute(3, 0, 1, 2)[None], pixel_mean, pixel_std)
    else:
        if pixels.ndim == 3:
            pixels = pixels.permute(2, 0, 1)[None, :, None]
        else:
            pixels = pixels.permute(3, 0, 1, 2)[None]
        posterior = vae.encode(_normalized(pixels, pixel_mean, pixel_std), return_dict=False)[0]
    latents = posterior.sample(generator=torch.Generator().manual_seed(seed))
    return latents.to(torch.float16).float()


def _normalized(
    pixels: torch.Tensor,
    pixel_mean: tuple[float, float, float],
    pixel_std: tuple[float, float, float],
) -> torch.Tensor:
    """Diffusers' `encode_vae_condition` pixel convention: uint8 to float32, over 255, minus
    the per-channel mean, over the per-channel std."""
    mean = torch.tensor(pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)
    return (pixels.to(torch.float32).div(255.0) - mean) / std
