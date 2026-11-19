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
batch, its rows cut block by block as the device holds them (the Runtime's `split`); its head
then runs a row of tiles at a time, each stitched as `_stitch_tiles` stitches it.

The arithmetic per tile is the same; the BYTES are a property of the kernel a larger GEMM
selects, so they are measured, never assumed. Measured 2026-09-07 at the release tile
geometry: the batched decode differs from the sequential one by at most ONE ULP of the
compute dtype at the output's peak magnitude — fp32 on CPU (1.5e-8, 157 dB) and fp16
autocast on an RTX 4070 (1.2e-4, 85-91 dB at every batch from 2 to 28); the H100 verdict
is banked on the tracker issue (h3a-017). Tile geometry, autocast and precision are untouched —
h3a-006 owns the lossy decode variants.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    AutoencoderKLMiniMaxH3,
    MiniMaxH3VideoViTDecoder3d,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_utils import get_parameter_dtype

from cozy_runtime import rng
from cozy_runtime.accel import spread
from cozy_runtime.author import Cut, split
from cozy_runtime.author import device as placed_device


#: A tile's activations at the encoder's widest level held at once, before a pass is measured: the eager causal conv
#: holds its input, the SiLU of its norm, the reflected copy, the causally padded one (three frames of one) and its
#: output, about 7 with cuDNN's workspace; the fused one about 4 (cr-323).
ENCODE_ACTIVATIONS = 8


def encode_holds(config: Mapping[str, Any]) -> int:
    """`split`'s `holds` for the encoder over a batch of tiles, in windows of its input: `Cut(0)` keeps the tile count,
    so the Runtime reckons a window's output at the window's own bytes. The widest level is `block_out_channels[0]`
    channels at the tile's size, in the input's dtype (var-030 13: 2700 windows of moments read as input windows asked
    13.6 GiB of a 16 MP picture)."""
    return -(-ENCODE_ACTIVATIONS * int(config["block_out_channels"][0]) // int(config["in_channels"]))


def block_holds(dim: int, inner: int, residual: torch.dtype, compute: torch.dtype, rows: torch.dtype) -> int:
    """`split`'s `holds` for a decoder block over a window of tile rows, in windows of its input (`rows`): the block's
    peak is its feed-forward's, a token's `residual` beside its normed row, SwiGLU's two-half projection, the gate's
    activation and their product (`compute`). The default 5 fell short at level 5 (release-gate 44412956639)."""
    peak = residual.itemsize * dim + compute.itemsize * (dim + 4 * inner)
    return -(-peak // (rows.itemsize * dim))


class TileBatchedVideoVAE(AutoencoderKLMiniMaxH3):
    """`AutoencoderKLMiniMaxH3` whose tiled clip decode batches the tiles, whose temporal
    decode hands out each finished chunk as it lands (h3a-017), and whose picture encode
    batches its tiles (cr-323)."""

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
        num_tokens = z.shape[2] + self.config["token_drop"]
        pad_tokens = (-num_tokens) % tokens_chunk_size
        intra_tail = self.config["clip_length"] % temporal_ratio
        pad_frames = sum(
            intra_tail
            if intra_tail and (z.shape[2] + k) % tokens_chunk_size == 0
            else temporal_ratio
            for k in range(pad_tokens)
        )
        # Release a piece only while at least `pad_frames` frames stay held, so the trailing
        # padding is never handed out and no length formula has to predict the loop.
        # A name left bound here would keep its piece on the card through the next clip's decode.
        held: list[torch.Tensor] = []
        held_frames = 0
        for piece in self._decode_pieces(z, pad_tokens):
            held.append(piece)
            held_frames += int(piece.shape[2])
            del piece
            while held and held_frames - int(held[0].shape[2]) >= pad_frames:
                first = held.pop(0)
                held_frames -= int(first.shape[2])
                yield first
                del first
        if held and held_frames > pad_frames:
            rest = held[0] if len(held) == 1 else torch.cat(held, dim=2)
            yield rest[:, :, : held_frames - pad_frames]

    def _decode_pieces(self, z: torch.Tensor, pad_tokens: int) -> Iterator[torch.Tensor]:
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config["token_drop"]
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
        # The fade's frames are copied out, so a clip leaves the card before the next one decodes.
        chunk: torch.Tensor | None = None
        for clip in spread(self._decode_clip, windows):
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    yield chunk
                else:
                    overlap = chunk.clone(memory_format=torch.contiguous_format)
            del clip
            chunk = None
        if overlap is not None:
            yield overlap

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(self.decode_chunks(z)), dim=2)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Upstream's temporal encode with its clips spread over the group: each
        `clip_length` clip encodes from its own frames alone, so a round runs on the ranks
        at once and the moments join here in order, as one rank computes them."""
        clip_length = self.config["clip_length"]
        if x.shape[2] == 1:
            return self._encode_clip(x)
        if x.shape[2] % clip_length:
            pad = x[:, :, -1:].repeat(1, 1, (-x.shape[2]) % clip_length, 1, 1)
            x = torch.cat([x, pad], dim=2)
        clips = ((x[:, :, i : i + clip_length],) for i in range(0, x.shape[2], clip_length))
        moments = torch.cat(list(spread(self._encode_clip, clips)), dim=2)
        token_drop = self.config["token_drop"]
        return moments[:, :, :-token_drop] if token_drop > 0 else moments

    def encode_frames(
        self,
        pixels: torch.Tensor,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
    ) -> DiagonalGaussianDistribution:
        """`encode` of uint8 `(1, 3, T, H, W)` pixels normalized as `encode_condition` does,
        each clip normalized on the rank that encodes it: a clip crosses to a follower as
        uint8, a quarter of its float32 bytes (run 1516: 3.5-3.9 s per continuation export).
        Normalization is elementwise and padding repeats the last frame, so every clip's
        floats are the ones `_encode` would slice from the normalized video."""
        clip_length = self.config["clip_length"]
        if pixels.shape[2] % clip_length:
            pad = pixels[:, :, -1:].repeat(1, 1, (-pixels.shape[2]) % clip_length, 1, 1)
            pixels = torch.cat([pixels, pad], dim=2)
        clips = (
            (pixels[:, :, i : i + clip_length], pixel_mean, pixel_std)
            for i in range(0, pixels.shape[2], clip_length)
        )
        moments = torch.cat(list(spread(self._encode_frames_clip, clips)), dim=2)
        token_drop = self.config["token_drop"]
        if token_drop > 0:
            moments = moments[:, :, :-token_drop]
        return DiagonalGaussianDistribution(moments)

    def _encode_frames_clip(
        self,
        clip: torch.Tensor,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
    ) -> torch.Tensor:
        dtype = get_parameter_dtype(self.encoder)  # as `encode` aligns its pixels
        return self._encode_clip(_normalized(clip, pixel_mean, pixel_std).to(dtype))

    def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Upstream's tiled clip encode; a picture's tiles (one frame each) go through the encoder as ONE batch, cut as
        the device holds them (cr-323). One by one, a 256 px tile is too little work for the card and its launches wait
        on the host (8 MP: 228 tiles at ~10 ms each on an RTX 5090, whatever the conv work). Every tile is the same size
        and the encoder is batch-independent (`MiniMaxH3VideoGroupNorm` normalizes each frame of each tile alone), so
        each tile's moments are the per-tile loop's, stitched by the unchanged `_stitch_tiles`. A video clip's tiles
        each fill the card already and keep upstream's loop."""
        if not self.use_tiling or x.shape[2] != 1:
            return super()._encode_clip(x)
        ys, heights, y_overlaps = self._split_tiles(
            x.shape[-2], self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        xs, widths, x_overlaps = self._split_tiles(
            x.shape[-1], self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        tiles = [
            x[..., y : y + h, left : left + w]
            for y, h in zip(ys, heights, strict=True)
            for left, w in zip(xs, widths, strict=True)
        ]
        batch = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
        moments = split(self._encode_tiles, batch, Cut(0), holds=encode_holds(self.config)).split(x.shape[0], dim=0)
        columns = len(xs)
        rows = [moments[row * columns : (row + 1) * columns] for row in range(len(ys))]
        ratio = self.spatial_compression_ratio
        return self._stitch_tiles(rows, [o // ratio for o in y_overlaps], [o // ratio for o in x_overlaps])

    def _encode_tiles(self, batch: torch.Tensor) -> torch.Tensor:
        moments: torch.Tensor = self.quant_conv(self.encoder(batch))
        return moments

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
        hidden, tail = self._decode_blocks(batch)
        # The head and the stitch go a row of tiles at a time, each row cut to the room: a row's blend reads only the
        # row above as decoded, so no step holds every tile's pixels (se-169). Same ops, same order, as
        # `_stitch_tiles` over every row.
        across = len(x_indices) * z.shape[0]
        canvas: torch.Tensor | None = None
        above: Sequence[torch.Tensor] = ()
        top = 0
        for i in range(len(y_indices)):
            row = split(tail, hidden[i * across : (i + 1) * across], Cut(0)).split(z.shape[0], dim=0)
            strip = self._stitch_row(row, above, i, len(y_indices), y_overlaps, x_overlaps)
            if canvas is None:
                size = (sum(y_lengths) - sum(y_overlaps), sum(x_lengths) - sum(x_overlaps))
                canvas = strip.new_empty((*strip.shape[:-2], *size))
            canvas[..., top : top + strip.shape[-2], :] = strip
            top += strip.shape[-2]
            above = row
        assert canvas is not None  # a grid has a row
        return canvas

    def _stitch_row(
        self,
        row: Sequence[torch.Tensor],
        above: Sequence[torch.Tensor],
        i: int,
        rows: int,
        height_overlaps: Sequence[int],
        width_overlaps: Sequence[int],
    ) -> torch.Tensor:
        """`_stitch_tiles`' row `i` of `rows`: each tile blended into the one `above` and the one before it, as decoded,
        and its overlaps cut."""
        strip = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = self._blend(above[j], tile, height_overlaps[i - 1], dim=-2)
            if j > 0:
                tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
            if i < rows - 1:
                tile = tile[..., : -height_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -width_overlaps[j]]
            strip.append(tile)
        return torch.cat(strip, dim=-1)

    def time_grid(self, frames: int, device: torch.device) -> torch.Tensor:
        """The decoder's rotary time of each of a clip's `frames` latent frames, upstream's."""
        return 2.0 * (torch.arange(0.5, frames, dtype=torch.float32, device=device) / frames) - 1.0

    def _decode_blocks(self, batch: torch.Tensor) -> tuple[torch.Tensor, _TileTail]:
        """Upstream's ViT decoder over a batch of tiles up to its head, block by block: every tile
        passes a block before any tile enters the next, and `split` cuts each block's rows to the room.
        Cutting the whole decoder instead re-read all 36 blocks once per piece: on a 24 GB
        RTX 4090 a chunk ran in 10 pieces at 3.2-3.5 s against 2.4 s with the weights resident
        (runs 11220, 11228); block by block it holds 2.36-2.46 s (11325, 11335). The tiles share
        one geometry, so one rotary table serves them all; every op is upstream's `forward`:
        uncut, the two orders are bit-identical on the release weights. The first block takes the
        embedding's dtype to the residual stream's (float32 under the release's float32 layer scales);
        every later block keeps it, so it writes its rows over its input: the rows are held once
        (se-169). Returns the last block's rows and the head that turns a window of them into pixels."""
        decoder = self.decoder
        x = self.post_quant_conv(batch)
        tiles, channels, frames, height, width = x.shape
        hidden: torch.Tensor = decoder.proj_in(
            x.permute(0, 2, 3, 4, 1).reshape(tiles, frames * height * width, channels)
        )
        patches = hidden.shape[1]
        registers = decoder.register_tokens.expand(tiles, -1, -1).to(hidden.dtype)
        hidden = torch.cat([hidden, registers, torch.zeros_like(hidden[:, :1, :])], dim=1)
        grids = [
            self.time_grid(frames, hidden.device),
            *(
                2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden.device) / size) - 1.0
                for size in (height, width)
            ),
        ]
        positions = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)[None]
        suffix = positions.new_zeros((1, decoder.num_register_tokens + 1, 3))
        rotary = decoder.rope(torch.cat([positions, suffix], dim=1))
        first, *rest = decoder.transformer_blocks
        dim, compute = int(hidden.shape[-1]), hidden.dtype
        inner = int(self.config["decoder_ffn_mult"]) * dim
        stream = torch.promote_types(compute, cast(torch.Tensor, first.scale1).dtype)  # `hidden + branch * scale1`
        first_holds = block_holds(dim, inner, stream, compute, compute)
        hidden = split(_BlockRows(first, rotary), hidden, Cut(0), holds=first_holds)
        holds = block_holds(dim, inner, hidden.dtype, compute, hidden.dtype)
        for block in rest:
            split(_BlockRows(block, rotary), hidden, Cut(0), out=hidden, holds=holds)
        return hidden, _TileTail(decoder, patches, (frames, height, width))

    def encode_condition(
        self,
        pixels: torch.Tensor,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
        seed: int,
    ) -> torch.Tensor:
        """`encode_condition` as this component's method, so `accel.spread` can run it on
        another rank of the group."""
        return encode_condition(self, pixels, pixel_mean, pixel_std, seed)


class _BlockRows:
    """One decoder block over a window of tile rows, with the tiles' shared rotary table; with `out`, written there."""

    __slots__ = ("block", "rotary")

    def __init__(self, block: torch.nn.Module, rotary: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.block = block
        self.rotary = rotary

    def __call__(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows: torch.Tensor = self.block(hidden, self.rotary)
        return rows if out is None else out.copy_(rows)


class _TileTail:
    """The decoder's head over a window of tile rows: `norm_out`, `proj_out`, and each token's patch back in place,
    upstream's `forward` after its blocks."""

    __slots__ = ("decoder", "patches", "grid")

    def __init__(self, decoder: MiniMaxH3VideoViTDecoder3d, patches: int, grid: tuple[int, int, int]) -> None:
        self.decoder = decoder
        self.patches = patches
        self.grid = grid

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder
        frames, height, width = self.grid
        tiles = hidden.shape[0]
        out = decoder.norm_out(hidden).to(get_parameter_dtype(decoder.proj_out))
        out = decoder.proj_out(out)[:, : self.patches, :]
        p, pt = decoder.patch_size, decoder.patch_size_t
        out = out.view(tiles, frames, height, width, decoder.out_channels, pt, p, p)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        pixels: torch.Tensor = out.reshape(tiles, decoder.out_channels, frames * pt, height * p, width * p)
        return pixels


def encode_condition(
    vae: AutoencoderKLMiniMaxH3,
    pixels: torch.Tensor,
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
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
    latents: torch.Tensor = posterior.sample(generator=rng.generator(seed))
    return latents.to(torch.float16).float()


def _normalized(
    pixels: torch.Tensor,
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
) -> torch.Tensor:
    """Diffusers' `encode_vae_condition` pixel convention: uint8 to float32, over 255, minus
    the per-channel mean, over the per-channel std."""
    mean = torch.tensor(pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)
    return (pixels.to(torch.float32).div(255.0) - mean) / std
