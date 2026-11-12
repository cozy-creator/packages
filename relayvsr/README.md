# RelayVSR video upscaling

[RelayVSR](https://kopperx.github.io/RelayVSR/) ([paper](https://arxiv.org/abs/2609.37850),
[code](https://github.com/kopperx/RelayVSR), Apache-2.0) upscales a video 4x as a stream. Wan2.2-TI2V-5B,
distilled to one step, makes a reference latent from a keyframe every `window_size - 1` frames
(default 15), attending to the previous four keyframes. A 32M-parameter decoder then upscales
every frame as a residual over bicubic, attending locally to the two keyframes around it and to
the frame before it.

`upscale` takes a video, `resolution` (`1080p`, `1440p`, `2160p`), a `seed` for the keyframes'
noise, `aspect_ratio` (unset keeps the source's shape; `16:9` letterboxes onto the exact canvas), and,
under Advanced, `window_size`. Frame timestamps and audio pass through Runtime's streaming media
API. A video already at `resolution` comes back as it is.

Each frame is upscaled 4x at the source's own size, then area-scaled to fit `resolution`'s 16:9
canvas (H3's 1344x768 becomes 5376x3072, then 3780x2160 at 2160p). Shrinking the input to a
quarter of the target first (946x540 for 2160p) halves the time (40 s) but loses 0.5 dB of
down-consistency PSNR (37.7 vs 38.2 dB, FL2VA): it discards source detail.

The keyframes' K/V (about 6 GB per keyframe for a 1344x768 source, four kept) are Runtime-held
state, so they spill to host memory on a smaller card. Both local attentions run as Triton
kernels on CUDA sm80+, built when the executor starts; elsewhere an exact masked-SDPA path serves.

## Measured

2026-11-11, one RTX PRO 6000, H3's 15 s Turbo originals (362 frames, seed 581203), warm: execution
time, and down-consistency PSNR (area-scaled back to 1344x768 against the original) for FL2VA / Ref2VA.
SeedVR2 7B 0.6.0 is FL2VA only. Details and review sheets: `minimax-h3/UPSCALE.md`.

| | 1440p | 2160p |
|---|---|---|
| RelayVSR 0.1.0 | 73 s, 38.3 / 36.8 dB | 77 s, 38.2 / 36.7 dB |
| SeedVR2 7B 0.6.0 | 374 s, 38.9 dB | 841 s, 40.3 dB |

The Runtime keeps up to 88 GB on a 96 GB card; under a 24 GiB cap 2160p takes 88.5 s (peak 16 GB).

## Weights

`kopper/relayvsr@0.2.0`, a mirror of `hf://kopper/RelayVSR@d43dc5545f9a3bd1c7730b68ebff4ab1f9199bd1`
(`wan.safetensors`, `flash.safetensors`; Apache-2.0 per the code repository, as the weights repository
states no license). TensorFS's `relayvsr` table keeps their keys as they are, as component `wan` (bf16,
with the prompt embeddings) and `flash` (F32, served in bf16 on CUDA), each with its file's own `config` as its
component config, which the package builds from, and architecture `RelayVSR`.
