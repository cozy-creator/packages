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
quarter of the target first is about twice as fast but discards source detail; see the table.

The keyframes' K/V (about 6 GB per keyframe for a 1344x768 source, four kept) are Runtime-held
state, so they spill to host memory on a smaller card. Both local attentions run as Triton
kernels on CUDA sm80+, built when the executor starts; elsewhere an exact masked-SDPA path serves.

## Weights

`kopper/relayvsr@0.1.1`, a mirror of `hf://kopper/RelayVSR@d43dc5545f9a3bd1c7730b68ebff4ab1f9199bd1`
(`wan.safetensors`, `flash.safetensors`; Apache-2.0 per the code repository, as the weights repository
states no license). TensorFS's `relayvsr` table keeps their keys as they are, as component `wan` (bf16,
with the prompt embeddings) and `flash` (F32, served in bf16 on CUDA), with architecture `RelayVSR`.
