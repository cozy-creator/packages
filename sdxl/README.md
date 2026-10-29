# SDXL

`generate` makes one image from a text prompt with Stable Diffusion XL 1.0 (VAE fix).

Write the request in a file, `request.yaml`:

```yaml
prompt: a lighthouse at dusk, oil painting
aspect_ratio: "16:9"
seed: 7
```

```sh
cozy run sdxl/generate --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run sdxl/generate prompt='a lighthouse at dusk, oil painting' seed=7
cozy run sdxl/generate prompt='a fox in fresh snow' aspect_ratio=16:9 megapixels=2 steps=30
```

Inputs are the shared image request, the same in every image package: `prompt`, `reference_images`, `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `steps`, `guidance`, `negative_prompt`, `seed` and `prompt_upsampling`. SDXL adds `hidiffusion`. Left out, each is SDXL's default: 1:1 at 1 MP, 20 steps at
guidance 7 (HiDiffusion's 30 at 7.5 above 1 MP, where it runs unless `hidiffusion=false`), no
negative prompt, a random seed. Sizes snap to the 64 px stride and stay within 1–8 MP (16 MP at
1:1), 1:4–4:1 and 4096 px sides; steps 1–50, guidance 0–20. Out of range clamps with a `clamp`
adjustment. SDXL takes no `reference_images`, and ignores `prompt_upsampling` with a warning. The
result is `image` (a WebP in `~/.cozy/outputs`), `width`, `height`, `seed` and `prompt`.

GPU: any NVIDIA card with 8 GB. The default 8-bit lane is 3.8 GiB (`sdxl@1.0.0`);
`--lane bf16` loads the 6.5 GiB bf16 lane. A 1024² image at 20 steps takes about 2.6 s on an
RTX 4090 and 17 s on an 8 GB RTX 4070 Laptop. Without an NVIDIA GPU, add `--on new` to rent
one.

Its recipes write the model's `fp8`, `int8` and `w4a8` lanes, for model authors
(`cozy model quantize <model> --lane int8 --package sdxl`).

Deterministic Euler runs as one `denoise_loop`: the Runtime runs the steps eagerly or as a native
program (resident or streamed UNet blocks), by card and free memory, and records which in the run
record's `loops`. Text encoding, VAE decode and HiDiffusion's DDIM run eagerly. Up to 1 MP the VAE decodes whole;
above it (HiDiffusion's sizes, up to 16 MP) in 1024 px tiles, 512 or 256 px where those do not fit. Every request holds
the VAE in its decode dtype for good (a cast of the stored bytes on its next fill; no request
converts VAE weights). CLIP conditioning calls the encoders' layers directly, avoiding
Transformers' lazy output-collection hooks; its original 5.16.1 forward is the numerical oracle.

It requires a Runtime with `cozy_runtime.author.denoise_loop`.
