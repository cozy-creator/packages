# SDXL

`generate` makes one image from a text prompt with Stable Diffusion XL 1.0 (VAE fix).

`image` is the canvas tool, its prompt `instructions`: `aspect_ratio`, `megapixels`, `negative_prompt` and `seed`; HiDiffusion runs above 1 MP.

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

Inputs: `prompt`, `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `seed`, `steps`, `guidance`, `negative_prompt` and `hidiffusion`. Left out, each is SDXL's default: 1:1 at 1 MP, 30 steps at
guidance 7 (HiDiffusion's 30 at 7.5 above 1 MP, where it runs unless `hidiffusion=false`), no
negative prompt, a random seed. Sizes snap to the 64 px stride and stay within 1–8 MP (16 MP at
1:1), 1:4–4:1 and 4096 px sides; steps 1–50, guidance 0–20. Out of range clamps with a `clamp`
adjustment. SDXL makes images from text only: its prompt is words, so a picture mentioned in it goes as its name ([references](../docs/references.md)). The
result is `image` (a WebP in `~/.cozy/outputs`), `width`, `height`, `seed` and `prompt`.

GPU: any NVIDIA card with 8 GB. The default 8-bit lane is 3.8 GiB (`stabilityai/sdxl@1.0.0`);
`--lane bf16` loads the 6.5 GiB bf16 lane. A 1024² image at 30 steps takes about 2.9 s on an
RTX 4090 (int8). Without an NVIDIA GPU, add `--on new` to rent
one.

Its recipes write the model's `fp8`, `int8` and `w4a8` lanes, for model authors
(`cozy model quantize <model> --lane int8 --package sdxl`).

Encoding, every step and the decode run in one call: a plain loop over DPM++ 2M Karras (Diffusers'
`DPMSolverMultistepScheduler`, epsilon or v), or with HiDiffusion its `DDIMScheduler` at eta 1. The VAE decodes whole where it fits, else in the fewest windows that do (a
`split` site, neighbours cross-faded over 256 px). Every request holds
the VAE in its decode dtype for good (a cast of the stored bytes on its next fill; no request
converts VAE weights). CLIP conditioning calls the encoders' layers directly, avoiding
Transformers' lazy output-collection hooks; its original 5.16.1 forward is the numerical oracle.

