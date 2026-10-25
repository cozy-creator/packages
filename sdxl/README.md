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

On Runtime 0.32.2, deterministic Euler with epsilon prediction can submit all steps after
step zero as one prepared native interval. Runtime accepts it only when the UNet is already
resident and its hooks and weight identity are safe for capture. HiDiffusion, other
schedulers, streamed models and capacity refusals keep eager execution. The prepared
interval closes before VAE decode; it is currently prepared per request. Native progress
reports completed steps without treating observation intervals as per-step GPU timings.

The coordinated native-pipeline candidate additionally supports a single resident program
for both text encoders, every Euler step and VAE decode. A construction retains at most one
variant; prompt and seed values update fixed input buffers, while shape, schedule, guidance,
representation or Runtime invalidation replaces the definition. Runtime prepares the VAE's
wide representation before capture and owns retirement under pressure.

`ResidentPipeline.prepare` is the explicit preparation entry for an admitted warm turn.
The ordinary request path only reuses it or captures already-resident representations; it
does not load the whole model speculatively. The coordinated Runtime candidate invokes
SDXL's typed preparation factory from an explicit GPU warm request:

```sh
cozy run sdxl/generate --warm gpu prompt='warm variant' steps=30 guidance=7.0 aspect_ratio=1:1 hidiffusion=false
```

Warm and ordinary requests use the same resolved dimensions, defaults and adjustment rows.
Preparation initializes fixed-shape token slots without drawing a seed, submitting an image
or writing outputs. Later requests with matching geometry and sampling settings can replace
prompt/seed values and reuse the program. HiDiffusion and capacity refusals leave the existing
streamed path available. Full-pipeline numerical parity, the production warm caller and
request reuse remain unqualified until their real release-gate runs pass. The existing
component and native-interval receipts do not prove this new path.

Whole-pipeline telemetry uses `native_pipeline` and `request_path.pipeline_native_steps`.
It does not fabricate separate text/denoise/VAE device durations or a first-noise metric.
CPU tokenization, input transfer, image verification and WebP/output delivery remain outside
the submitted GPU program. Tensor-only CLIP conditioning avoids Transformers' lazy output
collection hooks; its original 5.16.1 forward is the independent numerical oracle.
