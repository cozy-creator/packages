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

Local inference uses Linux and NVIDIA CUDA. The default `int8` model lane is about
3.8 GiB; the `bf16` catalog lane stores F16 weights and is about 6.5 GiB. Actual fit
also depends on activations, graph scratch and the selected image dimensions. Rentals
remain available through `--on new`.

Its recipes write the model's `fp8`, `int8` and `w4a8` lanes, for model authors
(`cozy model quantize <model> --lane int8 --package sdxl`).

SDXL 0.1.1 requires Runtime 0.34 or newer and TensorFS 0.17 or newer. These floors
cover the native sequence, source-stage and graph-pool APIs used by this package;
older SDKs must not be selected merely because their version accepted an earlier
package release.

For plain epsilon Euler, the package can load missing UNet blocks and run the complete
denoising interval through native stages at the model's existing virtual addresses.
The standalone interval shares one owned graph scratch pool. Native code advances the
stages and repeated steps; Python observes progress. HiDiffusion, unsupported layouts
and safe preparation refusals retain their existing execution paths. A failure after
native submission is propagated and never reruns the image automatically.

An admitted preparation can also build a reusable sequence covering both text encoders,
all Euler steps and VAE decode. A construction retains one variant. Prompt and seed
values update fixed buffers; changed geometry, schedule, guidance, weight representation
or Runtime invalidation closes the old definition. The wide typed VAE representation
remains resident through this sequence. Its phase graphs currently use separate scratch
pools; the later whole-phase pool refinement is not included in this release.

`ResidentPipeline.prepare` is the preparation entry for an admitted warm turn. The
ordinary request path reuses a valid program or captures only already-resident
representations; it does not speculatively load the entire pipeline. The public warm
command accepts model and lane choices, but no request payload or `--gpus`:

```sh
cozy run sdxl/generate --warm gpu --on <machine>
```

This warms the function's default variant. Do not treat it as proof that a different
30-step, shape or guidance variant has already been prepared. Matching requests reuse
the prepared buffers; capacity refusal leaves the existing denoising path available.
Idle pressure can retire a cached program and its boundary buffers after checked
completion. A later request prepares a fresh definition as needed.

The merged implementation has real CUDA proof for full-size 1024-square, 30-step streamed
denoising on an 8 GiB device; the complete phase sequence has reduced-model execution,
changed-input reuse and pressure-retirement proof. These are distinct scopes. They do
not establish a full-pipeline fit at every image size or a ComfyUI performance advantage.

Telemetry reports `native_pipeline` and `request_path.pipeline_native_steps` for a
complete sequence, or native denoising metrics for an interval. It does not invent
per-step GPU timings from polling. CPU tokenization, input transfer, image validation,
WebP encoding and output delivery remain outside the native GPU program.
