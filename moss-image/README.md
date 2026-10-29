# MOSS-Image

OpenMOSS's MOSS-Image, a text-to-image model known for rendering text, reading long prompts (up to
4,096 tokens) and placing objects and text in boxes. It is a 5B single-stream DiT over Qwen3.5-9B's last
hidden state, decoded by FLUX.2's VAE with its small decoder.

| function | checkpoint | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | MOSS-Image-SFT | 50 | 3.0 (CFG; the negative branch skips block 22) |

Upstream ships no distilled checkpoint, so there is no `generate_turbo`. MOSS-Image-Base, the LoRA
training base, runs with the same function: `model.model=moss-image-base@<release>`.

Write the request in a file, `request.yaml`:

```yaml
prompt: 'A sticky note on a fridge, handwritten: "Buy milk and eggs"'
```

```sh
cozy run moss-image/generate --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run moss-image/generate prompt='A sticky note on a fridge, handwritten: "Buy milk and eggs"'
cozy run moss-image/generate prompt='A minimalist travel poster, cream background' megapixels=1 \
  'regions:=[{"box": [0.1, 0.06, 0.9, 0.2], "prompt": "Large dark-blue uppercase title", "text": "EXPLORE"},
             {"box": [0.2, 0.32, 0.8, 0.9], "prompt": "A flat blue mountain illustration"}]'
```

Fields: `prompt`, `regions`, `aspect_ratio` (any `W:H`, default 1:1), `megapixels` (default 1), `steps`,
`guidance`, `negative_prompt` (default empty, as upstream's; upstream's user guide suggests
`nsfw, worst quality, bad anatomy, watermark, jpeg artifact`) and `seed` (random when omitted,
returned). MOSS-Image has no reference images and no prompt upsampler: `reference_images`,
`reference_boxes` and `prompt_upsampling` are accepted, warned about and ignored. Unknown fields are
dropped with a warning.

The result has the image, its `width` and `height`, the `seed`, and `prompt`: what the model read.

## Layout

Each region is `{box, prompt, text?}`. `box` is `[x0, y0, x1, y1]`, corners normalized 0–1 on the output
canvas from its top-left, the same box as every image package. `prompt` describes what goes in the box,
and `text`, when given, is the exact string to render there (`\n` breaks a line). With regions, the
prompt describes the whole scene.

The package compiles them into MOSS's own layout prompt, upstream's JSON form: the prompt as
`summary.caption`, each region an item labelled `text` (with its `text`) or `object`, and its box in pixels
from the canvas centre, `[y0, x0, y1, x1]`. As upstream's compiler does, left and top edges are floored
and right and bottom ceiled. The result's `prompt` is that JSON. Without regions the prompt reaches the
model as written, so a hand-written MOSS layout JSON works too. Upstream calls layout control
experimental.

## Sizes

The user guide recommends 1–2 MP and says the model extrapolates from 0.25 to 4 MP. `megapixels`
(1 MP = 1024² pixels) is held to 0.25–4, sides to 256–4096 px on the 16 px grid, and each clamp is
reported as an adjustment.

## Sampling

Upstream's (`pipeline_moss_image.py`, OpenMOSS-Team/MOSS-Image-SFT `f972bd4b`): flow-match Euler at shift
5, its sigmas snapped to the nearest of the 1,000 training sigmas, CFG 3 whose negative branch runs
without block 22, f32 noise from `seed` drawn on the CPU and f32 latents between steps. `guidance=1`
runs one forward per step. The text encoder reads the prompt through Qwen3.5's chat template with
thinking open; Qwen3.5's vision tower is never built or read.

## Execution

The denoise is one `denoise_loop`. Each prompt's text stream and RoPE are built once a request; with
CFG both branches pass through each DiT block in turn (their text lengths differ, so they are two
streams, as ComfyUI's two forwards). On a GPU a block runs as ComfyUI's MOSS-Image does: each modulated
RMSNorm and SiTU comes out of the int8 leaves' quantization pass, the q/k norms and split-half RoPE are
one kernel and each gated residual one `addcmul`, all prebuilt in the Runtime (no compile on a cold
request; the run log's `dit block` line says which served). The 42 DiT blocks attend at 8 bits
(comfy-kitchen INT8 on RTX 40-class cards), as Comfy-Org's int8 file declares, on every lane;
`--attention-kernel exact` runs them exactly. The VAE decodes
in bf16 where the GPU computes it natively. On the CPU, and for the bf16 lane's norms and SiTU, a block
is upstream's eager composition with each gated residual rounded once.

## Prepare the models

```sh
cozy model upload hf://OpenMOSS-Team/MOSS-Image-SFT@f972bd4b4b9e3852133cc5da954d87bc32cd8fb9 openmoss-team/moss-image --lane bf16 --on <machine>
cozy model quantize openmoss-team/moss-image@sha256:<bf16 commit> --lane int8 --package moss-image --on <machine>
```

The upload is the bf16 lane (the VAE stays f32). The `int8` lane (`int8.convrot`, W8A8 on int8 tensor
cores) takes the DiT blocks' q, k, v, o and FFN projections, as Comfy-Org's int8 file does, and every
projection of Qwen3.5's 32 language layers. The text refiner, embedders, modulation, token embedding
and VAE keep their bytes.

## Measured (one RTX 4090, 1 MP, 50 steps, CFG 3)

Against upstream's own pipeline at the same seed (bf16, its model-offload mode), the bf16 lane's images
are bit-identical in all three cases: plain text-to-image, text in boxes, and objects in boxes. The int8
lane reads 29.1, 31.0 and 37.9 dB against bf16, with the same pictures and legible text. Its encoded
weights equal Comfy-Org's int8 ConvRot files (per-weight error equal to 1e-10 over 118 sampled weights).

| s/step | bf16 | int8 |
|---|---|---|
| `generate` (CFG, two forwards) | 0.95–0.98 | 0.62 |

Upstream's pipeline took 57–63 s an image. The text encoder and the DiT take turns on the GPU: 15 and
9.5 GiB at bf16, 8.4 and 5.0 GiB at int8.

## Licence

MOSS-Image's weights and code are Apache-2.0 (`moss_image/LICENSE.model`). The DiT and sampler here
are adapted from upstream's `pipeline_moss_image.py`.
