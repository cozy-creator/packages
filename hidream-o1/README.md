# HiDream-O1-Image

HiDream.ai's HiDream-O1-Image: an 8B pixel-space Unified Transformer for text-to-image generation
and editing. It is Qwen3-VL-8B with a 32 px patch embedder, a timestep token and a pixel head, and
has no VAE and no separate text encoder. Text and condition tokens attend causally, and the noisy
patches attend to everything (tech report arXiv:2605.11061 §3.3).

| function | checkpoint | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | HiDream-O1-Image | 50 | `guidance` 5.0 (CFG) |
| `generate_turbo` | HiDream-O1-Image-Dev (distilled) | 28 (its only schedule) | none |

One function does both jobs: with `references`, it edits them, or composes several in order.
HiDream recommends the full model, `generate`, for editing. A pose is one more reference: upstream's
try-on example passes a face, a background, an OpenPose skeleton and three garments, in that order.

Write the request in a file, `request.yaml`:

```yaml
prompt: 'A dog holds a sign that says "HiDream-O1-Image release."'
```

```sh
cozy run hidream-o1/generate_turbo --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run hidream-o1/generate_turbo prompt='A dog holds a sign that says "HiDream-O1-Image release."'
cozy run hidream-o1/generate prompt='remove the earphones' references:='[{"name": "Photo", "image": "photo.jpg"}]'
```

Inputs are the shared image request, the same in every image package: `prompt`, `references`, `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `steps`, `guidance`, `negative_prompt`, `seed` and `prompt_upsampling`. `references` are up to 10, and a reference's `box` places it (below). HiDream-O1 has no words of its own for its references (upstream's prompts describe them, "the provided clothing"), so a `<Name>` mention reads as the name itself ([references](../docs/references.md)). HiDream-O1 adds `guidance_interval` (below). Left out, each
is the model's default: 50 steps (1–200) at guidance 5.0 (1–20) for `generate`, upstream's unconditional
caption (a single space) as the negative prompt, and a random seed. `generate_turbo` runs its 28 distilled
steps and ignores `guidance`, `negative_prompt` and `guidance_interval` with a warning; neither has a prompt
upsampler. Unknown fields are dropped with a warning.

The result has the image, its `width` and `height`, the `seed`, and `prompt`: the caption used.

## Placing references

A reference's `box` is `[x0, y0, x1, y1]`, corners from 0 to 1 on the output, origin top-left; a
reference without one is placed by the model itself. This is upstream's layout
pipeline (`--layout_bboxes`), exactly: each reference is framed in its box's colour, and a black canvas
with the coloured boxes follows the references as one more. Only the five largest boxes are drawn.
Placed references are elements, not the canvas: the output is 1:1 at 4 MP unless `aspect_ratio` or
`megapixels` says otherwise, and the full model samples at shift 1.

Placement is soft. HiDream-O1 matches references to boxes mostly by their order, not by the frame
colours, so list the references in the boxes' left-to-right order. In three two-person tests that
asked for the opposite of the reference order, the right person landed in each box once in three as
upstream orders them, and twice in three with the references sorted by their boxes. Upstream's own code
puts the same people in the same wrong boxes, and its examples only place references in their own order.
Our layout inputs are pixel-identical to upstream's, and a single pass matches its code at 40–57 dB.

```sh
cozy run hidream-o1/generate prompt='City council members pose on a sunlit terrace' \
  references:='[{"name": "A", "image": "a.jpg", "box": [0.205, 0.488, 0.439, 0.742]},
                {"name": "B", "image": "b.jpg", "box": [0.576, 0.088, 0.801, 0.342]}]'
```

## Sizes

The model was refined at 2048² (report §4.1), and upstream serves only about 4 MP, from 2560×1440 to
2048². Text-to-image defaults to 1:1 at 4 MP, 2048×2048. `megapixels` (1 MP = 1024² pixels) is held to
3.52–4: at 1024² and below the model falls apart (upstream issue #9 shows pure noise at 800²). Sides
are 32 px multiples from 256 to 4096, and a ratio past that is clamped. Each clamp is reported as an
adjustment.

With references, any size input you leave out comes from the first reference: its ratio, its pixel
count, or both. With one reference and neither input, the output takes the reference's own size on the
32 px grid. This is upstream's `keep_original_aspect`: the reference is resized to that area, and the
output matches it. Otherwise each reference is resized to the area of a square on the output's longer
side, smaller when there are more references (upstream's `generate_image`).

## Sampling

All recipes are upstream's latest (HiDream-ai/HiDream-O1-Image `main` at 2c2d29f, `inference.py`):

- **`generate`:** Wan's flow UniPC (bh2, order 2) at shift 3, with CFG against a single-space caption.
  With two or more references, or boxes, it uses shift 1, as upstream's README runs subject-driven
  generation.
  The initial noise is 8 standard normals per pixel. CFG stops for the last 10 of the 50 steps
  (`guidance_interval`, limited-interval guidance; `false` keeps upstream's CFG to the end).
- **`generate_turbo`:** the 28 fixed Dev timesteps. Text-to-image and two or more references use the
  flash sampler: predict the clean image, then noise it again to the next sigma, at noise scale 7.5
  clipped at 2.5 standard deviations. A one-reference edit uses flow-match Euler, upstream's editing
  scheduler for Dev since 2026-05-12.

As in upstream, the noise comes from `seed + 1`, and the state between steps is bf16. The vision tower
encodes each reference at 384² (288² for 5–8 references, 192² for more). Its tokens go into the prompt,
and its deepstack features are added after the first three layers, as upstream added on 2026-05-13.
ComfyUI's port drops them. The text and condition tokens' keys and values are computed once per request.

`scripts/hidream-o1-model-proof.py` checks the cached forward against Transformers' own Qwen3-VL text
model over the whole sequence under the hybrid mask. It also checks the UniPC timesteps against
upstream's.

## Prepare the models

Upstream ships f32 weights (35 GB per checkpoint). Upload each repository as it is:

```sh
cozy model upload hf://HiDream-ai/HiDream-O1-Image@0b0901d99f200389e138c61946af1185f5f49a13 hidream-ai/hidream-o1 --lane f32 --on <machine>
cozy model upload hf://HiDream-ai/HiDream-O1-Image-Dev@c0bada0e15c54a9f96a6d1ecc35575b32bc21544 hidream-ai/hidream-o1-dev --lane f32 --on <machine>
```

Each lane is written from the f32 upload. `bf16` rounds every f32 tensor to bf16. `int8` (the default
lane, `int8.convrot`, W8A8 on int8 tensor cores) takes the q, k, gate and up projections of the 36
decoder layers: 63% of the decoder's linear weights. v, o and down stay bf16, as do the patch embedder,
timestep MLP, pixel head, embeddings and vision tower. With every projection in int8, the full model's
CFG 5 turns W8A8's per-pass error (33–37 dB teacher-forced) into visible grain (17–24 dB after CFG).
Keeping v, o and down in bf16 was the best of nine measured keep-lists (23/28/36 dB at t 999/702/290).

```sh
cozy model quantize hidream-ai/hidream-o1@sha256:<f32 commit> --lane bf16 --package hidream-o1 --on <machine>
cozy model quantize hidream-ai/hidream-o1@sha256:<f32 commit> --lane int8 --package hidream-o1 --on <machine>
```

## Measured (one L40S, 4 MP)

Against HiDream's own code at the same seed (bf16), both edits match: 38.6 dB full, 33.9 dB Dev.
Text-to-image keeps the same image, but details drift: 29.9 dB full, 22.1 dB Dev. In a single
teacher-forced pass the full model matches upstream at 45–57 dB, so CFG and Dev's 28 re-noisings amplify
rounding differences. The int8 lane against bf16 at the same seed reads 17.6 dB on text-to-image (the
details differ) and 21.7 dB on an edit, with no grain visible at 1× or 2×.

| s/step | bf16 | int8 |
|---|---|---|
| `generate`, text-to-image (CFG) | 0.81–0.88 | 0.67 |
| `generate`, one-reference edit (CFG) | 1.97 | 1.65 |
| `generate_turbo`, text-to-image | 0.44 | 0.31 (all-projection lane) |

Upstream's own code ran at 1.02 and 0.51. The first step adds 1.4–3.5 s. Black dots in some edits
(upstream #13) appear in upstream's own images too.

## The 32 px grid

With CFG 5 to the last step, the full model's flat areas show a grid of slightly miscoloured 32 px
patches (upstream #1), in upstream's own images as much as ours. Each patch's pixel-head output
carries a small colour error, and CFG multiplies it while the colours settle. Without CFG over the last
fifth, the patch offsets on flat areas fall 4–5× (1.6 to 0.3 levels on white) and the grid is at Dev's
level, 10% faster. Faces lose the blotches and keep their detail. The edge step over
the mid-patch step on flat areas, on one 4090 at 4 MP (int8):

| | upstream (CFG to the end) | `guidance_interval` | also rolled half a patch | Dev |
|---|---|---|---|---|
| two-reference sheet, white | 1.78 | 1.20 | 1.05 | |
| text-to-image, white | 1.65 | 1.18 | 1.05 | 1.13 |
| edit, grey backdrop | 1.70 | 1.25 | 1.07 | |

The last column comes from adding ComfyUI's patch-seam smoothing (`HiDreamO1PatchSeamSmoothing`) on
top. Over the same last fifth it also predicts on the image rolled 16 px and averages the two. It reaches
1.05–1.07 at upstream's speed. It is not shipped; it is the stronger option if a case needs the margin.
ComfyUI's smoothing alone halves the steps but moves half of them to 16 px, and CFG-rescale changes the
image without removing the grid.

## Licence

HiDream-O1-Image and its code are MIT (`hidream_o1/LICENSE.model`, Copyright (c) 2026 HiDream.ai).
The UniPC solver follows Wan's (Apache-2.0).
