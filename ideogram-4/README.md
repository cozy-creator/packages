# Ideogram 4

Ideogram's 9.3B flow-matching DiT, trained from scratch on structured JSON captions: placed objects
and literal text by bounding box, colour palettes, 256–2048 px sides. It is a pair of DiTs: a
conditional one reads the caption beside the image tokens, a separate unconditional one the image
tokens alone, and both run every step ("asymmetric CFG"). The text encoder is a frozen
Qwen3-VL-8B-Instruct (13 of its layers' hidden states, concatenated); the VAE is FLUX.2's.

| function | checkpoints | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | `ideogram-ai/ideogram-4@0.1.0`, text encoder `qwen/qwen3-vl-8b-instruct@0.1.1` (int8) | 48 | 7, then 3 on the last `ceil(steps/16)` |

There is no distilled checkpoint, so no `generate_turbo`; Ideogram's 20- and 12-step presets are
`steps=20` and `steps=12`.

Write the request in a file, `request.yaml`:

```yaml
prompt: A jazz festival poster with a saxophone made of brass leaves
aspect_ratio: "2:3"
```

```sh
cozy run ideogram-4/generate --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run ideogram-4/generate prompt='A jazz festival poster with a saxophone made of brass leaves' aspect_ratio=2:3
cozy run ideogram-4/generate prompt='A bakery storefront at dawn' \
  'regions=[{"box":[0.15,0.05,0.85,0.25],"prompt":"hand-painted gold sign lettering","text":"MAISON LUNE"}]'
```

Inputs are the shared image request: `prompt`, `aspect_ratio` (any `W:H`), `megapixels`, `steps`,
`guidance`, `seed` (random when omitted, returned) and `prompt_upsampling` (default on). Text to image
defaults to 1:1 at 1 MP (1024×1024, the reference CLI's); sizes are held to 256–2048 px sides, at most
6:1 and 4 MP, on the 16 px grid, and anything outside is a `clamp` adjustment. Ideogram 4 takes no `pictures` (refused before upload), so a `<Name>` in its prompt stays as written, with a warning ([references](../docs/references.md)). `negative_prompt` is ignored
with a warning (the unconditional DiT reads no text). Outputs: `image` (PNG), `width`, `height`, `seed`
and `prompt`, the caption the image was made from.

## Captions and regions

Ideogram 4 reads only JSON captions (`high_level_description`, `style_description`,
`compositional_deconstruction` with `background` and typed `elements`; Ideogram's
[prompting guide](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md)). A JSON
`prompt` is used as given. A plain one is expanded by the upsampler: Diffusers' local prompt enhancer,
the same Qwen3-VL with its language head, Ideogram's system message and the target ratio, sampled at
temperature 1 from the request's seed and constrained to the caption schema (`outlines`). A plain prompt
with `prompt_upsampling=false` is refused: the model reads only captions, and draws its own refusal card
on a bare description.

Ideogram 4's weights draw a flat grey "Image blocked by safety filter" card for some captions and seeds,
harmless ones included, at any precision (its post-training; Ideogram's own code shows the same). The
package does not retry around it. A run whose image looks like that card carries a warning; a more
detailed prompt or another seed usually renders.

`regions` places elements by box, in the shared box format: `box` is `[x0, y0, x1, y1]`, normalized
0–1, origin top-left. `prompt` describes the element; with `text` it is that literal text, rendered in
the style `prompt` gives. Each region is appended to the caption's elements as Ideogram's
`{"type": "obj" | "text", "bbox": [y_min, x_min, y_max, x_max], ("text"), "desc"}` on its 0–1000 grid.

## Fidelity

Conditioning, sampling and decoding are Diffusers' `Ideogram4Pipeline` (0.41.0), run per component:
the caption is read unpadded (Ideogram's own code; Diffusers left-pads to 2048 tokens), the logit-normal
schedule is shifted by the image size with the preset's (mu, std) nearest the step count, latents stay
float32 and noise is drawn on the CPU from the seed. With tiny random modules on CPU the package's text
features, latents and images equal the pipeline's exactly (three sizes and step counts), and its
upsampler's text equals the pipeline's `upsample_prompt`.

## Prepare the models

Ideogram publishes only weight-only fp8 (per-row scales on every linear, its own key layout), and nf4; no
16-bit release exists, so the package serves the official fp8 as Ideogram 4's only lane, with no bf16 or int8
written from it (owner, 2026-10-24). TensorFS ingests it with Diffusers' keys (tfs-134). fp8 needs an fp8
GPU (sm_89 and newer: RTX 4090, L40S, H100, Blackwell); 3090-class cards have no Ideogram lane yet. The
repository is gated: accept Ideogram's licence on Hugging Face with the account whose token cozy uses.

```sh
cozy model upload hf://ideogram-ai/ideogram-4-fp8@ee79a7237b519f1402ceacf952f30c8a31ec5073 ideogram-ai/ideogram-4 --lane fp8 --on <machine>
cozy model upload hf://Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b qwen/qwen3-vl-8b-instruct --lane bf16 --on <machine>
cozy model quantize qwen/qwen3-vl-8b-instruct@sha256:<bf16 commit> --lane int8 --package ./ideogram-4 --on <machine>
```

The two DiTs are 19 GB at fp8. The text encoder's `int8` (from Qwen's bf16, 16 GB to 8 GB) takes its 36
language layers' linears; Ideogram's own fp8 text encoder is exactly the per-row fp8 of stock Qwen3-VL-8B-Instruct
(checked byte for byte, 368 fp8 and 381 plain tensors), so stock Qwen serves as the encoder.

## Licence

Ideogram 4's weights are Ideogram's, under the Ideogram Non-Commercial Model Agreement
(`ideogram_4/LICENSE.model.md`, notice in `ideogram_4/NOTICE.model.md`); the inference code is
Apache-2.0. Use is limited to Non-Commercial Purposes (no revenue, or a company's non-production
testing and research), and offering the model as a hosted service is Distribution under the same
terms; commercial use needs a separate agreement with Ideogram. The agreement makes whoever serves it
responsible for content filters and AI-generated disclosure where law requires it, and forbids training
competing models on its outputs. Qwen3-VL-8B-Instruct is Apache-2.0. This wrapper grants no model rights.
