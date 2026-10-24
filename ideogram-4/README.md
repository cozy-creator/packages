# Ideogram 4

Ideogram's 9.3B flow-matching DiT, trained from scratch on structured JSON captions: placed objects
and literal text by bounding box, colour palettes, 256–2048 px sides. It is a pair of DiTs: a
conditional one reads the caption beside the image tokens, a separate unconditional one the image
tokens alone, and both run every step ("asymmetric CFG"). The text encoder is a frozen
Qwen3-VL-8B-Instruct (13 of its layers' hidden states, concatenated); the VAE is FLUX.2's.

| function | checkpoints | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | `ideogram-4@0.1.0`, text encoder `qwen3-vl-8b-instruct@0.1.0` | 48 | 7, then 3 on the last `ceil(steps/16)` |

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
6:1 and 4 MP, on the 16 px grid, and anything outside is a `clamp` adjustment. Ideogram 4 takes no
references, so `reference_images` and `reference_boxes` must be empty; `negative_prompt` is ignored
with a warning (the unconditional DiT reads no text). Outputs: `image` (PNG), `width`, `height`, `seed`
and `prompt`, the caption the image was made from.

## Captions and regions

Ideogram 4 reads only JSON captions (`high_level_description`, `style_description`,
`compositional_deconstruction` with `background` and typed `elements`; Ideogram's
[prompting guide](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md)). A JSON
`prompt` is used as given. A plain one is expanded by the upsampler: Diffusers' local prompt enhancer,
the same Qwen3-VL with its language head, Ideogram's system message and the target ratio, sampled at
temperature 1 from the request's seed and constrained to the caption schema (`outlines`). With
`prompt_upsampling=false` a plain prompt becomes the caption's one-line description.

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

Ideogram publishes only weight-only fp8 (per-row scales on every linear, its own key layout);
TensorFS ingests it as the release's `fp8` lane with Diffusers' keys (tfs-134). The repository is gated:
accept Ideogram's licence on Hugging Face with the account whose token cozy uses.

```sh
cozy model upload hf://ideogram-ai/ideogram-4-fp8@ee79a7237b519f1402ceacf952f30c8a31ec5073 <account>/ideogram-4 --on <machine>
cozy model upload hf://Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b <account>/qwen3-vl-8b-instruct --on <machine>
cozy package install ./ideogram-4 --on <machine>
cozy model quantize <account>/ideogram-4 --recipe bf16 --on <machine>
cozy model quantize <account>/ideogram-4 --format int8 --on <machine>
cozy model quantize <account>/qwen3-vl-8b-instruct --format int8 --on <machine>
```

The `bf16` lane is the fp8 weights decoded exactly (f32 code × scale, rounded once). `int8` puts the
attention and SwiGLU linears of both DiTs' 34 blocks in `int8.convrot` and decodes the rest to bf16;
the text encoder's `int8` takes its 36 language layers' linears. The int8 lanes are the defaults; the
two DiTs are 37 GB at bf16 and 19 GB at int8, the text encoder 16 GB and 8 GB.

## Licence

Ideogram 4's weights are Ideogram's, under the Ideogram Non-Commercial Model Agreement
(`ideogram_4/LICENSE.model.md`, notice in `ideogram_4/NOTICE.model.md`); the inference code is
Apache-2.0. Use is limited to Non-Commercial Purposes (no revenue, or a company's non-production
testing and research), and offering the model as a hosted service is Distribution under the same
terms; commercial use needs a separate agreement with Ideogram. The agreement makes whoever serves it
responsible for content filters and AI-generated disclosure where law requires it, and forbids training
competing models on its outputs. Qwen3-VL-8B-Instruct is Apache-2.0. This wrapper grants no model rights.
