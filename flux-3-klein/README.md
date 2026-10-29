# FLUX.3 Image Klein

Black Forest Labs' FLUX.3 Image Klein: a 5.6B DiT for text-to-image generation and editing with up
to ten reference images. Its text encoder is Qwen3-VL-4B.

| function | checkpoint | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | Klein Base | 50 | `guidance` 4 (CFG) |
| `generate_turbo` | Klein (distilled flow map) | 16 | none |

Each makes an image from the prompt, or edits and composes up to ten ordered `reference_images`.

Write the request in a file, `request.yaml`:

```yaml
prompt: A lighthouse on a basalt cliff in a winter storm
aspect_ratio: "16:9"
seed: 7
```

```sh
cozy run flux-3-klein/generate_turbo --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run flux-3-klein/generate_turbo prompt='A lighthouse on a basalt cliff in a winter storm' \
  aspect_ratio=16:9 seed=7
cozy run flux-3-klein/generate prompt='Make it night, the lamp lit' \
  --asset reference_images.0=lighthouse.png
```

Inputs are the shared image request, the same in every image package: `prompt`, `reference_images`, `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `steps`, `guidance`, `negative_prompt`, `seed` and `prompt_upsampling`. Klein adds `reference_boxes`, `regions` (below) and `prompt_drafting`. Left out, each
is the default in the table, no negative prompt (Base's CFG), upsampling on and a random seed; steps run
1–200 (Base) or 1–100, guidance 1–20, and `generate_turbo` ignores `guidance` and `negative_prompt` with a
warning. Without references the default is 1:1 at 1 MP (1024×1024, BFL's `default_size`); the size is
that area at that ratio, rounded to 32 px. With references, what is not given is the first reference's: given neither,
its own size (fitted to at most 4 MP, generated on the 16 px grid and cropped back); given one, its ratio
or its pixel count. Sides stay within BFL's 64–4096 px; a size moved to fit is reported as a `clamp`
adjustment. The result returns the image, its size, the seed and `prompt`: the caption the image was made
from.

## Layout boxes

Klein places elements by box natively: it is trained on captions that cite each element as `<id>` and
end with a JSON table of boxes (BFL's `flux3/bbox.py`). A box is `[x0, y0, x1, y1]`, 0–1 of the output,
origin top-left.

- `reference_boxes`: one entry per reference, in order, where it goes in the output; `null` places none.
  When any reference is placed, the references are elements, not the canvas: the size defaults to
  text-to-image's (1:1 at 1 MP).
- `regions`: `[{box, prompt, text?}]`, elements described in words; `text` is the literal string to render
  there.

```sh
cozy run flux-3-klein/generate_turbo prompt='A picnic on a hillside' \
  'regions:=[{"box": [0.05, 0.55, 0.45, 0.95], "prompt": "a wicker basket"},
             {"box": [0.1, 0.05, 0.9, 0.2], "prompt": "a hand-painted banner", "text": "SUMMER FAIR"}]'
cozy run flux-3-klein/generate_turbo prompt='The mug on a desk by a window' \
  --asset reference_images.0=mug.png 'reference_boxes:=[[0.55, 0.5, 0.85, 0.9]]'
```

The boxes become table rows on BFL's 0–1000 canvas (`[y_min, x_min, y_max, x_max]`). The upsampler then
writes the caption around them with BFL's `t2i_bbox`/`i2i_bbox` prompts, adds rows for what else the scene
holds, and the given boxes are kept exactly. With `prompt_upsampling=false` the rows are appended to the
prompt as typed. A prompt that already is a layout caption (its whole table cited as `<id>`s) goes to
Klein as it is. Boxes guide placement and scale; they are not hard masks.

## Prompt upsampling

BFL's prompt upsampler is a Qwen3.5-9B fine-tune. It turns a short request, plus the references when
editing, into the full caption Klein is trained on. It runs first, as its own memoized stage, with
BFL's system prompts and decoding: greedy, presence penalty 1.5 for text-to-image and 0 for edits.
Text-to-image also tells it the canvas (`TARGET CANVAS: 16:9`). Set `prompt_upsampling=false` to send
the prompt as typed, which then should be a full scene caption. Its token loop declares
`min_bits=4, phase="decode"`, so it serves the release's `nvfp4` lane where there is one.

## Sampling

The sampling follows BFL's reference code (`black-forest-labs/flux-3-image-klein` at `bfc8889`): Euler
steps with a constant shift (alpha) of 16, CFG 4 for the base model, and noise from the seed in bf16
with an fp32 state across steps. BFL raised alpha from the release's 6.93 and CFG from 3.5 on
2026-10-17; Diffusers and ComfyUI still default to 6.93.
The distilled model also embeds the timestep each step jumps to. The base model runs CFG with an empty
negative prompt. A lone reference is padded to the 16-pixel grid at up to 4 MP, with no upscaling.
With several references, each is cropped to the grid at up to 1 MP. The modules are Diffusers'
(main `7ecb1e1b`; Flux3 is in no release yet).

## Prepare the models

Upload each repository as it is. The served checkpoint is the upload: TensorFS takes a Diffusers repo's
index components (not BFL's `variants/`) and a Transformers repo's config, tokenizer, processor and chat
template. Each upload is lane `bf16`.

```sh
cozy model upload hf://black-forest-labs/flux-3-image-klein@8454017f9ebaada30c998fccd9f67a80c7580f68 \
  flux-3-klein --lane bf16 --on <machine>
cozy model upload hf://black-forest-labs/flux-3-image-klein-base@1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4 \
  flux-3-klein-base --lane bf16 --on <machine>
cozy model upload hf://black-forest-labs/flux-3-image-klein-pup@8b069751eb5dcab23829d9c5eccb470eb8b8a4ed \
  flux-3-klein-upsampler --lane bf16 --on <machine>
```

Klein never builds Qwen3-VL's vision tower, its last four layers or its head, so it never loads those
tensors either. BFL's upsampler system prompts ship in `flux_3_klein/system_prompts.json`.

## The int8 lane

`int8` writes the default lane: every linear of the 43 DiT blocks, the modulation heads, the context
embedder, the output norm and the text encoder's 32 language layers as `int8.convrot`, served
W8A8 on the int8 tensor cores. On real Klein activations each linear is 0.5–1.3% from bf16 (fp8
row-scaled: 1.5–2.8%), and the images match bf16's. The VAE stays f32. As in BFL's int8 release,
the image and joint blocks declare 8-bit attention (`attention_bits`; Runtime's `kitchen-int8` on
RTX 30/40); the text and reference early blocks, and `--lane bf16`, keep exact attention.

```sh
cozy model quantize black-forest-labs/flux-3-klein@sha256:<bf16 commit> --lane int8 --package flux-3-klein --on <machine>
cozy model publish black-forest-labs/flux-3-klein 0.1.0 sha256:<bf16 commit> sha256:<int8 commit>
```

Do the same for `flux-3-klein-base` and `flux-3-klein-upsampler`: each lane is one recipe that reads
which checkpoint it was given (`--lane nvfp4` writes the 4-bit lane the same way). Warm 1 MP seconds per step (16-step distilled; 50-step base
with CFG), and the weights the lane holds:

| card | distilled bf16 → int8 | base bf16 → int8 | weights |
|---|---|---|---|
| RTX 5090 | 0.245 → 0.146 | 0.496 → 0.295 | 17.6 → 9.4 GiB |
| RTX 4090 | 0.390 → 0.255 | 0.781 → 0.506 | |

## Licence

FLUX.3 Image Klein is under the FLUX Kommunity License v1.0 (`flux_3_klein/LICENSE.model.md`,
`NOTICE.model.md`): non-commercial use, plus commercial outputs for Qualifying Users under the terms
there. Its Qwen3-VL text encoder and the prompt upsampler are Apache-2.0. This wrapper grants no model
rights.
