# Fibo 2.0

Bria's Fibo 2.0: one 8B flow-matching DiT for text-to-image and image editing, trained on structured
JSON captions. It has 44 single-stream blocks over the FLUX.2 VAE's latents. Qwen3-VL-4B encodes the
prompt and the images to edit, and a Perceiver turns it into 256–512 gist tokens; five blocks also
cross-attend to Qwen3-VL's inner layers. It ranked #1 for editing and #2 for text-to-image among
open-weights models on Artificial Analysis (2026-10-20).

| function | checkpoint | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | Fibo 2.0 Dev | 50 | `guidance` 5.0 (CFG), `negative_prompt` |
| `generate_turbo` | Fibo 2.0 Edit Turbo, distilled from Dev | 4 | none |

Both generate from text, or edit when `reference_images` are given (up to four, `<image_1>`…`<image_4>`).
Bria ships Edit Turbo for editing; it also renders text-to-image.

Write the request in a file, `request.yaml`:

```yaml
prompt: An origami whale swimming through paper clouds
aspect_ratio: "16:9"
seed: 7
```

```sh
cozy run fibo-2/generate --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run fibo-2/generate prompt='An origami whale swimming through paper clouds' aspect_ratio=16:9 seed=7
cozy run fibo-2/generate_turbo prompt='Replace the jacket with a red leather one' \
  --asset reference_images.0=photo.jpg
```

Both take the shared image request (`ImageRequest`). Unset knobs take Fibo's defaults: `steps` 50 (Dev) or 4
(Turbo), `guidance` 5.0 and `negative_prompt` Bria's (Dev only; Turbo warns and ignores them), and
`prompt_upsampling` on. A `seed` is drawn when omitted and returned. Fibo 2.0 was trained at about 1 MP on a
16 px grid (model card). Text-to-image defaults to 1:1 at 1024×1024. With references, the output takes the
first one's aspect ratio and pixel count; `aspect_ratio` or `megapixels` replaces only that one. Sizes are
held to 0.25–1 MP and 64–8192 px sides, steps to 200 (Dev) or 50 (Turbo) and guidance to 1–20, each move a
`clamp` adjustment. A lone reference is cropped to the output's size; several are each cropped to about 1 MP
at their own ratio (Diffusers). The result is `image`, its size, the `seed` and `prompt`: the prompt the
image was made from.

## Prompts

Fibo 2.0 reads a JSON caption (Bria's VGL), a short description, or for an edit the instruction alone.
A JSON `prompt` is used as given; an edit's needs an `edit_instruction`. Otherwise
`prompt_upsampling` writes the caption with Bria's prompt-to-JSON VLMs, the local route Diffusers
points to (Bria's own API path, `bria-vgl`, calls Gemini):

- text-to-image: `briaai/FIBO-vlm` (`<generate>`);
- one reference: `briaai/FIBO-edit-vlm` (`<edit>`, the image thumbnailed to 1024 px), whose restated
  request (`reworded_prompt`) becomes the caption's `edit_instruction`;
- several references: neither VLM reads more than one image, so the instruction goes as typed, with a
  warning.

Both are Qwen3-VL-4B fine-tunes, decoded greedily (Bria samples at temperature 0.2/0.4), each fetched
on first use. `prompt_upsampling=false` sends the prompt as typed. The returned `prompt` reproduces
the image with `prompt_upsampling=false` and the same seed.

## Sampling

Diffusers' `BriaFibo2Pipeline` (main `48cdbaef`, #14991, verified by Bria against its reference code),
run per component. Euler from sigma 1 to 1/steps, shifted by the image size; float32 latents; noise
from the seed on the CPU. Dev runs CFG 5 at 50 steps (the model card; Diffusers' default is 30), with
the card's text-to-image negative prompt, or the empty caption the model was trained with for edits.
Turbo runs 4 steps at guidance 1. The caption's text is also drawn and shown to Qwen3-VL, in the fonts
training used (`fibo_2/fonts`). On an L40S, its 16-bit images are 37–54 dB PSNR from Diffusers'
own bf16 pipeline (same caption, seed and size, five cases: runs 14911–14923 against 14925/14926); the
int8 lane's are 22.6–27.6 dB from them, the same pictures by eye (14929–14951). The VAE's BatchNorm
statistics need Runtime ≥ 0.30.16 (cozy-runtime #1700).

## Prepare the models

Upload each repository as it is, as lane `bf16`, and publish it with its quantized lanes (below).
The Fibo repositories are gated: accept Bria's licence on Hugging Face first.

```sh
cozy model upload hf://briaai/Fibo-2.0-dev@fb2b781f9e9e5572a6c348a5af7a4d155a2b1ed0 briaai/fibo-2-dev --lane bf16 --on <machine>
cozy model upload hf://briaai/Fibo-2.0-edit-turbo@901c02672d969ed21e44ff58f1a0872c7581ddb0 briaai/fibo-2-edit-turbo --lane bf16 --on <machine>
cozy model upload hf://briaai/FIBO-vlm@bd13da5e0b73c808014d6260963e311c8cdf0d8b briaai/fibo-vlm --lane bf16 --on <machine>
cozy model upload hf://briaai/FIBO-edit-vlm@61d852ebd380ac79e04ec171552f5d7cca1a04cc briaai/fibo-edit-vlm --lane bf16 --on <machine>
```

## Quantized lanes

`int8` writes the default lane from a 16-bit commit: the DiT's attention linears and FFN gate/up
projections (blocks, noise refiner, Perceiver) and Qwen3-VL's language linears as `int8.convrot`, W8A8
on the int8 tensor cores. The FFN down projections (9,536 columns, not a multiple of ConvRot's 256),
adaLN modulation, embedders, final layer, vision tower and VAE keep their bytes. For a VLM it takes the
language linears, 0.86–1.04% each from bf16. Every linear both quantize has the error of Comfy-Org's
`fibo2_dev_int8_convrot` and `qwen3vl_4b_int8_convrot` to 1e-8 (same encoding; se-075 run 14879). Comfy
also pads the FFN to 9,728 to quantize its down projection (xs-070). `nvfp4` puts the DiT's 44 blocks, down projections
included, in NVFP4 (Blackwell), and a VLM's linears in NVFP4 (W4A16 decode).

```sh
cozy model quantize briaai/fibo-2-dev@sha256:<bf16 commit> --lane int8 --package fibo-2 --on <machine>
cozy model publish briaai/fibo-2-dev 0.1.0 sha256:<bf16 commit> sha256:<int8 commit>
```

## Licence

Fibo 2.0, FIBO-vlm and FIBO-edit-vlm are Bria AI's, under CC BY-NC 4.0 (`fibo_2/LICENSE.model.md`):
non-commercial use; commercial use needs an agreement with Bria. This wrapper grants no model rights.
