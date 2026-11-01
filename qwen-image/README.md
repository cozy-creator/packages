# Qwen Image

Qwen's 20B dual-stream DiT with a Qwen2.5-VL-7B text encoder. One function generates from text with
Qwen-Image-2512, or edits and composes `pictures` with Qwen-Image-Edit-2511. The two
checkpoints store the same text encoder and VAE byte for byte, so one copy serves both, and only the
DiT changes between requests.

| function | without references | with references | default steps | guidance |
| --- | --- | --- | --- | --- |
| `generate` | `qwen/qwen-image-2512@0.1.1` | `qwen/qwen-image-edit-2511@0.1.0` | 50 / 40 | true CFG 4.0 |
| `generate_turbo` | the same, with `lightx2v/qwen-image-2512-lightning-lora@0.1.0` | the same, with `lightx2v/qwen-image-edit-2511-lightning-lora@0.1.0` | 8 | none |

Write the request in a file, `request.yaml`:

```yaml
prompt: A red vintage bicycle against a pastel wall
aspect_ratio: "3:2"
```

```sh
cozy run qwen-image/generate --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run qwen-image/generate prompt='A red vintage bicycle against a pastel wall' aspect_ratio=3:2
cozy run qwen-image/generate prompt='Make <Guitarist> play a violin instead of the guitar' \
  pictures:='[{"name": "Guitarist", "image": "guitarist.png"}]' seed=42
cozy run qwen-image/generate_turbo prompt='<Cat> wearing the space suit from <Astronaut>' \
  pictures:='[{"name": "Cat", "image": "cat.jpg"}, {"name": "Astronaut", "image": "astronaut.jpg"}]'
```

Inputs are the shared image request: `prompt`, `pictures` (ordered, up to ten; Qwen recommends
one to three; each `<Name>` mention reads as Qwen-Image-Edit's own "Picture N", the label its template
writes before each image, [references](../docs/references.md)), `aspect_ratio`
(any `W:H`), `megapixels` (1024² pixels each), `steps`, `seed` (random when omitted, returned).
`generate` also takes `guidance` and `negative_prompt`; by default each model card's: Qwen-Image-2512's
Chinese quality negative, and Qwen-Image-Edit-2511's single space. A picture's `box` and
`prompt_upsampling` are accepted and ignored with a warning: the model has no layout control or
upsampler. With pictures, a size input you leave out is the first picture's. Unknown fields are ignored with a warning. Outputs: `image` (PNG), `width`, `height`,
`seed` and `prompt`.

## Slots

`model` is a whole checkpoint: the text-to-image DiT, the text encoder, the VAE and the tokenizer.
`edit_model` binds only the edit checkpoint's DiT, and the first request with references fetches and
builds it (and `edit_turbo_lora`). Runtime keeps each DiT's residency: going from one to the other
pages weights, it does not reload them. `--lane` picks the lane of `model`; the edit DiT takes the
machine's default lane.

## Size

Text to image defaults to Qwen-Image-2512's 1328×1328 (1.68 MiP); an explicit `aspect_ratio` keeps that
area. With references and neither size input, the output takes the first reference's aspect ratio and
pixel count (upstream takes the last reference's ratio at 1 MiP). Sides are rounded to the 16 px grid.
The model serves 256² to 2 MiP with sides from 128 to 4096 px; anything outside is clamped and
reported as a `clamp` adjustment.

## Fidelity

Conditioning and sampling are Diffusers' `QwenImagePipeline` and `QwenImageEditPlusPipeline` (0.41.0),
Qwen's reference code: with tiny random modules the package's output equals both pipelines exactly.
Text alone is read with the text-to-image template, at most 512 tokens. Each reference goes to the
vision encoder at a 384² area and to the VAE at a 1024² area, and its latents sit beside the target
at t = 0 (`zero_cond_t`). Noise is seeded on the CPU, the timesteps are `linspace(1, 1/steps)` with
the checkpoint's dynamic shift, and true CFG keeps the conditional prediction's norm. References are
read upright, with any transparency composited over white. `generate_turbo` binds lightx2v's 8-step
Lightning LoRA (V1.0, fp32) for each DiT at the file's `alpha / rank` (8/64) and samples at
lightx2v's constant shift of 3. Runtime bakes the LoRA into the weights of whichever lane serves the
DiT; the merged weights exist only on the GPU.

## Quality and speed (Qwen-Image-Edit-2511, A40, warm)

Same seed and size as Diffusers' bf16 pipeline: the bf16 lane is the same image (61.0 dB on a
one-reference edit, 55.4 dB on a two-reference compose); the Lightning LoRA per step is 30.5 dB from
Diffusers' 8-step Lightning, and the default bake is visually the same image. The int8 lane
re-samples the same image (18–26 dB from bf16) and keeps every reference's identity in
two-reference composes.

| request | bf16 | int8 |
| --- | --- | --- |
| `generate`, one 0.5 MP reference, 40 steps | 3.1 s/step | 2.7 s/step |
| `generate`, two references, 0.9 MP out, 40 steps | 6.6 s/step | 5.5 s/step |
| `generate_turbo`, 8 steps | | 13–23 s sampling |

## Lanes

The `int8` recipe writes the default lane: every linear of the 60 DiT blocks, their modulation heads
included, as `int8.convrot` (W8A8). The text encoder stays bf16, as in v1's 8-bit lane: with it int8
too, a two-reference compose lost the first reference's identity. `nvfp4` (`--lane nvfp4`, Blackwell) uses
NVFP4 for the blocks' attention and MLP linears and int8 for the modulation heads. Both are made from
the bf16 upload of either checkpoint, and one release names all three lanes.

```sh
cozy model upload hf://Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26 \
  qwen-image-2512 --lane bf16 --on <machine>
cozy model upload hf://lightx2v/Qwen-Image-2512-Lightning@4609161e308ce82018a42c2a0c9b107e865bb751/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors \
  qwen-image-2512-lightning-lora --lane f32 --on <machine>
cozy model quantize qwen/qwen-image-2512@sha256:<bf16 commit> --lane int8 --package qwen-image --on <machine>
cozy model quantize qwen/qwen-image-2512@sha256:<bf16 commit> --lane nvfp4 --package qwen-image --on <machine>
cozy model publish qwen/qwen-image-2512 <version> sha256:<bf16 commit> sha256:<int8 commit> sha256:<nvfp4 commit>
```

The edit checkpoint is `hf://Qwen/Qwen-Image-Edit-2511@6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`, its
LoRA `hf://lightx2v/Qwen-Image-Edit-2511-Lightning@107d238d68b8387e32843d0b4a2e24ceb0d3ad86/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors`.

## Licence

Qwen-Image-2512, Qwen-Image-Edit-2511 and both Lightning LoRAs are Apache-2.0. The architecture is
Diffusers' (Apache-2.0), at the release `pyproject.toml` pins.
