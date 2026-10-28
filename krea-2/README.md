# Krea 2

Krea's open text-to-image model: a 12.8B single-stream DiT conditioned on twelve hidden states of
Qwen3-VL-4B, decoded by the Qwen-Image VAE. It ships as two checkpoints, and the package has a function
for each:

| function | default model | default steps | guidance | default size |
| --- | --- | --- | --- | --- |
| `generate` | Krea 2 Raw (`krea-2`) | 52 | 3.5 (CFG) | 1024² |
| `generate_turbo` | Krea 2 Turbo (`krea-2-turbo`) | 8 | none | 2048² |

Krea recommends training LoRAs on Raw and running them on Turbo. The function sets the regime: `generate`
samples as Raw (CFG, the token count's shift), `generate_turbo` as Turbo (no CFG, shift 1.15). The two
checkpoints have identical configs, so pass a fine-tune of each to its own function (`model.model=`).

Write the request in a file, `request.yaml`:

```yaml
prompt: A tiny harvest mouse clinging to a slender branch, macro photograph, creamy bokeh
aspect_ratio: "3:2"
```

```sh
cozy run krea-2/generate_turbo --input request.yaml
cozy run krea-2/generate prompt='a red fox in fresh snow at golden hour' seed=7
cozy run krea-2/generate_turbo --input request.yaml --lora model:transformer=<org/krea-2-lora>,0.8
```

Fields: `prompt`, `aspect_ratio` (any `W:H`, default 1:1), `megapixels` (Raw 1, Turbo 4), `steps` (1–100),
`guidance` (0–20; 0 runs no negative branch), `negative_prompt` (default empty, as Krea's) and `seed`
(random when omitted, returned). Krea 2 makes images from text only and has no prompt upsampler:
`reference_images` is refused, `reference_boxes` and `prompt_upsampling` are warned about and ignored.
Unknown fields are dropped with a warning. The result is `image`, `width`, `height`, `seed` and `prompt`.

Guidance is Krea's: `cond + g·(cond − uncond)`, so Krea's 3.5 is ComfyUI's cfg 4.5, and Turbo's 0 is
ComfyUI's cfg 1.

## Sizes

Each checkpoint defaults to the most Krea gives it: Raw is trained up to 1 MP, Turbo makes 1–4 MP and Krea
runs it at 2048². `megapixels` is held to 0.25–4, sides to 256–4096 px on the 16 px grid, and each clamp is
reported as an adjustment. The decode runs whole, or in 512 px tiles (256 px where those do not fit) when
the canvas does not fit the card.

## Pipeline

Diffusers' `Krea2Pipeline` (0.42.1), Krea's reference code (`krea-ai/krea-2` db3984f) in its form: the
Qwen-Image chat template, hidden states 2, 5, … 35 of Qwen3-VL-4B's language model, Euler on
`linspace(1, 1/steps)` time-shifted by mu (Raw: from the token count, 0.5 at 256 tokens to 1.15 at 6400;
Turbo: 1.15), bf16 noise from `seed` drawn on the CPU, and the VAE in bf16. Two departures, both exact:

- Diffusers pads every prompt to 512 tokens and masks the padding. The prompt is encoded unpadded: real
  tokens keep their positions and values, the DiT reads text at rotary position 0, and the masked keys
  carried nothing. It saves the padding's compute, and with no mask every attention kernel can serve.
  On tiny real modules in f32 the package's latents equal the pipeline's to 1.7e-6 (bf16: its rounding).
- Only the text encoder's layers 0–34 are built; layer 35, the final norm and the vision tower are never
  read.

The steps are one `author.denoise_loop` (Runtime ≥ 0.35.0), which the Runtime runs eagerly, as one
resident program or as streamed blocks. The step is `Krea2Transformer2DModel.forward` op for op, with what
no step changes built once: each prompt's text fusion and projection, and its rotary tables. With CFG the
two branches pass through each block in turn, so a streamed block comes in once per step, not twice.

## LoRAs

Caller LoRAs adapt the DiT, at any strength: `--lora model:transformer=<ref>[,strength]`, in order. The
Runtime bakes them into the resident weights. Upload a LoRA file as it is:

```sh
cozy model upload hf://krea/Krea-2-LoRA-darkbrush@af7645f3502694cfc9fd69b9c2dca113344df33d <org/name> --lane bf16 --on <machine>
cozy model upload civitai://<version> <org/name> --lane bf16 --on <machine>
```

TensorFS's `krea2.lora` table reads every spelling a published loader does (Diffusers' and Krea's,
ComfyUI's and AI-Toolkit's reference names, musubi-tuner's kohya names).

## Prepare the models

```sh
cozy model upload hf://krea/Krea-2-Raw@6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603 krea-2 --lane bf16 --on <machine>
cozy model upload hf://krea/Krea-2-Turbo@98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7 krea-2-turbo --lane bf16 --on <machine>
cozy model quantize krea-2@sha256:<bf16 commit> --lane int8 --package krea-2 --on <machine>
cozy model quantize krea-2-turbo@sha256:<bf16 commit> --lane int8 --package krea-2 --on <machine>
```

The upload reads the Diffusers folders (`raw.safetensors` and `turbo.safetensors`, the reference
checkpoints, are not taken); the text encoder and VAE are byte-identical in both. The `int8` lane
(`int8.convrot`, W8A8 on int8 tensor cores, the default) takes what Comfy-Org's int8 ConvRot files do:
every linear of the 28 DiT blocks and every projection of Qwen3-VL's language layers. The text fusion,
embedders, modulation, final layer, token embedding and VAE keep their bytes.

## Licence

Krea 2 Raw and Turbo are under the Krea 2 Community License (`krea_2/LICENSE.model.md`,
`NOTICE.model.md`); the int8 lanes are modified versions. This wrapper grants no model rights.
