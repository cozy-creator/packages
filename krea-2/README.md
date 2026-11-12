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
`image` is the canvas tool, its prompt `instructions`: one box, Turbo or Raw, which loads only the checkpoint a render asks for.

Write the request in a file, `request.yaml`:

```yaml
prompt: A tiny harvest mouse clinging to a slender branch, macro photograph, creamy bokeh
aspect_ratio: "3:2"
```

```sh
cozy run krea-2/generate_turbo --input request.yaml
cozy run krea-2/generate prompt='a red fox in fresh snow at golden hour' seed=7
cozy run krea-2/generate_turbo --input request.yaml --lora <org/krea-2-lora>,0.8
```

Fields: `prompt`, `aspect_ratio` (any `W:H`, default 1:1, or the first picture's), `megapixels` (Raw 1, Turbo 4,
or the first picture's), `steps` (1–100) and `seed` (random when omitted, returned). `generate` also takes
`guidance` (0–20; 0 runs no negative branch) and `negative_prompt` (default empty, as Krea's): Turbo samples
without guidance. Unknown fields are dropped with a warning. The result is `image`, `width`, `height`, `seed`
and `prompt`.

## Pictures

Krea 2 is text-only upstream; its community adapters make it edit. Each is a LoRA the package arms itself,
from its author's mirror on the Hub (`--model <slot>=<ref>` swaps one).

- **Edit:** the pictures a prompt mentions (`<amy> on a beach at dusk`; up to 2, the first the scene, the
  second the person) are edited with Identity Edit V2 (`edit` slot). A mention reads as its name
  ([references](../docs/references.md)).
- **Image-to-image:** `image` is a picture to start from, `denoise` how much of it is redrawn (default 0.7).
- **Control:** `control` is a list of the Runtime's `Control` (`{image, kind, strength}`), `image` the ready map,
  of the four kinds below; a map of another kind is dropped with a warning. Krea 2 reads the first.

| input | adapter (slot) | how it conditions | Raw | Turbo |
| --- | --- | --- | --- | --- |
| mentioned pictures | conradlocke Identity Edit V2 (`edit`) | VAE tokens before the target + Qwen3-VL sees the pictures | yes | yes |
| `pose` (OpenPose/DWPose skeleton) | thedeoxen pose ControlNet LoRA (`pose`) | ai-toolkit edit: VAE tokens after the target at t=0, attending only to themselves, + `Picture 1:` | untested | yes (trained on Turbo) |
| `lineart` (white lines on black) | Cierpliwy lineart edit (`lineart`) | ai-toolkit edit, as pose | yes | yes |
| `inpaint` (the picture with the area to redraw transparent, or black) | Cierpliwy inpaint edit (`inpaint`) | ai-toolkit edit, as pose | yes | yes |
| `depth` (near is white) | Patil depth ControlNet-LoRA (`depth`) | the map's latents beside each image token (`img_in` 64 → 128) | yes | yes |
| `sketch`, `canny` | none | dropped with a warning | | |

With pictures, the output is at most 2 MP (the adapters' range). A control with mentioned pictures goes as its
VAE tokens alone, the person beside the canvas. The pose adapter follows a skeleton loosely, and with a person
it can repeat them; two pictures (a scene and a person) composite stiffly. General Edit V2 (`--model edit=conradlocke/general-edit@0.1.0`) is the same author's
instruction editor (remove, recolor, relight); the two are not stacked. Identity Edit V2 and General Edit V2 are
free for non-commercial use only; commercial use needs the author's licence (their model card).

Guidance is Krea's: `cond + g·(cond − uncond)`, so Krea's 3.5 is ComfyUI's cfg 4.5, and Turbo's 0 is
ComfyUI's cfg 1.

## Text encoder

`text_encoder` is its own slot: it reads the Qwen3-VL-4B of any Krea 2 checkpoint, the function's own by default
(Raw's and Turbo's are byte-identical); `model` reads only the DiT and VAE. Quiho's abliterated encoder
([Krea2_NSFW_Uncensored_Complete](https://huggingface.co/Quiho/Krea2_NSFW_Uncensored_Complete),
`quiho/qwen3-vl-4b-abliterated`) is Qwen3-VL-4B with its refusal direction removed:

```sh
cozy run krea-2/generate_turbo prompt='...' model.text_encoder=quiho/qwen3-vl-4b-abliterated@0.1.0
```

Quiho publishes it in fp8 only. Its `int8` lane is this package's recipe, written from those fp8 values, and serves
wherever the card has int8 tensor cores; `fp8` serves the rest.
On an RTX 4090 it changed details but not what an explicit prompt renders, with or without an NSFW LoRA: what
Turbo leaves out (a "topless" prompt drawn in a bikini) it leaves out with either encoder.

## Sizes

Each checkpoint defaults to the most Krea gives it: Raw is trained up to 1 MP, Turbo makes 1–4 MP and Krea
runs it at 2048². `megapixels` is held to 0.25–4, sides to 256–4096 px on the 16 px grid, and each clamp is
reported as an adjustment. The VAE decodes the whole canvas, each block in strips of rows (exact, as
ComfyUI does), which keeps the decode to a few GB, so the DiT stays resident beside it on a 24 GB card.

## Pipeline

Diffusers' `Krea2Pipeline` (0.42.1), Krea's reference code (`krea-ai/krea-2` db3984f) in its form: the
Qwen-Image chat template, hidden states 2, 5, … 35 of Qwen3-VL-4B's language model, Euler on
`linspace(1, 1/steps)` time-shifted by mu (Raw: from the token count, 0.5 at 256 tokens to 1.15 at 6400;
Turbo: 1.15), bf16 noise from `seed` drawn on the CPU, and the VAE in bf16. Two departures, both exact:

- Diffusers pads every prompt to 512 tokens and masks the padding. The prompt is encoded unpadded: real
  tokens keep their positions and values, the DiT reads text at rotary position 0, and the masked keys
  carried nothing. It saves the padding's compute, and with no mask every attention kernel can serve.
  On tiny real modules in f32 the package's latents equal the pipeline's to 1.7e-6 (bf16: its rounding).
- Only the text encoder's layers 0–34 are built; layer 35 and the final norm are never read. Its vision
  tower runs only for a prompt with pictures.

The steps are a plain loop over Diffusers' `FlowMatchEulerDiscreteScheduler`. Each step is `Krea2Transformer2DModel.forward` op for op, with what
no step changes built once: each prompt's text fusion and projection, and its rotary tables. With CFG the
two branches pass through each block in turn, so a streamed block comes in once per step, not twice.

## LoRAs

Caller LoRAs adapt the DiT, at any strength: `--lora <ref>[,strength]`, in order. The
Runtime bakes them into the resident weights. Upload a LoRA file as it is:

```sh
cozy model upload hf://krea/Krea-2-LoRA-darkbrush@af7645f3502694cfc9fd69b9c2dca113344df33d <org/name> --lane bf16 --on <machine>
cozy model upload civitai://<version> <org/name> --lane bf16 --on <machine>
```

TensorFS's `krea2.lora` table reads every spelling a published loader does (Diffusers' and Krea's,
ComfyUI's and AI-Toolkit's reference names, musubi-tuner's kohya names).

## Prepare the models

```sh
cozy model upload hf://krea/Krea-2-Raw@6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603 krea/krea-2 --lane bf16 --on <machine>
cozy model upload hf://krea/Krea-2-Turbo@98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7 krea/krea-2-turbo --lane bf16 --on <machine>
cozy model quantize krea/krea-2@sha256:<bf16 commit> --lane int8 --package krea-2 --on <machine>
cozy model quantize krea/krea-2-turbo@sha256:<bf16 commit> --lane int8 --package krea-2 --on <machine>
cozy model upload hf://Quiho/Krea2_NSFW_Uncensored_Complete@4ecd0d82d30daec24691db30495d752f2d7d7f64/qwen3VL4BAbliteratedComfyui_FP8.safetensors \
  quiho/qwen3-vl-4b-abliterated --lane fp8 --base krea/krea-2-turbo@sha256:<bf16 commit> \
  --mirror https://huggingface.co/Quiho/Krea2_NSFW_Uncensored_Complete --on <machine>
cozy model quantize quiho/qwen3-vl-4b-abliterated@sha256:<fp8 commit> --lane int8 --package krea-2 --on <machine>
```

The upload reads the Diffusers folders (`raw.safetensors` and `turbo.safetensors`, the reference
checkpoints, are not taken); the text encoder and VAE are byte-identical in both. The `int8` lane
(`int8.convrot`, W8A8 on int8 tensor cores, the default) takes what Comfy-Org's int8 ConvRot files do:
every linear of the 28 DiT blocks and every projection of Qwen3-VL's language layers. The text fusion,
embedders, modulation, final layer, token embedding and VAE keep their bytes.

Quiho's encoder is one ComfyUI file (`model.layers.*`, `model.visual.*`), which TensorFS's `qwen3vl.comfy` table reads
as the `text_encoder`; its linears are plain fp8 with no scales, which TensorFS reads as `fp8.rowwise` with every row's
scale 1, the values ComfyUI upcasts, so the int8 recipe reads them exactly.
The upload takes the DiT, VAE and tokenizer from `--base`, so the mirror is a whole Krea 2 Turbo checkpoint, of which
the slot reads only the encoder.

## Licence

Krea 2 Raw and Turbo are under the Krea 2 Community License (`krea_2/LICENSE.model.md`,
`NOTICE.model.md`); the int8 lanes are modified versions. This wrapper grants no model rights.
