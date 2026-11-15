# Qwen Image 2

`generate` creates or edits one lossless image using Qwen-Image-2.1, and `generate_turbo` does
the same with Alibaba PAI's Fun Acc overlay on it (`alibaba-pai/qwen-image-2.1-fun-acc`, a 330 MiB
4-step distillation). Both take the same inputs. On a canvas they're one tool, `image` (Qwen Image 2):
its `instructions`, a Turbo toggle (the overlay loads only when asked for), `aspect_ratio`, `megapixels`, `seed` and,
under Advanced, `prompt_upsampling` and `creativity`. The model checkpoint is selected
independently from the package. TensorFS owns model downloads and bytes;
Runtime owns construction, memory residency and component execution.

Each picture's `description` follows the prompt as `<name>: description`, in the words tuned on this model: a mention reads as `<image3>` alone, and with no words for it a third picture was dropped. In a `cozy project`, each `<name>` the text mentions is that box's picture and description, and a text box goes in as its words.

`prompt`, then those words, goes to Qwen's prompt enhancer first (below), and its rewrite reaches the model. The image keeps the decoder's alpha when anything in it is
transparent, and is RGB otherwise; alpha within 4 levels of 0 or 255 (the decoder's noise) snaps
to it. `seed` is optional and the chosen seed is
returned. Seeds range from 0 through 9007199254740991 (the interoperable 53-bit
integer range); omitting one chooses a random seed in that range. Inputs: `prompt` (its text and the pictures it mentions), `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `seed`, `prompt_upsampling` and `creativity`. The full model runs 40 steps, Turbo 4 on Fun Acc's
grid, with its own output head each step and an F32 state (`qwenimage21_pdd.py`). Qwen runs no CFG, so it has no `guidance` or `negative_prompt`. Unknown fields are ignored with a warning. The result is `image`, `width`, `height`, `seed` and `prompt`, the prompt the image was made from.

`upscale` (the Upscale tool on a picture box) makes the box's picture larger, at 2 or 4 MP: Turbo edits it with the picture
as its reference and, Lanczos-upscaled to the canvas, below `denoise` (0.2) as the start. Downscaled back, 29 dB PSNR to
the input on a photo and a woodblock print, against SeedVR2's 26.5 and 30.3; at `denoise` 1 it redraws (16-19 dB, se-123).
`megapixels`, `denoise` and `seed`.

## Prompt upsampling

Qwen ships two prompt enhancers with Qwen-Image-2.1 ([`prompt_rewrite/`](https://github.com/QwenLM/Qwen-Image-2.1/tree/ab96caa944157ca28a4f36b695c3047549f48f98/prompt_rewrite)),
each a Qwen3.5-9B fine-tune: `Qwen-Image-2.1-PE-I2I` rewrites an edit instruction, from its pictures, into a precise
one that names what changes, pushes it to full strength and holds the rest; `Qwen-Image-2.1-PE-T2I` expands a
request from words into a long description of the finished picture. `prompt_upsampling` (on by default) runs the
one that fits before generating; off, the prompt goes as written and neither enhancer is fetched.

Each runs as upstream's `pe_core.py`: its own `system_prompt.txt`, thinking first, then a JSON answer; temperature
1.0, top-p 0.95, presence penalty 0 for edits and 1.5 for words; each picture over white at most 1 MP. Runtime's
sampler has no top-k, so upstream's top-k 20 is not applied. The rewrite is sampled from the render's `seed`: a new
render rewrites anew, and a given seed repeats its rewrite. The rewrite is the result's `prompt`, and the run's log has it with its
seed, canvas and token counts.

Upstream lets the enhancer think for up to 24,000 tokens, which on a 4090 added about 69 s to a 5 s edit. Here its
thought stops at 128 tokens and it answers, closed as Qwen3 closes a thinking budget, and an edit's rewrite is asked
to stay within 80 words. That adds about 9 s to an edit and keeps the rewriter's gains: a designed scene for a new
shot, and no props the new picture has no use for. It holds the person's clothes as often as Turbo alone does (3 of 6),
where upstream's full thought held them 6 of 6. With no thought at all, the enhancer reasons in its answer and never
gives one. With no `aspect_ratio`, the enhancer's canvas is used, as upstream passes it to the renderer: its
`wh_ratio`, or the shape of the picture its `ratio_follow` names.

`creativity` steers how far the rewrite goes beyond the words: `balanced` (the default) is Qwen's own rewrite;
`faithful` and `creative` add one sentence after the request, where the enhancer reads intent: change only what was
asked, or make a striking new picture, bold with expression, pose, styling, setting, light and camera.

## Abliterated by default

The text encoder and both enhancers are abliterated by default: each is the stock model with its refusal direction
projected out of the weights that write the residual stream, so it no longer leans a request it would refuse as a
chat model towards a refusal. Each is one setting away from Qwen's own:

| slot | default | Qwen's own |
|---|---|---|
| `text_encoder` | `atomicchat/qwen-image-2.1-text-encoder-abliterated` ([AtomicChat](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF)) | `model.text_encoder=qwen/qwen-image-2.1@0.1.0` |
| `expander` | `pottokao/qwen-image-2.1-pe-t2i-abliterated` ([pottokao](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-T2I-Heretic)) | `model.expander=qwen/qwen-image-2.1-pe-t2i@0.1.0` |
| `rewriter` | `darrellbest/qwen-image-2.1-pe-i2i-abliterated` ([darrellbest](https://huggingface.co/darrellbest/Qwen-Image-2.1-PE-I2I-Heretic)) | `model.rewriter=qwen/qwen-image-2.1-pe-i2i@0.1.0` |

`text_encoder` is its own slot, one for every function: it reads the `text_encoder` of a Qwen-Image-2.1 checkpoint,
so any of them serves it (`qwen-image-2.1`'s is Qwen3-VL-8B-Instruct unmodified). `model` reads only the DiT.
Each slot builds only what it reads, so a model of that part alone serves it: a transformer with its scheduler for
`model`, a text encoder with its processor for `text_encoder` (`cozy model upload <model>:transformer,scheduler`); the
other parts come from the other slots' models.

`vae` is its own slot too, one for every function: it reads a checkpoint's `vae`, which encodes the pictures and decodes
the image. It is the base model's own by default; `model.vae=madebyollin/texture-fix-vae` takes madebyollin's
[fine-tune of it](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) (cleaner textures, no
checkerboard).

Outputs use lossless WebP to preserve the same decoded pixels and alpha while reducing the
bytes returned to the client. A canvas with either side at least 16384 pixels uses PNG,
because that exceeds WebP's dimension limit. The requested canvas is preserved. The image
asset reports its actual media type; clients should use it rather than assume a filename
extension.

The size is upstream's `calculate_dimensions` at `megapixels` (held to 1–4 MP), in multiples of 32 px;
text-to-image defaults to `1:1` at 2 MP; 4 MP is the model's native 2048×2048. A ratio past the model's rotary
range (a 32768 px side) is served at the nearest size it admits, reported as a `clamp` adjustment.

## Transparent images

Qwen-Image-2.1 generates RGBA natively when the prompt asks for it in the wording its model card
recommends; "on a transparent background" alone draws a white one:

Write the request in a file, `request.yaml`:

```yaml
prompt: >-
  This is an RGBA image with transparency. A cute cartoon dragon sticker.
  The image has alpha channel and the background is transparent.
seed: 42
```

```sh
cozy run qwen-image-2/generate --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run qwen-image-2/generate seed=42 \
  prompt='This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.'
```

## Image editing and references

Mention one through ten pictures in `prompt` as `<name>` and describe the desired edit or composition
([references](../docs/references.md)); past ten, the last mentioned go, with a warning. Mention none for
ordinary text-to-image generation. With two or more, each mention reads as Qwen-Image-2.1's
own `<image1>`, `<image2>`, …, the tags its template writes before each image; a lone picture's
mention reads as its name, as Qwen's prompt rewriter writes one. The same
Qwen-Image-2.1 checkpoint handles both modes; editing runs its complete instruction
sampling schedule, with no image-to-image strength parameter.

```sh
# in a project with boxes `portrait`, `person` and `room`
cozy run qwen-image-2/generate seed=42 \
  prompt='Change the background of <portrait> to a sunset beach; preserve the person and clothing.'
cozy run qwen-image-2/generate seed=42 prompt='Place <person> in <room>.'
```

References are ordered. Each is any image Runtime decodes, up to 32 MiB encoded, read upright
(EXIF orientation) with its alpha and scaled down to 4 MP when larger. As upstream, the VAE reads
the alpha and the vision encoder a copy composited over white. Each reference is resized to the
upstream 1 MP conditioning area independently of the output's `megapixels`.

With references, what is not given is the FIRST reference's (the image being edited): an omitted
`aspect_ratio` takes its exact ratio and an omitted `megapixels` its pixel count, held inside 1–4 MP
(reported as a `clamp` adjustment when held). Upstream takes the LAST reference's; every Cozy image
package takes the first.

The VAE runs in bf16 on GPUs that compute it natively (the base release
stores F32), else in its stored dtype. It decodes the whole canvas, each block in strips of rows
(exact, as ComfyUI does), so its working set stays a few GB at 4 MP and the DiT and text encoder
stay resident beside it on a 24 GB card.

Reference VAE encoding (without Diffusers' per-convolution feature cache) and multimodal prompt
encoding each run once before denoising, in separate component residency leases. Step 0 extracts the
text and reference tokens' K/V into the Runtime's store (`retained`), and the later steps run over the
target's tokens alone. With a padded prompt, every step is the whole joint pass. The steps are a plain
loop over Diffusers' `FlowMatchEulerDiscreteScheduler`. Reference images add tokens and memory demand, so compare editing
benchmarks separately from text-only runs.

The upstream model is under the Qwen Research License, for noncommercial research
and evaluation. This wrapper is not an authorization for commercial model use.
The licence and Notice ship in this package (`qwen_image_2/LICENSE.model.txt`, `qwen_image_2/Notice`).
The abliterated enhancers keep that licence (pottokao's and darrellbest's `LICENSE` are the same file). The text
encoder, Qwen3-VL-8B-Instruct, is Apache-2.0, as is AtomicChat's edit (`qwen_image_2/LICENSE.text-encoder.txt`).
The architecture is Diffusers' (Apache-2.0), at the release `pyproject.toml` pins.


## LoRAs

Any Qwen-Image 2.1 LoRA in Diffusers', PEFT's, kohya's or ComfyUI's spelling uploads with
`cozy model upload` (TensorFS's `qwenimage21.lora`) and adapts the DiT, Turbo or not, at any
strength: `model.lora=<org/name>` on a run, or dropped on the box. Runtime bakes it into the
weights. A step-distillation LoRA (Lightning, Turbo8) is trained for its own schedule, which these
functions don't run.

## Prepare the checkpoint

Use an existing rental to download, convert and upload the model. The checkpoint
carries the upstream `processor/` files (tokenizer, chat template, image and video
processor configs) as assets; the package builds its processor from them:

```sh
cozy model upload \
  hf://Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758 \
  qwen-image-2.1 --lane bf16 --on YOUR_MACHINE
cozy model upload \
  hf://alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs@f7545234760e1847cd8e89e52bd951cb0b7e327f/models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors \
  alibaba-pai/qwen-image-2.1-fun-acc --lane bf16 --mirror --on YOUR_MACHINE
cozy model publish alibaba-pai/qwen-image-2.1-fun-acc 0.1.0 sha256:COMMIT
```

The enhancers are mirrored as uploaded, and `qwen3.5`'s int8 recipe writes their lane (the same Qwen3.5-9B):

```sh
cozy model upload hf://Qwen/Qwen-Image-2.1-PE-I2I@72927bc08afc99b7888ceb7d7d51a12db3700bbd \
  qwen/qwen-image-2.1-pe-i2i --lane bf16 --mirror --on YOUR_MACHINE
cozy model upload hf://Qwen/Qwen-Image-2.1-PE-T2I@f3ed7985c788ad75b3ab7223e0c4c51e2a43545b \
  qwen/qwen-image-2.1-pe-t2i --lane bf16 --mirror --on YOUR_MACHINE
cozy model quantize qwen/qwen-image-2.1-pe-i2i@sha256:BF16_COMMIT --lane int8 --package qwen3.5 --on YOUR_MACHINE
```

The abliterated enhancers upload the same way (`hf://pottokao/Qwen-Image-2.1-PE-T2I-Heretic@96e92081`,
`hf://darrellbest/Qwen-Image-2.1-PE-I2I-Heretic@0402e068`). AtomicChat publishes its encoder as a BF16 GGUF only,
and TensorFS's GGUF import has no `qwen3vl` name map, so its commit is Qwen's checkpoint with the 73 edited tensors
(36 attention outputs, 36 MLP down projections and the token embedding) taken from the imported GGUF by reference
([`scripts/abliterated_ingest.py`](../scripts/abliterated_ingest.py)). Its DiT and VAE are Qwen's; the package reads
only its `text_encoder`, and its int8 lane is this package's recipe:

```sh
cozy model upload hf://AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF@881a778e306c44920e42536def3e52c3a3ecc7cd/Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf \
  atomicchat/qwen-image-2.1-text-encoder-abliterated --lane gguf --mirror https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF --on YOUR_MACHINE
cozy run ./scripts/abliterated_ingest.py/assemble kind=te model.stock=qwen/qwen-image-2.1@0.1.0 \
  model.gguf=atomicchat/qwen-image-2.1-text-encoder-abliterated@sha256:GGUF_COMMIT --lane bf16 --on YOUR_MACHINE
cozy model upload run:RUN atomicchat/qwen-image-2.1-text-encoder-abliterated --lane bf16
cozy model quantize atomicchat/qwen-image-2.1-text-encoder-abliterated@sha256:BF16_COMMIT --lane int8 --package qwen-image-2 --on YOUR_MACHINE
```

Replace `YOUR_MACHINE` and `COMMIT` with the selected machine name and
returned commit id. Upload creates the commit; the separate publish
command creates the package's default release (qwen-image-2.1's with its int8 lane,
below). Preparation and descriptor checks alone do not qualify GPU inference or the
generated image.

## The int8 lane

`int8` writes the fast lane for consumer cards: every linear of the 32 DiT blocks and of the
text encoder's 36 language layers as `int8.convrot` (each 256 input channels rotated by a
Hadamard, int8 per output row), served as W8A8 on the int8 tensor cores. Embeddings, the vision
tower, the input/output projections, the modulation and the VAE keep their source bytes.
The 32 DiT blocks also attend at 8 bits (comfy-kitchen INT8 on RTX 40-class cards); `--lane bf16`, or
`--attention-kernel exact`, attends exactly.

```sh
cozy model quantize qwen/qwen-image-2.1@sha256:BF16_COMMIT --lane int8 --package qwen-image-2 --on YOUR_MACHINE
cozy model publish qwen/qwen-image-2.1 0.1.0 sha256:BF16_COMMIT sha256:INT8_COMMIT
```

## Python dependency

This is an ordinary Python distribution (`qwen-image-2`) published on the
publishing account's Tensorhub package index. A package published by the same
account names that index in its `pyproject.toml` and never spells its URL:

```toml
[project]
dependencies = ["qwen-image-2>=0.1.0"]

[tool.uv.sources]
qwen-image-2 = { index = "tensorhub" }
```

Creator writes the `tensorhub` index (`<hub>/v1/index/<account>/simple/`, explicit)
for the command's Hub and account, so lock with `task lock -- <package>` (or any `cozy run` of it),
not raw `uv lock`. uv resolves this distribution only from that index. Run its functions as managed
calls:

```python
from cozy_runtime.author import call
from cozy_runtime.fields import Prompt
from qwen_image_2 import GenerateInput, generate

image = await call(
    generate,
    payload=GenerateInput(prompt=Prompt(text="An explorer in a blue coat"), aspect_ratio="3:4", megapixels=1),
)
```

The Hub package is the first-party `qwen-image-2` (no owner); its callables are `generate`,
`generate_turbo` and the tool `image`. Their default models are `qwen/qwen-image-2.1@0.1.0` and,
for Turbo, `alibaba-pai/qwen-image-2.1-fun-acc@0.1.1`; all read
`atomicchat/qwen-image-2.1-text-encoder-abliterated@0.1.0` (`text_encoder`; the slot reads only its text encoder),
`darrellbest/qwen-image-2.1-pe-i2i-abliterated@0.1.0` (`rewriter`) and
`pottokao/qwen-image-2.1-pe-t2i-abliterated@0.1.0` (`expander`).
A plain `generate(...)` is an ordinary in-process Python call; `call` runs it as its own job on
the machine.
