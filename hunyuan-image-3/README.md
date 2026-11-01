# HunyuanImage 3.0 Instruct

Tencent's HunyuanImage 3.0 Instruct: one 80B mixture-of-experts transformer (32 layers, 64 experts with 8
routed per token plus one shared, about 13B active) that reads text and images and makes images. It first
writes its chain of thought and a rewritten prompt token by token, then denoises the image (flow matching on
32-channel VAE latents) in the same transformer. One function makes images from text, or edits and fuses up to
three ordered reference images. Use `generate_turbo`.

| function | checkpoint | default `steps` | `guidance` (default 2.5) |
| --- | --- | --- | --- |
| `generate_turbo` | Instruct-Distil | 8 | the value the distilled model is conditioned on |
| `generate` | Instruct | 50 | classifier-free (both halves run every step) |

Write the request in a file, `request.yaml`:

```yaml
prompt: A paper boat sailing down a rain gutter at dusk
seed: 7
```

```sh
cozy run hunyuan-image-3/generate_turbo --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run hunyuan-image-3/generate_turbo prompt='A paper boat sailing down a rain gutter at dusk' seed=7
cozy run hunyuan-image-3/generate_turbo prompt='Put <Cat> in a tiny astronaut suit' \
  pictures:='[{"name": "Cat", "image": "cat.png"}]'
```

The request and result are the shared image interface (`ImageRequest`, `ImageResult`): `prompt`,
`pictures` (0–3, in order), `aspect_ratio` (any `W:H`; unset, the model picks), `megapixels`, `steps`,
`guidance`, `seed` (random when omitted, returned) and `prompt_upsampling` (unset: on). `negative_prompt` and
a picture's `box` warn and are ignored: the model's unconditional branch is the prompt dropped, and it has no
layout control. A `<Name>` mention of a picture reads as 图1, 图2, 图3, the way every prompt Tencent documents
names them ([references](../docs/references.md)); the images precede the prompt with no label of their own. The result's `prompt` is the caption the image was made from: the model's rewrite when it thought.

## Sizes

The model makes about 1 MP, at Tencent's sizes for `image_base_size` 1024: 33 ratios from 1:4 to 4:1 on a
64 px step (width + height = 2048), plus 1024×768, 1280×720 and their portraits. With no `aspect_ratio` the
model picks the size itself, as Tencent's `image_size="auto"`: after its recaption when `prompt_upsampling` is
on, else in one greedy step over its size tokens.
An `aspect_ratio` is served at the nearest size (Tencent's `get_target_size`); one past 1:4 or 4:1, or a
`megapixels` more than 10% off the size made, is reported as a `clamp` adjustment. Tencent's edit postprocess,
which resizes the output to the reference's exact ratio, is not applied: the output is the model's own size.

## Prompt upsampling

`prompt_upsampling` is the Instruct models' own think-then-recaption stage (`bot_task="think_recaption"`, Tencent's
default): the model thinks about the request and its references, rewrites the prompt, and draws from both. It
samples as Tencent's `generation_config.json` says (temperature 0.6, top-k 1024, top-p 0.95, at most 2048
tokens), from a generator seeded with `seed`, so a seed repeats the same thought. Off (`bot_task="image"`), the
prompt goes straight to the image stage.

## Memory

Each of the 2,048 routed experts (37.7M parameters, 37.8 MB at int8, 93% of the model) is its own block for
Runtime's weight paging, resident or paged on its own; attention, the shared expert, routers, embeddings and the
image layers stay resident (about 8 GB at int8). Denoising sends 4,096 image tokens through every layer, so
nearly every expert is used every step; the thinking stage uses 8 of 64 per layer per token. Every run logs both
stages' routing: experts hit per layer, tokens per expert, and how often an expert cache would have held them.

## Implementation

`hunyuan_image_3/tencent/` is Tencent's code (`Tencent-Hunyuan/HunyuanImage-3.0` at `6e9113a6`, the same as the
checkpoints' remote code), vendored because it is not importable as a package (its `__init__` imports a
top-level `utils`, and its requirements pin Gradio). The changes are in its git history: no Transformers
model/generation/cache classes, the token and denoising loops written out over Tencent's own input preparation
and logits processors, the next token's logits only, routed experts as their own class (the paging blocks),
and routing that gathers each expert's rows with one sort instead of a mask per expert. As Tencent's
`from_pretrained(torch_dtype="auto")` builds it, every module runs in bfloat16 except the expert routers
(float32) and the VAE. The VAE computes as Tencent's float16 autocast runs it, without the autocast: its convs
in float16, the operands the `bf16` lane stores, and its group norms in float32. `scripts/hunyuan-image-3-model-proof.py` checks the port against Tencent's code on CPU
with a small random-weight model: images bit-identical, the same thought and the same picked size.

## Prepare the models

Upload each checkpoint as it is (168.5 GB each; neither repository is gated):

```sh
cozy model upload hf://tencent/HunyuanImage-3.0-Instruct-Distil@c8ffd07206f1b843697606968196e8f59f8ff38c \
  hunyuan-image-3-instruct-distil --lane bf16 --on <machine>
cozy model upload hf://tencent/HunyuanImage-3.0-Instruct@2ec2c78bee7d4b94157341fba86c4c2c7b1858b2 \
  hunyuan-image-3-instruct --lane bf16 --on <machine>
```

## Lanes

`bf16` writes the served 16-bit lane from the upload: the VAE's conv weights and biases in float16 (2.5 GB
instead of 5.0 GB of float32; the decode is bit-identical, since Tencent's autocast computed them in float16),
everything else the upload's own. The quantized lanes are written from that `bf16` lane.

```sh
cozy model quantize tencent/hunyuan-image-3-instruct-distil@sha256:<upload> --lane bf16 --package ./hunyuan-image-3 --on <machine>
```

`int8` writes the default lane from the `bf16` lane: every linear of the 32 decoder layers (attention, the
shared and the 64 routed experts: 99% of the parameters) and the LM head as `int8.convrot`, about 85 GB in all.
`nvfp4` puts the routed experts in NVFP4, 43.5 GB of experts instead of 77.3. The routers, embedding, vision
tower, VAE and image layers keep their source bytes.

```sh
cozy model quantize tencent/hunyuan-image-3-instruct-distil@sha256:<bf16 commit> --lane int8 --package ./hunyuan-image-3 --on <machine>
cozy model quantize tencent/hunyuan-image-3-instruct-distil@sha256:<bf16 commit> --lane nvfp4 --package ./hunyuan-image-3 --on <machine>
```

## Licence

Tencent Hunyuan Community License (`hunyuan_image_3/LICENSE.model.txt`, `NOTICE.model.md`). It does not apply in
the European Union, the United Kingdom or South Korea, and a licensee with more than 100 million monthly active
users needs Tencent's licence. This wrapper grants no model rights.
