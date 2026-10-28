# Qwen Image 2

`generate` creates or edits one lossless image using Qwen-Image-2.1, and `generate_turbo` does
the same with Qwen-Image-2.1-Turbo. Both take the same inputs. The model checkpoint is selected
independently from the package. TensorFS owns model downloads and bytes;
Runtime owns construction, memory residency and component execution.

`prompt` reaches the model exactly as given. The image keeps the decoder's alpha when anything in it is
transparent, and is RGB otherwise; alpha within 4 levels of 0 or 255 (the decoder's noise) snaps
to it. `seed` is optional and the chosen seed is
returned. Seeds range from 0 through 9007199254740991 (the interoperable 53-bit
integer range); omitting one chooses a random seed in that range. Inputs are the shared image request, the same in every image package: `prompt`, `reference_images`, `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `steps`, `guidance`, `negative_prompt`, `seed` and `prompt_upsampling`. Steps default to 40 (1–100); `generate_turbo` runs Turbo's distilled schedule
(its 8 `sample_sigmas`), and a different `steps` is reported as a `clamp` adjustment. Qwen
runs no CFG and has no upsampler, so `guidance`, `negative_prompt` and `prompt_upsampling` are ignored
with a warning, as are unknown fields. The result is `image`, `width`, `height`, `seed` and `prompt`.

Outputs use lossless WebP to preserve the same decoded pixels and alpha while reducing the
bytes returned to the client. A canvas with either side at least 16384 pixels uses PNG,
because that exceeds WebP's dimension limit. The requested canvas is preserved. The image
asset reports its actual media type; clients should use it rather than assume a filename
extension.

The size is upstream's `calculate_dimensions` at `megapixels` (held to 1–4 MP), in multiples of 32 px;
text-to-image defaults to `1:1` at 4 MP, the model's native 2048×2048. A ratio past the model's rotary
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

Set `reference_images` to one through ten images and describe the desired edit or
composition in `prompt`. Omit it for ordinary text-to-image generation. The same
Qwen-Image-2.1 checkpoint handles both modes; editing runs its complete instruction
sampling schedule, with no image-to-image strength parameter.

```sh
cozy run qwen-image-2/generate \
  prompt='Change the background to a sunset beach; preserve the person and clothing.' \
  --asset reference_images.0=portrait.png seed=42

cozy run qwen-image-2/generate \
  prompt='Place the person from image 1 in the room from image 2.' \
  --asset reference_images.0=person.png --asset reference_images.1=room.png \
  seed=42
```

References are ordered. Each is any image Runtime decodes, up to 32 MiB encoded, read upright
(EXIF orientation) with its alpha and scaled down to 4 MP when larger. As upstream, the VAE reads
the alpha and the vision encoder a copy composited over white. Each reference is resized to the
upstream 1 MP conditioning area independently of the output's `megapixels`.

With references, what is not given is the FIRST reference's (the image being edited): an omitted
`aspect_ratio` takes its exact ratio and an omitted `megapixels` its pixel count, held inside 1–4 MP
(reported as a `clamp` adjustment when held). Upstream takes the LAST reference's; every Cozy image
package takes the first.

Reference VAE encoding and multimodal prompt encoding each run once before
denoising, in separate component residency leases. Denoising reuses their tensors
and the upstream prefix KV cache across steps. Reference images add tokens and
memory demand, so compare editing benchmarks separately from text-only runs.

The upstream model is under the Qwen Research License, for noncommercial research
and evaluation. This wrapper is not an authorization for commercial model use.
The licence and Notice ship in this package (`qwen_image_2/LICENSE.model.txt`, `qwen_image_2/Notice`).
The architecture is Diffusers' (Apache-2.0), at the release `pyproject.toml` pins.


## Prepare the checkpoint

Use an existing rental to download, convert and upload the model. The checkpoint
carries the upstream `processor/` files (tokenizer, chat template, image and video
processor configs) as assets; the package builds its processor from them:

```sh
cozy model upload \
  hf://Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758 \
  qwen-image-2.1 --lane bf16 --on YOUR_MACHINE
cozy model upload \
  hf://Qwen/Qwen-Image-2.1-Turbo@fc13ba1986fcb905854dc8de2f52eee395f49b5b \
  qwen-image-2.1-turbo --lane bf16 --on YOUR_MACHINE
cozy model publish qwen-image-2.1-turbo 0.1.0 sha256:COMMIT
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

```sh
cozy model quantize qwen-image-2.1@sha256:BF16_COMMIT --lane int8 --package qwen-image-2 --on YOUR_MACHINE
cozy model publish qwen-image-2.1 0.1.0 sha256:BF16_COMMIT sha256:INT8_COMMIT
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
for the command's Hub and account, so lock with `cozy package lock`, not raw
`uv lock`. uv resolves this distribution only from that index. Import its generated
managed caller:

```python
from qwen_image_2 import generate

image = await generate(
    prompt="An explorer in a blue coat",
    aspect_ratio="3:4",
    megapixels=1.0,
)
```

The Hub package is the first-party `qwen-image-2` (no owner); its callables are `generate` and
`generate_turbo`. Their default models are the first-party `qwen-image-2.1@0.1.0` and
`qwen-image-2.1-turbo@0.1.1`: a model with no org is the root namespace's.
Installing the wheel alone does not turn GPU execution into an in-process call;
Cozy supplies the managed caller when composing package functions.
