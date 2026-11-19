# SDXL

`generate` makes one image from a text prompt with Stable Diffusion XL 1.0 (VAE fix).

`image` is the canvas tool, its prompt `instructions`: `aspect_ratio`, `megapixels`, `negative_prompt` and `seed`; HiDiffusion runs above 1 MP.
It takes pictures as Anima's does:

- `image` and `strength` (0.9): image to image, Diffusers' img2img over the picture's latents.
- `control`, a list of `{image, kind, strength}`: `depth`, `pose`, `lineart`, `canny`, `sketch` and `inpaint`, all through
  xinsir's [ControlNet Union ProMax](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) (one model, a mode per kind;
  Diffusers' `ControlNetUnionModel`). `inpaint`'s map is the picture with the area to redraw transparent: that area
  is redrawn, the rest put back each step. Control runs base SDXL, without HiDiffusion.
- A picture mentioned in `instructions` steers subject and style through h94's
  [IP-Adapter Plus](https://huggingface.co/h94/IP-Adapter) (`ip-adapter-plus_sdxl_vit-h`, LAION's ViT-H/14 encoder), at
  `reference_strength` (0.4: the words set the scene, the picture gives its subject and style; from 0.5 up the picture's
  own backdrop wins over the words, at 0.3 its subject fades); up to 4 pictures. The mention reads as its description,
  else as nothing.
- A picture whose subject is a face steers its identity instead, through h94's
  [IP-Adapter-FaceID-PlusV2](https://huggingface.co/h94/IP-Adapter-FaceID) and its LoRA: insightface's buffalo_l finds
  the face (SCRFD-10G; one face covering at least 1% of the picture and twice any other's size) and gives its ArcFace
  identity, and ViT-H reads the face aligned at 224². At the default `reference_strength` it runs at its authors' scale
  1 (its scale is 2.5 times `reference_strength`). A face A/B (se-175): ArcFace similarity 0.65 against Plus's 0.42, the
  words still followed. Pictures without such a face keep Plus, beside it in the same request.
  insightface's models and FaceID are for non-commercial research use only.

Each adapter is a model slot (`controlnet`, `image_encoder`, `ip_adapter`, `faces`, `face_identity`, `face_adapter`,
`face_lora`), fetched only when a request uses it: buffalo_l's detector (`faces`) with any picture, its ArcFace
(`face_identity`) and FaceID's two once a run finds a face.

`upscale` (the Upscale tool on a picture box) makes the box's picture larger, at 2, 4 or 8 MP: img2img over the picture
Lanczos-upscaled to the canvas, the last `denoise` (0.2) of 30 DPM++ 2M Karras steps, no HiDiffusion. Downscaled back from
8 MP, 29-30 dB PSNR to the input on a photo and a woodblock print, against SeedVR2's 26.5 and 30.3 (se-123). `megapixels`,
`denoise` and `seed`.

Write the request in a file, `request.yaml`:

```yaml
prompt: a lighthouse at dusk, oil painting
aspect_ratio: "16:9"
seed: 7
```

```sh
cozy run sdxl/generate --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run sdxl/generate prompt='a lighthouse at dusk, oil painting' seed=7
cozy run sdxl/generate prompt='a fox in fresh snow' aspect_ratio=16:9 megapixels=2 steps=30
```

Inputs: `prompt`, `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `seed`, `steps`, `guidance`, `negative_prompt` and `hidiffusion`. Left out, each is SDXL's default: 1:1 at 1 MP, 30 steps at
guidance 7 (HiDiffusion's 30 at 7.5 above 1 MP, where it runs unless `hidiffusion=false`), no
negative prompt (guidance then runs against SDXL's zero unconditional, as Diffusers and Stability's own sampler do), a
random seed. Sizes snap to the 64 px stride and stay within 1–8 MP (16 MP at
1:1), 1:4–4:1 and 4096 px sides; steps 1–50, guidance 0–20. Out of range clamps with a `clamp`
adjustment. `generate` takes the same `image`, `strength`, `control` and `reference_strength`; with a picture and no
`aspect_ratio` or `megapixels`, the size is the first picture's ([references](../docs/references.md)). The
result is `image` (a WebP in `~/.cozy/outputs`), `width`, `height`, `seed` and `prompt`.

GPU: any NVIDIA card with 8 GB. The default 8-bit lane is 3.8 GiB (`stabilityai/sdxl@1.0.0`);
`--lane bf16` loads the 6.5 GiB bf16 lane. A 1024² image at 30 steps takes about 2.9 s on an
RTX 4090 (int8). Without an NVIDIA GPU, add `--on new` to rent
one.

Its recipes write the model's `fp8`, `int8` and `w4a8` lanes, for model authors
(`cozy model quantize <model> --lane int8 --package sdxl`).

Encoding, every step and the decode run in one call: a plain loop over DPM++ 2M Karras (Diffusers'
`DPMSolverMultistepScheduler`, epsilon or v), or with HiDiffusion its `DDIMScheduler` at eta 1. The VAE decodes whole where it fits, else in the fewest windows that do (a
`split` site, neighbours cross-faded over 256 px). Every request holds
the VAE in its decode dtype for good (a cast of the stored bytes on its next fill; no request
converts VAE weights). CLIP conditioning calls the encoders' layers directly, avoiding
Transformers' lazy output-collection hooks; its original 5.16.1 forward is the numerical oracle.

