# Anima

`generate` makes one anime-style image from a text prompt with Anima Base 1.0. The model's
licence is non-commercial.

The model card's quality tags open the prompt; the package adds nothing to it:

Write the request in a file, `request.yaml`:

```yaml
prompt: masterpiece, best quality, 1girl, silver hair, rain, neon city at night
seed: 11
```

```sh
cozy run anima/generate --input request.yaml
```

Inline inputs work too, and override the file's:

```sh
cozy run anima/generate prompt='masterpiece, best quality, 1girl, silver hair, rain, neon city at night' seed=11
cozy run anima/generate prompt='masterpiece, best quality, a cat samurai under cherry blossoms' aspect_ratio=16:9 megapixels=1
```

Leave the tags out when they fight the look you want ("crayon drawing, childlike"). The default
`negative_prompt` is the card's: `worst quality, low quality, score_1, score_2, score_3, artist
name, blurry, jpeg artifacts, chromatic aberration`. No rating tag is added, so a neutral prompt
can return NSFW output; write `safe` in the prompt to ask for it.

Inputs are the shared image request, the same in every image package: `prompt`, `reference_images`, `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `steps`, `guidance`, `negative_prompt`, `seed` and `prompt_upsampling`. Anima adds `cfg_interval_start`/`cfg_interval_stop` (the part of the schedule that
uses guidance, default 0.15–0.70) and `first_block_cache` (0 = off). Left out, each is Anima's
default: 1:1 at its 1536 class (1536², 2.25 MP), 30 steps at guidance 4.5, the card's negative
prompt, a random seed. Sizes round to 32 px within 1–2.25 MP and the rotary range (2048 px sides);
steps 8–50, guidance 1–10. Out of range clamps with a `clamp` adjustment. Anima takes no
`reference_images`, and ignores `prompt_upsampling` with a warning. The result is `image` (a WebP in
`~/.cozy/outputs`), `width`, `height`, `seed` and `prompt`.

GPU: any NVIDIA card with 8 GB. The default 8-bit lane is 3.2 GiB (`circlestone-labs/anima@1.0.0`); the
bf16 lane is 5.2 GiB. A 1024² image at 30 steps takes about 8.4 s on an RTX 4090 and 70 s on an
8 GB RTX 4070 Laptop. Without an NVIDIA GPU, add `--on new` to rent one.

`fp8` and `int8` write the model's 8-bit lanes; they are for model authors.
