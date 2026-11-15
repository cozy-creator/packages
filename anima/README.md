# Anima

`generate` makes one anime-style image with Anima Base 1.0: from a text prompt, steered by the pictures it
mentions and by control maps, or from a start picture. The model's licence is non-commercial, as are its adapters'.

`image` is the canvas tool, its prompt `instructions`, with the same inputs: a box with pictures edits.

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

Inputs: `prompt`, `aspect_ratio` (any W:H), `megapixels` (any size, 1024² pixels each), `seed`, `steps`, `guidance` and `negative_prompt`. Anima adds `cfg_interval_start`/`cfg_interval_stop` (the part of the schedule that
uses guidance, default 0.15–0.70) and `first_block_cache` (0 = off). Left out, each is Anima's
default: 1:1 at its 1536 class (1536², 2.25 MP), 30 steps at guidance 4.5, the card's negative
prompt, a random seed. Sizes round to 32 px within 1–2.25 MP and the rotary range (2048 px sides);
steps 8–50, guidance 1–10. Out of range clamps with a `clamp` adjustment. The result is `image` (a WebP in
`~/.cozy/outputs`), `width`, `height`, `seed` and `prompt`.

## Pictures

- **Mentioned pictures** (`<amy> sitting in a cafe`), up to 4, steer the image through LuciferTC's SigLIP2
  IP-Adapter (`lucifertc/ip-adapter`, with `google/siglip2-base-patch16-512`): the character's hair, outfit and
  accessories carry over. A mention reads as its picture's description, else as nothing ([references](../docs/references.md)).
  `reference_strength` (0–2, default 1) scales them.
- **Image to image:** `image` is the start picture, `strength` (default 0.9) how much of it is redrawn: below about
  0.8 it keeps the picture's colours and outfit whatever the words say.
- **Control maps:** `control` is a list of `{image, kind, strength}`, each map at any size (it is fitted to the
  output), through its kind's adapter:

  | kind | map | adapter |
  |---|---|---|
  | `pose` | a DWPose/OpenPose skeleton on black | `claquasse/control-pose` (Claquasse's control-LoRA, Base 1.0) |
  | `sketch`, `lineart` | lines or a scribble | `kohya-ss/lllite-any-test-like` (kohya's ControlNet-LLLite, Base 1.0) |
  | `depth` | a depth map | `kohya-ss/lllite-depth` (LLLite, Preview3) |
  | `inpaint` | the picture with the area to redraw transparent | `kohya-ss/lllite-inpainting` (LLLite, Base 1.0) |

  The pose adapter also brings its own LoRA on the DiT and the LLM adapter, applied only while a pose steers.
  An inpaint redraws the transparent area and blends into the rest, which keeps its own pixels: the edge is
  redrawn in a ring about 40 px wide. Its start is the map's picture, fully redrawn there, or `image` at
  `strength`. `canny` has no Anima adapter yet, so it is ignored with a note. Another kohya LLLite file works by
  pinning an LLLite kind's slot (`depth`, `lineart`, `sketch`, `inpaint`).

```yaml
prompt: masterpiece, best quality, <amy> reading in a library
control:
  - {image: pose.png, kind: pose}
```

GPU: any NVIDIA card with 8 GB. The default 8-bit lane is 3.2 GiB (`circlestone-labs/anima@1.0.0`); the
bf16 lane is 5.2 GiB. A 1024² image at 30 steps takes about 8.4 s on an RTX 4090 and 70 s on an
8 GB RTX 4070 Laptop. Without an NVIDIA GPU, add `--on new` to rent one.

`fp8` and `int8` write the model's 8-bit lanes; they are for model authors.
