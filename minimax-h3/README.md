# MiniMax H3

Video with sound from MiniMax H3: 5 to 15 seconds at 768p, optionally taken to 1080p, 1440p
or 2160p by SeedVR2 or NVIDIA's Regenerate.

| Function | Makes a video from |
|---|---|
| `fl2va` | a prompt and up to two keyframes: the first and the last frame |
| `ref2va` | a prompt and references: characters, backgrounds, audio, action videos |
| `long_form` | several segments sharing their references, joined into one video ([LONG_FORM.md](LONG_FORM.md)) |
| `regenerate` | an existing H3 video and its prompt, re-rendered at 768p to 2160p ([UPSCALE.md](UPSCALE.md)) |

The model's timestep plans set the step counts: `fl2va` and `ref2va` take `steps` 30, 40 or 50
(default 30), and `fl2va_turbo` and `ref2va_turbo` run the turbo LoRA's eight. The functions
default to `minimax-h3@1.0.0-rc.6` and `minimax-h3-turbo-lora@1.0.1`. Releases before
`minimax-h3@1.0.0-rc.4` carry no plans and refuse with `model_asset_missing`.

Write the request in a file, `request.yaml`:

```yaml
prompt: The camera pulls back from a lighthouse as a storm rolls in.
duration_s: 6
```

```sh
cozy run minimax-h3/fl2va_turbo --input request.yaml --asset first=./first.png --asset last=./last.png
```

Inline inputs work too, and override the file's:

```sh
cozy run minimax-h3/fl2va_turbo prompt='The camera pulls back from a lighthouse as a storm rolls in.' \
  duration_s=6 --asset first=./first.png --asset last=./last.png
cozy run minimax-h3/ref2va_turbo --input scene.json
cozy run minimax-h3/long_form --input story.json
```

`scene.json` names its references, by file path or description, and uses them in the prompt as
`<Name>`; a long-form `story.json` adds `segments` ([example](examples/dialogue-garden.json)):

```json
{
  "prompt": "<Rover> waves in a bright studio.",
  "duration_s": 6,
  "references": [
    {"name": "Rover", "kind": "character", "image": "./rover.png"},
    {"name": "Studio", "kind": "background", "description": "A bright studio."}
  ]
}
```

A `kind: video` reference guides motion; with `guide: true` it is the source the output follows
frame by frame, for edit LoRAs such as head swap and style transfer
([LONG_FORM.md](LONG_FORM.md#action-videos)).

Supplied character images have their backgrounds removed (BiRefNet) and are described in one
sentence (Qwen3.5); `clean_references=false` and `describe_references=false` turn
either off. Write prompts in MiniMax's
[full-reference format](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md);
`resolution` and `upscale` are in [UPSCALE.md](UPSCALE.md). The video is an MP4 in
`~/.cozy/outputs`.

LoRAs: add one `--lora <org/lora>[,strength]` per LoRA (strength 1 by default), in the order they
stack, or list them in the request file:

```sh
cozy run minimax-h3/ref2va_turbo --input scene.yaml --lora alissonerdx/minimax-h3-live-wallpaper,1.0
```

```yaml
lora:
  - {ref: alissonerdx/minimax-h3-live-wallpaper, strength: 1.0}
```

Each function adapts the transformer it runs: `transformer` for `fl2va`, `transformer_ref` for
`ref2va` and `long_form`. A LoRA trained for the other one still loads, but its effect is untested.

GPU: one NVIDIA card with 80 GB or more (H100, RTX PRO 6000, H200); the 8-bit lane is 70 GiB,
and `--on new` rents a card that holds it. Measured on one H100 at 30 steps: about 3 minutes
for 5 s of video and 16 minutes for 14 s; turbo runs 8 steps. Most computers rent: add
`--on new`.

`encode_text`, `assemble_video` and `long_form_cuts` are the workflow's parts, callable alone.
