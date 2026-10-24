# Higher-resolution H3 video

Direct `ref2va` and `ref2va_turbo` calls also accept the long-form `references`
syntax. As in long-form, BiRefNet removes the background of each supplied character
image and a one-sentence description is added to each undescribed supplied character;
`clean_references: false` or `describe_references: false` turns either off. Background,
audio and video files are used as supplied. The prompt is then compiled like one long-form
segment. Descriptions without images are generated once with Qwen Image 2.1. The resolved
assets are then handed to the H3 child, so preprocessing is not repeated. Send
either `references` or attached `assets`; a references-only request sends `"assets": []`.
`kind: video` references guide action as in long-form ([Action videos](LONG_FORM.md#action-videos)).

```json
{
  "prompt": "<Rover> waves in a bright studio.",
  "duration_s": 6,
  "references": [
    {"name": "Rover", "kind": "character", "image": "./rover.png"},
    {"name": "Studio", "kind": "background", "description": "A bright studio."}
  ],
  "assets": []
}
```

FL2VA and Ref2VA, including their Turbo variants, accept `resolution` as a closed enum:
`768p` (default, the native 1344×768), `1080p`, `1440p`, or `2160p` (4K UHD). Above 768p,
SeedVR2 7B restores the original to 1920×1080, 2560×1440, or 3840×2160, fitted without
stretching: the 7:4 frame gets narrow letterboxing. Audio and frame timing are retained.
SeedVR2 matches its colour to the original (`lab`) by default.

Higher-resolution calls publish the original 768p video (`originals`) before the upscale
finishes. CPU orchestration runs H3 and SeedVR2 as separate GPU children. Omit `--gpus` so
single-GPU SeedVR2 and multi-GPU H3 can each use their supported count. An explicit count
applies to every GPU child and is never silently reduced.

```sh
cozy run minimax-h3/ref2va_turbo --asset character.png \
  prompt='<Picture 1> waves in a bright studio.' duration_s=15 seed=581203 \
  resolution=1440p
```

The requested 15 seconds follows H3's native frame grid (362 frames at 24 fps, about
15.083 seconds). Upscaling can restore or synthesize texture; it cannot correct wrong
motion or character identity. Review moving faces, hands and clothing at normal speed.

Qualified 2026-10-13 on one RTX Pro 6000 with 15 s Turbo originals (seed 581203; FL2VA and
Ref2VA; seedvr2 0.1.3, 17-frame windows, no colour correction): minutes per upscale of the
362-frame clip, and down-consistency PSNR (output area-scaled back to 1344×768 against the
original) for FL2VA / Ref2VA:

| 1080p | 1440p | 2160p |
|---|---|---|
| 7.4–8.2 min, 34.5 / 34.1 dB | 12.5 min, 38.1 / 36.2 dB | 27.6–27.7 min, 39.3 / 37.6 dB |

The owner chose SeedVR2 7B on 2026-10-13: it keeps faces closest to the original while adding
detail. SeedVR2 3B was sharper but over-processed, and the H3 latent upscaler with H3
refinement was soft, drifted identity and could not reach 2160p on one GPU; both were deleted.

## Tuning

`upscale` passes SeedVR2's ComfyUI controls (seedvr2 README, Tuning) to the upscale, under the
upstream names; it applies only above 768p. Restoration can invent skin texture and fine detail;
`latent_noise_scale` softens it:

```sh
cozy run minimax-h3/fl2va_turbo prompt='...' duration_s=15 resolution=1080p \
  'upscale:={"latent_noise_scale": 0.1}'
```

Measured on the Ref2VA original above at 1080p (seedvr2 README has the full table): colour
correction removes a ~15% chroma over-saturation (ΔE76 3.90 → 2.87 with `lab`) at no detail
cost; `latent_noise_scale` 0.1 and 0.2 lower the detail measure by 7% and 12% and visibly remove
invented skin blemishes while staying sharper than the original; `input_noise_scale` 0.2 changes
little. The default five-frame windows took 5.5 min against 7.2 min for 17-frame windows that
share five frames, with no seam or flicker penalty.
