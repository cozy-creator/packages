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
`768p` (default, the native 1344×768), `1080p`, `1440p`, or `2160p` (4K UHD). `upscale.method`
picks how a size above 768p is reached. Audio and frame timing are retained either way.

```yaml
prompt: A street musician plays a violin under a neon sign that reads OPEN LATE.
duration_s: 5
resolution: 1440p
upscale:
  method: regenerate   # seedvr2 (default) | regenerate
  strength: 0.909      # regenerate only: 0.909 (default), 0.8, 0.7
```

| | `seedvr2` (default) | `regenerate` |
|---|---|---|
| What runs | SeedVR2 7B restores the decoded 768p original | NVIDIA's [Open-H3-Regenerate-2K](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/): the render's own latent is upscaled in latent space and re-rendered by H3 at the target, 3 steps (2 at 0.8 and 0.7) |
| Frame | 7:4 letterboxed into 16:9 | **fills 16:9: the 7:4 frame is stretched 1.6% horizontally**, as NVIDIA trained it |
| Original 768p video | published (`originals`) before the upscale finishes | not made: the latent is handed on in the same call, never decoded at 768p |
| Controls | SeedVR2's (Tuning, below) | `strength` |
| Faces, small text | kept | redrawn: at 0.909 a face can change expression; small text becomes other letters at every strength (large text stays legible) |

`regenerate` at `768p` is a refine-only pass at the native size. `strength` is the refine's
start noise: lower keeps more of the source (faces, on-screen text) and is faster; 0.909 is
NVIDIA's setting. The refine conditions on the prompt as text alone, so with references
(Ref2VA) it may redraw a character's face away from them. Where a size's rows do not fit the
card (2160p 15 s below 80 GB), the refine runs in overlapping temporal windows: never a
refusal. `minimax-h3/regenerate` runs the same route on an existing video (`video`, `prompt`,
`resolution`, `strength`, `seed`); its 768p frames are encoded first.

CPU orchestration runs H3 and SeedVR2 as separate GPU children. Omit `--gpus` so
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
detail. SeedVR2 3B was sharper but over-processed, and the in-house H3 latent upscaler with an
untrained H3 refine was soft and drifted identity; both were deleted. NVIDIA's trained refiner
(Open-H3-Regenerate-2K, 2026-10-28) came back beside SeedVR2 as `regenerate` on 2026-10-29.

## Tuning

With `method: seedvr2`, `upscale` passes SeedVR2's ComfyUI controls (seedvr2 README, Tuning) to
the upscale, under the upstream names; they apply only above 768p. Restoration can invent skin texture and fine detail;
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
