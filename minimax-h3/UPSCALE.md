# Higher-resolution H3 video

`upscale` takes one whole shot, of any length, to `resolution`: `1080p`, `1440p` (default) or `2160p`.
`upscale.model` picks how. Audio and frame timing are kept either way. A project upscales
each approved shot, whole, before the edit: per-piece upscaling shows a change of detail at
the joins (run 20682).

```yaml
# cozy run minimax-h3/upscale --input up.yaml --file video=./shot.mp4
resolution: 1440p
upscale:
  model: regenerate-2k   # seedvr2 (default) | regenerate-2k
  strength: 0.909        # regenerate-2k only: 0.909 (default), 0.8, 0.7
prompt: A street musician plays a violin under a neon sign that reads OPEN LATE.   # regenerate-2k only
```

| | `seedvr2` (default) | `regenerate-2k` |
|---|---|---|
| What runs | SeedVR2 7B restores the 768p frames | NVIDIA's [Open-H3-Regenerate-2K](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/): the shot's 768p latent is upscaled in latent space and re-rendered by H3 at the target, 3 steps (2 at 0.8 and 0.7) |
| Frame | H3's 7:4 letterboxed into 16:9; any other video keeps its aspect | **fills 16:9: the 7:4 frame is stretched 1.6% horizontally**, as NVIDIA trained it |
| Controls | SeedVR2's (Tuning, below) | `strength`, and the shot's `prompt` |
| Faces, small text | kept | redrawn: at 0.909 a face can change expression; small text becomes other letters at every strength (large text stays legible) |

`regenerate-2k` at `768p` is a refine-only pass at the native size. `strength` is the
refine's start noise: lower keeps more of the source and is faster. Where a size's rows do not
fit the card (2160p 15 s below 80 GB, or a long shot), the refine runs in overlapping temporal
windows. It holds the shot's frames in host memory: about 3 GB per 40 s of 768p source, plus
the output (11 GB for 40 s at 1440p).

CPU orchestration runs H3 and SeedVR2 as separate GPU children. Omit `--gpus` so
single-GPU SeedVR2 and multi-GPU H3 can each use their supported count. An explicit count
applies to every GPU child and is never silently reduced.

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
(Open-H3-Regenerate-2K, 2026-10-28) came back beside SeedVR2 as `regenerate-2k` on 2026-10-29.

## Tuning

With `model: seedvr2`, `upscale` passes SeedVR2's ComfyUI controls (seedvr2 README, Tuning) to
the upscale, under the upstream names; they apply only above 768p. Restoration can invent skin texture and fine detail;
`latent_noise_scale` softens it:

```sh
cozy run minimax-h3/upscale --file video=./shot.mp4 resolution=1080p \
  'upscale:={"latent_noise_scale": 0.1}'
```

Measured on the Ref2VA original above at 1080p (seedvr2 README has the full table): colour
correction removes a ~15% chroma over-saturation (ΔE76 3.90 → 2.87 with `lab`) at no detail
cost; `latent_noise_scale` 0.1 and 0.2 lower the detail measure by 7% and 12% and visibly remove
invented skin blemishes while staying sharper than the original; `input_noise_scale` 0.2 changes
little. The default five-frame windows took 5.5 min against 7.2 min for 17-frame windows that
share five frames, with no seam or flicker penalty.
