# Higher-resolution H3 video

`upscale` (MiniMax H3 Upscale) takes one whole take, of any length, `to` `1080p`, `1440p` (default) or `2160p` with
NVIDIA's [Open-H3-Regenerate-2K](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/): the take's 768p latent is
upscaled in latent space and re-rendered by H3 at the target in 3 steps, its sound and frame timing kept. A project
upscales each approved take, whole, before the edit: per-segment upscaling shows a change of detail at the joins
(run 20682). SeedVR2 is its own package and box, for any video.

```yaml
# cozy run minimax-h3/upscale --input up.yaml   (in a project, video: <shot_1>)
video: <shot_1>
to: 1440p
strength: 0.7          # under Advanced: 0.7 (default), 0.8 or 0.909 (NVIDIA's)
```

- **Source.** The take's own latents when H3 made it and its machine still has them (no encode, as NVIDIA's
  `--input-latents`), and the text they carry; else its frames, encoded, and its description as the text. With
  neither latents nor a description it refuses before any GPU: give the video a description.
- **Frame.** It fills 16:9: H3's 7:4 frame is stretched 1.6% horizontally, as NVIDIA trained it.
- **Faces, small text.** Redrawn: at 0.909 a face becomes another person's and a hat or sunglasses can change (0.7
  keeps them); small text becomes other letters at every strength (large text stays legible).
- **Strength** is the refine's start noise: lower keeps more of the take and is faster.
- **Size.** Where a size's rows do not fit the card (2160p 15 s below 80 GB, or a long take), the refine runs in
  overlapping temporal windows. It holds the take's frames in host memory: about 3 GB per 40 s of 768p source, plus the
  output (11 GB for 40 s at 1440p). A video already that big comes back as it is.

Upscaling can restore or synthesize texture; it cannot correct wrong motion or character identity. Review moving
faces, hands and clothing at normal speed.
