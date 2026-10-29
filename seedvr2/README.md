# SeedVR2 video restoration

Managed SeedVR2 7B restoration with the upstream Apache-2.0 architecture, one-step sampler,
fixed conditioning, standard PyTorch SDPA and Runtime-owned DiT/VAE weights. `upscale` takes a
video, `resolution` (`1080p`, `1440p`, `2160p`), a deterministic `seed`, and `aspect_ratio`
(`source` by default, or `16:9` for an exact letterboxed canvas). Frame presentation timestamps
and audio pass through Runtime's streaming media API.

Resolution defines the maximum 16:9 canvas; source-aspect output fits inside it. Internally,
padding meets the model's 16-pixel grid and is cropped from the result. Audio is decoded and
re-encoded by Runtime with its source clock; it is not copied byte-for-byte. `aspect_ratio=16:9`
outputs exactly 1920×1080, 2560×1440 or 3840×2160 using Pillow letterboxing; the subject is never
stretched. The Runtime sizes the VAE's tiles and the DiT's pieces from the GPU's free memory, and
frames stay in host memory between passes, so a whole clip restores on an 8 GB card.

## Tuning

`upscale` takes the ComfyUI SeedVR2 Video Upscaler node's controls under their upstream names
(numz/ComfyUI-SeedVR2_VideoUpscaler, Parameters):

| field | default | effect |
|---|---|---|
| `color_correction` | `lab` | `lab`, `wavelet`, `wavelet_adaptive`, `hsv`, `adain` or `none`: match the restoration's colour to the source. Upstream defaults to `wavelet` and recommends `lab` for fidelity; we measured `lab` closest. |
| `latent_noise_scale` | 0 | 0–1: noise on the conditioning latent; softens excessive invented detail. Try 0.1–0.2. |
| `input_noise_scale` | 0 | 0–1: noise on the input frames; reduces artifacts at high resolutions. |
| `batch_size` | whole clip | most frames per temporal window; must be 4n+1 (1, 5, 9, 13, 17, …). The DiT runs each window in the fewest pieces that fit the GPU. |
| `temporal_overlap` | 0 | 0–16 frames shared and crossfaded between windows. |
| `prepend_frames` | 0 | 0–32 mirrored frames restored before the start, then dropped. |
| `uniform_batch_size` | false | pad the last window to `batch_size`. |

`max_resolution`, offload and debug are not offered: `resolution` bounds the output, and the
Runtime memory manager places and pages the model.

Measured 2026-10-13 on one RTX Pro 6000 at 1080p (seed 581203) with the 15 s Ref2VA original of
H3 run 5515: execution time, and colour fidelity against the source after area-scaling back to
1344×768: mean CIELAB ΔE76, PSNR, chroma spread (σa\*/σb\* output/source); detail is the mean
|Laplacian| of the 1080p luma; flicker the mean |frame-to-frame luma change − the source's|.

| run | settings | time | ΔE76 | PSNR | σa\*/σb\* | detail | flicker |
|---|---|---|---|---|---|---|---|
| 5574 | `none`, 17/5 windows | 426 s | 3.90 | 30.29 dB | 1.15 / 1.16 | 2.53 | 4.72 |
| 5575 | `wavelet`, 17/5 | 426 s | 2.92 | 31.70 dB | 1.02 / 1.03 | 2.53 | 4.12 |
| 5576 | `lab`, 17/5 | 430 s | 2.87 | 31.89 dB | 1.00 / 0.99 | 2.52 | 4.06 |
| 5588 | `lab`, `batch_size` 5 (the 0.3 default) | 332 s | 2.64 | 32.25 dB | 1.00 / 0.99 | 2.47 | 3.34 |
| 5587 | `lab`, 17/5, `input_noise_scale` 0.2 | 424 s | 2.83 | 32.03 dB | 1.00 / 0.99 | 2.45 | 3.96 |
| 5585 | `lab`, 17/5, `latent_noise_scale` 0.1 | 433 s | 2.64 | 32.31 dB | 1.00 / 0.99 | 2.35 | 3.70 |
| 5586 | `lab`, 17/5, `latent_noise_scale` 0.2 | 427 s | 2.54 | 32.40 dB | 1.00 / 0.99 | 2.21 | 3.43 |

Without correction the restoration over-saturates chroma by about 15% (FL2VA run 5521: 14% / 7%);
`lab` and `wavelet` both remove it at no visible or measured detail cost and look alike, `lab`
slightly closer (FL2VA: ΔE 2.64 vs 2.66 vs 3.41 for none). `latent_noise_scale` visibly removes
invented skin blemishes on faces: 0.1 softens them, 0.2 mostly removes them, while staying sharper
than the source; `input_noise_scale` 0.2 barely changes this clip. Five-frame windows with no
overlap (upstream's default) were 23% faster than 17/5 with no seam or flicker penalty here.
Frames and outputs: `~/cozy_v2/outputs/seedvr2-color-ab-20261013/`.

## Weights

Weight source: `numz/SeedVR2_comfyUI` revision `09ced71023636e9bc8cdf9cdecfb2625d1e691e8`,
Apache-2.0 FP16 conversions of the ByteDance-Seed model. Upstream provenance and modifications
are in `seedvr2/upstream/NOTICE`.

TensorFS's `seedvr2` converter table makes the checkpoint at import: `dit` and `vae` from
`seedvr2_ema_7b_fp16.safetensors` and `ema_vae_fp16.safetensors`, and `conditioning` from
upstream's `pos_emb.pt`/`neg_emb.pt` (SeedVR `e4de8c24`), read by pinned digest and offset, never
unpickled. The installed package carries no model files and inference performs no file loading
or upstream download.

```sh
tfs import \
  hf://numz/SeedVR2_comfyUI@09ced71023636e9bc8cdf9cdecfb2625d1e691e8/seedvr2_ema_7b_fp16.safetensors \
  hf://numz/SeedVR2_comfyUI@09ced71023636e9bc8cdf9cdecfb2625d1e691e8/ema_vae_fp16.safetensors \
  https://raw.githubusercontent.com/ByteDance-Seed/SeedVR/e4de8c24441a67e1b7df56abea10645059bb1185/pos_emb.pt \
  https://raw.githubusercontent.com/ByteDance-Seed/SeedVR/e4de8c24441a67e1b7df56abea10645059bb1185/neg_emb.pt \
  --name seedvr2-7b
```

Publish the result as `bytedance-seed/seedvr2-7b@0.1.0`, lane `bf16`: the package's default release.
