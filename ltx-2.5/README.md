# LTX-2.5

`generate` makes synchronized video and stereo sound from text and an optional first frame.
It is the **LTX-2.5** video tool in the web UI. This package uses the LTX-2.5 22B Distilled model,
with its 8-step first stage and 3-step spatial refinement, at 24 fps.

```yaml
prompt: A small robot taps a ceramic mug with a spoon, listens to the ringing sound, then waves at the camera. Warm workshop lighting, a slow camera push, clear metallic taps and gentle room tone.
duration_s: 5
resolution: 1080p
seed: 20261115
```

```sh
cozy run ltx-2.5/generate --input request.yaml
cozy run ltx-2.5/generate --input request.yaml --file first_frame=opening.webp
```

Describe sounds and any spoken dialogue in the prompt. `audio: false` returns a silent MP4.
The output carries the actual width, height, fps, frame count, seed and prompt.

`resolution` is `1080p` (1920x1088) or `720p` (1280x704), on the model's native two-stage grid.
An explicit `aspect_ratio` overrides the first frame's ratio; without a frame it defaults to
the listed landscape size. Other ratios keep approximately the same pixel count. `duration_s`
accepts 1-20 seconds, snapped up to the causal `8k+1` frame grid.

The package stages the customized Gemma4-12B encoder, text connectors, transformer, spatial
upsampler, convolutional video VAE, audio VAE and 48 kHz vocoder under Runtime. Configs and
tokenizer assets travel with the model. The supplied diffusion decoder and DFR workflow are
separate from this convolutional-decode tool.

Sampling follows the released Diffusers Euler path. The newer native LTX pipeline's Euler
ancestral path has different intermediate arithmetic; matching its numerical output is a
separate comparison. See the [source assessment](../research/ltx-2.5/README.md).

The model uses the **LTX-2.x Community License**, including its use and redistribution terms;
the upstream inference code is Apache-2.0. Retain the model license and notices with weights.

Status: initial implementation; checkpoint normalization and real GPU qualification are in progress.
