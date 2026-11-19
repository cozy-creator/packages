# LTX-2.5

`generate` makes synchronized video and stereo sound from text and an optional first frame.
It is the **LTX-2.5** video tool in the web UI. This package uses the LTX-2.5 22B Distilled model,
with its 8-step first stage and 3-step spatial refinement, at 24 fps.

```yaml
prompt: A small robot taps a ceramic mug with a spoon, listens to the ringing sound, then waves at the camera. Warm workshop lighting, a slow camera push, clear metallic taps and gentle room tone.
duration_s: 20
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
An explicit `aspect_ratio` overrides the first frame's ratio, and the frame's centre is cropped to it (as native LTX
and its ComfyUI template do), never stretched; without a frame it defaults to the listed landscape size. Other ratios keep approximately the same pixel count. `duration_s`
accepts 1-20 seconds, snapped up to the causal `8k+1` frame grid.

The package stages the customized Gemma4-12B encoder, text connectors, transformer, spatial
upsampler, convolutional video VAE, audio VAE and 48 kHz vocoder under Runtime. Configs and
tokenizer assets travel with the model. The supplied diffusion decoder and DFR workflow are
separate from this convolutional-decode tool.

`lightricks/ltx-2.5@0.2.0` has two lanes: `bf16` and `int8` (int8 ConvRot on the transformer blocks and Gemma's language
layers, [`quantize.py`](ltx_2_5/quantize.py)). A GPU with int8 tensor cores runs `int8`; `--lane bf16` pins 16 bits.
`ENHANCER_INT8` writes the same lane of the Gemma 4 E2B prompt enhancer; no function runs an enhancer yet.

```sh
cozy model quantize lightricks/ltx-2.5@0.1.0 --lane int8 --package ./ltx-2.5 --on <machine>
```

Each stage conditions on the first frame encoded at that stage's size: the refinement re-encodes it at full size, as
native LTX-2.5 and its ComfyUI template do, rather than keeping the upsampled half-size encode (se-161).
Sampling follows the released Diffusers Euler path. The newer native LTX pipeline's Euler
ancestral path has different intermediate arithmetic; matching its numerical output is a
separate comparison. See the [source assessment](../research/ltx-2.5/README.md).

## Continuing a take, and redoing a range of one

Both run on one canvas of at most 20 s, where known frames and sound are held and the rest is made: upstream's retake
mask (`TemporalRegionMask`), sampled as LTX Desktop samples distilled 2.5's retake and extend (an ancestral 8-step
stage, then a same-size 3-step Euler refine). The take is read two seconds at a time, never whole.

- **Continue:** `generate` with a `pickup` (the page's "pick up from here") holds the take's last 25 frames and their
  sound (upstream's long-video carry) and makes `duration_s` after them. A long video is the project's chained boxes.
- **Redo a range:** `retake` makes `start_s`-`end_s` of `video` again from `prompt`, holding the frames and sound on
  both sides, its context filling the canvas (Desktop's window) with the mask widened 0.5 s into it. A range longer
  than one canvas runs in pieces left to right. A shot change within 0.8 s of an edge (a jump in the frame-to-frame
  difference) becomes that edge, with nothing held across it. `redo` keeps the take's sound or picture.

```sh
cozy run ltx-2.5/retake --file video=take.mp4 start_s=10 end_s=16 prompt="She turns to the window and laughs."
```

The result is the whole take. Only the window is encoded again: the edit, a linear blend over the 16 frames just
outside it, and a 40 ms equal-power sound crossfade (in the held context; just inside at a shot change).
`Outputs.splice_video` copies every other packet of the take byte for byte. A take it can't splice (not H.264/AAC MP4,
an edit list, not 24 fps) is written again whole, with a warning.

The model uses the **LTX-2.x Community License**, including its use and redistribution terms;
the upstream inference code is Apache-2.0. Retain the model license and notices with weights.

Real CLI renders cover synchronized sound, first-frame conditioning, portrait output and the
maximum 20-second path on H200. See the [execution report](../research/video-models/2026-11-15.md)
for exact cohorts and the separate publication/installed-package gates.

The [CPU importer](../examples/client-scripts/ingest-ltx25.py) reproduces the pinned Distilled
checkpoint with BF16 transformer/video and F32 audio VAE/vocoder. It runs before serving;
`HF_TOKEN` supplies gated source access, or `--token-stdin` reads it from standard input.

## Text encoder

The Gemma 4 12B encoder is its own slot, `text_encoder`; `model` reads everything else. Its default is abliterated
(Paul, 11-17: "use abliterated models wherever possible"): `igorls/gemma-4-12b-it-heretic-v1-ltx-2.5`, igorls'
[Heretic](https://huggingface.co/igorls/gemma-4-12B-it-heretic-v1) of Gemma 4 12B (0/100 refusals, KL 0.028) moved onto
Lightricks' encoder. That encoder is Lightricks' fine-tune of Gemma 4 12B (diverofdark measured 1-11% from Google's from
layer 11 on), so the edit goes on as its difference: Heretic changed 22 tensors (`down_proj` in layers 22-34, `o_proj` in
29-37), and each is Lightricks' plus `heretic - google`, rounded to bf16, as
[diverofdark's int8 file](https://huggingface.co/diverofdark/LTX-2.5-Gemma4-12B-heretic-int8-convrot) did it. The other
655 tensors are Lightricks'. The slot's class, `Ltx25TextEncoder`, builds only the encoder, so Lightricks' own is any
LTX-2.5 checkpoint: `--model text_encoder=lightricks/ltx-2.5@0.2.0`.

It is made on a machine by `transplant` ([`encoder.py`](ltx_2_5/encoder.py)) from
`igorls/gemma-4-12B-it-heretic-v1@192d14f6` and `google/gemma-4-12B-it@707f0a3b`, written to `int8` by this package's
recipe, and each lane cut to the encoder, its tokenizer and Lightricks' licence (a header alone, on the same chunks):

```sh
cozy model quantize igorls/gemma-4-12b-it-heretic-v1-ltx-2.5@sha256:<bf16 commit> --lane int8 --package ./ltx-2.5 --on <machine>
cozy model upload 'igorls/gemma-4-12b-it-heretic-v1-ltx-2.5@sha256:<commit>:text_encoder,tokenizer,model' \
  igorls/gemma-4-12b-it-heretic-v1-ltx-2.5 --lane <lane> --mirror https://huggingface.co/igorls/gemma-4-12B-it-heretic-v1 --on <machine>
```

Licences: LTX-2.x Community License (Lightricks' encoder); Gemma 4, Apache-2.0 (Google's card); igorls' Heretic, `gemma`
(its card).

## Several GPUs

A single video can use Ulysses sequence parallelism across 2, 4 or 8 GPUs:

```sh
cozy run ltx-2.5/generate --on renderbox --gpus=2 prompt="A quiet workshop." duration_s=20 resolution=1080p audio=true
```

Two H200 GPUs have produced full 20-second text-to-video and first-frame videos with stereo sound. Four/eight-rank transformer parity is checked on CPU; production GPU timing at those degrees remains unmeasured. See the [qualification report](../research/video-models/2026-11-15.md) for actual timings, thermal conditions and ComfyUI comparisons.
