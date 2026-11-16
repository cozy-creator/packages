# LTX-2.5

`generate` makes synchronized video and stereo sound from text, and optional first and last frames. A picture the
prompt mentions is a reference (a character, prop or place the video keeps), a sound it mentions is the soundtrack,
and Pick up from here continues a take. `sound` makes sound alone, and `retake` makes part of a take again.
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
cozy run ltx-2.5/generate --input request.yaml --file first_frame=opening.webp --file last_frame=closing.webp
```

A first frame replaces the video's first latent; a last frame guides its very last frame, as upstream's
`--image PATH FRAME_IDX 1.0` does (`ImageGuide`). Both are re-encoded at each stage's size, at the model's CRF 18.

## Prompts

Write what happens in your own words, as one flowing paragraph: the shot, the scene, the action, the people, the
camera and the sound, dialogue in quotes (Lightricks' [prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide)).
`prompt_upsampling` (on by default, as Lightricks' templates have it) first rewrites it with Lightricks' enhancer,
Gemma 4 E2B (`google/gemma-4-e2b-it`, fetched the first time it is used), and their own text-to-video and
image-to-video system prompts, into the caption style LTX-2.5 was trained on. The result's `prompt` is the text the
model read.

## References, soundtrack and pickup

- **Mentioned pictures** (`<amy>`, `<teapot>`) are panels of a reference sheet, in the order first mentioned, on
  black; Lightricks' Ingredients IC-LoRA (`lightricks/ingredients`) reads it in the first stage, and the prompt takes
  its two-part form: `Reference sheet: Left: amy. Right: teapot.` then `Generated video: <your text>`. It was
  trained on 768x448, 5-second clips; other sizes and longer clips work, less faithfully.
- **A mentioned video** sets the structure or the motion: Lightricks' Union Control IC-LoRA (`lightricks/union-control`,
  the LTX-2.3 adapter their LTX-2.5 workflow runs) follows its control maps in the first stage, at half the stage's
  size, as the adapter was trained. `control` picks the map, made as Lightricks' workflow makes it: `depth` (default,
  Video-Depth-Anything Small), `pose` ("Motion transfer": DWPose's body, hands and face, so the subject moves like the
  people in the clip), `edges` (Canny), or `map` (the clip is already a control map). Say what to do with the clip in
  the prompt ("<amy> dances the way <clip> does"), with the subject as the first frame or a mention.
  Pictures and a video can't be references together: their adapters take references at different sizes.
- **Keyframes** (`keyframes`: a picture and a time) each land on their exact frame, as upstream's `ImageGuide` does.
- **A sound mentioned after a picture** is that character's voice (`<amy> says, in <amyvoice>: "Hello."`): Licon
  MSR V2 (`liconstudio/multiple-subject-reference`, Apache-2.0) keeps up to five pictures and two of their voices,
  on both stages. Its prompt names them by slot: `Image 1: amy.`, `Audio 1: amyvoice, belongs only to Image 1.`,
  then your text with `<amy>` as Image 1. A voice is at most 5 seconds.
- **A sound mentioned with no picture** is the soundtrack: held as it is while the video is made to it (upstream's
  audio-to-video), and returned unchanged. A clip longer than the sound is shortened to it.
- **Pick up from here** keeps the take up to its frame and carries its last 25 frames and their sound into the
  new part, as upstream's long-video windows carry, so the new part continues the take.
- **LoRAs** (`--lora`, the page's LoRAs) apply to both stages, as upstream's distilled pipeline applies them.

Describe sounds and any spoken dialogue in the prompt. `audio: false` returns a silent MP4.
The output carries the actual width, height, fps, frame count, seed and prompt.

`resolution` is `1080p` (1920x1088) or `720p` (1280x704), on the model's native two-stage grid.
An explicit `aspect_ratio` overrides the first (else last) frame's ratio; without a frame it defaults to
the listed landscape size. Other ratios keep approximately the same pixel count. `duration_s`
accepts 1-20 seconds, snapped up to the causal `8k+1` frame grid.

The package stages the customized Gemma4-12B encoder, text connectors, transformer, spatial
upsampler, convolutional video VAE, audio VAE and 48 kHz vocoder under Runtime. Configs and
tokenizer assets travel with the model. The supplied diffusion decoder and DFR workflow are
separate from this convolutional-decode tool.

`lightricks/ltx-2.5@0.2.0` has two lanes: `bf16` and `int8` (int8 ConvRot on the transformer blocks and Gemma's language
layers, [`quantize.py`](ltx_2_5/quantize.py)). A GPU with int8 tensor cores runs `int8`; `--lane bf16` pins 16 bits.
`ENHANCER_INT8` writes the same lane of the Gemma 4 E2B prompt enhancer.

```sh
cozy model quantize lightricks/ltx-2.5@0.1.0 --lane int8 --package ./ltx-2.5 --on <machine>
```

Sampling follows the released Diffusers Euler path. The newer native LTX pipeline's Euler
ancestral path has different intermediate arithmetic; matching its numerical output is a
separate comparison. See the [source assessment](../research/ltx-2.5/README.md).

The model uses the **LTX-2.x Community License**, including its use and redistribution terms;
the upstream inference code is Apache-2.0. Retain the model license and notices with weights.

Real CLI renders cover synchronized sound, first-frame conditioning, portrait output and the
maximum 20-second path on H200. See the [execution report](../research/video-models/2026-11-15.md)
for exact cohorts and the separate publication/installed-package gates.

The [CPU importer](../examples/client-scripts/ingest-ltx25.py) reproduces the pinned Distilled
checkpoint with BF16 transformer/video and F32 audio VAE/vocoder. It runs before serving;
`HF_TOKEN` supplies gated source access, or `--token-stdin` reads it from standard input.

## Sound and Retake

- **`sound`** (LTX-2.5 Sound) makes sound alone from text: speech, music, a soundscape. It runs only the model's
  audio branch, as Lightricks' text-to-audio template does (`LTXVAudioOnlyModel`), and returns 48 kHz stereo FLAC.
  Quote what is said: `A woman saying: "Oh, what a lovely day we are having!"`.
- **`retake`** (LTX-2.5 Retake) makes part of a take again from text and keeps the rest, as upstream's
  `RetakePipeline` does: `start_s` to `end_s` is redone, its picture and sound, the picture only, or the sound
  only (`redo`). It runs one distilled stage at the take's size (sides on 32, at most 1080p's area), on at most
  20 s around the window; the rest of a longer take is kept as it was.

```sh
cozy run ltx-2.5/sound prompt='Rain on a tin roof, distant thunder.' duration_s=8
cozy run ltx-2.5/retake --file video=take.mp4 start_s=2 end_s=3.5 prompt='She laughs and claps her hands.'
```

## Several GPUs

A single video can use Ulysses sequence parallelism across 2, 4 or 8 GPUs:

```sh
cozy run ltx-2.5/generate --on renderbox --gpus=2 prompt="A quiet workshop." duration_s=20 resolution=1080p audio=true
```

Two H200 GPUs have produced full 20-second text-to-video and first-frame videos with stereo sound. Four/eight-rank transformer parity is checked on CPU; production GPU timing at those degrees remains unmeasured. See the [qualification report](../research/video-models/2026-11-15.md) for actual timings, thermal conditions and ComfyUI comparisons.
