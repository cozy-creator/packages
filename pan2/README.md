# PAN2

PAN2 turns a scene and a timeline of actions into silent video. Its `instructions` take a picture as a mention
(`<opening>`), as every canvas box does, and the first picture mentioned is the first frame the scene opens on.
`generate` is also the **PAN2** video tool in the web UI. The default is 10 seconds at 24 fps,
at the model's native 1248x704 size. The maximum is 20 seconds (481 frames).

```yaml
instructions: A red ceramic cup stands on a wooden table. A hand lifts it, pours water into a clear glass, then puts the cup down. One continuous close shot in natural window light.
duration: 20
seed: 20261115
```

```sh
cozy run pan2/generate --input request.yaml
cozy run pan2/generate instructions='<opening> A hand lifts the cup.' \
  'references:=[{"name": "opening", "image": "opening.webp"}]'
```

The prompt rewriter is PAN2's own for a local model (ifm-ai/PAN2@0fb907b6 `inference/prompts_rewritter.py`): Qwen3.5-9B,
huihui-ai's abliterated one (`huihui-ai/huihui-qwen3.5-9b-abliterated`, refusals removed) by default, or Qwen's own with
`--model reasoner=qwen/qwen3.5-9b@0.2.0`. Its instruction and the request go in one user turn (the first frame before them
with its "Reference image" line when one is mentioned), thinking off, sampled at temperature 0.2 and top-p 0.9 from the
render's seed, and its dense natural-language caption is used as written. Ours alone: an empty answer renders the
instructions as written, with a warning. PAN2's agent skill (`inference/pan-prompt-rewriter/`, a structured JSON caption
for a reasoning agent) is the other upstream path; with a thinking model it is a possible later improvement, unmeasured.
`prompt_upsampling: false` sends the instructions to the encoder as written, as PAN2's `generate.py` does. The encoder
reads 1536 caption tokens; past them a caption is cut, with a warning, as upstream cuts it. The returned `prompt` is the actual caption used.

Advanced controls are `steps` (50 by default), `guidance` (2.5), `negative_prompt`, and the seed.
The model's own negative caption is used by default. Output sides snap to 32 pixels at approximately
the native pixel count; `aspect_ratio` overrides the first frame's ratio. `duration` takes 1-20 whole seconds, as H3's beats do; frames snap up to `4k+1`.
The result reports actual width, height, frame count, fps and seed with its MP4.

The checkpoint includes its Qwen text encoder, Wan2.2 VAE, configs and tokenizer. Runtime owns
loading and residency; inference does not fetch models from Hugging Face. The separate conceptual
reasoner is an on-demand model slot (`reasoner`), defaulting to `huihui-ai/huihui-qwen3.5-9b-abliterated@0.1.0`.

The text encoder (stock Qwen3.5-9B's language model) is its own slot, `text_encoder`; `model` reads the DiT and the VAE.
By default it is pottokao's Heretic abliteration of Qwen3.5-9B (`pottokao/moss-image-text-encoder-heretic-pan2`: 32 of
its tensors changed, 2/100 refusals against 100/100, KL 0.050), the weights moss-image encodes with (se-160). PAN2's own
is any PAN2 checkpoint: `model.text_encoder=ifm/pan2@0.3.0`.

`ifm/pan2@0.3.0` has two lanes: `bf16` and `int8` (int8 ConvRot on the DiT blocks and the Qwen encoder's language
layers, [`quantize.py`](pan2/quantize.py)). A GPU with int8 tensor cores runs `int8`; `--lane bf16` pins 16 bits.
The `int8` DiT is in the curve form (ComfyUI's, se-140): its modulation input is a table of 192 coordinates at 4097
points of t, and each AdaLN linear takes those coordinates, 0.8 GB in place of 17.7 GB of F32 (a 24 GiB DiT, not 41).
[`lanes.py`](pan2/lanes.py) `curve` writes it from the full form; the recipe then quantizes it:

```sh
cozy run ./pan2/curve ifm/pan2@0.2.0 ifm/pan2 --on <machine>
cozy model quantize ifm/pan2@<the curve commit> --lane int8 --package ./pan2 --on <machine>
```

PAN2 code and weights are Apache-2.0, with upstream notices retained. The optional upstream
`pan2-guardrail` package is separate from this generator. See the [assessment](../research/pan2/README.md)
for pinned sources, component/dtype details and remaining execution evidence.

Real CLI renders cover the native 10-second and maximum 20-second paths on H200. See the
[execution report](../research/video-models/2026-11-15.md) for exact cohorts, timings, UI checks,
comparison caveats and the separate publication/installed-package gates.

## Several GPUs

The simulator supports Ulysses sequence parallelism across 2, 4 or 8 GPUs. Use `--gpus=2` with a machine that has two GPUs; the video and text token streams are split inside the transformer. The optional caption reasoner remains an auxiliary model.

A full 20-second, 1248×704, 50-step video completed on two H200s in 1247.600 seconds. One card thermally throttled, and the earlier single-H200 run used a different worker cohort, so this is not a clean scaling ratio. Four/eight-rank transformer parity is checked on CPU; production GPU runs at those degrees remain unmeasured. See the [qualification report](../research/video-models/2026-11-15.md).
