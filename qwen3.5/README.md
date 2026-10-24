# Qwen3.5

`ask` answers a single-turn question about text, images and videos with Qwen3.5, a natively
multimodal model. Decoding is greedy unless `temperature` is set; a sampled answer replays from
its `seed`, so the same request always gives the same answer, and the answer is memoized. The
request and the model code are Runtime's shared Qwen-VL ones (`cozy_runtime.vlm`), as `qwen3.6`'s.

Write the request in a file, `request.yaml`:

```yaml
prompt: Describe this character's clothing.
```

```sh
cozy run qwen3.5/ask --input request.yaml --asset images=./character.png
```

Inline inputs work too, and override the file's:

```sh
cozy run qwen3.5/ask "Describe this character's clothing." --asset images=./character.png
cozy run qwen3.5/ask "What happens, and when?" --asset videos=./clip.mp4
```

Inputs:
- `prompt` with `images` and `videos`: one question. The model sees the images, then the videos, then
  the text. Alternatively, `prompts` holds up to 16 questions, each `{text, images, videos}`, answered
  in one pass: one prefill and one decode for all of them.
- `system`, `max_tokens` (default 1024), `presence_penalty`, and `stop`, where an answer ends before
  the first stop string.
- `temperature` (default 0: greedy), `top_p` and `seed`.
- `thinking`: the model reasons first; only the answer after its thought is returned.
- Video reading, as Qwen-VL defines it:
  - about `fps` frames a second, at least `min_frames` and at most `max_frames`, an even count;
  - each frame is the one showing at the middle of its slot, stamped with its `<t seconds>`;
  - null values take the checkpoint's video processor (2 fps, 4 to 768 frames);
  - `max_edge` downscales larger frames before the processor does;
  - `video_end` stops each video at that second.

The output is `texts`, one answer per question.

The token loop is a graph-captured decode drafted by Qwen3.5's multi-token-prediction head. It
declares `min_bits=4, phase="decode"`, so it serves the release's `nvfp4` lane where there is one.
Concurrent calls spread over the run's GPUs.

## Sizes

A dense Qwen3.5 checkpoint (0.8B, 2B, 4B, 9B or 27B) loads in the `model` slot as uploaded. The MoE
sizes (35B-A3B, 122B-A10B, 397B-A17B) store their MTP experts per expert, with gate and up
projections apart, and need their own converter before they load.

`qwen3.5-9b@0.2.0` is the default and the only published release. Pick another uploaded size with
`model.model=qwen3.5-27b@<release>`.

## Model preparation

The served checkpoint is the plain upload. TensorFS keeps the repository's `config.json` as the
construction config, and its tokenizer, processor and chat template as assets. Parameters load in
their stored dtype; Qwen3.5 keeps `A_log` and its gated norms in f32. Inference never downloads
anything.

```sh
cozy model upload hf://Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a qwen3.5-9b --lane bf16 --on <name>
```

The `int8` recipe writes every attention, gated-delta-net and MLP projection of a dense checkpoint's
language layers as `int8.convrot` (W8A8 on the int8 tensor cores). The vision tower, embeddings and
LM head keep their bytes. The 9B drops from 17.5 to 11.1 GiB, so it fits a 16 GB card.

```sh
cozy model quantize qwen3.5-9b@sha256:<bf16 commit> --lane int8 --package qwen3.5 --on <name>
cozy model publish qwen3.5-9b 0.2.0 sha256:<bf16 commit> sha256:<int8 commit>
```
