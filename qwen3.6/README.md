# Qwen3.6

`ask` answers single-turn questions about text, images and videos with Qwen3.6, a natively
multimodal model (Apache-2.0). Decoding is greedy unless `temperature` is set; a sampled answer
replays from its `seed`, so the same request always gives the same answer, and it is memoized.

Write the request in a file, `request.yaml`:

```yaml
prompt: Describe this character's clothing.
```

```sh
cozy run qwen3.6/ask --input request.yaml --asset images=./character.png
```

Inline inputs work too, and override the file's:

```sh
cozy run qwen3.6/ask "Describe this character's clothing." --asset images=./character.png
cozy run qwen3.6/ask "What happens, and when?" --asset videos=./clip.mp4
```

The request and the model code are Runtime's shared Qwen-VL ones (`cozy_runtime.vlm`), the same
as the `qwen3.5` package's:
- `prompt` with `images` and `videos`, or `prompts`: up to 16 questions `{text, images, videos}`
  answered in one prefill and one decode.
- `system`, `max_tokens` (default 1024), `presence_penalty`, `stop`.
- `temperature` (default 0: greedy), `top_p`, `seed`. Qwen recommends 0.7 and 0.8 without
  thinking, 1.0 and 0.95 with it, and a `presence_penalty` of 1.5 for both.
- `thinking`: the model reasons first; only the answer after its thought is returned. A thought
  that `max_tokens` cuts off leaves an empty answer.
- Video reading as Qwen-VL defines it: `fps`, `min_frames`, `max_frames` (null: the checkpoint's
  2 fps, 4 to 768 frames), `max_edge`, `video_end`.

The output is `texts`, one answer per question.

## Sizes

| model | parameters | bf16 | int8 lane | 4-bit |
| --- | --- | --- | --- | --- |
| `qwen3.6-35b-a3b` (default) | 35.95B, 256 experts, 8 routed + 1 shared, ~3B active | 71.9 GB | 38.3 GB | 23.6 GB |
| `qwen3.6-27b` | 27.78B dense | 55.6 GB | 31.2 GB | 20.6 GB |

The 35B-A3B is the default: on images and video it is within about a point of the 27B, at a ninth
of the compute per token. Each routed expert is its own weight block, so a 24 GB card keeps the
hot experts resident and pages the rest. The 27B leads on coding and agent tasks.

## Model preparation

The upload keeps the repository's `config.json`, tokenizer, processor and chat template. TensorFS
stores 35B-A3B's fused experts one per expert at ingest.

```sh
cozy model upload hf://Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0 qwen/qwen3.6-35b-a3b --lane bf16 --on <name>
cozy model upload hf://Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 qwen/qwen3.6-27b --lane bf16 --on <name>
cozy model quantize qwen/qwen3.6-35b-a3b@sha256:<bf16 commit> --lane int8 --package qwen3.6 --on <name>
cozy model quantize qwen/qwen3.6-35b-a3b@sha256:<bf16 commit> --lane w4a8 --package qwen3.6 --on <name>
```

The `int8` recipe writes every attention, gated-delta-net, MLP and expert projection of the
language layers and of the multi-token-prediction layer as `int8.convrot`. The vision tower,
embeddings, routers and LM head keep their bytes. The 4-bit lanes keep that and write every MLP (35B-A3B's
routed and shared experts) in 4 bits: `w4a8` before Blackwell (int8 tensor cores), `nvfp4` on
Blackwell. Decoding serves a release's 4-bit lane where it has one (`min_bits=4`).

`scripts/qwen3.6-model-proof.py` checks the model code against Transformers on CPU with small
random-weight models, dense and MoE.
