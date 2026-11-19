# Qwen3

`ask` answers single-turn questions about text, images and videos with any Qwen3.5-family checkpoint: Qwen3.5,
Qwen3.6 and Qwen3.8, dense or mixture-of-experts, natively multimodal (Apache-2.0). Its default is the newest,
abliterated (refusals removed); a request picks another by `model`. Decoding is greedy unless `temperature` is set; a
sampled answer replays from its `seed`, so the same request always gives the same answer, and it is memoized.

Its canvas tools, under LLM:
- **Write** (`write`, boxes named `text_1`, …): one answer to a box's `instructions`, which may mention pictures and
  videos (`<dog>`), each shown to the model right after its first mention. It samples at Qwen's instruct settings
  (temperature 0.7, top_p 0.8, presence_penalty 1.5), from a new seed each render unless `seed` is set:
  `cozy run qwen3/write "instructions=Continue: The sea at night"`.
- **Caption** (`caption`, boxes named `caption_1`, …): a detailed, literal description, in one paragraph of at most 120
  words, of the pictures and videos its `instructions` mention, the box's own words after the package's question. Let go
  on a picture or a video, it makes a box of words beside it reading `<that box>`. Its box and sampling are Write's. Another
  tool reads it by mention: `<caption_1>` in Upscale's `prompt` is its words.

Write the request in a file, `request.yaml`:

```yaml
prompt: Describe this character's clothing.
```

```sh
cozy run qwen3/ask --input request.yaml --file images=./character.png
```

Inline inputs work too, and override the file's:

```sh
cozy run qwen3/ask "Describe this character's clothing." --file images=./character.png
cozy run qwen3/ask "What happens, and when?" --file videos=./clip.mp4
cozy run qwen3/ask "What happens, and when?" --file videos=./clip.mp4 --model model=llmfan46/qwen3.6-35b-a3b-uncensored-heretic-native-mtp-preserved@0.1.0
```

The request and the model code are Runtime's shared Qwen-VL ones (`cozy_runtime.vlm`):
- `prompt` with `images` and `videos`, or `prompts`: up to 16 questions `{text, images, videos}`
  answered in one prefill and one decode.
- `system`, `max_tokens` (default 1024), `presence_penalty`, `stop`.
- `temperature` (default 0: greedy), `top_p`, `seed`. Qwen recommends 0.7 and 0.8 without
  thinking, 1.0 and 0.95 with it; Qwen3.8's card gives a `presence_penalty` of 1.5 without thinking and 0 with it.
- `thinking`: the model reasons first; only the answer after its thought is returned. A thought
  that `max_tokens` cuts off leaves an empty answer. The checkpoint's chat template sets the effort (Qwen3.8's `reasoning_effort`: upstream's
  defaults to xhigh, the default's to none).
- Video reading as Qwen-VL defines it: `fps`, `min_frames`, `max_frames` (null: the checkpoint's
  2 fps, 4 to 768 frames), `max_edge`, `video_end`.

The output is `texts`, one answer per question.

## Models

Each is its author's abliterated checkpoint (refusals removed), mirrored privately until its release. Refusals and KL
divergence from the stock model are the author's own numbers (Heretic's 100 harmful and 100 harmless prompts).

| model | of | parameters | bf16 | refusals (stock) | KL |
| --- | --- | --- | --- | --- | --- |
| `llmfan46/qwen3.8-27b-uncensored-heretic-native-mtp-preserved` (default) | Qwen3.8-27B | 27.78B dense | 55.6 GB | 3/100 (91) | 0.024 |
| `llmfan46/qwen3.6-35b-a3b-uncensored-heretic-native-mtp-preserved` | Qwen3.6-35B-A3B | 35.95B, 256 experts, ~3B active | 71.9 GB | 10/100 (83) | 0.002 |
| `llmfan46/qwen3.6-27b-uncensored-heretic-v2-native-mtp-preserved` | Qwen3.6-27B | 27.78B dense | 55.6 GB | 6/100 (92) | 0.002 |
| `huihui-ai/huihui-qwen3.5-9b-abliterated` | Qwen3.5-9B | 9B dense | 19.3 GB | not stated | not stated |
| `huihui-ai/huihui-qwen3.5-0.8b-abliterated` | Qwen3.5-0.8B | 0.8B dense | 1.7 GB | not stated | not stated |

Each keeps the vision tower (unedited, by its card), video processing and multi-token-prediction head. The 35B-A3B reads images and video
within about a point of the 27Bs at a ninth of the compute per token; the 27Bs lead on coding and agent tasks.

Each serves its int8 lane where it fits the GPU and a 4-bit lane where only that fits (`min_bits=4`): `w4a8` before
Blackwell, `nvfp4` on Blackwell. A routed expert is its own weight block, so a 24 GB card keeps 35B-A3B's hot experts
resident and pages the rest.

## Model preparation

Each checkpoint is its author's, mirrored as uploaded under the author's org: the upload keeps the repository's
`config.json`, tokenizer, processor and chat template, and TensorFS stores a MoE's fused experts one per expert. The lanes
are written from the bf16 commit on the same machine:

```sh
cozy model upload hf://llmfan46/Qwen3.8-27B-Uncensored-Heretic-Native-MTP-Preserved@4b9e1517a43fdca25872318ed272a1c88e72c360 llmfan46/qwen3.8-27b-uncensored-heretic-native-mtp-preserved --lane bf16 --mirror https://huggingface.co/llmfan46/Qwen3.8-27B-Uncensored-Heretic-Native-MTP-Preserved --on <name>
cozy model upload hf://llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved@9599ac17d26daf33daf0fdd8f6c897ff4c6dc89a llmfan46/qwen3.6-35b-a3b-uncensored-heretic-native-mtp-preserved --lane bf16 --mirror https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved --on <name>
cozy model upload hf://llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved@4928dd14830859583b97f35dfaba9018f7310b3e llmfan46/qwen3.6-27b-uncensored-heretic-v2-native-mtp-preserved --lane bf16 --mirror https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved --on <name>
cozy model upload hf://huihui-ai/Huihui-Qwen3.5-9B-abliterated@05b9e7c9b978ba29bdb8f50a49c30e4b91183339 huihui-ai/huihui-qwen3.5-9b-abliterated --lane bf16 --mirror https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated --on <name>
cozy model upload hf://huihui-ai/Huihui-Qwen3.5-0.8B-abliterated@4813135658fe51b2e535b8e77927d2f23909ce42 huihui-ai/huihui-qwen3.5-0.8b-abliterated --lane bf16 --mirror https://huggingface.co/huihui-ai/Huihui-Qwen3.5-0.8B-abliterated --on <name>
cozy model quantize <org/name>@sha256:<bf16 commit> --lane int8 --package ./qwen3 --on <name>    # and nvfp4, w4a8
```

The `int8` recipe writes every attention, gated-delta-net, MLP and expert projection of the language layers and of the
multi-token-prediction layer, where the checkpoint has one, as `int8.convrot`. The vision tower, embeddings, routers and
LM head keep their bytes. The 4-bit lanes keep that and write every MLP (a MoE's routed and shared experts) in 4 bits.

`scripts/qwen3-model-proof.py` checks the model code against Transformers on CPU with small random-weight models, dense
and MoE.
