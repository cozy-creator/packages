# PAN2: candidate video package

Assessment date: 2026-11-15. Status: research; no Cozy implementation or inference qualification.

Task owner: Codex. Purpose: assess IFM/PAN2 as a supported Cozy video tool and model.
Branch: `docs/pan2-assessment`. Base: `origin/master` at
`e4b179bbd099f681658dc7ef5a059f484969cf94`. Worktree:
`/home/fidika/cozy/.worktrees/packages/pan2-assessment`.

## Upstream assessment

PAN2 is a plausible additional video generator, especially for text-described actions and timelines.
Its public checkpoint supports text-to-video and image-to-video from one starting frame.
The smallest integration can wrap the official Diffusers modular pipeline; a new transformer/kernel
implementation is a separate, much larger project. Quality, latency and memory in Cozy remain unmeasured.
[Released pipeline source][df-pipeline], [model configuration][hf-transformer].

This assessment used source and metadata only. No weights were downloaded, inference run, paid GPU
rented, or upstream author contacted. Numerical performance below is attributed to upstream.

## Exact sources inspected

| Source | Revision | Date / qualification |
| --- | --- | --- |
| `IFM/PAN2` model repository | `82f54c6cea3e20a9c451b61d7360016e5970503d` | HF API last modified 2026-11-12; configs, indices, notices and model card inspected |
| `ifm-ai/PAN2` reference code | `0fb907b6e16a82ead6d691d1de2eef6299c71e9a` | 2026-11-05; inference, architecture and dependency sources inspected |
| Diffusers stable release | `v0.43.0`, commit `423ab90580b2847998591669b8326f47b6d14d8b` | Released 2026-11-12; contains PAN2 modular blocks and transformer |
| Diffusers main | `43e30028af5120d8b0868cb6975b9fad39cc4fab` | 2026-11-13; version `0.44.0.dev0`, also contains PAN2 |
| vLLM-Omni PAN2 support | PR #8602, merge `171ab6a930b254851aeeb4ad180f5a02f17cf5f9` | Merged 2026-11-10; PR benchmark is historical evidence, not final-default qualification |
| Supplied paper | arXiv `2507.05169v5` | *Critique of World Model*, revised 2026-06-17 |

Sources: [HF revision][hf-tree], [reference commit][native-commit], [Diffusers release][df-release],
[release tag object][df-tag], [vLLM PR][vllm-pr], [paper version][paper].

## Released behavior and boundaries

| Capability | Evidence-supported public boundary |
| --- | --- |
| T2V / I2V | One checkpoint; text plus an optional single first-frame image. |
| Duration | Up to 481 frames, approximately 20 seconds at 24 fps; native default 241 frames, Diffusers/vLLM default 313. |
| Output size | Published recipe is height 704 × width 1248. Diffusers requires spatial dimensions divisible by 32 and frames of form `4k+1`; other sizes are configurable, not quality-qualified here. |
| Actions / camera control | Structured caption fields describe timed events, subjects and camera motion; these are text conditioning, not a numeric controller, trajectory tensor or guaranteed action execution. |
| Video continuation | No public video-input workflow found. Chaining a previous clip's last frame through I2V is possible compositionally, but preserves only one image plus text; it is not a released video-conditioned continuation model. |
| Multiple references / keyframes | Public workflow accepts one first frame. No last-frame, multiple-image or arbitrary-keyframe workflow is exposed. |
| Audio | No audio-generation component or audio output is exposed by the inspected pipeline; native output writer writes video frames. |
| Persistent world state | No persistent perceptual memory is maintained by a pipeline call. The model card explicitly reports appearance drift across chained clips. |

Sources: [native inference contract][native-inference], [Diffusers input and condition construction][df-before],
[Diffusers image encoder][df-encoders], [native video writer][native-generate], [model limitations][model-card].
The absence statements describe inspected public interfaces, not every possible research extension.

## Prompting and the supplied paper

The diffusion model consumes captions directly. A separate LLM rewrites a free-text request into
structured JSON describing style, lighting, initial state, subjects and a timeline of events/actions.
The native generator does not run this rewrite in process. The rewriter can use a local instruction
model or an agent following the published caption policy; there is no dedicated released conceptual-
reasoner checkpoint required by the video pipeline. Caption capacity is 1,536 tokens; the rewriter
uses a 1,480-token soft limit. [Workflow and token contract][native-inference], [rewriter policy][rewriter].

Use compact structured JSON for a reference qualification. The reference README says free-text
prompts can work, while the vLLM recipe warns short unstructured prompts yield unusable video;
Diffusers' short demo prompt establishes API usage, not comparable quality. A user-facing free-text
interface therefore needs a deliberate rewriting stage. [Reference examples][native-readme],
[vLLM prompt contract][vllm-recipe], [Diffusers guide][df-guide].

The supplied paper is an architectural essay, not the PAN2 release report. It proposes Generative
Latent Prediction with mixed discrete/conceptual and continuous/perceptual representations, and a
broader Physical, Agentic and Nested system. It says detailed PAN results belong to a separate
manuscript; its v5 text contains neither `PAN2` nor the 29B release specification. PAN2's model card
explicitly adopts that GLP framing. The project site still labels its PAN2 research paper as forthcoming. A separate official technical
blog supplies evaluation and ablation detail: PAN2 reports 75.1% step completion and 34.0% full-plan
completion, versus H3's 75.1% and 32.5%, under that study. Its fixed-diffusion ablation reports
0% full-plan completion from raw action instructions, 27.8% after format-only rewriting, and 34.0%
with conceptual reasoning. These are upstream study results, not an independent Cozy comparison.
The actual integration authorities are the checkpoint configs and executable source.
[Paper, §§1 and 5][paper], [project site][project], [technical blog][technical-blog].

## Architecture and checkpoint census

| Component | Released architecture / metadata |
| --- | --- |
| Diffusion backbone | 29,201,996,992 parameters; 44 dual-stream joint-attention blocks, width 4096, 32 heads × 128; four caption-refiner blocks. |
| Conditioning | 97 input channels = 48 noisy latent channels + 48 condition channels + one mask; 48 output channels. |
| Position / MLP | 3-axis RoPE `[16,56,56]`, theta 256, temporal scale 2/3; video MLP ratio 8, text ratio 4. |
| Text encoder | Qwen3.5-9B text model; packaged text weights contain 7,936,684,544 parameters. Hidden width 4096; full and linear attention are mixed. Conditioning uses `hidden_states[-3]`, skipping the last two layers. |
| VAE | Wan2.2 `AutoencoderKLWan`, 48 latent channels, 16× spatial and 4× temporal compression; DiT patch `[1,2,2]`. |
| Tokens | At 704×1248, latent spatial grid 44×78 becomes 22×39 = 858 video tokens per latent frame. At 481 output frames this is 121×858 = 103,818 video tokens, before caption tokens. |
| Sampling | Flow-matching Euler, shift 9; defaults 50 steps, CFG 2.5. I2V's given latent frame uses the positive prediction without CFG; denoising latents stay FP32 between updates. |

Sources: [DiT config][hf-transformer], [text config][hf-text], [text index][hf-text-index],
[VAE config][hf-vae], [HF parameter census][hf-api], [Diffusers text encoding][df-encoders],
[Diffusers denoising][df-denoise], [native defaults][native-inference]. Token counts are arithmetic
from those public shapes, not a measured memory profile.

| Stored component | Safetensors files | Exact file bytes | GiB |
| --- | ---: | ---: | ---: |
| DiT | 14 | 67,369,648,792 | 62.743 |
| Text encoder | 4 | 15,873,415,904 | 14.783 |
| VAE | 1 | 2,818,777,808 | 2.625 |
| Total | 19 | 86,061,842,504 | 80.151 |

This sums HF tree metadata for `.safetensors` files, including their headers and excluding tokenizer,
configs, demo assets and optional guardrail weights. DiT tensor payload metadata is 67,369,482,624
bytes. HF reports 24,719,252,672 BF16 and 4,482,744,320 FP32 DiT parameters, explaining why the
stored DiT is larger than a pure-BF16 29.2B checkpoint. Runtime BF16 loading and stored precision
are distinct. These are storage facts, not peak VRAM estimates. [Pinned file census][hf-files],
[DiT index][hf-dit-index], [parameter dtypes][hf-api].

## Three execution paths

| Path | What exists | Integration implications |
| --- | --- | --- |
| Official Diffusers | Released in 0.43.0; `PAN2ModularPipeline` / `ModularPipeline`, with explicit text, VAE encode, denoise and decode blocks. | Use `ModularPipeline.from_pretrained`, `load_components(dtype=torch.bfloat16)` and `output="videos"`; the generic HF `DiffusionPipeline(...).images` snippet is unsuitable. |
| IFM reference code | Training plus inference; HF DiT weights must be converted to its native tensor layout once. Each GPU worker holds a complete model and receives different prompts. | CUDA/FA4/quack assumptions and checkpoint conversion add work. Multi-GPU launch is throughput sharding, not one-video context parallelism. |
| vLLM-Omni | Native PAN2 pipeline; CFG parallelism, Ulysses, tensor parallelism and layerwise CPU offload profiles. | Optional alternative serving runtime. Its memory/latency evidence cannot be transferred directly to a Diffusers/Cozy wrapper. |

Sources: [Diffusers stable pipeline][df-pipeline], [stable blocks][df-blocks], [native inference/conversion][native-inference],
[vLLM recipe][vllm-recipe], [vLLM merge][vllm-pr]. HF ships `modular_model_index.json`, rather than a
conventional `model_index.json`; its exported configs say Diffusers `0.41.0.dev0`, but that metadata
does not establish stable-release availability. Verified usable PAN2 source is in `v0.43.0`.
PAN2 should declare its own resolved dependency environment; current H3 pins are not evidence
that PAN2 runs on the older version. [Modular index][hf-modular], [release][df-release].

Diffusers supports its attention-processor backends and context parallelism; inference uses
`ulysses_anything` / `ring_anything` because text and video token lengths need not divide the device
count. The reference repo's dense inference supports FA4 only; its training context-parallel code
does not change the generator's explicit `cp=1`. The reference environment pins Python 3.14.4,
PyTorch 2.12.1/CUDA13.2, FA4 4.0.0b29, quack0.6.4, Transformers5.13 and Diffusers0.38.
Those pins describe the native path, not requirements proven for the released modular wrapper.
[Diffusers guide][df-guide], [reference compatibility][native-compat], [generator][native-generate].

## Hardware and latency: upstream evidence only

The reference generator offers `none`, `models`, `blocks`, and `auto` offload modes, with block
prefetch; its shell launcher selects `none`. No measured minimum VRAM or absolute latency is
published in the inspected native docs. `auto` checks whether BF16 DiT weights fit, so that test
alone does not establish that activations, VAE and text encoder also fit. [Native memory][native-inference],
[fit predicate and runtime implementation][native-pipeline].

The current vLLM recipe sizes profiles for 481 frames at 704×1248: one ≥141GB GPU peaks at
107.4GiB; CFG2×Ulysses2 uses four such GPUs at 110.7GiB per GPU; one 96GB RTX PRO6000 with
layerwise and guard-model offload peaks at 84.4GiB; two RTX PRO6000 with TP2 peak at 84.5GiB
per GPU. Offload consumes pinned system RAM and parallel CFG branches add copies. These are
vLLM-published profiles, not independent measurements here. [Hardware table][vllm-recipe].

The merged vLLM PR reports B300 SXM6/driver595.58.03, vLLM0.31, Torch2.13/CUDA13.0,
FlashInfer0.7.0.post1, 704×1248×313 frames, 50 steps, CFG3, seed42, one warmup plus three timed
`Omni.generate` calls: 403.4s on one GPU, 126.8s with CFG2×Ulysses2 on four GPUs, and 98.4s
with that four-GPU layout plus Cache-DiT. It reports 2.6s guard overhead and about 11GiB on the
first GPU. The PR's shift7/CFG3 defaults, old guard model names and unfinished provenance fields
differ from the final recipe's shift9/CFG2.5 and updated guards. Preserve this as historical upstream
performance evidence, not a current-default or Cozy latency promise. [Exact benchmark report][vllm-pr].

## License, reproducibility and quality caveats

- PAN2's own code and weights carry Apache2.0. `NOTICE` identifies Wan2.2, Qwen3.5-9B and Diffusers; `THIRD_PARTY_LICENSES` includes their Apache2.0 texts and attributions. Retain applicable license/NOTICE/copyright material and mark modified copied files. This is an inventory of inspected files, not an audit of every transitive dependency or training-data right. [License][license], [notice][notice], [third-party texts][third-party].
- The model card's intended/out-of-scope guidance and recommended separate guardrail package should be recorded independently of the license text. Diffusers and vLLM enable that package's checks by default; the reference generator documents it separately. No new Cozy-wide mandatory policy is proposed here. [Model guidance][model-card], [Diffusers checker source][df-pipeline].
- Upstream states that roughly two-thirds of evaluated multi-step plans miss at least one step, that prompts outside training and case studies needed hand refinement, that text cannot precisely specify dexterous contact, and that intermediate physics can be bypassed. Plans designed to fail have not been quantitatively evaluated. Treat world-simulation language and headline benchmarks as bounded research claims. [Published limitations][model-card].
- Exported HF text config says Transformers5.19; native pins say5.13. Pin and test a coherent modular pipeline/Transformers/model revision cohort. Native checkpoint conversion changes names/layouts and handles text type rows/constants; avoid unnecessary conversion when wrapping Diffusers. [HF text config][hf-text], [native conversion contract][native-inference].
- A full reimplementation would additionally own Qwen3.5 hybrid attention, the four-block caption refiner, dual video/text projections, exact RoPE/normalization/modulation, first-frame channel/mask semantics, FP32 latent updates and WanVAE. Wrapping official components avoids that separate parity burden. [Transformer][df-transformer], [encoders][df-encoders], [denoiser][df-denoise].

## Sources

[hf-tree]: https://huggingface.co/IFM/PAN2/tree/82f54c6cea3e20a9c451b61d7360016e5970503d
[hf-api]: https://huggingface.co/api/models/IFM/PAN2/revision/82f54c6cea3e20a9c451b61d7360016e5970503d
[hf-files]: https://huggingface.co/api/models/IFM/PAN2/tree/82f54c6cea3e20a9c451b61d7360016e5970503d?recursive=true&limit=1000
[hf-transformer]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/transformer/config.json
[hf-dit-index]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/transformer/diffusion_pytorch_model.safetensors.index.json
[hf-text]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/text_encoder/config.json
[hf-text-index]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/text_encoder/model.safetensors.index.json
[hf-vae]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/vae/config.json
[hf-modular]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/modular_model_index.json
[model-card]: https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/MODEL_CARD.md
[native-commit]: https://github.com/ifm-ai/PAN2/commit/0fb907b6e16a82ead6d691d1de2eef6299c71e9a
[native-readme]: https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/README.md
[native-inference]: https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/inference.md
[native-compat]: https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/compatibility.md
[native-generate]: https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/generate.py
[native-pipeline]: https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/pan/pipelines/pan_video_pipeline.py
[rewriter]: https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/inference/pan-prompt-rewriter/SKILL.md
[df-release]: https://github.com/huggingface/diffusers/releases/tag/v0.43.0
[df-tag]: https://api.github.com/repos/huggingface/diffusers/git/tags/dc180cc1a10e8d6d5573e3620ad947df174d2b2e
[df-guide]: https://github.com/huggingface/diffusers/blob/423ab90580b2847998591669b8326f47b6d14d8b/docs/source/en/api/pipelines/pan2.md
[df-pipeline]: https://github.com/huggingface/diffusers/blob/423ab90580b2847998591669b8326f47b6d14d8b/src/diffusers/modular_pipelines/pan2/modular_pipeline.py
[df-blocks]: https://github.com/huggingface/diffusers/blob/423ab90580b2847998591669b8326f47b6d14d8b/src/diffusers/modular_pipelines/pan2/modular_blocks_pan2.py
[df-before]: https://github.com/huggingface/diffusers/blob/423ab90580b2847998591669b8326f47b6d14d8b/src/diffusers/modular_pipelines/pan2/before_denoise.py
[df-encoders]: https://github.com/huggingface/diffusers/blob/423ab90580b2847998591669b8326f47b6d14d8b/src/diffusers/modular_pipelines/pan2/encoders.py
[df-denoise]: https://github.com/huggingface/diffusers/blob/423ab90580b2847998591669b8326f47b6d14d8b/src/diffusers/modular_pipelines/pan2/denoise.py
[df-transformer]: https://github.com/huggingface/diffusers/blob/423ab90580b2847998591669b8326f47b6d14d8b/src/diffusers/models/transformers/transformer_pan2.py
[vllm-pr]: https://github.com/vllm-project/vllm-omni/pull/8602
[vllm-recipe]: https://recipes.vllm.ai/IFM/PAN2
[project]: https://ifm.ai/pan/
[technical-blog]: https://ifm.ai/blog/pan2/
[paper]: https://arxiv.org/html/2507.05169v5
[license]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/LICENSE
[notice]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/NOTICE
[third-party]: https://huggingface.co/IFM/PAN2/blob/82f54c6cea3e20a9c451b61d7360016e5970503d/THIRD_PARTY_LICENSES
