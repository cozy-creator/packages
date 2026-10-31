# Cozy serverless packages (v2)

First-party package sources authored against `cozy_runtime.author`. The tracker is the design
authority; this repository contains package schemas and model-specific computation. Functions declare each model slot's default release (see Model defaults). Private scripts
select their inputs explicitly and run without publishing code to the hub.

Follow [release practices](RELEASING.md) and the
[dependency isolation qualification checklist](DEPENDENCIES.md) when changing
dependencies or qualifying a worker image.

## Layout

| path | responsibility |
|---|---|
| `minimax-h3/` | MiniMax H3 video with sound: `shot` (one continuous take of any length) and `upscale` |
| `sdxl/` | SDXL text-to-image package plus the family's `int8`/`fp8`/`w4a8` recipes (se-008, cr-073, se-023) |
| `anima/` | Anima text-to-image package for local development and benchmarking (se-009; model license is non-commercial) |
| `minimax-h3-tools/` | H3 timestep-table and structural precompute functions (cr-071) |
| `qwen-image-2/` | Qwen-Image-2.1 image generation and reference editing with native Runtime execution (research model licence) |
| `qwen-image/` | Qwen-Image-2512 text-to-image and Qwen-Image-Edit-2511 editing on one shared text encoder and VAE (Apache-2.0) |
| `fibo-2/` | Bria Fibo 2.0 text-to-image and editing, Dev and Edit Turbo (CC BY-NC 4.0 model licence) |
| `ideogram-4/` | Ideogram 4 text-to-image from JSON captions, with layout regions and rendered text (Ideogram Non-Commercial model licence) |
| `moss-image/` | OpenMOSS MOSS-Image text-to-image with layout boxes for objects and text (Apache-2.0) |
| `krea-2/` | Krea 2 text-to-image: Raw (`generate`) and its 8-step Turbo (`generate_turbo`), with caller LoRAs (Krea 2 Community License) |
| `qwen3.6/` | Qwen3.6-35B-A3B and 27B: questions about text, images and videos, on Runtime's shared Qwen-VL code (Apache-2.0) |
| `examples/` | private `main()` scripts for conversions, uploads and diagnostics; run directly with `cozy run` |
| `*/pyproject.toml` | release metadata; its one `cozy.application` entry point names the package's App |
| `Taskfile.yaml` | `task publish -- <package>` (release with its dependents) |

## Boundaries

- Code states capability and tolerances, and can declare a default release beside a callable.
  Runtime validates and publishes that metadata; the machine resolves it before execution.
- Runtime alone hydrates and decodes request assets, places model components, and encodes outputs.
  Package code receives public decoded values and never opens a request path, URL, container, or
  packet. The model package does read its own five immutable tokenizer/processor data files.
- Model code owns model-specific normalization, conditioning, schedules, inference, and output
  semantics and quantization policy. Runtime owns generic quantization machinery, device
  placement, and offload; TensorFS owns artifact storage and transfers.
- Private H3 configs, plans, and proof receipts are identified by their package member and closed
  field shape; they do not add globally versioned schema tags beside the PackageInterface.
- Import libraries at module scope. Metadata inspection reads declarations without executing
  package code. Model construction happens during admitted worker preparation.
- Every wheel exposes exactly one standard `cozy.application` entry point, pyproject's; source and
  installed discovery both read it (a leftover `package.toml` is ignored).
- Community implementations are research evidence only. No community node, graph, prompt parser,
  checkpoint reader, or serving dependency is present here.

## Model defaults

A package publishes model slots (`<callable>.models.<param>`) and, beside the callable, each
slot's default release: `defaults={"model": "stabilityai/sdxl@1.0.0"}`. A default names a release, never a
lane. Upstream models live in their authors' parked orgs (`stabilityai/sdxl`); an org-less one names a model we
make ourselves. To change a default, publish the package
again. A request overrides one slot with `model.<param>=<org/model@release>`. A lane is one
commit in the release; `<org/model>@sha256:<commit>` pins one.

The machine picks the lane: 8 bits on the GPU's most accurate native format (`int8`; `fp8` where
int8 has no tensor cores, as on B300), else whatever lane works. `--lane nvfp4` or `--lane bf16`
asks for only that lane of the function's main model, and a release without it is refused; its
LoRAs, other models and calls into other packages keep the machine's choice. The package declares
only its tolerances, as `Model` class keywords:

- `min_bits=` (16, 8 or 4; default 8): the fewest bits the machine may pick by itself. A slot at
  4 takes `nvfp4` when it decodes; memory never lowers a lane.
- `phase="decode"` for an LM's token loop (qwen3.5's and Klein's upsampler's).
- `attention_bits=` (16, 8 or 4, or a mapping from a component-qualified module prefix to one;
  default 16): the fewest bits attention may run at. H3 declares 4 on its DiTs, Klein 8 on its
  image and joint blocks. `AttentionLayout.exact_paths` holds sites at 16.

Code never reads the GPU's compute capability or name, or a kernel name, to choose.

## SDXL sources

A Civitai SDXL single file uploads servable: `cozy model upload civitai://<version> <org/model> --lane fp16`.
TensorFS's `sdxl.single_file` table writes Diffusers' keys (`text_encoder.text_model.…` included)
in constructor order (f16), with the index, configs and tokenizers of its pinned SDXL base
reference. `cozy model quantize` (below) writes its quantized lanes.

## Quantization

Each family package declares its quantization recipes as data; `cozy_runtime` owns the encodings
(`int8.convrot`, `fp8.rowwise`, `nvfp4`, `w4a8.convrot`), the checkpoint transaction and the
round-trip tripwire. A recipe names the lane it writes, the model it reads and its plan:

```python
INT8 = Recipe(lane="int8", model=SdxlModel, plan=QuantizationPlan(
    encoding="int8.convrot", components=("unet", "text_encoder", "text_encoder_2"),
    include=("unet.*.attn1.to_q.weight", ...), output_precision="preserve"))
```

The static describe records every recipe a package's modules bind (`recipes` in its interface).
`cozy model quantize <model>@<release> --lane int8 --package <package>[@release]` runs the package's
recipe for that lane and model on the machine holding the checkpoint; the package is always
named, since packages that serve the same model can each declare one. The written lane's header
records its provenance: package, release, recipe, fingerprint (the digest of its plan), the
source checkpoint and the plan. A lane that needs code is a `DerivationHook` (H3's,
`h3_tables.job`). SDXL's fp8 encodes the UNet's block-aligned rank-2 weights, Anima's its
transformer's; every other component inherits unchanged.

## MiniMax H3

`minimax-h3` exposes two node functions for Cozy Creator's projects, `shot` and `upscale`
([minimax-h3/README.md](minimax-h3/README.md), [UPSCALE.md](minimax-h3/UPSCALE.md)). A shot renders
in pieces of at most 15 s, each an internal serving child on one of H3's official trunks: FL2VA
(text and keyframes) when it attaches no reference, else Ref2VA, which also carries the previous
piece's latents ([continuation](minimax-h3/docs/continuation.md)).

- **Models.** `base_model: H3TurboBase` is the AdaLN-pruned five-root checkpoint
  (`minimaxai/minimax-h3@1.0.0-rc.7`; pruned lanes bf16, fp8, int8 and nvfp4, and a FULL reference
  lane), and `turbo_lora: H3TurboLoRA` holds `fl2va_turbo` and `ref2va_turbo`
  (`alibaba-pai/minimax-h3-turbo-lora@1.0.1`, PDD-8). Both DiTs share one BF16 Qwen3-VL conditioner
  (layers 0–49, the pre-norm `hidden_states[50]`), one video VAE and one audio VAE; Runtime's
  component leases keep only the running trunk's DiT resident.
- **Steps.** Turbo runs 8 evaluations (4 dense, 4 sparse); the full model 30, 40 or 50 (10 dense),
  from the `timestep-plan.json` each DiT carries (`python scripts/h3_plans.py --output <dir> 30 40 50`,
  `--turbo 8` for the adapter). A pruned lane carries every served schedule's table rows.
- **Length.** A piece is 5 to 15 whole seconds, snapped up to the video VAE's 17n+5 frame grid (15 s
  is 362 frames, 15.083 s). Attention is quadratic in the frames, so length is the largest cost lever.
- **References.** Per piece at most 9 images, 3 videos, 3 standalone audio clips and 12 in all.
  Clips are at least 2 s; a video over 15 s is truncated (with a warning), audio over 15 s refuses.
  Images start at a 1024 px short edge and step down together until the 32,768-token vision budget
  fits; a world state reads at 512 px.
- **Attention.** H3's DiTs declare [4-bit attention](minimax-h3/docs/attention-defaults.md); Runtime
  picks the kernel by the GPU's family.
- **Construction order** comes only from Runtime's construction contract:
  `python scripts/h3_order.py model-construction-contract.json > construction-order.json`.

Results carry only the video, frames and warnings; checkpoint, plan, geometry and digest facts are
Telemetry.

### Release closure

Packages declare Python 3.12 or newer; workers select a compatible interpreter from their
current supported window (3.12, 3.13, and 3.14). A lock records the resolved dependencies,
not a restriction to one Python minor version. Locks pin released Diffusers, Transformers, and the
PyTorch 2.14/CUDA 13.0 family. Cozy Runtime, TensorFS, and Cozy Eval resolve from PyPI like any
dependency; the lock records the exact resolution. `cozy-runtime` and `tensorfs` take a floor
only, never an upper bound (`cozy-runtime>=0.31`, `tensorfs>=` what the package needs): a machine
runs every package on its own Runtime/TensorFS pair, so an upper bound can only refuse a run, and a
real break shows as a named import error instead. The five minimal tokenizer/processor data files are bundled from the pinned official
MiniMax snapshot and are the sole unweighted authority; the bound artifact owns only weighted
component configs.

The lock is necessary but not sufficient for a production claim. Promotion also requires a
platform-specific verified wheel set, an offline empty-environment install receipt, codec census,
the uniform dual FULL model snapshot, and output proof on the admitted provider GPU.

## Anima

### The default negative prompt is the model's quality knob

`generate.negative_prompt` defaults to the official Anima card negative, verbatim:

```
worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration
```

This is not stylistic taste. Anima's score-bucket conditioning acts through the NEGATIVE, so
what the negative pushes away decides the aesthetic. The owner-labelled bank of 2026-09-02
(se-026, `~/.cozy/outputs/paul-anima/research-20260902/`) holds same-seed pairs with seed,
steps, guidance and geometry fixed: both labelled winners carry this string and the labelled
loser carries the previous default `low quality, worst quality, blurry`. The loser also
carried `score_7, safe` in its POSITIVE prompt and still lost, and one winner carried no
positive quality prefix at all — so the negative is the isolated cause and the positive
prefix is not.

`artist name` is a literal Danbooru tag; it suppresses rendered signature text rather than
naming an artist.

Passing `negative_prompt` replaces this default whole. `negative_prompt=""` is a legal way to
run with no negative at all.

### The package injects nothing into the prompt

`prompt` reaches the sampler exactly as the caller wrote it. There is no server-side quality
preamble and no content-rating tag.

**Recorded decision (Paul, 2026-09-03).** se-026 proposed a typed `rating` enum
(`safe`/`sensitive`/`nsfw`/`explicit`, default `safe`) prepended server-side, because the
research observed two neutral prompts producing NSFW output. The owner declined it: *"don't
include `safe` in the prompt."* The consequence is recorded rather than mitigated — **a
neutral prompt can return NSFW output, by design, and callers who want a rating apply the tag
themselves.** The owner labelled one such output a winner (`117dafeb…`, "NSFW-leaning, no
`safe`"), so this is an informed choice about what the model is for, not an oversight.

The same reasoning covers a quality preamble: prepending `masterpiece, best quality` to every
request would be invisible and would fight a caller who asked for `crayon drawing, childlike,
naive`. Conventions that help belong in the prompt the caller writes, which is what the rest
of this section is for.

### Sampling shortcuts: one on, one off, both exposed

Both were measured on an sm89 RTX 4070 Laptop (8 GiB) at the package's own default 1536
class, bf16 verbatim lane, seed 1005, 30 steps, guidance 4.5 (se-026, 2026-09-03).

| configuration | denoise loop | verdict |
|---|---:|---|
| full CFG, no cache (previous default) | 205 s | the owner-approved reference pixels |
| `cfg_interval` 0.15–0.7 | 156 s (-24%) | **ON by default** |
| `first_block_cache` 0.075 alone | — | **refused**: `device_shortfall` on 8 GiB, 3/3 runs |
| both | 86 s (-58%) | survives, but only because CFG interval masks the cache's cost |

**`cfg_interval_start` / `cfg_interval_stop` default to 0.15 / 0.7.** Guidance contributes
almost nothing outside a middle band of the noise schedule, so the unconditional forward is
skipped there. Every block still runs on every step: the mechanism is subtractive and
bounded, and its failure mode is uniform rather than structural. Pass `0.0` / `1.0` for
full-range CFG.

**`first_block_cache` defaults to 0.0 (off).** It skips 27 of 28 blocks whenever the first
block's residual moves less than the threshold, on the assumption the whole network moves
like its first block. Measured at 0.075 rather than assumed:

- it extrapolates **44% of forwards** (alternating skip/compute; the final steps always compute);
- against the same seed it consistently **smooths faces and coarsens fine texture** — the two
  places where the first block is a poor proxy for the other 27;
- the cached residuals stay live, so it **costs VRAM**: alone at the 1536 class it refuses
  `device_shortfall` on an 8 GiB card, reproducibly. It survives only when `cfg_interval` is
  also narrowing the live cache states.

Turn it on per request when throughput matters more than faces.

### Card conventions, for whoever is writing the prompt

`circlestone-labs/anima@1.0.0` is CircleStone Labs **Anima Base v1.0** (`circlestone-labs/Anima-Base-v1.0-Diffusers`).

- **Two independent quality-tag systems**, and any combination of them works: the
  aesthetic-classifier buckets `score_1`…`score_9` (`score_9` is the maximum; the card
  suggests `score_7` to stay out of slop territory) and the human-scored ladder
  `masterpiece / best quality / good quality / normal quality / low quality / worst quality`.
  **Aesthetic-variant checkpoints omit `score_*` entirely** — those tags belong to Base.
- **Rating tags**: `safe`, `sensitive`, `nsfw`, `explicit`. Nothing supplies one by default.
- **Tags** are lowercase Danbooru, spaces not underscores (Gelbooru spelling on conflicts).
- **Artists** are `@name`.
- **Tag order**: quality / meta / year / rating → count → character → series → artist → general.
- **Year** tags are `year 2025`.
- **Weighting** is `(tag:2)`-scale — Anima needs distinctly higher factors than SDXL does.
- **CFG 4–5 over 30–50 steps** for this checkpoint; the package defaults (guidance 4.5,
  steps 30) sit inside that and are confirmed by the same bank. Below ~20 steps the same seed
  goes murky and then degraded; that is a scheduler and checkpoint limit, not a default to
  tune.


## Merging

PRs into `master` merge through the merge queue: `gh pr merge <n>` (the queue squashes; never `--admin`). CI runs there, on the final PR + branch state; nothing runs on a push.
