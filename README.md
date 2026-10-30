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
| `minimax-h3/` | MiniMax H3 dual-task package over AdaLN-pruned weights |
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

The H3 package exposes both official actions from one model generation. Its normal
`model: H3Model` slot contains only the two DiTs, text encoder and VAEs.

`long_form` defaults to `mode="turbo"`: each shot uses eight PDD evaluations, the
base release and the separate PDD-8 adapter. Omit `steps` in turbo mode.
For standard sampling, select `mode="standard"`; omitted `steps` then means 30,
with explicit 30/40/50 also supported. In either long-form composer, a segment's
`turbo` defaults to `true`. Set `turbo: false` for a standard 30-step segment within
a Turbo story. Global `mode="standard"` takes precedence and keeps every segment
standard, including segments whose `turbo` is `true`. The CPU parent renders and assembles remotely,
returning only the joined MP4. Intermediate shot clips stay inside the serving calls. A
partial delivery includes its playable completed portion.
Four-GPU placement is chosen when submitting the run, not by the composition function.

The `fl2va_turbo` and `ref2va_turbo` functions take two independent model slots:
`base_model: H3TurboBase` holds an AdaLN-pruned five-root checkpoint, and
`turbo_lora: H3TurboLoRA` holds only `fl2va_turbo` and `ref2va_turbo`. Both
checkpoints must have matching DiT configurations. They default to `minimaxai/minimax-h3@1.0.0-rc.4` and
`alibaba-pai/minimax-h3-turbo-lora@1.0.1`; callers can select other releases that carry timestep plans.
Each sampling scope holds its base DiT and separate overlay together. LoRA hooks
are installed during construction, before Runtime chooses fusion or Ulysses
execution. All three model classes declare degrees 2 and 4.

H3's DiTs declare [4-bit attention](minimax-h3/docs/attention-defaults.md); Runtime picks the
kernel by the GPU's family.
Turbo preserves four dense plus four sparse evaluations; regular 30-step sampling
preserves ten dense plus twenty sparse evaluations when a sparse backend is selected.
Explicit CLI attention pins take precedence.

For source qualification, install the local package from this repository:

```sh
cozy package install ./minimax-h3
```

This needs no package publication. A published release can instead be installed
with `cozy package install minimax-h3`
once available. With both checkpoint references available, a turbo invocation
selects their exact releases independently, and `--lane` the base model's lane:

```sh
cozy run local/minimax-h3/fl2va_turbo \
  'prompt=<your H3 prompt>' duration_s=5 seed=24680 \
  'model.base_model=<org/base-model>@<release>' \
  'model.turbo_lora=<org/turbo-lora>@<release>' \
  --on <existing-machine-name>
```

For the published installation, use `minimax-h3/fl2va_turbo` instead.
Repeat with explicitly selected two-H100 and four-H100 rentals for the GPU proof.
The rental's device group selects the parallel degree; there is no request field
for it. Record the installed package release, both checkpoint digests, worker
Runtime version, actual GPU identities, completed video/audio and measured timings.

The ordinary action syntax is:

- `fl2va` accepts zero, first, last, or first-and-last keyframes.
- `ref2va` accepts an ordered Runtime `Assets` input containing image, video,
  and standalone-audio references. Repeated CLI `--file` arguments preserve this order;
  callers do not construct tagged reference JSON.

For a reference call, the package reads the supplied assets through Runtime's media handles.
File delivery works the same locally and on a rental:

```sh
cozy run minimax-h3/ref2va \
  'prompt=<your H3 prompt>' \
  --file="$HOME/Pictures/character-1.png" \
  --file="$HOME/Pictures/character-2.png" \
  duration_s=14 steps=30 seed=24680 --on any
```

The published function supplies the default model unless an owner Hub override exists.
`--model.model=...` overrides either for one call. FL2VA accepts up to two images through the same Assets input. Images
fill the first and last frame roles in order; labels `first` and `last` select those
roles explicitly. For example, `--file="last=ending.png"` supplies only a last frame.
Other labels retain positional meaning. An empty collection generates without
keyframes. The generated interface declares supported input shapes and bounds;
the endpoint selects roles.

Assets may have caller-supplied labels, such as `--file="alice=~/Pictures/alice.png"`.
Labels are unique within the collection and preserve their exact spelling. Package
code can use `assets["alice"]` or `assets.info("alice")`; unlabelled references remain valid. H3
preserves the supplied order when building its references. A label does not change
the model's positional prompt syntax or rewrite the prompt.

Ref2VA declares a conservative preparation cap of 8192 pixels on the longest edge
and 16,777,216 pixels in total. This preserves room for a high-fidelity 2048px short
edge at the allowed 4:1 aspect ratio. Creator can fit oversized originals before
upload; Runtime applies the same preparation when it receives an original. The
package then performs its normal per-reference sizing below. FL2VA keeps its
canvas-dependent preprocessing.

Reference-image fidelity is an optional per-occurrence hint:

```sh
cozy run minimax-h3/ref2va \
  'prompt=<your H3 prompt>' duration_s=5 \
  --file="woman=~/Pictures/woman.png" --file-fidelity=woman=high \
  --file="background=~/Pictures/background.png" --file-fidelity=background=low \
  --on any
```

For Ref2VA, `low`, `medium`, and `high` select a 256, 1024, or 2048 pixel short edge,
respectively; `auto` starts at `reference_image_short_edge` (default 1024) and steps down
if needed to fit the combined reference budget. These are H3's
package rules. Runtime preserves source resolution and the original file. The H3
adapter supplies each resolved size to the official preprocessing step, preserves aspect
ratio and the upstream grid, and reports actual normalized dimensions through
telemetry. Keyframes in FL2VA still follow the generated canvas. These input-resolution
choices do not change reference conditioning strength.

Both actions use the official Diffusers MiniMax H3 implementation and share one exact BF16
Qwen3-VL conditioner, one video VAE and one audio VAE at their official FP32 destination dtypes,
and one scheduler contract. The conditioner retains the complete vision stack, token embedding,
and language layers 0–49; its parameterless final norm exposes the same pre-norm
`hidden_states[50]` as full Qwen, while layers 50–63 and the language head are absent from both
fill and execution. The same construction carries task-stamped `transformer` and `transformer_ref`
components (Diffusers' names); Runtime's component leases make only the selected task DiT
resident while its action runs. The interim community curve
architecture was deleted instead of retained as a second graph.

Artifact production takes ordering only from Runtime's exact construction contract. The
mechanical projection preserves `destination_sets` order and strips only each component prefix:

```sh
python scripts/h3_order.py model-construction-contract.json > construction-order.json
```

There is no source-header, lexical, or hand-maintained tensor-order fallback.

The two DiT components are distinct weight instances of the same class and exact topology. They
cannot be two names for one module: Runtime fills, fences, stages, and evicts by component name, so
an alias would let the second checkpoint fill overwrite the first and an eviction through either
name invalidate both. Artifact config equality and the constructed destination topology are both
checked before fill; only checkpoint contents and the task-specific upstream interface differ.

Two request knobs set the cost. `steps` is the number of transformer evaluations, from the
committed plans' set (30, 40 or 50; the fastest is the default). Each step count is the official
grid `MiniMaxH3Scheduler.set_timesteps(steps + 1)` per modality (video shift 12, audio shift 3):
the terminal zero is a grid point with no evaluation, so `steps` evaluations run and the progress
callback counts exactly them.

`duration_s` is the clip length in whole seconds, **5 to 15, and required** (15 is the full-length
ceiling): a call without it fails validation. Length is the larger cost lever of the two: the DiT
attends over ONE packed sequence whose rows scale with the frame count, and attention is quadratic
in it. The video VAE encodes `17n + 5` frames per clip, so a request's seconds snap UP to the next
such count (never down). The served set is 5 to 15 whole seconds; 15 snaps to 362 frames
(15.083 s), which the package admits under a scoped frame ceiling:

| `duration_s` | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frames | 124 | 158 | 175 | 192 | 226 | 243 | 277 | 294 | 328 | 345 | 362 |
| delivered s | 5.167 | 6.583 | 7.292 | 8.000 | 9.417 | 10.125 | 11.542 | 12.250 | 13.667 | 14.375 | 15.083 |
| denoise rows at 1344x768 | 37,710 | 47,902 | 53,000 | 58,096 | 68,290 | 73,386 | 83,580 | 88,676 | 98,870 | 103,966 | 109,062 |

4 and 16 s refuse typed at decode. A timestep plan holds one row per (timestep, modality)
and no row depends on the frame count, so every length runs on the same AdaLN-pruned tables and no
lane is re-tabled for one. Every attempt logs its length and resulting packed-sequence rows in the
`h3 output geometry` row. Task-selector, graph-selector, and AdaLN-mode request fields are absent.
The artifact config fixes both DiTs to one modulation structure before construction: reference
artifacts use the official FULL AdaLN path, while production artifacts use exact table rows for the
TimestepPlan each DiT carries as its `timestep-plan.json` asset, and contain no replaced
timestep/AdaLN projection destinations. The plans are the one authority for step counts, sigmas
and scheduler shifts; the package holds no copy. A pruned lane carries the union of every served
schedule's rows, so one checkpoint serves every `steps` value; a plan its rows do not cover
refuses at construction. The
AdaLN-pruned extension inherits Diffusers' forward and changes only those projection modules; it
does not interpolate a curve or carry a second model port. The machine selects the lane before
fetch (Model defaults above); model computation and environment variables do not choose one.

`minimaxai/minimax-h3@1.0.0-rc.4` serves AdaLN-pruned lanes `bf16`, `fp8`, `int8` and `nvfp4`, and the
FULL reference lane `bf16-full`. `adaln-curve` is reserved for approximate community curve
artifacts and is not a lane.

Ref2VA preserves request order and enforces the official product bounds: at most 9 images, 3 videos,
3 standalone audio clips, and 12 entries total. Video and audio clips are at least 2 seconds. A video
longer than 15 seconds is truncated to its first 15 (Runtime records the adjustment and `warnings`
says so); audio over 15 seconds refuses, and standalone audio totals at most 15 seconds. A video's
embedded soundtrack belongs to that video and does not consume the standalone-audio count. Standalone audio cannot be the only reference modality.
The package selects qwen-image-2 sizes automatically. Images with `auto` fidelity
start at a 1024-pixel short edge; `low`, `medium`, and `high` fidelity select 256, 1024,
and 2048 pixels respectively. Diffusers rounds both axes to its 32-pixel grid. Explicit
fidelity hints keep their sizes; automatic images step down together through 1024,
768, 512 and 256 until the combined
32,768-token vision budget fits. A set that still exceeds the budget refuses with the
arithmetic before entering a component scope. Video and audio presentation is unchanged.
The resolved sizes are request-local and appear in telemetry. Smaller references reduce
attention work while the generated video retains its resolution; output quality and speed
still require measurement. Prompts are bounded to 4,096 characters. The token budget is a
conservative capacity limit, not a measured maximum-size GPU fit guarantee.

The task-stamped canonical plans live in the model components as `timestep-plan.json`:
`transformer` and `transformer_ref` in the base model, and `fl2va_turbo` and `ref2va_turbo`
in the turbo adapter. Serving and producer tools read the selected model's assets. To generate
replacement model assets from the official scheduler, use
`python scripts/h3_plans.py --output /path/to/model-assets 30 40 50`, or `--turbo 8`
for the adapter. Each base plan lists one schedule per step count — its exact 31/41/51 video and audio
sigmas, 30/40/50 timesteps, and all five finite modulation classes (including the 0.999 clean-video
and 1.0 clean-audio anchors) — plus one deduplicated union order of 315 block-modulation keys and 207
final-normalization keys. The selected plan must be covered by its component's table layout;
there is no packaged-plan fallback. Each task has one table bank for the whole plan.

Both actions return exactly one muxed MP4 and its warnings — the customer result carries nothing
else. Checkpoint, plan, geometry, and digest facts are attempt observations emitted through
Telemetry, never result fields.

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
