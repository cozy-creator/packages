# minimax-h3-tools

The H3 producer package. Its conversions are functions that take one primary model input and a
destination repository, run on the rental that holds the source:

```sh
cozy run minimax-h3-tools/<function> <input model ref> <org/model> --on NAME
```

Each declared output becomes a commit in the lane named by its slot; publish lanes with
`cozy model publish <org/model> <version> <commit>…`. On the rental that
ingested or already fetched the input, its objects are in the pod Store and nothing moves.

| Function | Input (first positional) | Output | Notes |
|---|---|---|---|
| `bf16-full` | a full-precision H3 checkpoint: the converted `MiniMaxAI/MiniMax-H3` release or a `bf16-full` lane | `bf16-full` | source-only rows dropped, video VAE encoder and decode operands f16 |
| `bf16-pruned` | same | `bf16` | AdaLN tables replace the 106 modulation rows per DiT; CUDA |
| `fp8-pruned` | same | `fp8` | tables + both DiTs and the conditioner's 350 decoder linears `fp8.rowwise`; CUDA; every serving rung |
| `turbo-lora` | the converted `alibaba-pai/MiniMax-H3-Acc-LoRAs` release (one fl2va and one ref2va component) | `pdd8` | `model.base=<full-precision H3>`; reads its modulation weights and heads only |
| `regenerate-lora` | the release's `lora/` folder of `Efficient-Large-Model/Open-H3-Regenerate-2K`, converted with `--source-format h3.lora` (one component) | `bf16` | `model.base=<full-precision H3>`; the merged rank-576 factors in bf16, and refine-plan tables with LMS Sharp's AdaLN curve added |
| `retable` | an AdaLN-pruned lane | `adaln-pruned`, `tables` | `model.full=<full-precision H3>`; recomputes tables only |
| `vae` | any H3 lane | `vae` | the video VAE alone, normalised; no base, so it fetches only that component. `cozy model upload <it> <org/model> --lane L --base <lane L commit>:vae` composes it into a lane by header |

Its quantized lanes are recipes (`FP8`, `INT8`, `NVFP4`, `W4A8` in `h3_tables.job`): code, as a
`DerivationHook`, since what a lane encodes depends on what the source still holds plain. Both
DiTs and the text conditioner's 350 decoder linears, and in int8, nvfp4 and w4a8 the video VAE
decoder's 216 block linears at int8.convrot, encoding what is still plain. Every recipe also
applies the video VAE's f16 normalisation, so a lane written before it re-casts only those rows, and
`--lane bf16` / `--lane bf16-full` on a plain pruned / full lane apply that normalisation alone; CPU:

```sh
cozy model quantize minimaxai/minimax-h3@sha256:<pruned commit> --lane fp8 --package minimax-h3-tools --on NAME
```

The conditioner keeps `embed_tokens` and its vision tower at bf16 (`lanes.TEXT_ENCODER_KEEP`); a
component already carrying the requested encoding is inherited and one carrying the other
encoding refuses, so `--lane fp8` on an `fp8` commit encodes only the conditioner. The lane functions run on the
ingest rental:

```sh
cozy run minimax-h3-tools/fp8-pruned minimaxai/minimax-h3@1.0.0 minimaxai/minimax-h3 --lane bf16-full --on NAME
cozy run minimax-h3-tools/turbo-lora minimax-h3-acc-loras@1.0.0 alibaba-pai/minimax-h3-turbo-lora --lane bf16 \
  model.base=minimaxai/minimax-h3@sha256:<bf16-full commit> --on NAME
```

The lane functions, `turbo-lora` and `retable` write weights, so they are never memoized; a
committed derivation replays for the same code and inputs. Tables, heads and quantized
tensors checkpoint as they complete; a request re-issued with the same inputs on the same
machine adopts a canceled or failed run's completed work and computes only the remainder. Quantized tensors encode on every available CPU. The metrics
`h3.reused_tensors`/`h3.computed_tensors` (`h3.turbo.*` for the LoRA) report the split.

The lane recipes read the granted checkpoint's structure once and drop only source-only
rows still present (the native `rope.inv_freq` buffer and the text model's layers 50–63,
final norm and head), so the converted upstream release and `bf16-full` produce the same
lanes. It does not accept a pruned or quantized checkpoint as a substitute for the full source.

Every source DiT carries its schedule as `timestep-plan.json` (`transformer` for
FL2VA, `transformer_ref` for Ref2VA). Turbo production reads each task's plan
from its adapter component, whose one schedule has an evaluation per distilled PDD
block (32 / 4 = 8); its shifts set the collapsed heads. Missing, wrong-task or
unsupported plans refuse; there is no packaged fallback or plan digest pin. Outputs,
projections and table banks retain the plan asset, and full-model derivations
preserve other component assets. The source checkpoint identity binds plan bytes
to resumable computation. See [client preparation](CLIENT-SCRIPTS.md) for the
migration boundary of older projection/bank bindings.

Pruned checkpoint configs describe the ordered AdaLN rows as `cozy_h3.table_keys`:
each final-normalization row names an exact float32 timestep, and each block-modulation
row names a timestep and modality. Sampling-plan hashes remain generation provenance;
they do not decide serving compatibility. Producer and inference use byte-identical
copies of the same small row-label parser, checked by CI.

## Lanes

A lane is a NAME plus, per component, what this producer does to it. The whole catalogue
lives in `h3_tables/lanes.py`; the request names which of its rows this attempt produces
and defaults to all of them. The recipes are code, never a request field: output slot names
are decorator-time facts, and every lane mints a MiniMax H3 Model Derivative under §I.11(i)
of the community licence, which is a reviewed act rather than a caller choice.

| lane | modulation | per-component treatment |
|---|---|---|
| `bf16-full` | full | none — every component inherited |
| `bf16` | AdaLN-pruned | none |
| `fp8` | AdaLN-pruned | both DiTs and the conditioner's decoder linears `fp8.rowwise` |

A component a lane does not name is **copied unchanged**: the output's copy keeps the
source's stored objects, so the 58.2 GiB conditioner/VAE trio is the same objects in every
lane above. Naming a component costs its bytes once per lane that names it, which is why the
treatment map is sparse and must stay sparse.

A treatment is a dtype `cast` over the component's float32 rows, an `encode` over its
block-aligned rank-2 float weights, or both; `keep` names exact keys the encoding must not
select. Adding a lane is one catalogue row, one `WeightsOutput` line and one `LaneName`
member — an import-time check proves the three agree, and refuses a lane that treats the
audio VAE, that uses an encoding no lane of the vocabulary stores, or whose declared byte
ceiling disagrees with its treatments.

A lane encodes only as the vocabulary's encoded lanes do (`LANE_ENCODINGS`): int8
`int8.convrot`, fp8 `fp8.rowwise`, nvfp4 `nvfp4`. One component
is refused outright. **The `audio_vae`, by name**: 637 of its 1,087 rows are rank-3 — including the BigVGAN decoder's
344 `weight_norm` `weight_g`/`weight_v` parameters — and the rank-2 encoding cannot
represent any of them; the only six rank-2 float weights the component carries are the
`pre_block` ENCODER attention/MLP linears — so a shape rule does not refuse
the component, it silently quantizes the audio conditioning path and leaves the BigVGAN
decoder untouched.

Each lane holds the components `transformer` (FL2VA), `transformer_ref` (Ref2VA),
`text_encoder`, `vae` and `audio_vae` (Diffusers' names), each with its own config;
the DiTs are the exact 638-row Diffusers DiTs (import drops the
native-only `rope.inv_freq`). Full assembly drops the official text model's layers 50–63,
final norm and language-model head (156 rows) to produce the reviewed 902-row, 50-layer
pre-norm conditioner without rewriting any retained payload. Each pruned task then
replaces 106 dynamic AdaLN rows with 51 BF16 timestep-table rows carrying the union of
every schedule in the task plan (30, 40 and 50 evaluations: 315 block rows and 207
final-normalization rows, 1,020,515,328 table bytes), so one pruned checkpoint serves every
served step count. FP8 is a child of those pruned BF16 task components.

The ordinary `retable` function recomputes only those tables for an existing AdaLN-pruned
checkpoint of any encoding: `pruned` (the checkpoint to inherit, BF16, FP8 or int8) and
`full` (the complete BF16 checkpoint whose modulation weights the rows are computed from).
Every non-table tensor carries over and nothing is requantized, so widening the plan's
schedule set costs table bytes only. It emits two outputs from one table pass: `adaln-pruned`
(the retabled checkpoint, `pruned` with its tables written again) and `tables` (a two-DiT
table bank). The retabled checkpoint commits first. It refuses before any read unless `pruned` carries table rows and no dynamic
modulation weights for both DiTs and `full` carries the exact modulation weights.

The package declares no GPU, SM, VRAM, or host-RAM guess. Creator derives accelerator-class work
from the typed model inputs; exact artifact residency and measured request/scratch envelopes drive
fit. A lane function opens its one output, copies what it keeps, computes each task's
timestep tables once and encodes what its treatments name. There is no workflow graph and
no nested call. Tensorhub owns final retention or publication;
the function receives no publisher credential or store path.

A retry computes only what TensorD has not journaled (`has`); a committed output replays.
A local commit does not imply remote custody; Creator still verifies uploads before
publishing a release.

The tools wheel carries the exact model config, full DiT shape contract and construction
order as immutable importlib resources. Task plans come from source model component assets;
the Runtime wheel carries the reviewed quantization tensor specifications. The ordinary
`h3_tables.operations.quantization_plan(encoding)` helper binds their exact two-DiT geometry and
this package's full/pruned construction-order digests for the single Runtime-owned
`cozy_runtime.derive.operations.quantize` operation. Its compact plan contains no
model bytes or executable policy. Noncanonical source order refuses; the lane
functions and AdaLN producers already emit the accepted orders. These are not
Creator-supplied assets.

The result preserves the Runtime quantizer's existing weight measurements in
`weight_fidelity_this_run`, a list of rows naming output slot, component, the treatment that
produced them and stats: worst per-tensor relative Frobenius error for the encoding and,
separately, for the cast — they measure different carriers against different bounds and a
maximum over both would hide which one moved — plus saturated element count, measured tensor
count and byte counters. Each
completed component also logs those same values before later stages run. Replayed outputs
are absent from this run's measurements; absence never means zero error. BF16 inheritance
and AdaLN table production do not claim quantizer measurements. Activation fidelity,
matched output distance and output quality require separate inference/evaluation evidence.


`turbo-lora` builds the PDD-8 overlays from the released
`MiniMax-H3-FL2VA-Acc-8Step.safetensors` and `MiniMax-H3-Ref2VA-Acc-8Step.safetensors`
(`alibaba-pai/MiniMax-H3-Acc-LoRAs`, revision `335001fb9e5455d68a0caa18ec2e319072150328`)
and the base model's original modulation weights. The adapter checkpoint must hold exactly two
components whose names contain `fl2va` and `ref2va`.

The output holds only the `fl2va_turbo` and `ref2va_turbo` components. Their six inference LoRA families are copied
with the adapters' objects. Only adapted AdaLN tables and eight collapsed output heads are
written. The consumed AdaLN factors are absent from the output. The output journals each
completed tensor, so a resubmitted run continues unfinished work and a
completed replay returns the same artifact. Ordinary `retable` keeps its two inputs
and two outputs; turbo preparation does not change the base schedule.

The output binds to `minimax-h3`'s single `H3TurboModel` slot. Ordinary `H3Model`
continues to consume a checkpoint with only the original five components.

The row-label parser is the producer's `h3_tables.table_layout`; `minimax-h3`
depends on this package and imports it rather than keeping a copy.

