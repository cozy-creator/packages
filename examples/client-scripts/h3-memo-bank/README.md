# Positive H3 table-bank qualification

Owner `/root/astra_memoization_finish`; branch
`qualify/h3-positive-bank-plan-20260925`, base
`56cce9cdaa62465009de51795aa942eaeafb5fc2`.
This is a historical execution plan and dry-run fixture. No GPU was rented or bank computed.
The old model references below predate the required timestep-plan assets: use a
migrated full model and a new adoption receipt before executing the current tools.
Planless projections/banks and their old generating-projection bindings must be
recomputed; adding an asset does not authorize rewriting a binding.
The later root-owned worker must be idle after its film; never fault another request.
All invocations use the global `cozy` CLI and normal `~/.cozy` history/outputs.

## Source and capacity

Use the actual full generator source, not an already-pruned model:

`minimaxai/minimax-h3@sha256:f1e86cb6b4935fbf9e79d90ee87366b6d13e2280b1846b4e9bfa0a00242793eb`

Current local-Hub metadata identifies it as release `1.0.0-rc.2`, lane `bf16-full`, with
190,176,693,539 logical bytes. The film's FP8-pruned source
`minimaxai/minimax-h3@sha256:302eec1e11fd773f6b3cc29a7746f6e5e41ac26f1015ea47225b9392de525433`
lacks the generating weights and cannot replace it. Source projections inherit
native parts without copying payloads, but ordinary model acquisition still binds
the full 190 GB checkpoint. Do not price the download as only 52 GB.

| Per task, FL2VA or Ref2VA | Exact geometry |
|---|---:|
| Generator tensors / payload bytes | 106 / 26,142,079,488 |
| Largest source tensor | 520,224,768 bytes |
| FP32 time-embedding reads per execution | 63,340,032 bytes |
| First block's weight and bias | 520,418,304 bytes |
| Tables / BF16 payload bytes | 51 / 1,020,515,328 |
| Largest checkpoint unit | 20,321,280 bytes |
| Union timesteps / block rows | 207 / 315 |

Both tasks total 52,284,158,976 generator bytes and 2,041,030,656 table bytes.
`geometry.py LOCAL_CHECKPOINT` derives these numbers from the migrated model
plan and current library topology, including dtypes. The table above records the
original approved 30/40/50-plan geometry.

The kernel holds one block projection at a time on one `ctx.device`. It does not
load the full model into VRAM, and extra GPUs do not automatically accelerate the
public helper's sequential banks. One GPU is sufficient architecturally; 8 GiB
VRAM and 16 GiB host RAM are conservative planning allowances, **not measured peak
requirements**. The root's already-owned 2/4-H100 worker has ample margin.

Reserve at least **400 GiB disk** for conservative coexistence of the full 190 GB
source, the roughly 100 GB film source, bank variants, environments and free space.
This does not multiply by GPU count. Actual native deduplication can reduce the
requirement; verify real free space and closure overlap before submission.

There is no measured completion-time estimate yet. The root's acquiring Shalsha
2-H100-NVL worker is quoted at **$6.4634/hour**: another10/30/60 minutes would
cost about $1.08/$3.23/$6.46, excluding any separately billed storage. These are
estimates, **not spending limits, safety cutoffs or stop times**. The user permits
continued paid work; continue while progress is real and stop only for an actual
blocker or explicit instruction. Full-source acquisition may dominate elapsed time
and must be measured. No new rental is requested by this plan.

## Ordinary CLI entry and genuine artifact boundary

First check the metadata-only adoption leaf's request contract with `--describe`, then
run it only on the chosen existing worker when the root authorizes the paid phase:

```sh
cozy run /ABSOLUTE/PATH/adopt_full.py --describe
cozy run /ABSOLUTE/PATH/adopt_full.py \
  'model.source=minimaxai/minimax-h3@sha256:f1e86cb6b4935fbf9e79d90ee87366b6d13e2280b1846b4e9bfa0a00242793eb' \
  --on ROOT_OWNED_WORKER --json
# Retain stdout as adoption.json, then:
python3 prepare_callers.py adoption.json
cozy run ~/.cozy/outputs/h3-bank-qualification/compute.py --describe
```

The adoption script refuses pruned inputs and produces a real native receipt with
zero tensor payload production. A bound Model is an attempt capability; it cannot
be passed directly to a managed child. `prepare_callers.py` therefore requires an
actual completed CLI ModelArtifact result and never invents a producer/receipt.
It emits full compute, single-task FL compute, caller-only edit, schedule40 and
wrong-plan callers under normal
outputs. A private tools checkout may be supplied for same-version code/plan tests;
that generated private path is never published.

Already passed: the actual adoption-script dry-run and the published
`minimax-h3-tools/select-adaln-weights` dry-run against the exact full source.
The complete composition dry-run awaits that genuine adoption result; no receipt
was fabricated merely to make a dry-run appear complete.

## Finite proof sequence

1. **Positive production plus compatible partial recovery.** Start the first
   `compute.py` on the full source. Observe the FL2VA bank's own native operation
   identity, then stop only that accepted operation after its first completed
   block checkpoint. Cancel it, then resubmit with an edited caller.
   An observer may hold the exact native checkpoint boundary, but must be scoped
   to this operation and must not mutate journals. Require 51 FL2VA tables, no
   repeated first-block projection/read, and 51 Ref2VA tables. FP32 time embeddings
   may be reconstructed: after one block checkpoint the expected extra source
   read is 63,340,032 bytes, not another 520,418,304-byte first block. Record
   `h3.adaln.source_bytes`, `h3.adaln.reused_tables`, all native part writes,
   checkpoint identities, attempt causes and complete results.
2. **Completed caller reuse.** Run `caller_edited.py`. The public helper composes
   two select, two compute and one attach calls. Require zero new body attempts
   for all five leaves, exact computation/result/receipt identity, and no new
   generator reads or table writes. Confirm the first compute really ran; existing
   table bytes in the film checkpoint are not proof of a positive bank execution.
3. **Public schedule equivalence and wrong-plan refusal.** Run `schedule40.py`:
   the expected result is reuse, because 30/40/50 all select the same approved
   union bank. Run `wrong_plan.py`: require named `adaln_plan` refusal before
   table/source-payload work. The minimum public setting is 30, but it still
   computes all 207 union timesteps; a one-timestep production bank is unavailable.
4. **Same-version callee invalidation.** Use one owned private copy of the exact
   tools package, unchanged version, and first establish its own control capture.
   Change real implementation spelling without altering math (for example the
   BF16 cast's positional argument to its equivalent dtype keyword). Require a
   new library revision/computation and actual bank production, with equal table
   bytes. Whole-package closure changes may also invalidate cheap projections;
   do not promise finer-grained invalidation than the current implementation.
5. **Coverage-plan refusal control.** `coverage_variant.py LOCAL_CHECKPOINT` reads the source component asset and retains the existing
   30-step schedule and regenerates first-distinct table keys. The private FL
   fixture has60 timesteps and92 block rows. It passes the structural validator
   and its digest differs from the source model's plan. Use this as a negative
   plan control only; never write it into a production model or call it a
   production positive bank. Positive changed-plan production remains dependent
   on an independently reviewed new production plan. `compute_fl.py` obtains the
   exact plan digest from its source selection reply to qualify a single task. Never substitute
   a30→40 wrapper-argument change for plan invalidation.

6. **Numerical and graft custody checks.** Compare all bank header/part hashes and
   coverage maps, including the exact 106 generator-key deletion set. Compare
   representative original-weight/full calculations with table lookups for both
   tasks, first/middle/last timesteps and every modality, separately reporting
   exact equality and numerical errors. Reuse job-010's actual comparator;
   where a trained full-checkpoint comparator is not available, leave that
   numerical gate open instead of inventing an acceptance tolerance. Attach the
   banks to the real BF16 source and, via existing `retable_adaln`, to the retained
   FP8 body; prove table part identity and zero model-sized copy. Run ordinary pruning only after all
   compute/invalidation arms; prove returned grafts still read exact table/body
   bytes after unused producer memo entries are collected. Prune alone is not
   proof that every explicitly retained producer was released.

No image/video inference or partial denoising is memoized here. Bank correctness,
full-vs-table numerical equivalence and whole-model perceptual quality are distinct
gates. Trusted cross-machine memo mappings remain outside this phase.

Validation: both ordinary CLI dry-runs pass, including the adoption rerun on
CLI `0.0.0-f9654a7190aa`. Geometry is source-derived and the coverage fixture passes
the structural parser while failing the production launch pin as expected. The
caller generator was syntax-checked using a genuine historical small-fixture
receipt; those generated test callers were not submitted and are not H3 proof.
