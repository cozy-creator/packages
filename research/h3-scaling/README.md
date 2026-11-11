# H3 one-, four- and eight-GPU benchmark captures

`prepare.py` makes local H3 packages from this checkout. The original functions,
weights, geometry and sampling schedules are unchanged. Each capture's base and
Turbo LoRA ladders have only one width (1, 4 or 8), for H100 and RTX 5090. This matters
on an eight-card machine: the shipping ladder currently stops at four cards.

The captures pin Runtime 0.22.0. They do not publish a package or change the user's
Hub. Obtain the configured account index from `cozy package lock`, then:

```sh
python research/h3-scaling/prepare.py /absolute/output/native \
  --account-index=http://127.0.0.1:8819/v1/index/paul/simple/
cd /absolute/output/native/package-degree-4
cozy package lock --upgrade-package=cozy-runtime
cozy run ./fl2va_turbo --describe --json
```

Repeat the lock and describe for `package-degree-8`. The explicit index in the
capture is required by `uv export --locked`; it must match the configured Hub's
account index. The script preserves the original model references:

- `minimax-h3@1.0.0-rc.3/fp8-pruned`
- `minimax-h3-turbo-lora@1.0.0-audit.1/pdd8`

Use `--degrees 1` to add only the same-host one-GPU control without touching
existing four/eight-GPU captures. Its base and LoRA rungs both request one GPU;
Runtime selects the single-GPU producer default automatically. Lock and describe
both `fl2va_turbo` and `fl2va` before assigning the existing rental's GPU 0.

The current rc.3 base differs from the prior rc.2 campaign: 350 text-encoder
decoder weights now use rowwise FP8. Header comparison proves both DiTs and both
VAEs retain exactly the same tensor descriptors and content hashes; only those
350 text-encoder rows change. Comfy's prepared default uses its BF16 text encoder.
Report this precision difference and retain fresh quality videos; the older
attention quality approval does not establish approval of the new text encoder.
The submitted prompts also pass through each application's own normalization:
native H3 appends `\n\nnon_diegetic_music: N/A` when that field is absent, while
the pinned Comfy node tokenizes the submitted text directly. Disclose this
conditioning difference; matching submitted text is not identical encoder text.

Invoke the local directory through ordinary `cozy run`, with an existing rental
and a unique idempotency key. The input is the full 15-second request. Turbo uses
8 steps (4 initial dense, 4 sparse); regular uses 30 (10 dense, 20 sparse).

An accepted request is not a degree proof. Before accepting a timing, retain
`cozy run watch <id> --once --json` and verify the reported degree, each participating
GPU UUID and rank, attention recipe and Sol counts, all denoising steps, and the
362-frame 1344×768 24-fps video with audio. Record compile/download warmup separately
from repeated steady runs. Do not overlap Cozy and Comfy inference.

On compatible 5090 systems, degree 4/8 uses the head-local Sage3 NVFP4 dense plus
Kitchen INT8 Sol recipe. The shared-QKV producer is single-GPU only. H100 defaults
remain BF16 Sol; its dense reference is selected from the ready Sage/FA3/SDPA
preferences and must be recorded from the actual run, not assumed.

The DiT is sequence parallel. The video VAE also distributes independent temporal
clips over the group's ranks through `author.spread`; the leader restores temporal
order and performs the cross-fades and output handling. Spatial tile batches stay
fixed at 28. Sampling and audio decode remain on the leader. Older H3Model and
context-parallel module prose saying both VAEs stay on the leader is stale.

H3 has 56 attention heads: degree 4 assigns 14 heads/rank and degree 8 assigns 7.
The current campaign's run 5125 produced a full 362-frame video on eight actual
RTX 5090 ranks, with the expected hybrid recipe and 4+4 counts on every rank.
This is video and execution evidence; quality still needs the user's review.

## Review gallery

`python research/h3-scaling/build_review.py /absolute/campaign` writes
`review/index.html`, a relative-path artifact index and its browser script. It
indexes validated timed native RTX 5090/H100 captures and qualified timed entries
from `comfy/results-ledger.json`, verifies their video hashes, and never copies
videos. Small posters are extracted from the originals without altering them.
Regenerate it after the Comfy ledger changes. Failed, partial and warmup captures
stay out of the timed comparison.

The gallery groups timing rows by hardware, GPU count, sampling mode and engine.
Its A/B input selector is for visual review only. One/four/eight-GPU playback can be
aligned, slowed and viewed beside Comfy. Comfy's baseline remains the initial
selection; qualified variants remain separately selectable. Format validation is
explicitly separate from the user's visual and audio quality decision.

The Miranjo four-H100 transport study is a separate gallery selection from the
Sanger baseline. Its peer and host-memory videos can be compared within either
engine or across engines on Miranjo. Timing means always stay within one host,
sampling configuration, engine and transport mode.

### Attention kernel comparison

The earlier single-5090 quality matrix is a separate study: four attention
recipes × two schedules (4 dense + 4 Sol or all 8 dense) × two matched inputs.
It used a different host, 575 W power limit, Runtime, and experimental linear
implementation from the scaling campaign. Its timings must stay within that
study; it does not provide an H100 kernel comparison.

Build a separate `review/kernels.html` from the archived benchmark table and the
original kernel review index:

```sh
python research/h3-scaling/build_kernel_review.py \
  --archive /absolute/tensorhub/docs/benchmarks/data/minimax-h3-2026-10-09.json \
  --source-index /absolute/quality-matrix-20261009/review/index.json \
  --output /absolute/campaign/review
```

The builder checks all 16 matrix identities, original video/input hashes,
recorded timings, complete pairs, and video/audio metadata with `ffprobe`.
Earlier full-decode validation records remain linked. It embeds the data for
direct `file://` use, links the original videos without copying or modifying
them, and adds navigation to an existing scaling gallery without rebuilding it.
Later scaling-gallery rebuilds retain that link when `kernels.html` exists.
Neither generator performs inference or assumes perceptual quality equivalence.

## Transport comparison

The [transport recipe](transport/README.md) captures otherwise unchanged
four-H100 Turbo packages with direct-peer or host-shared-memory NCCL transport.
It retains actual-route validation, ordinary CLI submission, and passive NVLink
counter analysis with explicit sampling bounds. The recipe never rents or resets
a worker; the operator controls those lifecycle boundaries.
