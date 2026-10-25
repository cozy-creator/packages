# Qwen-Image-2.1, 30 steps on RTX 3090

Runs **17335–17338 passed** on Cheval (one RTX 3090, 24 GiB, SM86, driver 580.95.05),
using the ordinary default Cozy CLI. Every call completed 30 observed steps at 1024×1024,
returned a downloaded PNG whose SHA matched its receipt, and reused executor PID **488**.
The four seeds and four pixel hashes differ. All four images were visually inspected: the
bookstore, legible “PAPER & LIGHT” sign, mustard coat, blue umbrella and wet reflections are
present. This is a qualitative output smoke check, not an upstream numerical-fidelity comparison.

The exact tested package source is `8d8d7243142c382188ee9c8fef5b1b7ffa623f6d`.
The installed cohort was Creator/TensorD `76656368e`, TensorFS `0.15.2+dev.g76656368`,
and Runtime 0.32.1 from candidate `75bce8937` (parent's install run 17333). No package or SDK
change occurred during these four calls. The GPU lane was returned after the controller's normal
CLI snapshot showed ready, zero running/queued calls and no live machine runs. No rental lifecycle
operation was performed by this lane.

## Measured timing and memory

| Run | Call | CLI wall s | Reported execution s | First CUDA step ms | Mean CUDA steps 2–30 ms | Mean host steps 2–30 ms | Peak allocated / reserved GiB | Store movement GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17335 | first/cold | 187.000 | 56.868 | 5980.472 | 563.994 | 553.187 | 16.712 / 20.453 | 17.134 |
| 17336 | warm 1 | 113.682 | 49.025 | 1526.102 | 1588.817 | 1588.853 | 15.975 / 18.092 | 211.176 |
| 17337 | warm 2 | 117.753 | 48.473 | 1546.574 | 1569.884 | 1570.013 | 13.614 / 18.092 | 211.176 |
| 17338 | warm 3 | 114.289 | 47.583 | 1471.741 | 1541.874 | 1541.976 | 13.614 / 18.092 | 211.176 |

CLI wall uses one controller monotonic clock and includes editable-source preparation and output
collection. Reported execution is the machine's run interval. CUDA-event and host step intervals
are not additive CPU/GPU totals. Cross-clock `accepted_to_started_ms` was negative on warm receipts,
so it is not used as queue latency. Initialization/construction facts repeat historically in warm
receipts; they were not treated as repeated warm work. There were no controller metric gaps.

The first call reported 13.210 s of executor initialization and 6.537 s of construction preparation;
these nested facts are retained as reported, not summed into a new total. Prompt encoding was
11.245 s cold versus 0.0004/0.020/0.020 s warm. VAE decode was 2.061 s cold and about 1.2 s warm.
Tracked peaks are process/native measurements, not a hard driver-wide VRAM cap. Source/disk
coldness was not forced by deleting data, and store movement is not network download traffic.

## Actual model and routes

Every receipt selected `_/qwen-image-2.1`, release `0.1.0`, lane/precision `int8`, slot `model`.
The reported model manifest was
`sha256:78705de45f7421fde51a0196913b11f1f7e3c95d468881d762d18af50da6fec3`.
The published Hub commit identity remains separately recorded in `model-snapshot.json`.

- Native `cozy-int8`, leaf `cozy.int8-convrot256.native-leaf/1`: 252 text-encoder sites and
  224 transformer sites.
- Attention selector: `sdpa`, 32 transformer sites. This does not identify an underlying
  FlashAttention kernel.
- CUDA graph replays: **zero**. The transformer reports `with opaque arguments`; no model graph
  acceleration or GPU-driven outer denoising-loop claim is made.

## Performance finding

The warm calls are functionally valid but denoising is slower. Cold loading read 8.526 GB of text
encoder data, 7.257 GB of transformer data and 1.351 GB of VAE data in 71 fills. Each warm call
read **209.825 GB of transformer data in 961 fills**, plus the VAE once: 962 fills and 211.176 GB.
The 32 transformer blocks were repeatedly loaded across all 30 steps. Native lookahead was active
(483 speculative transformer fills per warm call), but did not eliminate the repeated reads.

Preferred backing fell from 7.231 GB after the first call to 4.696 GB after warm 1. This is consistent
with the preferred-prefix/phase-policy issue under investigation; a policy A/B is still needed to
attribute the whole regression. No model or memory-policy change was slipped into this baseline.
The prompt cache and executor were reused, while the denoising working set was not retained well.

The 24 GiB int8 case is now an actual fit/output result for this payload. It does not qualify 8 GiB,
bf16, Turbo, reference editing, or temporary-Arena teardown under unrelated graph activity.

## Evidence

Compact committed evidence: `cheval-766-75.json` (raw receipt hashes, exact image paths/hashes,
model/route facts, step summaries and per-component fill counts).

Original report and full receipts/images:
`/home/fidika/cozy_v2/outputs/varena-remote-20261024/cohort-4-qwen24/qwen-30/report.json`.
Each `cold` / `warm-1` / `warm-2` / `warm-3` directory keeps its input, raw CLI events, full run
snapshot, stderr and image. Images are preserved there, not recompressed into this repository.
