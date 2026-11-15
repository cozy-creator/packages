# H3 producer/consumer boundary experiment

Owner: Codex `/root/h3_boundary_fusion`. Purpose: remove repeated activation quantization at
H3's Q/K/V fan-out through existing Runtime contracts. Branch:
`codex/h3-qkv-boundary-sharing-20261115`. Packages base:
`e4b179bbd099f681658dc7ef5a059f484969cf94`. Worktree:
`/home/fidika/cozy/.worktrees/packages/h3-qkv-boundary-sharing-20261115`.

Runtime source inspected: `af2a4753cb70598b75405942901e4f8f3b24eda6`. No native interface,
attention-kernel internals, graph ownership, model precision, or automatic kernel selection changes.

## Remaining boundaries, ranked

1. **Same input quantized separately for Q, K and V during capture or inference mode.**
   `glue._Heads.__call__` projects one input through three `Rows` calls. Runtime's
   `shared_operand` does not share inference tensors implicitly (no mutation version counter),
   or a leaf recording a graph. The existing `quantize_once` scope provides explicit sharing.
   Close it immediately after projections, before RoPE, key-mean seeding and attention. Scope
   only these two cases; ordinary no-grad inputs already share across windows and keep doing so.
   This is the selected narrow experiment.
2. **Projection → norm/RoPE → attention quantization/packing.** Q/K normalization already
   overwrites the projection output in `rmsnorm_rope`; no duplicate BF16 Q/K allocation remains
   there. Sparse `producer.pack` already writes GEMMs directly into packed Q|K|V ranges and
   scopes shared activation quantization. Combining norm/RoPE with attention packing needs
   the existing whole-token key mean, per-tile scale, rotation, protected-row and sparse-layout
   rules; it is a larger change than removing an avoidable repeated quantizer.
3. **Up projection → activation → down projection.** INT8 already folds SwiGLU into its
   activation quantizer. FP8's SwiGLU intentionally recomputes a wide row twice to preserve
   whole-row amax without register spilling. The available profile does not establish this
   variant as a significant cost; a speculative schedule rewrite is not selected.

`FusedMiniMaxH3TransformerBlock` already windows its tail and reuses the attention output.
The added explicit scope is per existing head window, with no new cache across windows or attention
calls. It retains one operand through Q/K/V and releases it before attention; separate quantizers
already need an operand while the final V projection coexists with Q/K. Existing implicit reuse
for versioned tensors is unchanged.

## Evidence and qualification

The existing SM89 15-second Turbo diagnostic (cr-297 session `43442514074`, cell `m-h1`) reports
426.68 ms of one INT8 quantizer family and 321.31 ms of another in a 49,445.33 ms dense sampled
call. These families cover more than Q/K/V. Event sums overlap and are not a critical-path bound;
they do not imply the candidate's full-model saving. The investigation starts from a concrete
repeated producer computation, not a claimed end-to-end bottleneck.

Sources: [H3 glue](../minimax_h3/glue.py), [sparse producer](../minimax_h3/producer.py),
[Runtime sharing contract](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/src/cozy_runtime/accel/kernels.py#L212),
[Runtime row projections](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/src/cozy_runtime/accel/rows.py#L114).

## Native projection screen

The [reproducible probe](../../research/h3-boundaries/qkv_probe.py) executes H3's actual `_Heads`
method with real Runtime native INT8 leaves and `Rows`, stopping before norm/RoPE/attention.
Its baseline is the original method at the packages base above. Synthetic weights/inputs use seed
20261115, BF16 activations/output, INT8 ConvRot256 payloads and whole-row FP32 scales. It adds no
test suite. The [final receipt](../../research/h3-boundaries/sm89-4096.json) records raw samples,
native kernel names and hashes.

Hardware: RTX 4070 Laptop, SM89, 8 GB; Torch 2.15.0+cu132; published Runtime 0.46.0 native kernels.
That wheel's `accel/kernels.py` and `accel/rows.py` byte-match the inspected Runtime revision above.
Native startup verification passed. Package `scripts/typecheck.sh` passes with zero known errors;
Ruff and `git diff --check` pass.

With 4096 tokens, hidden width 5376, 16 synthetic heads of dimension 128, and two 8-head windows:

| Case | Native quantizer calls, before → after | Q/K/V output |
|---|---:|---|
| Ordinary no-grad tensor | 1 → 1 | Bit-identical |
| Inference tensor | 6 → 2 | Bit-identical |
| Inference tensor with armed LoRA hooks | 6 → 2 | Bit-identical |
| Prequantized input | 0 → 0 | Bit-identical |
| CUDA capture, versioned no-grad input | 6 → 2 | Replay bit-identical to eager |
| CUDA capture, inference input | 6 → 2 | Replay bit-identical to eager |

An unknown pre-hook still declines row splitting. Weak references prove the explicitly shared
operand is gone before the RoPE boundary. The CUDA profiler independently records six versus
two native quantizer kernel events for the inference case. Payload/scale bytes materialized
across both windows fall from 132,218,880 to 44,072,960, with 176,160,768 logical BF16 input bytes
avoided. These are logical tensor bytes, not measured DRAM traffic. Allocator peak above the
same live inputs/weights is unchanged at 57,688,576 bytes.

The inference projection-only medians over 30 alternating samples were 4.3003 ms and 3.1205 ms.
The corresponding sample ranges were 4.0008–4.8660 ms and 2.9481–3.5205 ms.
This was a warm laptop and synthetic operator screen; the measurements do not establish full-H3
performance or performance on another GPU. A smaller 1024-token screen also passed every case.
Earlier unconditional-scope results are retained under
`/home/fidika/cozy_v2/outputs/h3-qkv-boundary-sharing-20261115/`: they exposed a no-grad 1 → 2
regression across windows and are superseded by the conditional implementation.

Reproduction (the GPU interpreter supplies CUDA Torch; dependencies are appended after it imports):

```bash
<cuda-python> research/h3-boundaries/qkv_probe.py \
  --dependency-site <compatible-site-packages> \
  --tokens 4096 --repetitions 30 --output <receipt.json>
```

## Serving boundary

H3's [denoising loops](../minimax_h3/steps.py) use `torch.no_grad()`. The current Runtime source
does not wrap ordinary executor invocation in inference mode. Therefore this change has **no
demonstrated ordinary eager H3 quantizer saving**; its demonstrated benefit is the capture/inference
boundary. It does not enable graph capture or change the graph owner's memory/binding work.
CUDA graph screening used PyTorch's allocator, not the complete Varena-managed model lifecycle.

No GPU rental or full-model run was performed. The local process exited and `nvidia-smi` independently
showed no compute processes before the GPU slot was released. Full-model qualification requires
matched ordinary-CLI 15-second H3 requests (362 frames,
1344×768, 30 steps and audio for regular H3; Turbo at its own fixed schedule), executed
precision, output parity, peak memory, timing and provenance. A kernel probe is only screening.
