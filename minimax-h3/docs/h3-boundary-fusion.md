# H3 producer/consumer boundary experiment

Owner: Codex `/root/h3_boundary_fusion`. Purpose: remove repeated activation quantization at
H3's Q/K/V fan-out through existing Runtime contracts. Branch:
`codex/h3-qkv-boundary-sharing-20261115`. Packages base:
`e4b179bbd099f681658dc7ef5a059f484969cf94`. Worktree:
`/home/fidika/cozy/.worktrees/packages/h3-qkv-boundary-sharing-20261115`.

Runtime source inspected: `af2a4753cb70598b75405942901e4f8f3b24eda6`. No native interface,
attention-kernel internals, graph ownership, model precision, or automatic kernel selection changes.

## Remaining boundaries, ranked

1. **Same input quantized separately for Q, K and V.** `glue._Heads.__call__` projects one
   input through three `Rows` calls. Runtime's `shared_operand` does not share inference
   tensors implicitly (no mutation version counter), or a leaf recording a graph. The existing
   `quantize_once` scope provides explicit sharing. Close it immediately after projections,
   before RoPE, key-mean seeding and attention. This is the selected narrow experiment.
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
The proposed scope is per existing head window, with no cache across windows or attention calls.
It retains one operand through Q/K/V and releases it before attention; separate quantizers already
need an operand while the final V projection coexists with Q/K.

## Evidence and qualification

The existing SM89 15-second Turbo diagnostic (cr-297 session `43442514074`, cell `m-h1`) reports
426.68 ms of one INT8 quantizer family and 321.31 ms of another in a 49,445.33 ms dense sampled
call. These families cover more than Q/K/V. Event sums overlap and are not a critical-path bound;
they do not imply the candidate's full-model saving. The investigation starts from a concrete
repeated producer computation, not a claimed end-to-end bottleneck.

Sources: [H3 glue](../minimax_h3/glue.py), [sparse producer](../minimax_h3/producer.py),
[Runtime sharing contract](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/src/cozy_runtime/accel/kernels.py#L212),
[Runtime row projections](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/src/cozy_runtime/accel/rows.py#L114).

Status: source investigation; execution probe pending. No GPU rental or full-model run.
Full-model qualification requires matched ordinary-CLI 15-second H3 requests (362 frames,
1344×768, 30 steps and audio for regular H3; Turbo at its own fixed schedule), executed
precision, output parity, peak memory, timing and provenance. A kernel probe is only screening.
