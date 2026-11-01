# Exact Kitchen INT8 projected heads

Extend the complete ProjectedHeads operation using the actual selected Kitchen INT8 dense/exact route. The pinned existing quantize and prequantized APIs keep all buffers caller-owned. Reuse consumed internal Q float scratch as contiguous BHND output, then copy its transpose into the declared full NHD output on the same stream. No C++ kernel or numerical-tiling change is planned.

Use the real pinned artifact identity and shared dense/exact/prefix-heavy policy. Validate the fused reference against split ABI execution, actual allocation classes/counts, zero-allocating final copy, both CTA sizes, seed/complete-processor behavior and refusal/retirement. Sparse producer remains a separate migration. The qualified INT4 cohort and its running full-model pair stay frozen.

This initial draft records implementation scope; no INT8 GPU qualification is claimed. Primary source inventory: outputs/xs138-attention-closure-review in the shared workspace.
