# Declared fused VAE block

`_BlockRows.run` is the real producer seam. It supplies the actual block, hidden state,
shared rotary table and the two direct layer-scale parameter slots. Runtime owns the
complete allocation menu and batch choice. The pre-launch cohort requires the new declaration API. Unsupported execution routes
may keep their ordinary call before acceptance; accepted failures never retry it.

The first Runtime executor targets the existing fused2048-wide FP16-autocast layout,
FP32 residual/norm/gate math, six biased dense projections and selected Kitchen INT8.
It runs ordinary block/source hooks once and validates current bindings before writes.
No opaque modulation callback or guessed scratch is supplied by the package.

The enclosing `_decode_tiles` is still inside a legacy split. Runtime deliberately
refuses the native block while that parent scope is uncommitted, so this declaration
alone does not claim full-H3 adoption. The actual standalone block consumer can be
qualified while the outer convolution/projection/assembly contract is migrated.
No package kernel source changed; GPU numerical, allocation and lifetime proof is pending.


The unchanged `VAE_BLOCK` declaration now lives beside `_BlockRows` in `vae_tiles`;
`model.VAE_BLOCK` remains the same imported object. This lets the actual standalone
consumer and its kernel build import the producer without importing the complete H3
model/tools stack. Kernel entry bodies and identities are unchanged; the origin module
metadata now names `minimax_h3.vae_tiles` and is recorded with the new package source.
