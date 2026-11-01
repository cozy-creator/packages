# Declared fused VAE block

`_BlockRows.run` is the real producer seam. It supplies the actual block, hidden state,
shared rotary table and the two direct layer-scale parameter slots. Runtime owns the
complete allocation menu and batch choice. Older Runtime versions keep the ordinary
call before any native acceptance; accepted failures never retry that call.

The first Runtime executor targets the existing fused2048-wide FP16-autocast layout,
FP32 residual/norm/gate math, six biased dense projections and selected Kitchen INT8.
It runs ordinary block/source hooks once and validates current bindings before writes.
No opaque modulation callback or guessed scratch is supplied by the package.

The enclosing `_decode_tiles` is still inside a legacy split. Runtime deliberately
refuses the native block while that parent scope is uncommitted, so this declaration
alone does not claim full-H3 adoption. The actual standalone block consumer can be
qualified while the outer convolution/projection/assembly contract is migrated.
No package kernel source changed; GPU numerical, allocation and lifetime proof is pending.
