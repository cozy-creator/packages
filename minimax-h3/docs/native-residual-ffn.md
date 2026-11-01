# Declared residual-FFN tail (unreleased source candidate)

The current in-place `_Tail.run` topology is retained. One frozen `ResidualFFN`
intent names its readonly residual/norm/gate/index operands, up/down projections
and three existing compiled GLUE entries. Runtime chooses one complete row menu
and preallocates four BF16 intermediates, shared quantization/scales and maximum
exact GEMM workspace before writing the exclusive attention output. It runs the
normal projection modules and their baked hooks, with no nested split or retry
after acceptance.

Current package master 28ff0e75 / H3 0.2.6 is the base. The older 93861 explicit
phase-input/key declaration is merged while preserving current Tail, stepcache,
continuation and release work. Both full-model arms must use this identical new
package; Runtime API-only control 1aca1d6cd runs the original callable. The frozen
938 retention pair remains separate evidence.

The initial supported contract is one GPU, contiguous BF16 rows and int8 mma.sync
projections with built-in baked hooks. FP8 and the unhooked fused-int8 prologue
continue through the original tail. Every index pointer retains the existing
16-byte ABI: whole odd row counts are legal, split extents are even and the final
odd tail starts at an even row. No index copy or kernel alignment change occurs.

Both GLUE sets and all 15 entries have unchanged Triton source/cache identities,
signatures, constants, options and alignments. Original `_Tail.run`, `_FFN`, kernel
bodies and wrappers have unchanged ASTs. The new intent import is local after the
kernels, preserving the source line numbers Triton includes in its cache keys.

This is an unreleased development cohort; no published Runtime version floor is
invented. Creator #2382 uses byte-for-byte copies of these real package modules
and compares current `_Tail.run` with the composite at whole and constrained
odd/full-tail sizes, checks actual workspace callbacks, refusal before mutation,
baked hooks and request-end cleanup. Its CPU construction passes; GPU arithmetic,
standalone-projection regression and matched full H3 qualification remain pending.
