# Declared residual-FFN tail (source draft)

The current in-place `_Tail.run` topology is retained. One author descriptor will
name its readonly residual/norm/gate/index operands, up/down projections and the
three existing compiled GLUE entries. Runtime chooses one complete row menu and
preallocates all scratch before writing the caller-owned attention output. It
runs no nested split, altered quantization fusion, per-chunk synchronization or
retry after the first write.

Current package master28ff0e75/H30.2.6 is the base. The older93861 explicit phase
input/key declaration is merged without reverting current Tail, stepcache,
continuation or release work. Both before/after full-model arms must use the same
new package source. No change to the in-flight frozen938 retention pair.

This is an unreleased dev-cohort API dependency, not a guessed published Runtime
version. Original-versus-composite arithmetic/workspace/alias/failure and full H3
qualification remain pending. Kernel bodies and GLUE entries are to remain exact.
