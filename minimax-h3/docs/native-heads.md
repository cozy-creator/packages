# Native H3 head operation

Implementation work in progress for xs-138. The first closed contract covers the actual projected `_Heads` call on a single GPU when its selected dense route is INT4QK. Q/K/V, norm/RoPE, optional key-mean seed, output and exact backend scratch belong to one native choice before legacy sizing. Actual Rows and baked hook semantics remain. Kitchen INT8, sparse producers and multiple GPUs are explicit later migrations.

No consumer implementation or GPU qualification is claimed by this initial planning commit. Runtime design: `docs/varena-native-heads.md` on `codex/xs138-p2-heads`.
