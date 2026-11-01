# Native H3 head operation

The package now declares the actual projected `_Heads` call on a single GPU when Runtime can serve its selected dense INT4QK route. Q/K/V, norm/RoPE, optional key-mean seed, output and exact backend scratch belong to one native choice before legacy sizing. Actual Rows and baked hook semantics remain. Kitchen INT8, sparse producers and multiple GPUs are explicit later migrations.

The successful full processor call records completed seed heads; each piece still writes the same key-mean bytes. Both GLUE sets retain exactly their previous compiled identities. An older Runtime without the intent uses the original callable. CPU construction and static checks pass; GPU qualification is pending. Runtime design: `docs/varena-native-heads.md` on `codex/xs138-p2-heads`.
