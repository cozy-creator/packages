# H3 attention

`H3Model` declares `attention_bits={"transformer": 8, "transformer_ref": 8, "vae": 8}` (`minimax_h3/model.py`):
its DiTs' and video VAE decoder's attention run at 8 bits (Kitchen INT8 on consumer cards, as ComfyUI's), and
the audio VAE and text encoder keep exact attention. Never 4: Paul, 11-11, after same-seed side-by-sides, "4-bit
attention is not worth it; gains are small in terms of speedup, and it reduces quality in the seagulls". The
package names no GPU and no kernel. Runtime serves each site with the first kernel of the GPU family's cell at
that level: Kitchen INT8 on RTX 30/40/50-class cards (`kitchen-sol-producer-fast-shared-qkv` on sparse steps),
Sage2 on Hopper, FA4 FP8 with rotated Q/K on datacenter Blackwell, and Sol on sparse steps. `--lane bf16`
serves every site exactly.

`AttentionLayout` (`minimax_h3/official.py`) supplies the rest:

- Turbo keeps eight evaluations: four dense, then four sparse. Regular sampling keeps the first
  ten dense (`sol_dense_steps`), then sparse.
- `dense_paths`: the token refiners and the first two transformer blocks are dense in every step;
  the protected text/audio prefix of a sparse block is dense too.
- Exact rows: the text refiner and block 1 (`exact_paths`), and blocks 45, 48 and 49, which Runtime
  finds by their key RMSNorm gain, take Kitchen INT8 where the route has it (else BF16), because FP4's
  and per-head FP8's scales do not carry their key outliers (on an RTX 5090: FP4 0.10–0.28 rel L2
  there, Kitchen INT8 0.008–0.030; FP8 0.10–0.42 on B200).

The 4-bit forms (Sage3 FP4 dense rows on RTX 50, int4-QK on RTX 30/40, and their Kitchen Sol producers)
serve only by pin.

## Explicit overrides

A pin takes precedence over the rules:

```sh
cozy run minimax-h3/shot --input shot.yaml --on my-machine \
  --attention-kernel=transformer=kitchen-sol-producer-fp4-lowmem-shared-qkv
```

A dense-only pin keeps every evaluation dense. An unsupported pin fails instead of substituting
another kernel. The run's observed attention implementation and dense/sparse call counts are the
evidence; a requested selector is not.
