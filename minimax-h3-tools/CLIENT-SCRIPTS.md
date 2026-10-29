# H3 preparation from a client script

`h3_tables.operations.precompute_adaln` is an ordinary async Python helper.
It composes managed operations alongside `quantize` in an unpublished client script:

```python
from h3_tables.operations import precompute_adaln, quantize

async def main(ctx):
    # full is a `bf16-full` lane, e.g. from `cozy run minimax-h3-tools/bf16-full`.
    pruned = await precompute_adaln(model=full, timesteps=50)
    fp8 = await quantize(source=pruned, encoding="fp8.rowwise")
```

Each source DiT must carry `timestep-plan.json`: `transformer` for FL2VA and
`transformer_ref` for Ref2VA. There is no package plan fallback; the model's plan
(30, 40 and 50 denoising steps in `minimaxai/minimax-h3@1.0.0-rc.4`) is the authority.
`timesteps` selects a step count that plan serves; the retained BF16 bank supports
all of them. The argument does not silently invent a new scheduler.

`precompute_adaln` runs in the calling script; it needs no `ctx` argument or
separate call of its own. It invokes two inexpensive generating-weight projections,
two independent table computations and one native metadata attachment; each one's
committed derivation replays for the same code and inputs.
The selection replies carry the source plan digest and supported step counts to
the caller. Each leaf runs with its own managed services and cancellation checks;
the source checkpoint identity includes its assets and binds the derivation.
Each table computation checkpoints completed block tables and final normalization;
a retry skips their projections and source-weight reads.

Projection content contains the task's original generating weights, required
topology and component assets, including its timestep plan. Changes to unrelated body weights therefore preserve the
projection identity. Body quantization has no influence on table bytes; changing
generating weights, the plan, numerical contract or captured code invalidates
compatible reuse. Runtime owns the memo index; these operations introduce no
package cache or manual run identifier.

The returned model carries each generating projection's digest inside its existing
`cozy_h3` config stamp. Attaching a bank from other generating weights refuses even
when table shape and timestep plan match. An already-pruned current-plan model
with these bindings can be returned after validation. To replace older tables,
pass `generating_model=full`; the source must contain the original dynamic weights.
Older artifacts with no generating binding refuse this reuse path because the
removed weights cannot be recovered from a table's shape. Planless projections
and banks also refuse: recompute them from a migrated full model. An existing
`generating_projection_digest` from a planless projection cannot be silently
rewritten when adding assets; its bank and body binding need recomputation. This
release retains the existing exact approved-plan digest boundary. It does not
claim retabling to a newly approved schedule across that older binding format.
