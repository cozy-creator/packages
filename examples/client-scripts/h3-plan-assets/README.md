# H3 timestep plan assets

Each H3 DiT carries its timestep plan as the `timestep-plan.json` asset of its component
(`transformer`, `transformer_ref`; the turbo LoRA's `fl2va_turbo`, `ref2va_turbo`), first in
`minimaxai/minimax-h3@1.0.0-rc.4` and `alibaba-pai/minimax-h3-turbo-lora@1.0.1`. The package reads them from the
selected model and holds no copy.

`scripts/h3_plans.py --output DIR 30 40 50` (Turbo: `--turbo 8`) generates plans from the official
scheduler. A plan reaches a model through the producer: `minimax-h3-tools` `retable`, the lane
functions and `turbo-lora` write it with the tables it labels. Adding a plan to an existing release
without rewriting tables needs a metadata-only TensorFS commit: `tensorfs.Writer(base=...)` fetches
every carried tensor chunk before it commits, so that is a TensorFS change, not a script here.

Read a published checkpoint's header and plan references without reading tensor values:
`cozy run ./examples/client-scripts/h3-plan-assets/inspect_headers.py model.source=org/model@sha256:<commit> --on MACHINE`.
