# FL2VA phase inputs

Plain single-GPU FL2VA declares the tensors its denoising step reads through
`denoise_loop(inputs=...)`. The two mutable latent states and ordered producer
mean slots remain `state` and `carried`. The twelve read-only inputs are the prompt
embeddings, packed timestep values and row indices, modality tags, positions,
video/audio/text packing indices, two Euler timestep tables and two sigma tables.
They are passed without copying, casting, reading device values or changing order.

The immutable phase key describes condition-row counts, the existing phase tuple,
dense/sparse attention layouts and ordered producer-site semantics. Tensor values,
seeds, Python object IDs and virtual addresses are absent. Runtime supplies model
content/fill identity, tensor geometry and the actual dispatch/cut observations.
The existing module IDs are only local routing keys for the carried mean slots.

This declaration lets Varena P2 associate successful eager observations with their
actual configuration. It changes neither Euler arithmetic nor the phase/producer
schedule. The official helper-selection predicate remains unchanged: continuation,
Ref2VA, overlay kwargs and multi-rank execution stay on their existing paths.

## Graph lifetime boundary

This change does not enable cross-call graph reuse. `phases` remains explicit and
`after` remains absent; Runtime refuses a retained whole-call graph under either
condition. Its first-call and idle-recording paths both enforce that refusal.

Request-local derived data still exists in `StepInvariants` (rotary tables and
refined text), producer rotary presentation and Kitchen's packed rotary cache.
Their creation and release order remains unchanged. Before future cross-call
reuse, that derived data needs explicit owned slots and per-call refresh/rebinding;
a semantic key or shape comparison alone cannot refresh old prompt/state values.
Do not add a dummy `after` or remove phases merely to permit reuse.

## Qualification

This source change needs ordinary full H3 qualification with paired P2 Runtime
and TensorFS candidates: 15 seconds, 362 frames, 1344×768, 30 steps and audio.
Require a complete successful eager observation; retries, incomplete execution
and capture must not replace the last successful record. A second same-geometry
call with changed prompt/seed checks fresh input binding and preserved quality.
Exact model-output bytes are informative but are not the owner's acceptance bar.
No GPU or numerical qualification is claimed by the static source checks.

Regenerate, Ref2VA, Turbo, cached presets and multi-rank observation publication
remain separate migrations. Their behavior is not changed here.
