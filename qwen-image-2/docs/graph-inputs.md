# Cached denoising graph inputs

Qwen Image 2.1 still executes step zero for each request to extract that request's prefix K/V. The remaining steps declare the latent as mutable state and the Euler tables, rotary table, target mask and every layer's prefix K/V as read-only inputs to `denoise_loop`.

The step closes over the placed transformer and its block list only. Small Diffusers cache wrappers are rebuilt from the supplied tensor arguments. When Runtime records using independent storage, or copies a new prompt and noise into a retained program, the step therefore reads that storage rather than tensors from the old request. The semantic key names this algorithm; Runtime also keys tensor layouts, step range and model/kernel identities. Step-cache presets retain Runtime's existing eager behavior.

This uses the companion Runtime correction allowing a keyed loop with no `after` callback. The recording implementation already has an optional suffix. No dummy callback, altered denoising schedule, byte estimate or application retry is introduced.

## Proof

Before: actual L40 run 26833 completed base 2048×2048 / 40 steps / int8 but recorded no graph: the loop declared no key, inputs or retained program.

After: the actual tiny Diffusers 0.41 Qwen 2.1 transformer on Torch 2.15.0+cu132 CPU produces exact stock cached-forward values for twelve steps across three requests. Changing both noise and prompt conditioning in the same input layouts changes the result; restoring them restores the exact result. Original request tensors collect while the step remains alive. This is CPU numerical/ownership proof, not CUDA capture or a simulation of the native Runtime.

Exact source hashes, client, commands and receipts are in `/home/fidika/cozy_v2/outputs/xs138-qwen-declared-inputs-20261103/cpu-001/`. The helper passes Ruff; `model.py` retains the same three pre-existing diagnostics as the base. Syntax and diff checks pass.

Required GPU gate: actual ordinary-Cozy base 2048×2048 / 40-step int8 calls with the matching Runtime, checked completion, capture and reuse counters, different prompt/seed values, output quality and source custody. The successful old eager runs do not close this gate. No merge or release is authorized.
