# Matched MiniMax H3 Plate guide benchmark

Draft experiment for h3a-100. The isolated package supports `guide_sampling: full`, `uniform`, and `motion` on its `edit` tool. Full is the default. Both reductions remove one eligible latent frame per equally sized time bin, normally 21 of 107 on a 362-frame input. Motion selects the least active frame in each bin. Each comparison uses the same retained count and full guide decode, VAE encode, and RNG draws.

## Paired runs

1. Use a qualified experimental package installed by exact version and digest on the same two H100s. Pin the Runtime, worker, base model, edit LoRA, Turbo adapter, attention lane, and package identity for the entire set. Start with a 1344×768, 362-frame Plate at 24 fps. Include a rapid pan, a short foreground action, a cut, and fine stationary detail. Test a longer Plate with a continuation join after the single-segment comparison.
2. Import the Plate once into a disposable benchmark project and name it `plate`. Prepare one ordinary `edit` request with `plate: <plate>`, `edit: vfx_edit`, identical summary and beats, `turbo: true`, `quality: standard`, and a fixed seed. Clone the request twice and change only `guide_sampling`. Use a separate seed for warmup or a repeated set, and the same seed across each measured set.
3. Run the three requests using the `cozy` CLI. Set `target` to the exact installed experimental `edit` function, `machine` to the qualified machine, and `lane` to the shared precision lane. Give each run its own output directory and retain its complete record.

```bash
cozy run "$target" --on "$machine" --gpus=2 --lane "$lane" \
  --input "$request" --json --full --output "$output"
```

Use the rendered MP4 and the segment's `h3 guide sampling` receipts to confirm execution and the matched row budget. Check actual denoising stages, four dense and four sparse steps, and evidence that the segment rendered rather than replayed a memo. Record queue, preparation, VAE condition, denoise, decode and end-to-end separately. Alternate run order across seeds so an improving warm cache is not assigned to one mode.

## Quality review

Keep the deliverable frame rate, total frames, dimensions, and sound duration equal across variants. Review anonymized outputs at ordinary speed and frame-by-frame around the source's marked cut, rapid motion, occlusions, and the continuation join. Check timing and shape of the moving subject, stationary texture, identity, instruction fidelity and sound sync. Compare against the source where it should be preserved and the full-guide output where the edit intentionally changed it. Record changes with frame ranges and compare them alongside the end-to-end and denoise timings.

Local CPU coverage exercises the production packing function with synthetic latents and no model weights. Run it from the repo root after installing the H3 package's dependencies, including PyTorch.

```bash
PYTHONPATH=minimax-h3 minimax-h3/.venv/bin/python checks/test_h3_guide_compression.py -v
```
