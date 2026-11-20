# H3 Plate patch selector

This benchmark-only candidate removes the same number of guide rows as the existing whole-frame `uniform` arm. It keeps a guide patch at every timestamp and in every overlapping 2×2 spatial region. Each spatial location retains an observation at least every third latent frame. The first and last guide frames and the carried head through the continuation join keep all rows. Motion, drift since the last guaranteed kept patch, and detail increase a candidate's keep priority. When much of a scene changes, filling the matched budget can still remove changed patches. The source motion test exercises a quiet background with a localized subject, static edge, and cut.

## Staged candidate

```text
OfficialH3Pipeline._pack_guide
  draw noise at the original guide shape
  patchify clean and noised guides in identical order
  score the clean guide and gather corresponding noised rows
  pack the retained rows with their original target offsets
```

Use the isolated Packages #1075 retry candidate at `outputs/h3a-100/matched-h100/project/candidate-retry` in the Cozy v2 workspace. The retry uses Packages RC `030fd0de` with the temporal sampler and its startup correction at `3b91948eb81430124775c55a7b81419c2ed30039`. Copy its package source into a separate directory. Apply [candidate.patch](candidate.patch) there with `git apply`, then copy [guide_patch_selector.py](../../minimax-h3/minimax_h3/guide_patch_selector.py) into that copy's `minimax_h3/` directory. The resulting overlay adds `patch` to the staged package only. `full` remains the default there, and the full, uniform and motion arms keep their previous execution paths. The guide count returned by `_pack_guide` drives the existing clip-selection and Sol prefix logic. The guide audio, anchor, target and continuation rows still use their existing paths. The sampling receipt counts a timestamp as kept when the guide retains any patches there.

The staged source, file hashes, offline wheel, schema and matched watchmaker request are under `outputs/h3a-100/patch-selector/prototype/`. The source manifest verifies the same 62 original inputs with changes to `minimax_h3/guide_rows.py` and `minimax_h3/official.py` and the new selector. The original temporal candidate files match their preserved hashes.

## CPU evidence

- The selector retained 86,688 of 107,856 rows (21,168 removed, or 19.63%) on synthetic `107 × 24 × 42` input with 96 features per patch and the first eight frames protected. Single-threaded PyTorch 2.13 took 303 ms median across five warm trials (300–334 ms). This measures CPU selection on synthetic features.
- Four selector tests confirmed exact row count, deterministic ordering on a flat guide, full protected endpoints and head, spatiotemporal coverage, and retention of the moving patch region, cut and edge in synthetic input. An independent staging test called the candidate's actual `_pack_guide` and confirmed equality of the selected noised rows, positions, tags and request RNG against the full arm, identical non-guide rows, and an equal budget to uniform.
- The staged wheel built offline and `cozy run --describe ./candidate-patch/edit --json` showed `patch` with `full` still the schema default.

```bash
PYTHONPATH=minimax-h3 OMP_NUM_THREADS=1 python checks/test_h3_plate_patch_selector.py -v
PYTHONPATH="<staged package>" OMP_NUM_THREADS=1 python research/h3-plate-patches/check_candidate.py -v
```

## Matched GPU qualification

Run a new full/uniform/patch comparison once a qualified two-H100 machine is assigned for this experiment. Use the same staged package, 1344×768 canvas, 362 delivered frames, stereo audio, seeds, model and adapters, precision, worker and Runtime for every arm. Keep four dense and four sparse steps. Use the existing matched request as a first control and include a clip with localized movement, fine stationary detail and a continuation join. Record selection and packing time, live rows, realized Sol density, dense and sparse step times, GPU peak, end-to-end time and actual decoded media. Review source registration and short-lived motion at original timestamps, details in preserved regions, audio alignment, join continuity and edit fidelity. [Tracker h3a-100](https://github.com/cozy-creator/tracker/blob/master/tracker/h3-acceleration/h3a-100-plate-guide-compression.md)

The candidate is ready for a matched run. GPU execution, speed and rendered quality still need measurement.
