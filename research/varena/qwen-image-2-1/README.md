# Qwen-Image-2.1 candidate preparation

This is preparation for actual image inference, not a GPU result. No machine, SDK, model download,
rental or publication was changed here. Use the parent's explicitly handed-off GPU lane.

## Exact identities

Default Cozy CLI metadata on 2026-10-25 identifies the first-party **package**
`qwen-image-2@0.11.3`, separately from **model** `qwen-image-2.1@0.1.0`. The package's
`generate` default names that base model. The source constructs Diffusers'
`QwenImage21Pipeline`/`QwenImage21Transformer2DModel`; it does not infer model identity from a
package version. The [official base model card](https://huggingface.co/Qwen/Qwen-Image-2.1)
identifies its 7B visual component and 40-step example. The package admits 1–100 steps, so the
requested 30-step, 1 MP (1024²) base-model run is ordinary supported input.

| Published model/lane | Immutable Hub commit | Stored bytes | GiB |
| --- | --- | ---: | ---: |
| `qwen-image-2.1@0.1.0`, int8 | `ce2918d01f2132226405f2f5aa8c702507591b0778abfc00cc03b0c1f8da14f8` | 19,218,339,764 | 17.898 |
| `qwen-image-2.1@0.1.0`, bf16 | `3feeda1b7c396addd8e2aeafd29d2dadd5bbe611ffb167ece0c18328c85b4b42` | 33,132,028,647 | 30.857 |
| `qwen-image-2.1-turbo@0.1.0`, bf16 | `7a5858cb36251790d8960c832d8d6bc1e75db5e8f3c841721b77af7b6f1a9c3e` | 32,452,129,570 | 30.223 |

Hashes above use the `sha256:` scheme. The base int8 lane includes `int8.convrot`, BF16 and F32;
it is not an all-int8 model. Hub provenance fields were null, so this preparation does not assert
an upstream conversion revision from the Hub hash alone. `model-snapshot.json` preserves the CLI facts.

Turbo is a separate checkpoint with its own **8-step** schedule. The package clamps a requested
30 to that schedule; the [Turbo card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) explains
that changing step count alone does not replace it. Do not substitute Turbo for this 30-step case.
The catalog also lists unpublished base mxfp8 and Turbo int8 commits; neither is selected here.

## Candidate and dependency boundary

The source associated with published package 0.11.3 (packages commit `8b87f6b3`) declares
`cozy-runtime[media]>=0.30,<0.32`, which excludes the candidate Runtime 0.32.1. The normal CLI
published contract describes successfully, but that is not SDK admission or inference proof.

This branch starts from fetched packages master `da31a14a`. Its requirements admit Runtime
0.32.1, but its new `vlm.processor(assets, prefix, config)` call is newer than exact Runtime
`75bce8937`, whose helper accepts only `assets`. For this qualification-only source candidate,
`model.py` retains the existing package-owned processor construction from preceding commit
`e023b0bd`. Device selection, sampling, source VAE precision and current request-size behavior
remain that source's. This avoids changing the installed SDK or adding runtime version dispatch.
Do not treat this candidate-only restoration as a publication or an endorsement of duplicating
processor construction in the long-term shared API. Two unused local bindings were removed for lint.

Declared direct constraints: Python >=3.12; Runtime[media] >=0.31; Diffusers >=0.41,<0.42;
Transformers >=5.17; Torch >=2.13; Torchvision >=0.28; Tokenizers >=0.23.1; Pillow >=12;
NumPy >=1.26; msgspec >=0.19. The existing lock records Diffusers 0.41.0, Transformers 5.17.0,
HF Hub 1.33.0, Tokenizers 0.23.2, Pillow 12.3.0, and Linux Torch 2.14.0/Torchvision 0.29.0.
Its Runtime 0.31/TensorFS 0.14 resolution is not the machine SDK: repo policy selects the installed
Runtime/TensorFS pair subject to the declared bounds. Preserve the actual preparation/software
receipts, including any private dependency installation; do not bypass resolution with `--no-deps`.

## Capacity planning, not a measured fit claim

Stored bytes are neither resident weight bytes nor peak VRAM. The source stages text encoding,
transformer denoising and VAE decoding separately, accepts encoded leaves, and skips the unused
stored language-model head. Activations, native decoder/library scratch, graph pools and CUDA
context memory must still fit. No current-cohort Qwen GPU measurement is claimed.

| GPU capacity | Planned case and limitation |
| --- | --- |
| 8 GB | Int8 at 1 MP is a streaming experiment, not all-resident. Largest live groups and VAE/attention peaks are not yet measured; allow an honest capacity failure. Do not silently reduce size or steps. |
| 24 GB | Int8 at 1 MP is the preferred first baseline. Its stored payload leaves materially more room than bf16, but activations still need measurement. BF16 exceeds capacity as a whole and needs component/weight staging. |
| 48 GB | BF16 provides the most comfortable full-weight capacity baseline; its 30.857 GiB payload still excludes runtime/activation overhead. This is a planning judgment, not a guarantee. |

Allow storage for the selected payload plus the package environment and outputs. Do not download
both lanes or quantize a new lane merely to run this prepared case.

## Invocation after GPU handoff

From this worktree, the exact source invocation is:

```sh
~/.cozy/bin/cozy --json run ./qwen-image-2/generate \
  --input research/varena/qwen-image-2-1/request.json \
  model.model=qwen-image-2.1@0.1.0 seed=2026102511 \
  --lane int8 --gpus=1 --on MACHINE --out OUTPUT/cold
```

For cold plus three warm calls, reuse the existing Creator controller (commit `56de2b5fa` or
later); it does not install software or rent a machine:

```sh
nice -n 19 python3 /home/fidika/cozy/.worktrees/cozy-creator/gpu-driven-plan-proof/release-gate/suites/execution/model-qualification/run.py \
  --target "$PWD/qwen-image-2/generate" \
  --input "$PWD/research/varena/qwen-image-2-1/request.json" \
  --machine MACHINE --lane int8 --steps 30 --step-stage 'generating image' \
  --expect-size 1024x1024 --seeds 2026102511 2026102512 2026102513 2026102514 \
  --output /home/fidika/cozy_v2/outputs/varena-models-20261025/qwen21-int8-1mp-30-MACHINE
```

Use a new output directory and fresh distinct seeds for a repeated experiment. Same-prompt
encoding reuse is intentional; differing seeds prevent whole-request memoization from masquerading
as warm inference. The controller requires all 30 actual step observations, equal warm executor PID,
unchanged software cohort, selected-lane facts, real image dimensions and downloaded SHA validation.
It saves original CLI events/full run snapshots, images, wall/request/execution/first/steady-step timing,
tracked memory, graph/route facts and metric gaps. Inspect the actual images for prompt fidelity too.
“Cold” means first observed invocation; cache coldness is reported, never forced by deleting data.
