# Checkpoint-owned loading data

XS170 moves tokenizer, processor and learned normalization data into the model's
checkpoint. Installed package code remains the constructor and inference implementation.

H3's five weighted components stay unchanged. Its processor files belong to
`text_encoder` as `processor/*` assets, exposed to the loader under
`text_encoder/processor/`. Both ordinary and turbo base constructors receive the
selected checkpoint's `ModelAssets`; they do not read the package's processor folder.

The latent upscaler's construction config must contain `latent_normalization`:
`enabled: true`, 24 finite `mean` values and 24 finite, positive `std` values. These
are learned model data, not package-wide constants.

H3 `1.0.0-rc.8` and the upscaler's `0.1.1` release carry this data. This consumer
adopts those independently published checkpoints. Their previous releases remain
immutable; the added data changes checkpoint identifiers without changing weights,
construction geometry or timestep plans. No package or model publication is
performed by this PR.

Anima's current `1.0.0` lanes already carry both tokenizers and their configuration.
Its consumer uses those selected checkpoint assets directly.

H3 lane production and retabling validate all source construction facts before
opening an output writer. Library version labels and intentional AdaLN/conditioner
stamps may change; learned VAE and audio normalization values may not be silently
replaced by the package's supported-constructor reference.

Lane production requires the source's five processor files. Retabling requires them
on its pruned base, whose assets are retained; its full generating model supplies
DiT weights and plans, so unused processor data is not required there. Assetless old
inputs refuse before an output writer opens. Runtime's `Output.inherit` preserves
all source component assets for fresh lanes; a retable Writer preserves its base
assets and replaces only the requested timestep plans.

P-H3-Image is a separate model. Its existing `prunaai/p-h3-image@0.1.0` checkpoints
already contain their own `processor` and `prompt_upsampler_processor` components.
Those consumers read the selected checkpoint's assets directly; they do not use
the official H3 folded processor namespace or its base-model guards.
