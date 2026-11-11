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

The current public H3 `1.0.0-rc.7` lanes and upscaler `0.1.0` omit this data. Their
replacement checkpoints and new release mappings must be qualified before this
consumer change merges or publishes. Preserve existing weights, configuration,
timestep plans and old immutable releases; the added data necessarily changes
checkpoint identifiers. No package or model publication is performed by this PR.

Anima's current `1.0.0` lanes already carry both tokenizers and their configuration.
Its consumer uses those selected checkpoint assets directly.

H3 lane production and retabling validate all source construction facts before
opening an output writer. Library version labels and intentional AdaLN/conditioner
stamps may change; learned VAE and audio normalization values may not be silently
replaced by the package's supported-constructor reference.
