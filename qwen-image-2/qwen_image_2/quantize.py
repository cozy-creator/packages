"""Qwen-Image-2.1's quantized lanes: which weights each encoding takes is this package's recipe;
the encodings and their kernels are Runtime's."""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import Context, EncodingRule, QuantizationPlan, Recipe, Telemetry

from .model import QwenImage21Model

#: Every linear of the 32 DiT blocks and of the text encoder's 36 language layers, in `int8.convrot`:
#: W8A8 on int8 tensor cores, the fast lane on consumer cards. Embeddings, the vision tower, the
#: timestep/text/image projections, the shared modulation and the output projection keep their
#: source bytes: small, and the ones the image is most sensitive to. So does the VAE.
INT8 = Recipe(
    lane="int8",
    model=QwenImage21Model,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            "transformer.transformer_blocks.*.attn.to_[qkv].weight",
            "transformer.transformer_blocks.*.attn.to_out.0.weight",
            "transformer.transformer_blocks.*.img_mlp.*.weight",
            "text_encoder.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
            "text_encoder.model.language_model.layers.*.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)

#: The 4-bit lane before Blackwell (xs-062): the int8 lane's weights with the DiT blocks in W4A8
#: (`w4a8.convrot`, multiplied on the int8 tensor cores: Ampere, Ada, Hopper); the text encoder
#: stays int8.
W4A8 = Recipe(
    lane="w4a8",
    model=QwenImage21Model,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule("transformer.*", "w4a8.convrot"),),
        output_precision="preserve",
    ),
)

#: What the svdq-nvfp4 lane is fitted to: subjects, styles, lighting and rendered text.
CALIBRATION = (
    "A bustling morning bakery counter, trays of croissants and seeded loaves, a chalkboard menu "
    'reading "FRESH SOURDOUGH 6.50", flour dust in the window light, documentary photography.',
    "A lone lighthouse on black basalt cliffs during a storm, waves exploding into white spray, "
    "lightning splitting a purple sky, a long-exposure photograph.",
    'A flat vector infographic titled "How Bees Make Honey" with five numbered steps, simple '
    "icons, a warm yellow and charcoal palette, clean sans-serif labels.",
    "An oil painting of a quiet canal in a European city at dusk, gondolas moored by striped poles, "
    "lamplight on the water, thick impasto brushstrokes.",
    "A macro photograph of a dew-covered spider web strung between dry grass stems, backlit by a "
    "low sun, bokeh highlights, extreme detail.",
    "A cheerful children's book illustration of a robot and a girl planting sunflowers on a "
    "rooftop garden, soft pastel colors, a city skyline behind them.",
    "A fashion editorial portrait of a man in a tailored mustard suit against a teal wall, hard "
    "flash, crisp shadows, a confident sideways glance.",
    "A cutaway technical illustration of a vintage mechanical wristwatch movement, gears, jewels "
    "and a balance wheel labeled with thin leader lines, a blueprint style.",
)


def calibrate(ctx: Context, tel: Telemetry, model: QwenImage21Model, record: Any) -> None:
    """Qwen-Image-2.1 at 16 bits generates each of `CALIBRATION` at 1 MP on the checkpoint's
    base schedule while each svdq linear's inputs are recorded."""
    steps = 20
    with record(model.graph.components), tel.stage("calibrating", overall_range=(0.0, 0.3)):
        for seed, prompt in enumerate(CALIBRATION):
            embeds, mask = model.encode(prompt)
            span = (0.3 * seed / len(CALIBRATION), 0.3 * (seed + 1) / len(CALIBRATION))
            model.denoise(
                embeds, mask, width=1024, height=1024, steps=steps, seed=seed,
                on_step=tel.step_callback(steps, stage="calibrating", overall_range=span),
                cancel=ctx.raise_if_cancelled,
            )  # fmt: skip


#: Blackwell's 4-bit lane (xs-066): the int8 lane's weights with the DiT blocks in SVDQuant
#: (`svdq.nvfp4`: per-channel smoothing, a 16-bit rank-32 branch holding the outliers, an nvfp4
#: residual) fitted to the model's own activations (`calibrate`, from the 16-bit source); the text
#: encoder stays int8. 6.7% per linear against plain nvfp4's 7.5% (xs-066).
SVDQ_NVFP4 = Recipe(
    lane="svdq-nvfp4",
    model=QwenImage21Model,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule("transformer.*", "svdq.nvfp4"),),
        output_precision="preserve",
    ),
    calibrate=calibrate,
)
