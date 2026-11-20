"""FLUX.2's quantized lanes, one recipe per lane for each text encoder ([dev]'s Mistral, [klein]'s Qwen3):
which weights each encoding takes is this package's; the encodings and their kernels are Runtime's. Of a
lane's two recipes, a checkpoint is quantized by the one whose patterns name its weights.

As BFL's own FLUX.2 8-bit and NVFP4 files (`FLUX.2-klein-4b-fp8`, `FLUX.2-dev-NVFP4`), only the DiT blocks'
linears are quantized: the modulation heads, the context, image and time embedders and the final layer keep
their bf16, and NVFP4 leaves the double blocks' text attention at a higher precision (int8 here). The VAE and
Mistral's vision tower keep their bytes too. The Turbo LoRA stays 16-bit.
"""

from __future__ import annotations

from cozy_runtime.author import EncodingRule, QuantizationPlan, Recipe

from .model import Flux2

#: Every linear of the DiT's blocks: attention and MLPs of both streams.
_DIT = (
    "transformer.*blocks.*.attn.to_*.weight",
    "transformer.*blocks.*.attn.add_*_proj.weight",
    "transformer.*blocks.*.ff*.linear_*.weight",
)
#: [dev]'s Mistral layer projections and the LM head its upsampler decodes with.
_MISTRAL = (
    "text_encoder.language_model.model.layers.*_proj.weight",
    "text_encoder.language_model.lm_head.weight",
)
#: [klein]'s Qwen3 layer projections.
_QWEN3 = ("text_encoder.model.layers.*_proj.weight",)
#: BFL's NVFP4 [dev] keeps the double blocks' text attention (`txt_attn.qkv`, `.proj`) out of FP4.
_NVFP4 = (
    EncodingRule("transformer.transformer_blocks.*.attn.add_*_proj.weight", "int8.convrot"),
    EncodingRule("transformer.transformer_blocks.*.attn.to_add_out.weight", "int8.convrot"),
    EncodingRule("transformer.*blocks.*", "nvfp4"),
)

#: The int8 lane (`int8.convrot`): W8A8 on int8 tensor cores, the default lane.
DEV_INT8 = Recipe(
    lane="int8",
    model=Flux2,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(*_DIT, *_MISTRAL),
        output_precision="preserve",
    ),
)
KLEIN_INT8 = Recipe(
    lane="int8",
    model=Flux2,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(*_DIT, *_QWEN3),
        output_precision="preserve",
    ),
)

#: The 4-bit lane: the DiT blocks as W4A4 on Blackwell's FP4 tensor cores, the rest as int8 has it.
DEV_NVFP4 = Recipe(
    lane="nvfp4",
    model=Flux2,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=DEV_INT8.plan.components,
        include=DEV_INT8.plan.include,
        encodings=_NVFP4,
        output_precision="preserve",
    ),
)
KLEIN_NVFP4 = Recipe(
    lane="nvfp4",
    model=Flux2,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=KLEIN_INT8.plan.components,
        include=KLEIN_INT8.plan.include,
        encodings=_NVFP4,
        output_precision="preserve",
    ),
)


#: The fp8 lane (`fp8.rowwise`), for cards with no int8 tensor cores (B300): the int8 lane's components and weights. One
#: recipe per lane on the whole checkpoint, so a whole checkpoint's fp8 lane is fp8 throughout; a checkpoint of the encoder
#: alone takes its fp8 lane as a cut of one (`cozy model upload <model>@<fp8>:text_encoder`).
DEV_FP8 = Recipe(
    lane="fp8",
    model=Flux2,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=DEV_INT8.plan.components,
        include=DEV_INT8.plan.include,
        output_precision="preserve",
    ),
)

KLEIN_FP8 = Recipe(
    lane="fp8",
    model=Flux2,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=KLEIN_INT8.plan.components,
        include=KLEIN_INT8.plan.include,
        output_precision="preserve",
    ),
)
