"""Qwen-Image's quantized lanes, for either checkpoint: which weights each encoding takes is
this package's recipe; the encodings and their kernels are Runtime's. Both are made from the bf16
lane."""

from __future__ import annotations

from cozy_runtime.author import EncodingRule, QuantizationPlan, Recipe

from .model import QwenImage, QwenImageTextEncoder

_BLOCKS = "transformer.transformer_blocks.*"

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default lane: every linear of
#: the 60 DiT blocks, their two modulation heads included (6.8B of the 20B weights). The text
#: encoder stays bf16, as in v1's 8-bit lane: its language layers carry the references' image
#: tokens, and with it int8 too a two-reference compose lost Picture 1's identity (run 14702). It
#: runs once per request. The input, timestep and output projections and the VAE keep their
#: source bytes too.
INT8 = Recipe(
    lane="int8",
    model=QwenImage,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer",),
        include=(
            f"{_BLOCKS}.attn.to_[qkv].weight",
            f"{_BLOCKS}.attn.add_[qkv]_proj.weight",
            f"{_BLOCKS}.attn.to_out.0.weight",
            f"{_BLOCKS}.attn.to_add_out.weight",
            f"{_BLOCKS}_mlp.net.*.weight",
            f"{_BLOCKS}_mod.1.weight",
        ),
        output_precision="preserve",
    ),
)

#: The int8 lane with the blocks' attention and MLP linears in NVFP4 (W4A4 on Blackwell's FP4
#: tensor cores); the modulation heads stay int8 and the text encoder bf16.
NVFP4 = Recipe(
    lane="nvfp4",
    model=QwenImage,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(
            EncodingRule(f"{_BLOCKS}.attn.*", "nvfp4"),
            EncodingRule(f"{_BLOCKS}_mlp.*", "nvfp4"),
        ),
        output_precision="preserve",
    ),
)

#: The encoder's own 8-bit lanes on a checkpoint of the encoder alone (the `text_encoder` slot's), Paul 11-19: "make
#: sure they're at int8-convrot and fp8 at minimum". Its 28 language layers' projections; the vision tower, embedding
#: and head keep their bytes. The whole-checkpoint lanes above still keep the encoder bf16 (run 14702).
TEXT_ENCODER_INT8 = Recipe(
    lane="int8",
    model=QwenImageTextEncoder,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("text_encoder",),
        include=("text_encoder.model.layers.*_proj.weight",),
        output_precision="preserve",
    ),
)

#: The same projections in `fp8.rowwise`, for cards with no int8 tensor cores (B300).
TEXT_ENCODER_FP8 = Recipe(
    lane="fp8",
    model=QwenImageTextEncoder,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=("text_encoder",),
        include=("text_encoder.model.layers.*_proj.weight",),
        output_precision="preserve",
    ),
)
