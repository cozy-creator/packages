"""Qwen-Image's quantized lanes, for either checkpoint: which weights each encoding takes is
this package's recipe; the encodings and their kernels are Runtime's. Both are made from the bf16
lane."""

from __future__ import annotations

from cozy_runtime.author import EncodingRule, QuantizationPlan, Recipe

from .model import QwenImage

_BLOCKS = "transformer.transformer_blocks.*"
_TEXT = "text_encoder.model.layers.*"

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default lane: every linear of
#: the 60 DiT blocks, their two modulation heads included (6.8B of the 20B weights), and the q/k/v/o
#: and MLP projections of the text encoder's 28 language layers (6.5B; Paul 11-16: text encoders
#: int8 ConvRot by default). v1 kept the encoder bf16 after a two-reference compose lost Picture 1's
#: identity (run 14702); se-143 checks that compose again. The vision tower (its `down_proj` K=3420
#: is off ConvRot's 256), the embeddings, the input, timestep and output projections and the VAE
#: keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=QwenImage,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            f"{_BLOCKS}.attn.to_[qkv].weight",
            f"{_BLOCKS}.attn.add_[qkv]_proj.weight",
            f"{_BLOCKS}.attn.to_out.0.weight",
            f"{_BLOCKS}.attn.to_add_out.weight",
            f"{_BLOCKS}_mlp.net.*.weight",
            f"{_BLOCKS}_mod.1.weight",
            f"{_TEXT}.self_attn.[qkvo]_proj.weight",
            f"{_TEXT}.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)

#: The int8 lane with the blocks' attention and MLP linears in NVFP4 (W4A4 on Blackwell's FP4
#: tensor cores); the modulation heads and the text encoder stay int8.
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
