"""Fibo 2.0's quantized lanes, one recipe per lane for each of this package's checkpoints (Fibo 2.0 and
Bria's prompt-to-JSON VLMs). The encodings and their kernels are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import EncodingRule, QuantizationPlan, Recipe

from .model import Fibo2
from .upsampler import Captioner

_TEXT = "text_encoder.model.language_model.layers.*"

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default: every attention linear of
#: the DiT (its 44 blocks, 2 noise-refiner blocks and 2 Perceiver layers, the five injection blocks'
#: cross-attention included), its FFNs' gate and up projections, and Qwen3-VL's 36 language layers.
#: The FFN down projection (`w2`, 9,536 columns, not a multiple of ConvRot's 256), the adaLN
#: modulation, the embedders, the final layer, the vision tower and the VAE keep their bytes.
INT8 = Recipe(
    lane="int8",
    model=Fibo2,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            "transformer.*.to_[qkv].weight",
            "transformer.*.to_out.0.weight",
            "transformer.*.w[13].weight",
            f"{_TEXT}.self_attn.[qkvo]_proj.weight",
            f"{_TEXT}.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)

#: The int8 lane with the DiT's 44 blocks, their down projections too, in NVFP4 (W4A4 on Blackwell).
NVFP4 = Recipe(
    lane="nvfp4",
    model=Fibo2,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=(*INT8.plan.include, "transformer.layers.*.w2.weight"),
        encodings=(EncodingRule("transformer.layers.*", "nvfp4"),),
        output_precision="preserve",
    ),
)

_LM = "model.model.language_model.layers.*"

#: A prompt-to-JSON VLM's language linears: the bytes each decoded token reads. Its embedding (tied
#: to its head) and vision tower keep their source bytes.
CAPTIONER_INT8 = Recipe(
    lane="int8",
    model=Captioner,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(f"{_LM}.self_attn.[qkvo]_proj.weight", f"{_LM}.mlp.*_proj.weight"),
        output_precision="preserve",
    ),
)

#: The same linears in NVFP4 (W4A16 for its token loop).
CAPTIONER_NVFP4 = Recipe(
    lane="nvfp4",
    model=Captioner,
    plan=QuantizationPlan(
        encoding="nvfp4",
        components=CAPTIONER_INT8.plan.components,
        include=CAPTIONER_INT8.plan.include,
        output_precision="preserve",
    ),
)


#: The fp8 lane (`fp8.rowwise`), for cards with no int8 tensor cores (B300): the int8 lane's components and weights. One
#: recipe per lane on the whole checkpoint, so a whole checkpoint's fp8 lane is fp8 throughout; a checkpoint of the encoder
#: alone takes its fp8 lane as a cut of one (`cozy model upload <model>@<fp8>:text_encoder`).
FP8 = Recipe(
    lane="fp8",
    model=Fibo2,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=INT8.plan.components,
        include=INT8.plan.include,
        output_precision="preserve",
    ),
)
