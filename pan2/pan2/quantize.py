"""PAN2's int8 lane: which weights it encodes is this package's recipe; the encoding and its kernels are
Runtime's. IFM ships bf16 with an F32 modulation; the lane is written from that bf16."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import Pan2

_BLOCKS = "transformer.transformer_blocks.*"
_LM = "text_encoder.layers.*"

#: `int8.convrot` (W8A8 on int8 tensor cores) on the 44 blocks' attention (video and text q/k/v/out) and
#: both feed-forwards, as Comfy-Org's `pan2_int8_convrot`; the Qwen3.5 encoder's language linears
#: (gated-delta-net and attention projections, MLPs). The token refiner, embedders, F32 modulation,
#: output norm and projection, the delta-net decay/gate projections and the VAE keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=Pan2,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            f"{_BLOCKS}.attn.to_[qkv].weight",
            f"{_BLOCKS}.attn.to_out.0.weight",
            f"{_BLOCKS}.attn.add_[qkv]_proj.weight",
            f"{_BLOCKS}.attn.to_add_out.weight",
            f"{_BLOCKS}.ff.net.*.weight",
            f"{_BLOCKS}.ff_context.net.*.weight",
            f"{_LM}.linear_attn.in_proj_[qz]*.weight",
            f"{_LM}.linear_attn.out_proj.weight",
            f"{_LM}.self_attn.[qkvo]_proj.weight",
            f"{_LM}.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)


#: The fp8 lane (`fp8.rowwise`), for cards with no int8 tensor cores (B300): the int8 lane's components and weights. One
#: recipe per lane on the whole checkpoint, so a whole checkpoint's fp8 lane is fp8 throughout; a checkpoint of the encoder
#: alone takes its fp8 lane as a cut of one (`cozy model upload <model>@<fp8>:text_encoder`).
FP8 = Recipe(
    lane="fp8",
    model=Pan2,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=INT8.plan.components,
        include=INT8.plan.include,
        output_precision="preserve",
    ),
)
