"""MOSS-Image's int8 lane: which weights it encodes is this package's recipe; the encoding and its
kernels are Runtime's. The upload is the bf16 lane as it is."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import MossImage

_TEXT = "text_encoder.language_model.layers.*"

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default: the 42 DiT blocks' q, k, v, o,
#: FFN up-gate and down projections (Comfy-Org's int8 file takes the same), and every projection of
#: Qwen3.5's 32 language layers: full attention q/k/v/o, Gated DeltaNet qkv/z/out, and the MLPs. The
#: text refiner, embedders, modulation, final layer, token embedding, DeltaNet's a/b gates, the vision
#: tower and the VAE keep their bytes.
INT8 = Recipe(
    lane="int8",
    model=MossImage,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            "transformer.layers.*.proj.*.weight",
            f"{_TEXT}.self_attn.[qkvo]_proj.weight",
            f"{_TEXT}.linear_attn.in_proj_qkv.weight",
            f"{_TEXT}.linear_attn.in_proj_z.weight",
            f"{_TEXT}.linear_attn.out_proj.weight",
            f"{_TEXT}.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)


#: The fp8 lane (`fp8.rowwise`), for cards with no int8 tensor cores (B300): the int8 lane's components and weights. One
#: recipe per lane on the whole checkpoint, so a whole checkpoint's fp8 lane is fp8 throughout; a checkpoint of the encoder
#: alone takes its fp8 lane as a cut of one (`cozy model upload <model>@<fp8>:text_encoder`).
FP8 = Recipe(
    lane="fp8",
    model=MossImage,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=INT8.plan.components,
        include=INT8.plan.include,
        output_precision="preserve",
    ),
)
