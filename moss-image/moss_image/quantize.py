"""MOSS-Image's int8 lane: which weights it encodes is this package's recipe; the encoding and its
kernels are Runtime's. The upload is the bf16 lane as it is."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import MossImage, MossTextEncoder

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

#: The encoder's fp8 lane (`fp8.rowwise`), for cards with no int8 tensor cores (B300): the int8 lane's
#: projections, on a checkpoint of the encoder alone (the `text_encoder` slot's).
TEXT_ENCODER_FP8 = Recipe(
    lane="fp8",
    model=MossTextEncoder,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=("text_encoder",),
        include=(f"{_TEXT}.self_attn.[qkvo]_proj.weight", f"{_TEXT}.linear_attn.in_proj_qkv.weight", f"{_TEXT}.linear_attn.in_proj_z.weight", f"{_TEXT}.linear_attn.out_proj.weight", f"{_TEXT}.mlp.*_proj.weight"),
        output_precision="preserve",
    ),
)
