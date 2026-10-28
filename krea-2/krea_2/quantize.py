"""Krea 2's 8-bit lane, for Raw and Turbo alike: which weights the encoding takes is this package's; the
encoding and its kernels are Runtime's. Written from the bf16 upload.

As Comfy-Org's `krea2_*_int8_convrot` and `qwen3vl_4b_int8_convrot` files: every linear of the 28 DiT blocks
(attention q, k, v, out and gate; the SwiGLU's gate, up and down) and every projection of Qwen3-VL's language
layers. The text-fusion blocks, the embedders, the shared modulation, the final layer, the token embedding
and the VAE keep their bytes.
"""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import Krea2

_BLOCKS = "transformer.transformer_blocks.*"

#: The int8 lane (`int8.convrot`): W8A8 on int8 tensor cores, the default lane.
INT8 = Recipe(
    lane="int8",
    model=Krea2,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            f"{_BLOCKS}.attn.to_*.weight",
            f"{_BLOCKS}.ff.*.weight",
            "text_encoder.language_model.layers.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)
