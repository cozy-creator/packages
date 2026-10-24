"""The text encoder's int8 lane, written from stock Qwen3-VL-8B-Instruct's bf16: which weights the encoding takes
is this package's recipe; the encodings and their kernels are Runtime's. Ideogram 4 itself is served as Ideogram
publishes it, weight-only fp8 (per-row scales on every linear): no 16-bit release exists, so no lane is written
from it (owner, 2026-10-24)."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .text_encoder import QwenEncoder

_LM = "model.model.language_model.layers.*"

#: Qwen3-VL-8B-Instruct's language linears in `int8.convrot`; its embedding, head and vision tower
#: keep their bf16 bytes. The same lane serves conditioning and the upsampler.
TEXT_ENCODER_INT8 = Recipe(
    lane="int8",
    model=QwenEncoder,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(f"{_LM}.self_attn.[qkvo]_proj.weight", f"{_LM}.mlp.*_proj.weight"),
        output_precision="preserve",
    ),
)
