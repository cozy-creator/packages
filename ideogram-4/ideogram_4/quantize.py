"""Ideogram 4's int8 lanes: which weights each encoding takes is this package's recipe; the encodings and their
kernels are Runtime's.

Ideogram publishes only weight-only fp8 (per-row scales on every linear), so the DiTs' int8 lane is written from that
fp8, as Comfy-Org's `ideogram4_*int8_convrot` files are (Paul 11-12: "8-bit by default (int8 convrot) wherever
possible"). The text encoder's is written from stock Qwen3-VL-8B-Instruct's bf16."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import Ideogram4
from .text_encoder import QwenEncoder

_DITS = ("transformer", "unconditional_transformer")
_LM = "model.model.language_model.layers.*"

#: Both DiTs' block linears (attention q, k, v and out, the SwiGLU's w1-w3, the block's modulation) in
#: `int8.convrot`, as Comfy-Org's. The embedders, the caption projection, the timestep MLP, the final layer and
#: the VAE keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=Ideogram4,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=_DITS,
        include=tuple(
            f"{dit}.layers.*.{linear}.weight"
            for dit in _DITS
            for linear in ("attention.to_[qkv]", "attention.to_out.0", "feed_forward.w[123]", "adaln_modulation")
        ),
        output_precision="preserve",
    ),
)

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
