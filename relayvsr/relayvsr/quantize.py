"""RelayVSR's int8 lane: which weights take the encoding is this package's recipe; the encoding and its kernel
are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import RelayModel

_BLOCKS = "wan.blocks.*"

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default (Paul 11-16: every model int8 ConvRot by
#: default): the Wan DiT's 30 blocks, their self- and cross-attention q/k/v/out and both FFN linears (4.9B of its 5B
#: weights). The condition embedders, `lq_proj`, `proj_out`, the norms and the Flash decoder keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=RelayModel,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("wan",),
        include=(
            f"{_BLOCKS}.attn[12].to_[qkv].weight",
            f"{_BLOCKS}.attn[12].to_out.0.weight",
            f"{_BLOCKS}.ffn.net.0.proj.weight",
            f"{_BLOCKS}.ffn.net.2.weight",
        ),
        output_precision="preserve",
    ),
)
