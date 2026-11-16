"""BiRefNet's int8 lane: which weights take the encoding is this package's recipe; the encoding and its kernel
are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import BackgroundRemovalModel

_LATE = "model.bb.layers.[23].blocks.*"

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default (Paul 11-16: every model int8 ConvRot by
#: default): the Swin-L backbone's last two stages (20 blocks' qkv, proj and MLP linears) and its three patch
#: merges. The first two stages' linears read 192 and 384 columns, off ConvRot's 256, and keep fp16 with the
#: decoder's convolutions and every other tensor.
INT8 = Recipe(
    lane="int8",
    model=BackgroundRemovalModel,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(
            f"{_LATE}.attn.qkv.weight",
            f"{_LATE}.attn.proj.weight",
            f"{_LATE}.mlp.fc[12].weight",
            "model.bb.layers.*.downsample.reduction.weight",
        ),
        output_precision="preserve",
    ),
)
