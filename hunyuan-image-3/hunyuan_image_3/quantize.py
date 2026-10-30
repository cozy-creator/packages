"""HunyuanImage 3.0's lanes, written by `cozy model quantize`: `bf16` from the uploaded checkpoint, the
quantized lanes from that `bf16` lane. Which weights each encoding takes is this package's recipe; the
encodings and their kernels are Runtime's."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from cozy_runtime.author import (
    Context,
    DerivationHook,
    EncodingRule,
    ModelArtifact,
    QuantizationPlan,
    Recipe,
    Telemetry,
    UnsupportedInput,
    Weights,
)
from cozy_runtime.derive.safetensors_io import from_f32, to_f32
from cozy_runtime.weights_store import Derivation

from .model import COMPONENT, HunyuanImage3

_LAYERS = "model.model.layers.*"


def _vae_operands(tensors: Any) -> list[str]:
    """The VAE's conv operands: each conv's weight (every rank >= 2 `vae.` weight) and its bias."""
    convs = [key.removesuffix(".weight") for key, info in tensors.items()
             if key.startswith("vae.") and key.endswith(".weight") and len(info.shape) >= 2]
    return [key for conv in convs for key in (f"{conv}.weight", f"{conv}.bias") if key in tensors]


def vae_operands_f16(ctx: Context, tel: Telemetry, source: Weights[Any], output: str) -> ModelArtifact:
    """The uploaded checkpoint with its VAE's conv operands in float16, everything else its own: Tencent
    computes the VAE's convs in float16 (`vae_autocast_dtype`), so the decode and encode give the same bits
    from half the bytes (5.0 GB of float32 become 2.5 GB) with no cast per call. The norms stay float32."""
    opened = ctx.tensorfs_source(source)
    tensors = opened.components[COMPONENT].tensors
    if any(info.encoding is not None for info in tensors.values()):
        raise UnsupportedInput("the bf16 lane is written from the uploaded checkpoint, and the quantized lanes "
                               "from it: this source is already quantized", code="recipe_source")
    keys = _vae_operands(tensors)
    order = [(COMPONENT, key) for key in tensors]
    cast = 0
    with ctx.output(output, Derivation(base=opened, order=order)) as out:
        for done, key in enumerate(keys, 1):
            ctx.raise_if_cancelled()
            info = tensors[key]
            if info.dtype != "F16" and not out.has(COMPONENT, key):
                shape = list(info.shape)
                raw = bytearray(math.prod(shape) * 4 if info.dtype == "F32" else math.prod(shape) * 2)
                tensors.read_into(key, raw)
                half = from_f32(to_f32(raw, info.dtype, shape), "F16")
                if not np.isfinite(np.frombuffer(half, dtype="<f2")).all():
                    raise UnsupportedInput(f"{COMPONENT}.{key} overflows float16", code="quantization_tripwire")
                out.put(COMPONENT, key, "F16", shape, half)
                cast += 1
            tel.progress(done / len(keys), stage="vae operands")
        tel.log("vae operands", operands=len(keys), cast=cast)
        return ctx.adopt_model(out)


#: The served 16-bit lane: the upload with its VAE's conv operands in float16 (`vae_operands_f16`).
BF16 = Recipe(lane="bf16", model=HunyuanImage3, plan=DerivationHook(vae_operands_f16))

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default 8-bit lane, from the `bf16` lane:
#: every linear of the 32 decoder layers (attention, the shared expert and the 64 routed experts: 99% of
#: the 80B parameters) and the LM head the token loop reads. The routers, embedding, vision tower, VAE
#: and the image patch/timestep layers keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=HunyuanImage3,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(
            f"{_LAYERS}.self_attn.qkv_proj.weight",
            f"{_LAYERS}.self_attn.o_proj.weight",
            f"{_LAYERS}.mlp.shared_mlp.*_proj.weight",
            f"{_LAYERS}.mlp.experts.*.*_proj.weight",
            "model.lm_head.weight",
        ),
        output_precision="preserve",
    ),
)

#: The int8 lane with the routed experts in NVFP4 (Blackwell's FP4 tensor cores): ~45 GB instead of ~83.
NVFP4 = Recipe(
    lane="nvfp4",
    model=HunyuanImage3,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule(f"{_LAYERS}.mlp.experts.*", "nvfp4"),),
        output_precision="preserve",
    ),
)

#: The 4-bit lane before Blackwell (xs-062): the routed experts in W4A8 (`w4a8.convrot`, 4-bit codes multiplied
#: on the int8 tensor cores, an RTX 4090's), everything else as the int8 lane has it: half the expert bytes a
#: paged step moves.
W4A8 = Recipe(
    lane="w4a8",
    model=HunyuanImage3,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule(f"{_LAYERS}.mlp.experts.*", "w4a8.convrot"),),
        output_precision="preserve",
    ),
)
