"""Qwen-Image's quantized lanes, for either checkpoint: which weights each encoding takes is
this package's recipe; the encodings and their kernels are Runtime's. Each is made from the bf16
lane."""

from __future__ import annotations

from cozy_runtime.author import EncodingRule, QuantizationPlan, Recipe

from .model import QwenImage

_BLOCKS = "transformer.transformer_blocks.*"
#: Every linear of the 60 DiT blocks, their two modulation heads included (6.8B of the 20B weights).
_DIT = (
    f"{_BLOCKS}.attn.to_[qkv].weight",
    f"{_BLOCKS}.attn.add_[qkv]_proj.weight",
    f"{_BLOCKS}.attn.to_out.0.weight",
    f"{_BLOCKS}.attn.to_add_out.weight",
    f"{_BLOCKS}_mlp.net.*.weight",
    f"{_BLOCKS}_mod.1.weight",
)
#: Qwen2.5-VL's 28 language layers' projections; its vision tower, embedding and head keep their bytes.
_TEXT = ("text_encoder.model.layers.*_proj.weight",)

#: The 8-bit lanes take both: `int8.convrot` (W8A8 on int8 tensor cores, the default) and `fp8.rowwise` (B300, which has
#: no int8 tensor cores). The text encoder was bf16 here (a two-reference compose lost Picture 1's identity at int8, run
#: 14702). Paul 11-19 asked for int8-convrot and fp8 at minimum. Measured on that compose (cat + astronaut, seed 14702,
#: CLIP ViT-B/16, pod-session 44653329960), int8 and fp8 hold both pictures within 0.004 of bf16. The input, timestep and
#: output projections and the VAE keep their source bytes. One recipe per lane, on the whole checkpoint: a checkpoint of
#: the encoder alone takes its lane as a cut of one (`cozy model upload <model>@<lane>:text_encoder`), since a second
#: recipe on the encoder alone would read the whole checkpoint too.
INT8 = Recipe(
    lane="int8",
    model=QwenImage,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(*_DIT, *_TEXT),
        output_precision="preserve",
    ),
)

FP8 = Recipe(
    lane="fp8",
    model=QwenImage,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=INT8.plan.components,
        include=INT8.plan.include,
        output_precision="preserve",
    ),
)

#: The int8 lane with the blocks' attention and MLP linears in NVFP4 (W4A4 on Blackwell's FP4 tensor cores); the
#: modulation heads and the text encoder stay int8.
NVFP4 = Recipe(
    lane="nvfp4",
    model=QwenImage,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(
            EncodingRule(f"{_BLOCKS}.attn.*", "nvfp4"),
            EncodingRule(f"{_BLOCKS}_mlp.*", "nvfp4"),
        ),
        output_precision="preserve",
    ),
)
