"""LTX-2.5's int8 lane: which weights it encodes is this package's recipe; the encoding and its kernels are
Runtime's. The lane is written from Lightricks' Diffusers bf16."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe
from cozy_runtime.vlm import VisionLM

from .model import Ltx25

_BLOCKS = "transformer.transformer_blocks.*"
_LM = "text_encoder.model.language_model.layers.*"
_CONNECTORS = "connectors.*_connector.transformer_blocks.*"

#: `int8.convrot` (W8A8 on int8 tensor cores) on the 48 blocks' six attentions (video, audio, both
#: cross-modal and both prompt attentions: q/k/v/out) and both feed-forwards, the Gemma4 encoder's
#: language linears and the text connectors' 8+8 blocks (same-seed frames match the connectors-in-bf16
#: lane, 1.9 GiB smaller). The attention gates, scale-shift tables, patch and output projections,
#: timestep and AdaLN embedders, the connectors' 188,160-wide input projections, Gemma's embeddings and
#: multimodal embedders, the VAEs, vocoder and upsampler keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=Ltx25,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder", "connectors"),
        include=(
            f"{_BLOCKS}.to_[qkv].weight",
            f"{_BLOCKS}.to_out.0.weight",
            f"{_BLOCKS}ff.net.0.proj.weight",
            f"{_BLOCKS}ff.net.2.weight",
            f"{_LM}.self_attn.[qkvo]_proj.weight",
            f"{_LM}.mlp.*_proj.weight",
            f"{_CONNECTORS}.to_[qkv].weight",
            f"{_CONNECTORS}.to_out.0.weight",
            f"{_CONNECTORS}.ff.net.*.weight",
        ),
        output_precision="preserve",
    ),
)

_ENHANCER = "model.model.language_model.layers.*"

#: The prompt enhancer's (Gemma 4 E2B) language linears in `int8.convrot`; its token and per-layer
#: embeddings, per-layer gates and projections, vision and audio towers keep their source bytes.
ENHANCER_INT8 = Recipe(
    lane="int8",
    model=VisionLM,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(f"{_ENHANCER}.self_attn.[qkvo]_proj.weight", f"{_ENHANCER}.mlp.*_proj.weight"),
        output_precision="preserve",
    ),
)


#: The fp8 lane (`fp8.rowwise`), for cards with no int8 tensor cores (B300): the int8 lane's components and weights. One
#: recipe per lane on the whole checkpoint, so a whole checkpoint's fp8 lane is fp8 throughout; a checkpoint of the encoder
#: alone takes its fp8 lane as a cut of one (`cozy model upload <model>@<fp8>:text_encoder`).
FP8 = Recipe(
    lane="fp8",
    model=Ltx25,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=INT8.plan.components,
        include=INT8.plan.include,
        output_precision="preserve",
    ),
)
