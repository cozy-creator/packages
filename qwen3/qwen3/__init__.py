"""Qwen3: single-turn questions about text, images and videos, on any Qwen3.5-family checkpoint (Qwen3.5, Qwen3.6,
Qwen3.8; dense or MoE). The default is the newest, abliterated; a request picks another by its `model`."""

from __future__ import annotations

import msgspec
from cozy_runtime import vlm
from cozy_runtime.author import App, Context, EncodingRule, QuantizationPlan, Recipe, Telemetry
from cozy_runtime.vlm import AskRequest, AskResult, TextBox, TextResult, VisionLM

app = App()

#: The default checkpoint: the newest Qwen3.x, abliterated (llmfan46's Heretic of Qwen3.8-27B, its MTP head kept).
MODEL = "llmfan46/qwen3.8-27b-uncensored-heretic-native-mtp-preserved@0.1.0"

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores): every attention, gated-delta-net and MLP projection of the
#: language layers and of the multi-token-prediction layer where the checkpoint has one; in a MoE every routed and shared
#: expert. The vision tower, embeddings, routers, the decay/gate projections, norms and the LM head keep their bytes.
INT8 = Recipe(
    lane="int8",
    model=VisionLM,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(
            "model.model.language_model.layers.*.linear_attn.in_proj_[qz]*.weight",
            "model.model.language_model.layers.*.linear_attn.out_proj.weight",
            "model.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
            "model.model.language_model.layers.*.mlp.*_proj.weight",
            "model.mtp.layers.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)

#: The MLPs (dense, routed and shared experts) of the language and multi-token-prediction layers.
_MLPS = ("model.model.language_model.layers.*.mlp.*", "model.mtp.layers.*.mlp.*")

#: The int8 lane with every MLP in NVFP4 (Blackwell's FP4 tensor cores): what a decoded token reads is mostly MLP
#: weights (all of a MoE's experts), so this nearly halves its bytes.
NVFP4 = Recipe(
    lane="nvfp4",
    model=VisionLM,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=tuple(EncodingRule(mlp, "nvfp4") for mlp in _MLPS),
        output_precision="preserve",
    ),
)

#: The 4-bit lane before Blackwell (xs-062): every MLP in W4A8 (`w4a8.convrot`, 4-bit codes multiplied on the int8
#: tensor cores), the rest as the int8 lane has it.
W4A8 = Recipe(
    lane="w4a8",
    model=VisionLM,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=tuple(EncodingRule(mlp, "w4a8.convrot") for mlp in _MLPS),
        output_precision="preserve",
    ),
)

#: llama.cpp's lanes (xs-213): every weight llama-quantize quantizes, in ggml's block types and byte for byte what it
#: writes (no importance matrix). The vision tower is llama.cpp's mmproj and keeps its bytes.
_GGML = (
    "model.model.language_model.embed_tokens.weight",
    "model.lm_head.weight",
    "model.model.language_model.layers.*.linear_attn.in_proj_*.weight",
    "model.model.language_model.layers.*.linear_attn.out_proj.weight",
    "model.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
    "model.model.language_model.layers.*.mlp.*_proj.weight",
    "model.mtp.fc.weight",
    "model.mtp.layers.*_proj.weight",
)

#: llama.cpp's Q8_0: every one of those weights in `ggml.q8_0`.
Q8_0 = Recipe(
    lane="q8_0",
    model=VisionLM,
    plan=QuantizationPlan(
        encoding="ggml.q8_0", components=("model",), include=_GGML, output_precision="preserve"
    ),
)

#: llama.cpp's Q4_K_M: Q4_K, with Q6_K where llama-quantize's per-tensor rule puts it (the LM head, and the value and
#: gated-delta QKV projections and MLP down projections of the first and last eighth of the layers and every third
#: between).
Q4_K_M = Recipe(
    lane="q4_k_m",
    model=VisionLM,
    plan=QuantizationPlan(
        encoding="ggml.q4_k_m", components=("model",), include=_GGML, output_precision="preserve"
    ),
)


#: Test only, on a proof branch, never merged (xs-213): llama.cpp's Q8_0 with the LM head kept bf16, timed beside `q8_0` to
#: tell the head's bytes from the kernels' speed.
Q8_0_HEAD16 = Recipe(
    lane="q8_0_head16",
    model=VisionLM,
    plan=QuantizationPlan(
        encoding="ggml.q8_0",
        components=("model",),
        include=(
            "model.model.language_model.embed_tokens.weight",
            "model.model.language_model.layers.*.linear_attn.in_proj_*.weight",
            "model.model.language_model.layers.*.linear_attn.out_proj.weight",
            "model.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
            "model.model.language_model.layers.*.mlp.*_proj.weight",
            "model.mtp.fc.weight",
            "model.mtp.layers.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)

@app.entrypoint(defaults={"model": MODEL}, memoize=True)
def ask(ctx: Context, payload: AskRequest, model: VisionLM, tel: Telemetry) -> AskResult:
    """Answer each prompt about its text, images and videos, all prompts in one pass."""
    return vlm.ask(ctx, payload, model, tel)


#: What a description asks: what it is, for whatever mentions it to read, and nothing of how it looks as a picture.
_CAPTION = "Say what this shows, in one or two plain sentences, for someone who can't see it: who or what it is and how it looks. Say only what you can see."


@app.entrypoint(defaults={"model": MODEL}, memoize=True)
def caption(ctx: Context, payload: AskRequest, model: VisionLM, tel: Telemetry) -> AskResult:
    """What a picture or a video shows, in this package's own question: `images` or `videos`, and nothing to ask."""
    return vlm.ask(ctx, msgspec.structs.replace(payload, prompt=_CAPTION), model, tel)


@app.tool(label="Write", named="text", defaults={"model": MODEL}, memoize=True)
def write(ctx: Context, payload: TextBox, model: VisionLM, tel: Telemetry) -> TextResult:
    """Words on a canvas: what the instructions ask, about the pictures and videos they mention."""
    return vlm.write(ctx, payload, model, tel)
