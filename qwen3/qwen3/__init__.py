"""Qwen3: single-turn questions about text, images and videos, on any Qwen3.5-family checkpoint (Qwen3.5, Qwen3.6,
Qwen3.8; dense or MoE). The default is the newest, abliterated; a request picks another by its `model`."""

from __future__ import annotations

import msgspec
from cozy_runtime import vlm
from cozy_runtime.author import App, Context, EncodingRule, InvalidRequest, QuantizationPlan, Recipe, Telemetry
from cozy_runtime.fields import Prompt
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


@app.entrypoint(defaults={"model": MODEL}, memoize=True)
def ask(ctx: Context, payload: AskRequest, model: VisionLM, tel: Telemetry) -> AskResult:
    """Answer each prompt about its text, images and videos, all prompts in one pass."""
    return vlm.ask(ctx, payload, model, tel)


#: Caption's question, before the box's own words: a detailed, literal description of what is visible, for any use. This
#: wording is unmeasured as an upscale's prompt. What se-151 measured was upscale-specific ("Describe this picture in detail
#: so a painter could recreate it at high resolution: …", through `ask` with 400 tokens), and a two-sentence caption gained
#: nothing there.
_CAPTION = (
    "Describe what this shows in detail, in one paragraph of at most 120 words: the subjects and every visible element, "
    "any visible text exactly as written, the materials, colours and lighting, the medium and style (for example photo, "
    "3D render, flat cel-shaded art, watercolour, woodblock print), the layout and the background; for a video, also "
    "what happens and how the camera moves. Describe only what you can see and invent nothing. Do not mention blur, "
    "noise, compression, resolution or image quality."
)


@app.tool(label="Caption", named="caption", defaults={"model": MODEL}, memoize=True)
def caption(ctx: Context, payload: TextBox, model: VisionLM, tel: Telemetry) -> TextResult:
    """A detailed, literal description of the pictures and videos the box mentions: this package's question, then the
    box's own words. Its box and sampling are Write's."""
    said = payload.instructions
    if not said.references:
        raise InvalidRequest("Caption has nothing to describe: its words mention no picture or video",
                             code="caption_nothing")  # fmt: skip
    asked = Prompt(text=f"{_CAPTION}\n\n{said.text}", references=said.references)
    return vlm.write(ctx, msgspec.structs.replace(payload, instructions=asked), model, tel)


@app.tool(label="Write", named="text", defaults={"model": MODEL}, memoize=True)
def write(ctx: Context, payload: TextBox, model: VisionLM, tel: Telemetry) -> TextResult:
    """Words on a canvas: what the instructions ask, about the pictures and videos they mention."""
    return vlm.write(ctx, payload, model, tel)
