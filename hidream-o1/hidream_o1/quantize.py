"""HiDream-O1's lanes, for both checkpoints (full and Dev), and its prompt agent's (Gemma-4-31B-it): which
weights each encoding takes is this package's recipe; the encodings and their kernels are Runtime's. HiDream
ships f32 weights: every lane is written from them, and every f32 tensor a lane does not encode is rounded to
bf16. Gemma's lanes are written from its bf16."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .agent import PromptAgent
from .model import HiDreamO1

_LAYERS = "model.model.language_model.layers.*"

#: The 16-bit lane: every f32 tensor rounded to bf16, nothing encoded.
BF16 = Recipe(lane="bf16", model=HiDreamO1, plan=QuantizationPlan())

#: The int8 lane (`int8.convrot`), W8A8 on int8 tensor cores, the default lane: the q/k, gate and up
#: projections of the 36 decoder layers, 63% of the decoder's linear weights. v_proj, o_proj and
#: down_proj keep 16 bits: with every projection int8, CFG 5 multiplies W8A8's per-pass error into
#: visible grain (teacher-forced CFG x0 17-24 dB vs bf16 at t 999/702/290); keeping these three is the
#: best measured trade (23-36 dB). The patch embedder, timestep MLP, pixel head, embeddings and vision
#: tower keep 16 bits too.
INT8 = Recipe(
    lane="int8",
    model=HiDreamO1,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(
            f"{_LAYERS}.self_attn.[qk]_proj.weight",
            f"{_LAYERS}.mlp.gate_proj.weight",
            f"{_LAYERS}.mlp.up_proj.weight",
        ),
    ),
)

_AGENT = "model.model.language_model.layers.*"

#: The agent's int8 lane (`int8.convrot`): Gemma 4's 410 language projections, the bytes each decoded token reads.
#: Its embeddings (the tied head) stay bf16; the vision tower is never built.
AGENT_INT8 = Recipe(
    lane="int8",
    model=PromptAgent,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(f"{_AGENT}.self_attn.*_proj.weight", f"{_AGENT}.mlp.*_proj.weight"),
        output_precision="preserve",
    ),
)
