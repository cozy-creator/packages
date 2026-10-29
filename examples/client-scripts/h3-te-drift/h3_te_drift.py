"""Weights-only drift of H3 text-encoder encodings, against the served bf16 conditioning.

    cozy run minimax-h3/encode_text --input encode-request.json --on NAME --out bf16 \
      model.model=minimaxai/minimax-h3@1.0.0-rc.2 --lane fp8
    cozy run ./examples/client-scripts/h3-te-drift/weight_drift --input weight-request.json \
      --asset states=bf16/<run>-states.bin --on NAME --json \
      model.bf16=... model.fp8=...

`states` is `encode_text`'s output for the bf16 lane: its token ids and hidden states are the
reference. Each arm replays those token ids through one bf16 decoder layer at a time whose
weights are that arm's exactly decoded values, with no activation quantization: our fp8
lane, and the community FP8 schemes of this encoder (Qwen's block-128, RedHat's
per-channel with bf16 scales, Comfy's per-tensor) applied to the bf16 lane's weights. The
community files are round-to-nearest encodings of these exact weights (audited against their
bytes, 2026-10-09); package code has no network and TensorFS no ingest profile for those three
encodings, so the files themselves cannot be inputs. The bf16 arm is the harness's own floor.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

import msgspec
import numpy as np
import torch
from cozy_runtime.author import App, AssetBound, Context, FileAsset, Telemetry, Weights
from safetensors.torch import load
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRotaryEmbedding,
)

app = App()

LAYERS = 50
DEVICE = torch.device("cuda")
PREFIX = "model.language_model.layers"
EMBED = "model.language_model.embed_tokens.weight"
E4M3_MAX = 448.0
LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
NORMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)

#: Community FP8 releases of this encoder (unmodified Qwen3-VL-32B-Instruct): the scheme each
#: applies and the bytes of its published files.
COMMUNITY: dict[str, tuple[str, str, int]] = {
    "qwen-fp8-block128": ("Qwen/Qwen3-VL-32B-Instruct-FP8@4bf2c2f3", "block", 35_516_966_184),
    "redhat-fp8-dynamic": (
        "RedHatAI/Qwen3-VL-32B-Instruct-FP8-dynamic@a78b2194",
        "channel",
        35_518_522_872,
    ),
    "comfy-fp8-scaled": (
        "Rudra-ai/MiniMax-H3@b7376f3a text_encoders/qwen3vl_32b_minimax_h3_fp8_scaled",
        "tensor",
        26_532_151_816,
    ),
}

States = Annotated[
    FileAsset, AssetBound(max_bytes=1 << 30, media_types=("application/octet-stream",))
]


class WeightRequest(msgspec.Struct):
    """The bf16 lane's `encode_text` output, and the conditioner's text config (JSON)."""

    states: States
    text_config: str


class Result(msgspec.Struct):
    report: str


Loader = Callable[[int], dict[str, torch.Tensor]]


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Per-token cosine, relative L2 (Frobenius) and max abs error of one [tokens, 5120] state.
    Token 0 carries Qwen's massive activation and dominates any norm, so the relative error is
    also given without it."""
    a, b = reference.double(), candidate.double()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    diff = a - b
    return {
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "rel_l2": (diff.norm() / a.norm()).item(),
        "rel_l2_no_tok0": (diff[1:].norm() / a[1:].norm()).item(),
        "max_abs": diff.abs().max().item(),
    }


def decoded(view: Any, key: str) -> torch.Tensor:
    """One tensor's exact logical values by Runtime's reviewed reader, as float32 on the GPU."""
    tensor = view.tensor("text_encoder", key)
    count = math.prod(tensor.shape)
    out = np.empty(count, dtype=np.float32)
    chunk = 16 << 20
    for offset in range(0, count, chunk):
        view.read_values_into("text_encoder", key, offset, out[offset : offset + chunk])
    return torch.from_numpy(out).reshape(tuple(tensor.shape)).to(DEVICE)


def checkpoint_layers(view: Any, pool: ThreadPoolExecutor) -> Loader:
    keys = [f"{name}.weight" for name in LINEARS] + list(NORMS)

    def layer(index: int) -> dict[str, torch.Tensor]:
        values = pool.map(lambda suffix: decoded(view, f"{PREFIX}.{index}.{suffix}"), keys)
        return dict(zip(keys, values, strict=True))

    return layer


def community(weight: torch.Tensor, scheme: str) -> torch.Tensor:
    """`weight` through one community scheme: round-to-nearest e4m3 under its own scales."""
    value = weight.float()
    if scheme == "tensor":
        scale = (value.abs().amax() / E4M3_MAX).clamp_min(1e-12)
    elif scheme == "channel":
        scale = (value.abs().amax(dim=1, keepdim=True) / E4M3_MAX).to(torch.bfloat16).float()
    else:
        rows, columns = value.shape
        padded = torch.nn.functional.pad(value, (0, -columns % 128, 0, -rows % 128))
        blocks = padded.reshape(padded.shape[0] // 128, 128, padded.shape[1] // 128, 128)
        block = (blocks.abs().amax(dim=(1, 3)) / E4M3_MAX).clamp_min(1e-12)
        scale = block.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :columns]
    codes = (value / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return codes.float() * scale


def emulated(layer: Loader, scheme: str) -> Loader:
    def apply(index: int) -> dict[str, torch.Tensor]:
        values = layer(index)
        for name in LINEARS:
            values[f"{name}.weight"] = community(values[f"{name}.weight"], scheme)
        return values

    return apply


def layer0_inputs(
    view: Any, token_ids: dict[str, list[int]], config: Any
) -> dict[str, tuple[torch.Tensor, dict[str, Any]]]:
    """The conditioner's layer-0 call for text-only token ids: the bf16 embedding rows and the
    rotary embedding at positions 0..T-1 on all three M-RoPE axes; causal attention."""
    width = int(config.hidden_size)
    rotary = Qwen3VLTextRotaryEmbedding(config=config).to(DEVICE)
    inputs = {}
    for prompt, ids in token_ids.items():
        rows = np.empty((len(ids), width), dtype=np.float32)
        for row, token in enumerate(ids):
            view.read_values_into("text_encoder", EMBED, token * width, rows[row])
        hidden = torch.from_numpy(rows).to(DEVICE, torch.bfloat16)[None]
        positions = torch.arange(len(ids), device=DEVICE)[None, None].expand(3, 1, -1)
        inputs[prompt] = (
            hidden,
            {
                "attention_mask": None,
                "position_ids": positions[0],
                "position_embeddings": rotary(hidden, positions),
            },
        )
    return inputs


def weight_only(
    ctx: Context, layer_weights: Loader, config: Any, inputs: dict[str, Any], wanted: list[int]
) -> dict[str, dict[int, torch.Tensor]]:
    """Every prompt through the decoder one layer at a time, keeping the wanted states."""
    layer = Qwen3VLTextDecoderLayer(config, 0).to(device=DEVICE, dtype=torch.bfloat16).eval()
    hidden = {prompt: values[0] for prompt, values in inputs.items()}
    states: dict[str, dict[int, torch.Tensor]] = {prompt: {} for prompt in inputs}
    with torch.no_grad():
        for index in range(LAYERS):
            ctx.raise_if_cancelled()
            layer.load_state_dict(
                {k: v.to(torch.bfloat16) for k, v in layer_weights(index).items()}
            )
            for prompt in hidden:
                output = layer(hidden[prompt], **inputs[prompt][1])
                hidden[prompt] = output[0] if isinstance(output, tuple) else output
                if index + 1 in wanted:
                    states[prompt][index + 1] = hidden[prompt].detach()[0].float().cpu()
    return states


def drift(
    reference: dict[str, dict[int, torch.Tensor]], candidate: dict[str, dict[int, torch.Tensor]]
) -> dict[str, Any]:
    rows = {
        prompt: {k: compare(reference[prompt][k], candidate[prompt][k]) for k in reference[prompt]}
        for prompt in reference
    }
    layers = sorted(next(iter(reference.values())))
    summary = {
        str(k): {
            "cos_mean": float(np.mean([rows[p][k]["cos_mean"] for p in rows])),
            "cos_min": float(np.min([rows[p][k]["cos_min"] for p in rows])),
            "rel_l2_mean": float(np.mean([rows[p][k]["rel_l2"] for p in rows])),
            "rel_l2_max": float(np.max([rows[p][k]["rel_l2"] for p in rows])),
            "rel_l2_no_tok0_mean": float(np.mean([rows[p][k]["rel_l2_no_tok0"] for p in rows])),
            "max_abs": float(np.max([rows[p][k]["max_abs"] for p in rows])),
        }
        for k in layers
    }
    return {"summary": summary, "final_per_prompt": {p: rows[p][layers[-1]] for p in rows}}


@app.entrypoint
def weight_drift(
    ctx: Context,
    payload: WeightRequest,
    bf16: Weights[Any],
    fp8: Weights[Any],
    tel: Telemetry,
) -> Result:
    config = Qwen3VLTextConfig.from_dict(json.loads(payload.text_config))
    config._attn_implementation = "sdpa"
    served = load(payload.states.read_bytes())
    prompts = sorted({int(key.split(".")[0]) for key in served}, key=int)
    token_ids = {str(i): served[f"{i}.token_ids"].tolist() for i in prompts}
    wanted = sorted({int(key.rsplit(".", 1)[1]) for key in served if ".hidden_states." in key})
    reference = {
        str(i): {k: served[f"{i}.hidden_states.{k}"].float() for k in wanted} for i in prompts
    }
    result: dict[str, Any] = {
        "device": torch.cuda.get_device_name(),
        "community_files": {
            name: {"files": files, "bytes": size} for name, (files, _, size) in COMMUNITY.items()
        },
        "weight_only": {},
        "errors": {},
    }
    with ThreadPoolExecutor(len(LINEARS) + len(NORMS)) as pool:
        inputs = layer0_inputs(bf16, token_ids, config)
        original = checkpoint_layers(bf16, pool)
        arms: dict[str, Loader] = {
            "bf16 (harness floor)": original,
            "fp8 (ours, weights only)": checkpoint_layers(fp8, pool),
            **{name: emulated(original, scheme) for name, (_, scheme, _) in COMMUNITY.items()},
        }
        for name, layer_weights in arms.items():
            try:
                states = weight_only(ctx, layer_weights, config, inputs, wanted)
            except Exception as error:  # an arm's failure is a result
                ctx.raise_if_cancelled()
                result["errors"][name] = f"{type(error).__name__}: {error}"[:2000]
                continue
            result["weight_only"][name] = drift(reference, states)
            tel.log(f"te weight-only {name}", final=str(result["weight_only"][name]["summary"]))
    return Result(json.dumps(result))
