"""Sequence ownership for LTX's video, audio, prompt and rotary streams.

Projection, normalization, rotary and gating arithmetic follow Diffusers' Apache-2.0
LTX2AudioVideoAttnProcessor (Copyright 2025 Lightricks and HuggingFace).
"""

from typing import cast

import torch
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelModelPlan, ContextParallelOutput, ParallelConfig
from diffusers.models.attention_dispatch import (
    AttentionBackendName, all_to_all_single_any_o_async, all_to_all_single_any_qkv_async, dispatch_attention_fn,
)
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2Attention, LTX2VideoTransformer3DModel, LTX2VideoTransformerBlock, apply_interleaved_rotary_emb, apply_split_rotary_emb,
)


def plan(rope_type: str) -> ContextParallelModelPlan:
    # Split RoPE is [batch, heads, tokens, dim]; interleaved RoPE is [batch, tokens, dim].
    rotary_dim, rotary_ndim = (2, 4) if rope_type == "split" else (1, 3)
    return {
        "": {
            **{name: ContextParallelInput(split_dim=1, expected_dims=3, split_output=False)
               for name in ("hidden_states", "audio_hidden_states")},
            # A canvas's per-token timesteps and the first frame's keyframe marker split with their tokens; the
            # pipelines' [batch] audio timestep has other dims and stays whole.
            **{name: ContextParallelInput(split_dim=1, expected_dims=2, split_output=False)
               for name in ("timestep", "audio_timestep")},
            "video_keyframes_mask": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        **{name: {index: ContextParallelInput(split_dim=rotary_dim, expected_dims=rotary_ndim, split_output=True)
                  for index in (0, 1)}
           for name in ("rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope")},
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "audio_proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }


class CrossAttention:
    """Local queries against replicated text, or Ulysses over unequal video/audio streams."""

    _attention_backend: AttentionBackendName | None = None
    _parallel_config: ParallelConfig | None = None

    def __init__(self, *, prompt: bool) -> None:
        self.prompt = prompt

    def __call__(self, attn: LTX2Attention, hidden_states: torch.Tensor,
                 encoder_hidden_states: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                 query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
                 key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, length, _ = context.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, length, batch)
            attention_mask = attention_mask.view(batch, attn.heads, -1, attention_mask.shape[-1])
        gate = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
        query, key, value = attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)
        query, key = attn.norm_q(query), attn.norm_k(key)
        if query_rotary_emb is not None:
            rotary = apply_split_rotary_emb if attn.rope_type == "split" else apply_interleaved_rotary_emb
            query = rotary(query, query_rotary_emb)
            key = rotary(key, query_rotary_emb if key_rotary_emb is None else key_rotary_emb)
        query, key, value = (x.unflatten(2, (attn.heads, -1)) for x in (query, key, value))
        group = None
        heads, rows = query.shape[2], query.shape[1]
        if not self.prompt and self._parallel_config is not None:
            cp = self._parallel_config.context_parallel_config
            assert cp is not None and cp._ulysses_mesh is not None and cp.ring_degree == 1
            group = cp._ulysses_mesh.get_group()
            q_wait = all_to_all_single_any_qkv_async(query, group)
            k_wait = all_to_all_single_any_qkv_async(key, group)
            v_wait = all_to_all_single_any_qkv_async(value, group)
            query, key, value = q_wait(), k_wait(), v_wait()
        result = dispatch_attention_fn(query, key, value, attn_mask=attention_mask,
                                       dropout_p=0.0, is_causal=False, backend=self._attention_backend)
        if group is not None:
            result = all_to_all_single_any_o_async(result, group, NUM_QO_HEAD=heads, Q_S_LOCAL=rows)()
        result = result.flatten(2, 3).to(query.dtype)
        if gate is not None:
            result = (result.unflatten(2, (attn.heads, -1)) * (2.0 * torch.sigmoid(gate)).unsqueeze(-1)).flatten(2, 3)
        return cast(torch.Tensor, attn.to_out[1](attn.to_out[0](result)))


def install(model: LTX2VideoTransformer3DModel) -> None:
    model._cp_plan = {name: value for name, value in plan(str(model.config["rope_type"])).items()}
    for module in model.transformer_blocks:
        block = cast(LTX2VideoTransformerBlock, module)
        # Prompt KV/masks stay whole. Cross-modality calls exchange unequal Q and KV lengths.
        for attention, prompt in ((block.attn2, True), (block.audio_attn2, True),
                                  (block.audio_to_video_attn, False), (block.video_to_audio_attn, False)):
            attention.set_processor(CrossAttention(prompt=prompt))  # type: ignore[arg-type]  # Diffusers' closed processor union excludes user processors.
