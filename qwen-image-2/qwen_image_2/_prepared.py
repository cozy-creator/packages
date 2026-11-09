"""Qwen-Image-2.1's transformer passes, their layouts derived once per request.

Step 0 runs in the transformer's ``extract`` mode, prefilling the per-layer text and
condition-image K/V cache over the full joint sequence. Steps ``1..N-1`` are the ``cached`` pass
over the target-image tokens only (`cached_pass`). A padded prompt keeps no prefix: every step is
the whole joint pass (`joint_pass`).

`QwenImage21Transformer2DModel.forward` derives its layout on the host every call (boolean
scatters, a `tolist`); `_Layout` and `_Joint` derive it once, so a step waits on nothing. The
passes call the transformer's own ``img_in``/``time_text_embed``/``modulation``/``norm_out``/
``proj_out`` and the installed blocks, so the numerics are the Diffusers/Runtime ones.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    _qwenimage21_prefix_segments,
)


def _timestep(t: torch.Tensor) -> torch.Tensor:
    """causal_condition: the sampled row for the target tokens plus the trailing t=0 row the cached
    text/condition tokens modulate from, exactly as the transformer forward builds it."""
    return torch.cat([t, t.new_zeros(1)], dim=0)


def cached_pass(
    transformer: Any, layout: _Layout, kv_cache: QwenImage21KVCache, t: torch.Tensor, latents: torch.Tensor,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    """The prediction at timestep `t` ([1], already over 1000) from the target tokens and the
    prefix's cached K/V. `projection`: a parallel decoder's output projection for this step (PDD),
    whose prediction is F32."""
    hidden = transformer.img_in(latents.to(transformer.dtype))
    temb = transformer.time_text_embed(_timestep(t), hidden)
    modulation = transformer.modulation(temb)
    for layer, block in enumerate(transformer.transformer_blocks):
        hidden = block(
            hidden_states=hidden,
            modulation=modulation,
            rotary_emb=layout.rotary_emb,
            attention_mask=None,
            target_token_mask=layout.modulation_mask,
            layer_cache=kv_cache.layer_caches[layer],
            kv_cache_mode="cached",
            cache_write_slice=None,
            segments=None,
            key_valid=None,
        )
    hidden = transformer.norm_out(hidden, temb, layout.modulation_mask)
    if projection is None:
        return transformer.proj_out(hidden)  # type: ignore[no-any-return]
    return F.linear(hidden, projection).float()


def joint_pass(
    transformer: Any, joint: _Joint, t: torch.Tensor, latents: torch.Tensor, embeds: torch.Tensor,
    references: tuple[torch.Tensor, ...], projection: torch.Tensor | None = None,
) -> torch.Tensor:
    """The prediction at `t` over the whole joint sequence: `forward` with no cache past its layout."""
    targets = latents.shape[1]
    images = transformer.img_in(torch.cat([*references, latents.to(transformer.dtype)], dim=1))
    text = transformer.txt_in(embeds)
    hidden = images.new_empty(images.shape[0], joint.length, images.shape[2])
    hidden[:, joint.text_index] = text[:, joint.text_source]
    hidden[:, joint.image_index] = images
    temb = transformer.time_text_embed(_timestep(t), images)
    modulation = transformer.modulation(temb)
    for block in transformer.transformer_blocks:
        hidden = block(
            hidden_states=hidden,
            modulation=modulation,
            rotary_emb=joint.rotary_emb,
            attention_mask=None,
            target_token_mask=joint.target_mask,
            layer_cache=None,
            kv_cache_mode=None,
            cache_write_slice=None,
            segments=joint.segments,
            key_valid=joint.key_valid,
        )
    hidden = transformer.norm_out(hidden, temb, joint.target_mask)
    if projection is None:
        return transformer.proj_out(hidden)[:, -targets:]  # type: ignore[no-any-return]
    return F.linear(hidden[:, -targets:], projection).float()


class _Layout:
    """The cached-mode layout tensors the transformer forward derives every step.

    This reproduces `QwenImage21Transformer2DModel.forward`'s preamble for ``kv_cache_mode ==
    "cached"`` using the model's own ``pos_embed`` and ``build_token_metadata``, so the values
    are bit-identical to the ones the eager forward would compute for the same layout.
    """

    #: Each vision-language image slot stands for a 2x2 group of latent tokens.
    _IMG_TOKENS_PER_SLOT = 4

    def __init__(
        self,
        transformer: Any,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        device: Any,
    ) -> None:
        repeats = torch.where(img_mask, self._IMG_TOKENS_PER_SLOT, 1)[0]
        expanded = torch.repeat_interleave(img_mask[0], repeats)
        rotary_emb = transformer.pos_embed(img_shapes[0], expanded, device=device)
        _, target_token_mask = transformer.build_token_metadata(expanded, img_shapes[0])

        prefix_len = int((~target_token_mask).sum())
        self.prefix_len = prefix_len
        # The target queries only: the cached prefix lives in the K/V cache, never recomputed.
        self.rotary_emb = rotary_emb[prefix_len:]
        self.modulation_mask = target_token_mask[prefix_len:]


class _Joint:
    """The whole joint sequence's layout, which `QwenImage21Transformer2DModel.forward` derives on
    the host every call (`repeat_interleave`, boolean scatters, a `tolist`), derived once: where
    the latent tokens and the text tokens land, the rotary table, the target mask and the
    prefix's block-causal segments (the default processors' form)."""

    def __init__(
        self,
        transformer: Any,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        text_mask: torch.Tensor | None,
        device: Any,
    ) -> None:
        repeats = torch.where(img_mask, _Layout._IMG_TOKENS_PER_SLOT, 1)[0]
        expanded = torch.repeat_interleave(img_mask[0], repeats)
        self.length = int(expanded.shape[0])
        # A padded prompt's keys: its text positions, in order, are the text encoder's non-image ones.
        self.key_valid = None
        if text_mask is not None:
            self.key_valid = torch.ones(1, self.length, dtype=torch.bool, device=device)
            self.key_valid[:, ~expanded] = text_mask.bool()[:, ~img_mask[0][: text_mask.shape[1]]]
        self.image_index = expanded.nonzero(as_tuple=True)[0]
        self.text_index = (~expanded).nonzero(as_tuple=True)[0]
        self.text_source = (~img_mask[0]).nonzero(as_tuple=True)[0]
        self.rotary_emb = transformer.pos_embed(img_shapes[0], expanded, device=device)
        image_ids, self.target_mask = transformer.build_token_metadata(expanded, img_shapes[0])
        self.segments = _qwenimage21_prefix_segments(image_ids, int((~self.target_mask).sum()))
