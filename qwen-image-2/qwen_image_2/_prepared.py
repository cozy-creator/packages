"""Qwen-Image-2.1's transformer with a sampling step's two passes, their layouts derived once per request.

Step 0 runs in the transformer's ``extract`` mode, prefilling the per-layer text and
condition-image K/V cache over the full joint sequence. Steps ``1..N-1`` are the ``cached`` pass
over the target-image tokens only (`Transformer.forward(cached=...)`). A padded prompt keeps no
prefix: every step is the whole joint pass (`forward(joint=...)`).

`QwenImage21Transformer2DModel.forward` derives its layout on the host every call (boolean
scatters, a `tolist`); `_Layout` and `_Joint` derive it once, so a step waits on nothing and is one
call of the transformer over tensors alone. The passes call the transformer's own ``img_in``/
``time_text_embed``/``modulation``/``norm_out``/``proj_out`` and the installed blocks, so the
numerics are the Diffusers/Runtime ones.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVLayerCache,
    QwenImage21Transformer2DModel,
    _qwenimage21_prefix_segments,
)

if TYPE_CHECKING:
    from cozy_runtime.accel.lora import Factors

#: A layer's prefix keys and values.
KV = tuple[torch.Tensor, torch.Tensor]
#: A layer's prefix K/V, read as that layer runs.
Prefix = Callable[[int], KV]
#: A cached step's inputs after the latents: t, the target rotary table, the target mask, each layer's prefix K/V.
Cached = tuple[torch.Tensor, torch.Tensor, torch.Tensor, Prefix]
#: `_Joint.inputs()`: image and text indices, text source, rotary table, target mask, key validity, segments, length.
JointLayout = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None,
    list[tuple[int, int, bool]], int,
]
#: A joint step's inputs after the latents: t, text embeds, reference latents, then `JointLayout`.
Joint = tuple[
    torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor | None, list[tuple[int, int, bool]], int,
]


def _timestep(t: torch.Tensor) -> torch.Tensor:
    """causal_condition: the sampled row for the target tokens plus the trailing t=0 row the cached
    text/condition tokens modulate from, exactly as the transformer forward builds it."""
    return torch.cat([t, t.new_zeros(1)], dim=0)


class Transformer(QwenImage21Transformer2DModel):
    """`QwenImage21Transformer2DModel` whose forward is also a step past its layout: `cached` =
    (t, rotary table, target mask, each layer's prefix K/V), or `joint` = (t, text embeds, reference
    latents, *`_Joint.inputs()`). `t` is [1], already over 1000. `projection`: a parallel decoder's
    output projection for this step (PDD), whose prediction is F32."""

    #: The armed overlay's factors by site (`turbo.FunAccLoRA`), else None.
    _cozy_overlay: Mapping[str, Factors] | None

    def forward(
        self, hidden_states: torch.Tensor, *args: object, cached: Cached | None = None,
        joint: Joint | None = None, projection: torch.Tensor | None = None, **kwargs: object,
    ) -> torch.Tensor | Transformer2DModelOutput:
        if cached is not None:
            return self._cached(hidden_states, *cached, projection=projection)
        if joint is not None:
            return self._joint(hidden_states, *joint, projection=projection)
        out: torch.Tensor | Transformer2DModelOutput = super().forward(hidden_states, *args, **kwargs)
        return out

    def _cached(self, latents: torch.Tensor, t: torch.Tensor, rotary: torch.Tensor, target: torch.Tensor,
                prefix: Prefix, projection: torch.Tensor | None) -> torch.Tensor:
        """The prediction from the target tokens and the prefix's cached K/V, each layer's read as it runs: a retained
        state is held on the card while its tensors live, so a step holds one layer's, not the stack's (ten 4 MP
        references keep 758 MiB a layer, 24 GB over 32: run 47471 at 8 GiB)."""
        hidden = self.img_in(latents.to(self.dtype))
        temb = self.time_text_embed(_timestep(t), hidden)
        modulation = self.modulation(temb)
        for layer, block in enumerate(self.transformer_blocks):
            cache = QwenImage21KVLayerCache()  # type: ignore[no-untyped-call]  # Diffusers' cache __init__ is unannotated
            cache.store(*prefix(layer))
            hidden = block(
                hidden_states=hidden,
                modulation=modulation,
                rotary_emb=rotary,
                attention_mask=None,
                target_token_mask=target,
                layer_cache=cache,
                kv_cache_mode="cached",
                cache_write_slice=None,
                segments=None,
                key_valid=None,
            )
        hidden = self.norm_out(hidden, temb, target)
        if projection is None:
            return self.proj_out(hidden)  # type: ignore[no-any-return]
        return F.linear(hidden, projection).float()

    def _joint(
        self, latents: torch.Tensor, t: torch.Tensor, embeds: torch.Tensor, references: tuple[torch.Tensor, ...],
        image_index: torch.Tensor, text_index: torch.Tensor, text_source: torch.Tensor, rotary: torch.Tensor,
        target: torch.Tensor, key_valid: torch.Tensor | None, segments: list[tuple[int, int, bool]], length: int,
        projection: torch.Tensor | None,
    ) -> torch.Tensor:
        """The prediction over the whole joint sequence: `forward` with no cache past its layout."""
        targets = latents.shape[1]
        images = self.img_in(torch.cat([*references, latents.to(self.dtype)], dim=1))
        text = self.txt_in(embeds)
        hidden = images.new_empty(images.shape[0], length, images.shape[2])
        hidden[:, text_index] = text[:, text_source]
        hidden[:, image_index] = images
        temb = self.time_text_embed(_timestep(t), images)
        modulation = self.modulation(temb)
        for block in self.transformer_blocks:
            hidden = block(
                hidden_states=hidden,
                modulation=modulation,
                rotary_emb=rotary,
                attention_mask=None,
                target_token_mask=target,
                layer_cache=None,
                kv_cache_mode=None,
                cache_write_slice=None,
                segments=segments,
                key_valid=key_valid,
            )
        hidden = self.norm_out(hidden, temb, target)
        if projection is None:
            return self.proj_out(hidden)[:, -targets:]  # type: ignore[no-any-return]
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
        transformer: Transformer,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        device: torch.device,
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
        transformer: Transformer,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        text_mask: torch.Tensor | None,
        device: torch.device,
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

    def inputs(self) -> JointLayout:
        """What `Transformer.forward(joint=...)` reads after the timestep, embeds and references."""
        return (self.image_index, self.text_index, self.text_source, self.rotary_emb, self.target_mask,
                self.key_valid, self.segments, self.length)
