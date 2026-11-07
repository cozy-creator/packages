"""Qwen-Image-2.1's denoise steps: its Euler tables and the layouts they read.

Step 0 runs in the transformer's ``extract`` mode, prefilling the per-layer text and
condition-image K/V cache over the full joint sequence. Steps ``1..N-1`` are the ``cached`` pass
over the target-image tokens only, identical from one step to the next but for the timestep and
the running latent, so they run as one `denoise_loop` the Runtime may capture. With no room to
keep that prefix (`retained` None), every step is the whole joint pass (`_full_step`).

The rotary tables and target-token modulation mask are derived once by `_Layout`, so the step
carries no host sync, no Python scalar read and no data-dependent shape. The prefix K/V is
prompt-specific. It joins the layout and schedules as explicitly declared per-call inputs;
no retained step closure owns a previous prompt's tensors. The step calls the transformer's
own ``img_in``/``time_text_embed``/
``modulation``/``norm_out``/``proj_out`` and the installed blocks, so the numerics are the
Diffusers/Runtime ones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVLayerCache,
    _qwenimage21_prefix_segments,
)


@dataclass(frozen=True)
class _FlowEuler:
    """FlowMatch Euler device schedules, indexed by the global step.

    The pipeline casts the timestep to the latent dtype and only then divides by 1000, so the
    captured step must round in that order; `ts` holds that bf16 value. The non-stochastic
    update is ``sample + (sigma_next - sigma) * model_output`` upcast to f32, so `dt` is the
    per-step sigma delta in f32 — the exact `FlowMatchEulerDiscreteScheduler.step` arithmetic.
    """

    ts: torch.Tensor
    dt: torch.Tensor

    @classmethod
    def from_scheduler(cls, scheduler: Any, device: Any) -> _FlowEuler:
        sigmas = scheduler.sigmas
        return cls(
            (scheduler.timesteps.to(torch.bfloat16) / 1000).to(device=device),
            (sigmas[1:] - sigmas[:-1]).to(device=device),
        )

    def timestep(self, index: torch.Tensor) -> torch.Tensor:
        # causal_condition: the sampled row for the target tokens plus the trailing t=0 row the
        # cached text/condition tokens modulate from, exactly as the transformer forward builds it.
        t = self.ts.index_select(0, index)
        return torch.cat([t, t.new_zeros(1)], dim=0)

    def delta(self, index: torch.Tensor) -> torch.Tensor:
        return self.dt.index_select(0, index).reshape(())


def _cached_step(transformer: Any) -> Callable[..., None]:
    """A step whose only closed-over owner is the placed transformer.

    Schedules, target layout and every prefix K/V arrive as declared tensors.
    Rebuilding the small cache wrappers from those arguments also works when
    Runtime records on its own copies and later copies a new prompt into them.
    """
    blocks = transformer.transformer_blocks

    def step(
        index: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        rotary: torch.Tensor,
        target: torch.Tensor,
        *prefix: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            euler = _FlowEuler(timesteps, deltas)
            hidden = transformer.img_in(latents)
            temb = transformer.time_text_embed(euler.timestep(index), hidden)
            modulation = transformer.modulation(temb)
            for layer, block in enumerate(blocks):
                cache = QwenImage21KVLayerCache()
                cache.store(prefix[2 * layer], prefix[2 * layer + 1])
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
            hidden = transformer.norm_out(hidden, temb, target)
            noise = transformer.proj_out(hidden)
            latents.copy_((latents.float() + euler.delta(index) * noise).to(latents.dtype))

    return step


class _Layout:
    """The cached-mode layout tensors the transformer forward derives every step, hoisted out
    of the captured body so the body is sync-free and shape-static.

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

    def inputs(self) -> tuple[torch.Tensor, ...]:
        masks = () if self.key_valid is None else (self.key_valid,)
        return self.image_index, self.text_index, self.text_source, self.rotary_emb, self.target_mask, *masks


def _full_step(
    transformer: Any, segments: list[tuple[int, int, bool]], length: int, targets: int, masked: bool,
) -> Callable[..., None]:
    """A step over the whole joint sequence, `forward` with no cache past its layout (`_Joint`):
    `masked`, its key mask follows the layout's inputs. `segments` and `length` are host values of
    one layout, so its loop takes no key."""
    blocks = transformer.transformer_blocks

    def step(
        index: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        image_index: torch.Tensor,
        text_index: torch.Tensor,
        text_source: torch.Tensor,
        rotary: torch.Tensor,
        target: torch.Tensor,
        *rest: torch.Tensor,
    ) -> None:
        key_valid, (embeds, *references) = (rest[0], rest[1:]) if masked else (None, rest)
        with torch.no_grad():
            euler = _FlowEuler(timesteps, deltas)
            images = transformer.img_in(torch.cat([*references, latents], dim=1))
            text = transformer.txt_in(embeds)
            hidden = images.new_empty(images.shape[0], length, images.shape[2])
            hidden[:, text_index] = text[:, text_source]
            hidden[:, image_index] = images
            temb = transformer.time_text_embed(euler.timestep(index), images)
            modulation = transformer.modulation(temb)
            for block in blocks:
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
            noise = transformer.proj_out(transformer.norm_out(hidden, temb, target))[:, -targets:]
            latents.copy_((latents.float() + euler.delta(index) * noise).to(latents.dtype))

    return step
