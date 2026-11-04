"""Qwen-Image-2.1's cached denoise step: its Euler tables and the layout it reads.

Step 0 runs in the transformer's ``extract`` mode, prefilling the per-layer text and
condition-image K/V cache over the full joint sequence. Steps ``1..N-1`` are the ``cached`` pass
over the target-image tokens only, identical from one step to the next but for the timestep and
the running latent, so they run as one `denoise_loop` the Runtime may capture.

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
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVLayerCache


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
        encoder_hidden_states_mask: torch.Tensor | None,
        device: Any,
    ) -> None:
        repeats = torch.where(img_mask, self._IMG_TOKENS_PER_SLOT, 1)[0]
        expanded = torch.repeat_interleave(img_mask[0], repeats)
        rotary_emb = transformer.pos_embed(img_shapes[0], expanded, device=device)
        _, target_token_mask = transformer.build_token_metadata(expanded, img_shapes[0])

        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(1, expanded.shape[0], dtype=torch.bool, device=device)
            text_positions = (~expanded).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[
                :, vlm_text_positions
            ]

        prefix_len = int((~target_token_mask).sum())
        self.prefix_len = prefix_len
        # The target queries only: the cached prefix lives in the K/V cache, never recomputed.
        self.rotary_emb = rotary_emb[prefix_len:]
        self.modulation_mask = target_token_mask[prefix_len:]
        self.attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
