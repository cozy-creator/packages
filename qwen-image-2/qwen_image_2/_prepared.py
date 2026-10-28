"""Qwen-Image-2.1's cached denoise step: its Euler tables and the layout it reads.

Step 0 runs in the transformer's ``extract`` mode, prefilling the per-layer text and
condition-image K/V cache over the full joint sequence. Steps ``1..N-1`` are the ``cached`` pass
over the target-image tokens only, identical from one step to the next but for the timestep and
the running latent, so they run as one `denoise_loop` the Runtime may capture.

Everything except the latent is a function of the request *layout* (prompt length, canvas), not
of the step: the rotary tables, the target-token modulation mask and the key-valid mask are
derived once, eagerly, by `_Layout`, so the step carries no host sync, no Python scalar read and
no data-dependent shape. The step calls the transformer's own ``img_in``/``time_text_embed``/
``modulation``/``norm_out``/``proj_out`` and the installed blocks, so the numerics are the
Diffusers/Runtime ones.
"""

from __future__ import annotations

from typing import Any

import torch


class _FlowEuler:
    """FlowMatch Euler device schedules, indexed by the global step.

    The pipeline casts the timestep to the latent dtype and only then divides by 1000, so the
    captured step must round in that order; `ts` holds that bf16 value. The non-stochastic
    update is ``sample + (sigma_next - sigma) * model_output`` upcast to f32, so `dt` is the
    per-step sigma delta in f32 — the exact `FlowMatchEulerDiscreteScheduler.step` arithmetic.
    """

    def __init__(self, scheduler: Any, device: Any) -> None:
        sigmas = scheduler.sigmas
        self.ts = (scheduler.timesteps.to(torch.bfloat16) / 1000).to(device=device)
        self.dt = (sigmas[1:] - sigmas[:-1]).to(device=device)

    def timestep(self, index: torch.Tensor) -> torch.Tensor:
        # causal_condition: the sampled row for the target tokens plus the trailing t=0 row the
        # cached text/condition tokens modulate from, exactly as the transformer forward builds it.
        t = self.ts.index_select(0, index)
        return torch.cat([t, t.new_zeros(1)], dim=0)

    def delta(self, index: torch.Tensor) -> torch.Tensor:
        return self.dt.index_select(0, index).reshape(())


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
