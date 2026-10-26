"""Qwen-Image-2.1's native cached-denoise interval; Runtime owns the transformer weights.

Step 0 runs eagerly in the transformer's ``extract`` mode, prefilling the per-layer text and
condition-image K/V cache over the full joint sequence. Steps ``1..N-1`` are the ``cached``
pass over the target-image tokens only, and they are identical from one step to the next but
for the timestep and the running latent. That repeated interval is captured once and replayed
as a single native program, mirroring SDXL's ``repeat_euler``.

What makes the cached step capturable is that everything except the latent is a function of the
request *layout* (prompt and reference length, canvas), not of the step: the rotary tables, the
target-token modulation mask and the key-valid mask are all derived once, eagerly, by `_Layout`
out of the body, so the captured body carries no host sync, no Python scalar read and no
data-dependent shape. The body calls the transformer's own ``img_in``/``time_text_embed``/
``modulation``/``norm_out``/``proj_out`` and the installed blocks, so the numerics are the
Diffusers/Runtime ones; the parity gate is this path against the eager Diffusers loop.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from cozy_runtime.author import CapabilityError, Context, OutputError, Telemetry
from cozy_runtime.compute import PreparedRepeat


class _FlowEuler:
    """FlowMatch Euler device schedules for the cached interval (steps ``start..N-1``).

    The pipeline casts the timestep to the latent dtype and only then divides by 1000, so the
    captured step must round in that order; `ts` holds that bf16 value. The non-stochastic
    update is ``sample + (sigma_next - sigma) * model_output`` upcast to f32, so `dt` is the
    per-step sigma delta in f32 — the exact `FlowMatchEulerDiscreteScheduler.step` arithmetic.
    """

    def __init__(self, scheduler: Any, device: Any, start: int) -> None:
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas
        self.ts = (timesteps[start:].to(torch.bfloat16) / 1000).to(device=device)
        self.dt = (sigmas[start + 1 :] - sigmas[start:-1]).to(device=device)

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
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

        prefix_len = int((~target_token_mask).sum())
        self.prefix_len = prefix_len
        # The target queries only: the cached prefix lives in the K/V cache, never recomputed.
        self.rotary_emb = rotary_emb[prefix_len:]
        self.modulation_mask = target_token_mask[prefix_len:]
        self.attention_mask = (
            None if joint_key_valid is None else joint_key_valid[:, None, None, :]
        )


def repeat_cached(
    *,
    transformer: Any,
    scheduler: Any,
    latents: torch.Tensor,
    kv_cache: Any,
    layout: _Layout,
    ctx: Context,
    tel: Telemetry,
) -> torch.Tensor | None:
    """After the eager extract step 0, run steps ``1..N-1`` as one captured native program.

    `latents` is the target latent after step 0 and `kv_cache` is that step's prefix K/V. A
    refusal before submission returns ``None`` so the caller finishes through the eager
    Diffusers loop; once submitted, errors and cancellation propagate and never rerun an
    interval that might have partially executed.
    """
    total = len(scheduler.timesteps)
    if (
        not latents.is_cuda
        or latents.dtype != torch.bfloat16
        or total < 2
        or scheduler.sigmas.shape[0] != total + 1
    ):
        return None
    start = 1

    blocks = transformer.transformer_blocks
    try:
        with torch.inference_mode(False), torch.no_grad():
            state = latents.clone()
            euler = _FlowEuler(scheduler, state.device, start)
        rot = layout.rotary_emb
        mask_t = layout.modulation_mask
        attention_mask = layout.attention_mask
        caches = [kv_cache.get_layer(index) for index in range(len(blocks))]

        def body(index: torch.Tensor) -> None:
            with torch.no_grad():
                hidden = transformer.img_in(state)
                temb = transformer.time_text_embed(euler.timestep(index), hidden)
                modulation = transformer.modulation(temb)
                for block, cache in zip(blocks, caches, strict=True):
                    hidden = block(
                        hidden_states=hidden,
                        modulation=modulation,
                        rotary_emb=rot,
                        attention_mask=attention_mask,
                        target_token_mask=mask_t,
                        layer_cache=cache,
                        kv_cache_mode="cached",
                        cache_write_slice=None,
                        segments=None,
                        key_valid=None,
                    )
                hidden = transformer.norm_out(hidden, temb, mask_t)
                noise = transformer.proj_out(hidden)
                state.copy_((state.float() + euler.delta(index) * noise).to(state.dtype))

        plan = PreparedRepeat.capture(
            body, state=(state,), steps=total - start, owners=(transformer,), resident_only=True
        )
    except CapabilityError as error:
        if error.code != "prepared_graph_ineligible":
            raise
        tel.log("eager denoise", reason=str(error))
        return None
    except torch.OutOfMemoryError:
        # Runtime has checked capture cleanup and restored the declared state, so capacity
        # pressure during preparation can keep the ordinary eager path.
        tel.log("eager denoise", reason="resident graph preparation exceeded capacity")
        return None

    try:
        execution = ctx.plan(plan)
        observed = -1
        while True:
            ctx.raise_if_cancelled()
            progress = execution.poll()
            if progress.completed_steps != observed:
                observed = progress.completed_steps
                fraction = (start + observed) / total
                tel.progress(
                    fraction, stage="generating image", overall_fraction=0.1 + 0.8 * fraction
                )
            if progress.done:
                break
            time.sleep(0.05)
        (result,) = execution.wait()
        if progress.canceled or progress.completed_steps != total - start:
            raise OutputError("prepared denoising stopped early", code="incomplete_denoise")
        tel.metric("request_path.denoise_native_steps", float(progress.completed_steps))
        return result
    finally:
        # Close joins checked completion on cancellation or unwind before releasing the owner.
        plan.close()
