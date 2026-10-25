"""SDXL's fixed Euler interval; Runtime owns admission and the resident graph lifetime."""

from __future__ import annotations

import time
from typing import Any

import torch
from cozy_runtime.author import CapabilityError, Context, OutputError, Telemetry
from cozy_runtime.compute import PreparedRepeat
from diffusers import EulerDiscreteScheduler


def repeat_euler(
    *,
    unet: Any,
    scheduler: Any,
    latents: torch.Tensor,
    prompt: torch.Tensor,
    pooled: torch.Tensor,
    time_ids: torch.Tensor,
    guidance: float,
    ctx: Context,
    tel: Telemetry,
) -> torch.Tensor | None:
    """After eager step zero, finish a supported resident interval or leave it to the caller.

    A refusal before submission preserves the caller's latent. Once submitted, errors and
    cancellation propagate; they never rerun an interval that might have partially executed.
    HiDiffusion/DDIM and other Euler prediction rules keep their ordinary scheduler loop.
    """
    if (
        type(scheduler) is not EulerDiscreteScheduler
        or scheduler.config.prediction_type != "epsilon"
        or scheduler.step_index != 1
        or not latents.is_cuda
        or latents.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or scheduler.sigmas.device.type != "cpu"
        or scheduler.sigmas.dtype != torch.float32
        or len(scheduler.timesteps) < 2
        or len(scheduler.sigmas) != len(scheduler.timesteps) + 1
    ):
        return None

    total = len(scheduler.timesteps)
    classifier_free = guidance > 1.0
    try:
        # Match Diffusers' CPU scalar arithmetic before moving the immutable schedule.
        # Torch's CUDA true-divide with a CPU denominator computes its reciprocal in
        # double, rounds it to float, then multiplies in float opmath. A GPU scalar uses
        # another kernel and can round to half before arithmetic, despite the same shape.
        sigmas = scheduler.sigmas[1:-1]
        scales = torch.stack([(sigma**2 + 1) ** 0.5 for sigma in sigmas])
        sigma_hat = torch.stack([sigma * 1 for sigma in sigmas])
        deltas = scheduler.sigmas[2:] - sigma_hat
        inverse_scales = scales.double().reciprocal().float()
        inverse_sigmas = sigma_hat.double().reciprocal().float()
        with torch.inference_mode(False), torch.no_grad():
            state = latents.clone()
            timesteps = scheduler.timesteps[1:].to(device=state.device)
            inverse_scales = inverse_scales.to(device=state.device)
            inverse_sigmas = inverse_sigmas.to(device=state.device)
            sigma_hat = sigma_hat.to(device=state.device)
            deltas = deltas.to(device=state.device)

        def body(index: torch.Tensor) -> None:
            with torch.no_grad():
                timestep = timesteps.index_select(0, index)
                inverse_scale = inverse_scales.index_select(0, index).reshape(())
                inverse_sigma = inverse_sigmas.index_select(0, index).reshape(())
                sigma = sigma_hat.index_select(0, index).reshape(())
                delta = deltas.index_select(0, index).reshape(())
                batch = torch.cat([state] * 2) if classifier_free else state
                noise = unet(
                    (batch.float() * inverse_scale).to(batch.dtype),
                    timestep,
                    encoder_hidden_states=prompt,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                ).sample
                if classifier_free:
                    uncond, cond = noise.chunk(2)
                    noise = uncond + guidance * (cond - uncond)
                # Keep EulerDiscreteScheduler.step's rounding and operation order. In
                # particular, sample + noise * delta is not numerically equivalent.
                sample = state.to(torch.float32)
                # The CPU sigma is an opmath float, but its product is rounded back to
                # noise.dtype before subtraction from the float32 sample.
                original = sample - (sigma * noise.float()).to(noise.dtype)
                derivative = (sample - original) * inverse_sigma
                state.copy_((sample + derivative * delta).to(noise.dtype))

        plan = PreparedRepeat.capture(
            body, state=(state,), steps=total - 1, owners=(unet,), resident_only=True
        )
    except CapabilityError as error:
        if error.code != "prepared_graph_ineligible":
            raise
        tel.log("eager denoise", reason=str(error))
        return None
    except torch.OutOfMemoryError:
        # Runtime has checked capture cleanup and restored the declared state. Capacity
        # pressure during preparation can therefore keep the existing streamed path.
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
                fraction = (1 + observed) / total
                # Observation intervals are not GPU step timings. The GPU continues while
                # this watcher sleeps; it does not need a Python callback per iteration.
                tel.progress(fraction, stage="denoise", overall_fraction=0.10 + 0.80 * fraction)
            if progress.done:
                break
            time.sleep(0.05)
        (result,) = execution.wait()
        if progress.canceled or progress.completed_steps != total - 1:
            raise OutputError("prepared denoising stopped early", code="incomplete_denoise")
        tel.metric("request_path.denoise_native_steps", float(progress.completed_steps))
        return result
    finally:
        # No request-path graph retains UNet pages while the VAE starts. Close also joins
        # checked completion on cancellation or unwind before releasing any graph owner.
        plan.close()
