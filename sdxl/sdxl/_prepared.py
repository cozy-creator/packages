"""SDXL's reusable resident pipeline and Euler interval; Runtime owns their memory."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from cozy_runtime import rng
from cozy_runtime.author import CapabilityError, Context, OutputError, Telemetry
from cozy_runtime.compute import PreparedRepeat
from diffusers import EulerDiscreteScheduler
from transformers.masking_utils import create_causal_mask


def _clip_text(model: Any, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CLIP's tensor orchestration, with an explicit penultimate result.

    Transformers 5.16 installs Python output-collection hooks when hidden states are
    requested. A replay needs the tensors, not those hooks. Keep its causal-mask, layer,
    normalization and EOS-pooling operations, calling the original numerical modules.
    The real parity gate compares this path with that pinned library forward.
    """
    ids = ids.view(-1, ids.shape[-1])
    hidden = model.embeddings(input_ids=ids)
    # This helper always encodes a dense sequence, without a padding mask or KV cache.
    # Keep SDPA's implicit causal path during capture too: Transformers materializes an
    # equivalent mask while tracing, which selects different attention math/rounding.
    mask = (
        None
        if model.config._attn_implementation == "sdpa"
        else create_causal_mask(
            config=model.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
        )
    )
    if not model.encoder.layers:
        raise ValueError("SDXL conditioning requires a nonempty CLIP encoder")
    penultimate = hidden
    for layer in model.encoder.layers:
        penultimate = hidden
        hidden = layer(hidden, mask, is_causal=True)
    hidden = model.final_layer_norm(hidden)
    positions = ids.to(dtype=torch.int, device=hidden.device)
    if model.eos_token_id != 2:
        positions = (positions == model.eos_token_id).int()
    pooled = hidden[
        torch.arange(hidden.shape[0], device=hidden.device),
        positions.argmax(dim=-1),
    ]
    return penultimate, pooled


def encode_tokens(
    first_encoder: Any,
    second_encoder: Any,
    ids: torch.Tensor,
    ids_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two-tower tensor computation, after inputs and weights have been placed."""
    first, _ = _clip_text(first_encoder, ids)
    second, pooled = _clip_text(second_encoder.text_model, ids_2)
    prompt = torch.cat([first, second], dim=-1)
    return prompt, second_encoder.text_projection(pooled)


def decode_latents(vae: Any, latents: torch.Tensor, scale: float) -> torch.Tensor:
    """Decode with the VAE's already selected representation; never change its weights."""
    return vae.decode(latents / scale).sample


class _Euler:
    """The existing Diffusers Euler operation order, with immutable device schedules."""

    def __init__(self, scheduler: Any, device: Any, start: int) -> None:
        # A CPU denominator takes reciprocal in double, then multiplies in float opmath.
        # Moving the original scalar to the GPU changes half rounding, so prepare the
        # same reciprocals before transfer. Both native paths use this one arithmetic.
        sigmas = scheduler.sigmas[start:-1]
        scales = torch.stack([(sigma**2 + 1) ** 0.5 for sigma in sigmas])
        sigma_hat = torch.stack([sigma * 1 for sigma in sigmas])
        self.timesteps = scheduler.timesteps[start:].to(device=device)
        self.inverse_scales = scales.double().reciprocal().float().to(device=device)
        self.inverse_sigmas = sigma_hat.double().reciprocal().float().to(device=device)
        self.sigmas = sigma_hat.to(device=device)
        self.deltas = (scheduler.sigmas[start + 1 :] - sigma_hat).to(device=device)

    def step(
        self,
        unet: Any,
        index: torch.Tensor,
        state: torch.Tensor,
        prompt: torch.Tensor,
        pooled: torch.Tensor,
        time_ids: torch.Tensor,
        guidance: float,
    ) -> torch.Tensor:
        timestep = self.timesteps.index_select(0, index)
        inverse_scale = self.inverse_scales.index_select(0, index).reshape(())
        inverse_sigma = self.inverse_sigmas.index_select(0, index).reshape(())
        sigma = self.sigmas.index_select(0, index).reshape(())
        delta = self.deltas.index_select(0, index).reshape(())
        batch = torch.cat([state] * 2) if guidance > 1.0 else state
        noise = unet(
            (batch.float() * inverse_scale).to(batch.dtype),
            timestep,
            encoder_hidden_states=prompt,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        ).sample
        if guidance > 1.0:
            uncond, cond = noise.chunk(2)
            noise = uncond + guidance * (cond - uncond)
        sample = state.to(torch.float32)
        original = sample - (sigma * noise.float()).to(noise.dtype)
        derivative = (sample - original) * inverse_sigma
        state.copy_((sample + derivative * delta).to(noise.dtype))
        return noise


class ResidentPipeline:
    """One reusable definition per construction, retired by Runtime under pressure.

    `prepare` is for an explicit admitted warm turn. Ordinary `run` may capture only
    already-resident representations; it never loads a whole model speculatively.
    Shape replacement closes the previous definition, while prompt/seed values reuse it.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._program: _Pipeline | None = None

    @contextmanager
    def _using(self, ctx: Context) -> Iterator[None]:
        while not self._lock.acquire(timeout=0.05):
            ctx.raise_if_cancelled()
        try:
            ctx.raise_if_cancelled()
            yield
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            old, self._program = self._program, None
            if old is not None:
                old.plan.close()

    def prepare(self, **parameters: Any) -> bool:
        """Build one known variant during an admitted warm turn, before request submission.

        Parameters match `_select`; seed is intentionally absent. Token values only
        instantiate fixed-shape capture slots and are replaced by each request. Runtime's
        ordinary prelaunch admission owns the weight/workspace capacity check.
        """
        ctx = parameters["ctx"]
        if ctx.device.type != "cuda":
            return False
        with self._using(ctx):
            return self._select(**parameters, resident_only=False) is not None

    def _select(
        self,
        *,
        components: dict[str, Any],
        scheduler_config: dict[str, Any],
        vae_scale: float,
        tokens: tuple[torch.Tensor, torch.Tensor],
        negative_tokens: tuple[torch.Tensor, torch.Tensor] | None,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        ctx: Context,
        tel: Telemetry,
        resident_only: bool,
    ) -> tuple[_Pipeline, bool] | None:
        vae = components["vae"]
        dtype = next(vae.parameters()).dtype
        if bool(getattr(vae.config, "force_upcast", False)):
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported(including_emulation=False)
                else torch.float32
            )
        key = (
            json.dumps(scheduler_config, sort_keys=True, separators=(",", ":")),
            vae_scale,
            width,
            height,
            steps,
            guidance,
            dtype,
            tuple(tuple(value.shape) for value in tokens),
            None
            if negative_tokens is None
            else tuple(tuple(value.shape) for value in negative_tokens),
        )
        program = self._program
        if program is not None and (program.key != key or not program.plan.valid()):
            self.close()
            program = None
        if program is not None:
            return program, True
        began = time.perf_counter()
        try:
            program = _Pipeline(
                key,
                components,
                scheduler_config,
                vae_scale,
                tokens,
                negative_tokens,
                width,
                height,
                steps,
                guidance,
                dtype,
                ctx,
                resident_only,
            )
        except CapabilityError as error:
            if error.code != "prepared_graph_ineligible":
                raise
            tel.log("eager pipeline", reason=str(error))
            return None
        except torch.OutOfMemoryError:
            tel.log("eager pipeline", reason="resident pipeline preparation exceeded capacity")
            return None
        self._program = program
        tel.metric("request_path.pipeline_prepare_ms", (time.perf_counter() - began) * 1000)
        return program, False

    def run(
        self,
        *,
        seed: int,
        ctx: Context,
        tel: Telemetry,
        **parameters: Any,
    ) -> torch.Tensor | None:
        if ctx.device.type != "cuda":
            return None
        with self._using(ctx):
            selected = self._select(**parameters, ctx=ctx, tel=tel, resident_only=True)
            if selected is None:
                return None
            program, reused = selected
            steps = parameters["steps"]
            # Once ctx.plan begins, errors/cancellation propagate and never rerun the image.
            try:
                program.update(parameters["tokens"], parameters["negative_tokens"], seed)
                tel.metric("request_path.pipeline_reused", float(reused))
                launches = int(program.plan.facts()["host_graph_launches"])
                with tel.stage("native_pipeline", overall_range=(0.02, 0.98)):
                    execution = ctx.plan(program.plan)
                    observed = -1
                    while True:
                        ctx.raise_if_cancelled()
                        progress = execution.poll()
                        if progress.completed_steps != observed:
                            observed = progress.completed_steps
                            tel.progress(
                                observed / steps,
                                stage="native_pipeline",
                                overall_fraction=0.02 + 0.96 * observed / steps,
                            )
                        if progress.done:
                            break
                        time.sleep(0.05)
                    latent, prompt, pooled, image = execution.wait()
                    if progress.canceled or progress.completed_steps != steps:
                        raise OutputError(
                            "prepared pipeline stopped early", code="incomplete_pipeline"
                        )
                tel.metric("request_path.pipeline_native_steps", float(progress.completed_steps))
                tel.metric(
                    "request_path.pipeline_host_launches",
                    float(int(execution.facts()["host_graph_launches"]) - launches),
                )
                # First-noise and per-stage timings are deliberately absent: collecting
                # them must not add a reduction or a host rendezvous to every iteration.
                for name, value in (
                    ("prompt_absmax", prompt),
                    ("pooled_absmax", pooled),
                    ("latent_absmax", latent),
                ):
                    tel.metric(
                        name, round(float(torch.nan_to_num(value, 0.0, 0.0, 0.0).abs().max()), 4)
                    )
                return image
            except BaseException:
                self.close()
                raise


class _Pipeline:
    def __init__(
        self,
        key: tuple[Any, ...],
        components: dict[str, Any],
        scheduler_config: dict[str, Any],
        vae_scale: float,
        tokens: tuple[torch.Tensor, torch.Tensor],
        negative_tokens: tuple[torch.Tensor, torch.Tensor] | None,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        dtype: Any,
        ctx: Context,
        resident_only: bool,
    ) -> None:
        self.key = key
        first, second, unet, vae = (
            components[name] for name in ("text_encoder", "text_encoder_2", "unet", "vae")
        )
        unet.begin_denoise_request(steps, False, 0)
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        if scheduler.config.prediction_type != "epsilon":
            raise CapabilityError(
                "resident pipeline requires epsilon Euler", code="prepared_graph_ineligible"
            )
        device = str(ctx.device)
        scheduler.set_timesteps(steps, device=device)
        self.initial_sigma = scheduler.init_noise_sigma
        classifier_free = guidance > 1.0
        if classifier_free != (negative_tokens is not None):
            raise ValueError("negative tokens must match the classifier-free branch")
        with torch.inference_mode(False), torch.no_grad():
            self.tokens = tuple(value.to(device).clone() for value in tokens)
            self.negative_tokens = (
                None
                if negative_tokens is None
                else tuple(value.to(device).clone() for value in negative_tokens)
            )
            self.latent = torch.zeros(
                (1, 4, height // 8, width // 8), dtype=torch.float16, device=device
            )
            batch = 2 if classifier_free else 1
            prompt = torch.zeros(
                (batch, tokens[0].shape[-1], first.config.hidden_size + second.config.hidden_size),
                dtype=torch.float16,
                device=device,
            )
            pooled = torch.zeros(
                (batch, second.config.projection_dim), dtype=torch.float16, device=device
            )
            image = torch.zeros((1, 3, height, width), dtype=dtype, device=device)
            time_ids = torch.tensor(
                [[height, width, 0, 0, height, width]] * batch, dtype=torch.float16, device=device
            )
            euler = _Euler(scheduler, device, 0)

        def prefix() -> None:
            positive, positive_pool = encode_tokens(first, second, *self.tokens)
            if self.negative_tokens is not None:
                negative, negative_pool = encode_tokens(first, second, *self.negative_tokens)
                prompt.copy_(torch.cat([negative, positive]))
                pooled.copy_(torch.cat([negative_pool, positive_pool]))
            else:
                prompt.copy_(positive)
                pooled.copy_(positive_pool)

        def body(index: torch.Tensor) -> None:
            euler.step(unet, index, self.latent, prompt, pooled, time_ids, guidance)

        def suffix() -> None:
            image.copy_(decode_latents(vae, self.latent.to(dtype=dtype), vae_scale))

        self.plan = PreparedRepeat.capture(
            body,
            state=(self.latent, prompt, pooled, image),
            steps=steps,
            owners=(first, second, unet, vae),
            resident_only=resident_only,
            prefix=prefix,
            suffix=suffix,
            dtypes={vae: dtype},
        )

    def update(
        self,
        tokens: tuple[torch.Tensor, torch.Tensor],
        negative_tokens: tuple[torch.Tensor, torch.Tensor] | None,
        seed: int,
    ) -> None:
        with torch.no_grad():
            for slot, value in zip(self.tokens, tokens, strict=True):
                slot.copy_(value)
            if self.negative_tokens is not None:
                assert negative_tokens is not None
                for slot, value in zip(self.negative_tokens, negative_tokens, strict=True):
                    slot.copy_(value)
            self.latent.copy_(
                rng.noise(
                    self.latent.shape, seed, dtype=self.latent.dtype, device=self.latent.device
                )
                * self.initial_sigma
            )


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
    try:
        with torch.inference_mode(False), torch.no_grad():
            state = latents.clone()
            euler = _Euler(scheduler, state.device, 1)

        def body(index: torch.Tensor) -> None:
            with torch.no_grad():
                euler.step(unet, index, state, prompt, pooled, time_ids, guidance)

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
