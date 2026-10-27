"""SDXL's reusable resident pipeline and Euler interval; Runtime owns their memory."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from cozy_runtime import rng
from cozy_runtime.author import CapabilityError, Context, OutputError, Telemetry
from cozy_runtime.author._model import PLACED
from cozy_runtime.compute import PreparedRepeat, PreparedSequence
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


def decode_dtype(vae: Any) -> torch.dtype:
    """The dtype every SDXL decode runs the VAE in. The SDXL VAE overflows fp16 (force_upcast):
    bf16 where the GPU computes it natively (fp32's range; 48 dB PSNR to fp32 on the 1024px fox,
    run 1642), else fp32."""
    if not getattr(vae.config, "force_upcast", False):
        return next(vae.parameters()).dtype
    return (
        torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32
    )


def hold_decode_dtype(vae: Any, tel: Telemetry) -> None:
    """Hold the VAE in `decode_dtype` for good, before any VAE scope opens: Runtime's cast of its
    stored bytes, filled on its next use, so nothing loads here and no request converts weights.
    Eager and native decode read the same weights, and a later ordinary request can capture
    the resident pipeline without a warm."""
    dtype = decode_dtype(vae)
    if next(vae.parameters()).dtype == dtype:
        return
    placed = PLACED.get(vae)
    if placed is None:
        vae.to(dtype=dtype)  # not Runtime-placed: an ordinary module
    else:
        placed[0].represent(placed[1], dtype)()
    tel.log("vae representation", dtype=str(dtype).removeprefix("torch."))


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
        self._region: Any = None
        self._region_owner: Any = None
        self._region_token: Any = None

    def _owned_region(self, components: dict[str, Any]) -> Any:
        """Register only a release callback; Torch already accounts for these allocations."""
        placed = PLACED.get(components["unet"])
        if placed is None:
            return None
        owner = placed[0]
        with self._lock:
            if self._region is not None:
                if self._region_owner is not owner:
                    raise CapabilityError(
                        "native pipeline cache belongs to another model construction",
                        code="prepared_graph_ineligible",
                    )
                return self._region
            token = object()

            def release() -> None:
                with self._lock:
                    if self._region_token is token:
                        self._close_locked()

            # Region registration creates only a release handle; it does not acquire its
            # lock, allocate GPU memory, account bytes or invoke the callback here.
            made = owner.region("sdxl-native-plan", "latent", (), release=release)
            self._region, self._region_owner, self._region_token = made, owner, token
            return made

    @contextmanager
    def _using(self, ctx: Context, components: dict[str, Any]) -> Iterator[None]:
        region = self._owned_region(components)
        with region.pin() if region is not None else nullcontext(False):
            while not self._lock.acquire(timeout=0.05):
                ctx.raise_if_cancelled()
            try:
                ctx.raise_if_cancelled()
                if region is not self._region:
                    raise RuntimeError("native pipeline cache ownership changed during use")
                yield
            finally:
                self._lock.release()

    def _close_locked(self) -> None:
        old = self._program
        if old is not None:
            # Failed/unproven native close must retain the complete program and its actual
            # graph/buffer owners. Only checked close authorizes releasing cached state.
            old.close()
            self._program = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def prepare(self, **parameters: Any) -> bool:
        """Build one known variant during an admitted warm turn, before request submission.

        Parameters match `_select`; seed is intentionally absent. Token values only
        instantiate fixed-shape capture slots and are replaced by each request. Runtime's
        ordinary prelaunch admission owns the weight/workspace capacity check.
        """
        ctx = parameters["ctx"]
        if ctx.device.type != "cuda":
            return False
        with self._using(ctx, parameters["components"]):
            selected = self._select(**parameters, resident_only=False)
            if selected is None:
                return False
            program, reused = selected
            program.observe(parameters["tel"], "prepared", reused)
            return True

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
        dtype = decode_dtype(vae)
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
        constructors = (
            (_SequentialPipeline, _Pipeline)
            if not resident_only and callable(getattr(PreparedRepeat, "capture_stage", None))
            else (_Pipeline,)
        )
        reasons = []
        for constructor in constructors:
            try:
                program = constructor(
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
                break
            except CapabilityError as error:
                if error.code not in ("prepared_graph_ineligible", "prepared_load_unavailable"):
                    raise
                reasons.append(str(error))
            except torch.OutOfMemoryError:
                reasons.append(f"{constructor.__name__} preparation exceeded capacity")
        else:
            tel.log("eager pipeline", reason="; ".join(reasons))
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
        with self._using(ctx, parameters["components"]):
            selected = self._select(**parameters, ctx=ctx, tel=tel, resident_only=True)
            if selected is None:
                return None
            program, reused = selected
            steps = parameters["steps"]
            # Once ctx.plan begins, errors/cancellation propagate and never rerun the image.
            try:
                program.update(parameters["tokens"], parameters["negative_tokens"], seed)
                tel.metric("request_path.pipeline_reused", float(reused))
                tel.metric(
                    "request_path.pipeline_sequential",
                    float(isinstance(program, _SequentialPipeline)),
                )
                launches = int(program.plan.facts()["host_graph_launches"])
                with tel.stage("native_pipeline", overall_range=(0.02, 0.98)):
                    execution = ctx.plan(program.plan)
                    observed = -1
                    while True:
                        ctx.raise_if_cancelled()
                        progress = execution.poll()
                        completed = program.completed(execution, progress)
                        if completed != observed:
                            observed = completed
                            tel.progress(
                                observed / steps,
                                stage="native_pipeline",
                                overall_fraction=0.02 + 0.96 * observed / steps,
                            )
                        if progress.done:
                            break
                        time.sleep(0.05)
                    latent, prompt, pooled, image = execution.wait()
                    completed = program.completed(execution, progress)
                    if progress.canceled or completed != steps:
                        raise OutputError(
                            "prepared pipeline stopped early", code="incomplete_pipeline"
                        )
                tel.metric("request_path.pipeline_native_steps", float(completed))
                tel.metric(
                    "request_path.pipeline_host_launches",
                    float(int(execution.facts()["host_graph_launches"]) - launches),
                )
                program.observe(tel, "completed", reused)
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
    decoder: Any = None  # the VAE decoder's binding the program owns (None: not Runtime-placed)

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

        if next(vae.parameters()).dtype != dtype:
            raise CapabilityError(
                "the VAE is not held in its decode dtype", code="prepared_graph_ineligible"
            )
        # Text-to-image only decodes: the program owns the VAE's decoder, never its encoder,
        # whose blocks no request fills (an encoder block kept every capture non-resident).
        placed = PLACED.get(vae)
        self.decoder = (
            None if placed is None else placed[0].retain(placed[1], ("decoder", "post_quant_conv"))
        )
        try:
            self.plan = PreparedRepeat.capture(
                body,
                state=(self.latent, prompt, pooled, image),
                steps=steps,
                owners=(first, second, unet, vae if self.decoder is None else self.decoder),
                resident_only=resident_only,
                prefix=prefix,
                suffix=suffix,
            )
        except BaseException:
            if self.decoder is not None:
                self.decoder.close()
            raise

    def close(self) -> None:
        self.plan.close()  # an unproven close raises and keeps every owner
        if self.decoder is not None:
            self.decoder.close()

    def completed(self, execution: Any, progress: Any) -> int:
        return int(progress.completed_steps)

    def observe(self, tel: Telemetry, phase: str, reused: bool) -> None:
        # Diagnostic identity is process-local, never a persisted cache or model key.
        tel.log(
            "resident pipeline definition",
            phase=phase,
            pid=os.getpid(),
            plan=f"{id(self.plan):x}",
            identity_scope="process-local",
            host_graph_launches=int(self.plan.facts()["host_graph_launches"]),
            reused=reused,
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


class _SequentialPipeline(_Pipeline):
    """The existing pipeline cache with native phase boundaries and streamed UNet blocks.

    The initial wide VAE remains resident through the whole submission. Its established
    typed preparation is safe; raw source stages cannot yet reload derived BF16/F32 final
    representations. Text/UNet stages keep source reservations without lifetime weight pins.
    """

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
        from ._streamed import StreamedEuler

        self.key = key
        first, second, unet, vae = (
            components[name] for name in ("text_encoder", "text_encoder_2", "unet", "vae")
        )
        unet.begin_denoise_request(steps, False, 0)
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        if scheduler.config.prediction_type != "epsilon":
            raise CapabilityError(
                "native pipeline requires epsilon Euler", code="prepared_graph_ineligible"
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
            latent = torch.zeros(
                (1, 4, height // 8, width // 8), dtype=torch.float16, device=device
            )
            batch = 2 if classifier_free else 1
            first_hidden = torch.zeros(
                (batch, tokens[0].shape[-1], first.config.hidden_size),
                dtype=torch.float16,
                device=device,
            )
            second_hidden = torch.zeros(
                (batch, tokens[1].shape[-1], second.config.hidden_size),
                dtype=torch.float16,
                device=device,
            )
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
        self.plan = None
        self._definitions: list[Any] = []
        self._interval = None
        self._capture_memory: list[dict[str, Any]] = []

        def observed(name: str, plan: Any) -> Any:
            self._definitions.append(plan)
            self._capture_memory.append(
                {
                    "phase": name,
                    "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                }
            )
            return plan

        try:
            self.latent = latent

            def decode(_index: torch.Tensor) -> None:
                image.copy_(decode_latents(vae, self.latent.to(dtype=dtype), vae_scale))

            # This real typed binding owns the VAE's converted final weights until close;
            # do not silently capture FP16 decode for a force_upcast model.
            vae_plan = observed(
                "wide_resident_vae",
                PreparedRepeat.capture(
                    decode,
                    state=(image,),
                    steps=1,
                    owners=(vae,),
                    dtypes={vae: dtype},
                    resident_only=resident_only,
                ),
            )

            # Wide VAE eligibility is checked before paying for every UNet graph.
            interval = StreamedEuler(
                unet=unet,
                scheduler=scheduler,
                latents=latent,
                prompt=prompt,
                pooled=pooled,
                time_ids=time_ids,
                guidance=guidance,
                _attach_sequence=False,
                _latent_state=latent,
            )
            self._interval = interval

            def encode_first(_index: torch.Tensor) -> None:
                positive, _ = _clip_text(first, self.tokens[0])
                if self.negative_tokens is None:
                    first_hidden.copy_(positive)
                else:
                    negative, _ = _clip_text(first, self.negative_tokens[0])
                    first_hidden.copy_(torch.cat([negative, positive]))

            first_plan = observed(
                "text_encoder",
                PreparedRepeat.capture_stage(
                    encode_first, state=(first_hidden,), owner=first, load_ordering="events"
                ),
            )

            def encode_second(_index: torch.Tensor) -> None:
                positive, positive_pool = _clip_text(second.text_model, self.tokens[1])
                positive_pool = second.text_projection(positive_pool)
                if self.negative_tokens is None:
                    second_hidden.copy_(positive)
                    pooled.copy_(positive_pool)
                else:
                    negative, negative_pool = _clip_text(second.text_model, self.negative_tokens[1])
                    negative_pool = second.text_projection(negative_pool)
                    second_hidden.copy_(torch.cat([negative, positive]))
                    pooled.copy_(torch.cat([negative_pool, positive_pool]))

            second_plan = observed(
                "text_encoder_2",
                PreparedRepeat.capture_stage(
                    encode_second,
                    state=(second_hidden, pooled),
                    owner=second,
                    load_ordering="events",
                ),
            )

            def condition(_index: torch.Tensor) -> None:
                prompt.copy_(torch.cat([first_hidden, second_hidden], dim=-1))

            condition_plan = observed(
                "conditioning", PreparedRepeat.capture(condition, state=(prompt,), steps=1)
            )
            # The interval's reset is a one-time prefix; its input..sampler range repeats.
            stages = (first_plan, second_plan, condition_plan, *interval.definitions, vae_plan)
            begin = 3 + 1
            end = 3 + len(interval.definitions)
            self.plan = PreparedSequence(
                stages, outputs=(self.latent, prompt, pooled, image), repeat=(begin, end, steps)
            )
        except BaseException:
            for definition in self._definitions:
                definition.close()
            if self._interval is not None:
                self._interval.close()
            raise

    def completed(self, execution: Any, progress: Any) -> int:
        return int(execution.facts()["sequence"]["completed_passes"])

    def observe(self, tel: Telemetry, phase: str, reused: bool) -> None:
        assert self.plan is not None and self._interval is not None
        tel.log(
            "native sequential pipeline definition",
            phase=phase,
            pid=os.getpid(),
            plan=f"{id(self.plan):x}",
            identity_scope="process-local",
            reused=reused,
            host_graph_launches=int(self.plan.facts()["host_graph_launches"]),
            vae_policy="wide resident typed binding held through whole submission",
            weight_phase_policy="native demand capacity; no per-phase preference update",
        )
        tel.metric(
            "request_path.pipeline_boundary_bytes", float(self._interval.facts()["boundary_bytes"])
        )
        tel.metric(
            "request_path.pipeline_torch_reserved_bytes",
            float(torch.cuda.memory_reserved(self.latent.device)),
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
    load_from_storage: bool = False,
) -> torch.Tensor | None:
    """Run an eligible resident Euler interval, optionally loading missing raw UNet blocks.

    A refusal before submission preserves the caller's latent. Once submitted, errors and
    cancellation propagate; they never rerun an interval that might have partially executed.
    HiDiffusion/DDIM and other Euler prediction rules keep their ordinary scheduler loop.
    """
    if load_from_storage and not getattr(PreparedRepeat, "supports_component_load", False):
        return None
    start = 0 if load_from_storage else 1
    if (
        type(scheduler) is not EulerDiscreteScheduler
        or scheduler.config.prediction_type != "epsilon"
        or scheduler.step_index not in ((None, 0) if load_from_storage else (1,))
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
            euler = _Euler(scheduler, state.device, start)

        def body(index: torch.Tensor) -> None:
            with torch.no_grad():
                euler.step(unet, index, state, prompt, pooled, time_ids, guidance)

        options: dict[str, Any] = {"resident_only": not load_from_storage}
        if load_from_storage:
            options.update(load_component=unet, load_ordering="auto")
        plan = PreparedRepeat.capture(
            body, state=(state,), steps=total - start, owners=(unet,), **options
        )
    except CapabilityError as error:
        if error.code not in ("prepared_graph_ineligible", "prepared_load_unavailable"):
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
                fraction = (start + observed) / total
                # Observation intervals are not GPU step timings. The GPU continues while
                # this watcher sleeps; it does not need a Python callback per iteration.
                tel.progress(fraction, stage="denoise", overall_fraction=0.10 + 0.80 * fraction)
            if progress.done:
                break
            time.sleep(0.05)
        (result,) = execution.wait()
        if progress.canceled or progress.completed_steps != total - start:
            raise OutputError("prepared denoising stopped early", code="incomplete_denoise")
        tel.metric("request_path.denoise_native_steps", float(progress.completed_steps))
        if load := execution.facts().get("native_load"):
            tel.metric("request_path.denoise_native_load_bytes", float(load["read_bytes"]))
            tel.log(
                "native denoise input",
                ordering=load["ordering"],
                regions=load["input_read_bytes"],
                completion_proven=load["completion_proven"],
            )
        return result
    finally:
        # No request-path graph retains UNet pages while the VAE starts. Close also joins
        # checked completion on cancellation or unwind before releasing any graph owner.
        plan.close()
