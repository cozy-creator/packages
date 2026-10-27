"""Qwen-Image-2.1's native cached-denoise interval; Runtime owns the transformer weights.

Step 0 runs eagerly in the transformer's ``extract`` mode, prefilling the per-layer text and
condition-image K/V cache over the full joint sequence. Steps ``1..N-1`` are the ``cached``
pass over the target-image tokens only, and they are identical from one step to the next but
for the timestep and the running latent. That repeated interval is captured once and replayed
as one native submission. A resident transformer uses one repeated graph; a capacity
refusal prepares source-backed block stages that share graph scratch and repeat the whole
cached step in native Runtime. The eager extract pass remains outside that submission.

What makes the cached step capturable is that everything except the latent is a function of the
request *layout* (prompt and reference length, canvas), not of the step: the rotary tables, the
target-token modulation mask and the key-valid mask are all derived once, eagerly, by `_Layout`
out of the body, so the captured body carries no host sync, no Python scalar read and no
data-dependent shape. The body calls the transformer's own ``img_in``/``time_text_embed``/
``modulation``/``norm_out``/``proj_out`` and the installed blocks, so the numerics are the
Diffusers/Runtime ones; the parity gate is this path against the eager Diffusers loop.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

import torch
from cozy_runtime.author import CapabilityError, Context, OutputError, Telemetry
from cozy_runtime.compute import PreparedRepeat, PreparedSequence


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
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[
                :, vlm_text_positions
            ]

        prefix_len = int((~target_token_mask).sum())
        self.prefix_len = prefix_len
        # The target queries only: the cached prefix lives in the K/V cache, never recomputed.
        self.rotary_emb = rotary_emb[prefix_len:]
        self.modulation_mask = target_token_mask[prefix_len:]
        self.attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]


class ResidentDenoise:
    """Prefer one resident repeat, then native block streaming when residency refuses.

    Only the transformer is retained. The eager prefix refreshes K/V for every prompt;
    those values, schedules, layout and initial latent are copied into stable inputs.
    Runtime's binding/fill and kernel identity checks decide whether reuse remains valid.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._program: _Denoise | None = None

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
            if self._program is not None:
                self._program.plan.close()
                self._program = None

    def run(
        self,
        *,
        transformer: Any,
        scheduler: Any,
        latents: torch.Tensor,
        kv_cache: Any,
        layout: _Layout,
        ctx: Context,
        tel: Telemetry,
    ) -> torch.Tensor | None:
        total = len(scheduler.timesteps)
        if (
            not latents.is_cuda
            or latents.dtype != torch.bfloat16
            or total < 2
            or scheduler.sigmas.shape[0] != total + 1
        ):
            self.close()
            return None
        with self._using(ctx):
            with torch.inference_mode(False), torch.no_grad():
                euler = _FlowEuler(scheduler, latents.device, 1)
            caches = [
                kv_cache.get_layer(index) for index in range(len(transformer.transformer_blocks))
            ]
            inputs = (
                latents,
                euler.ts,
                euler.dt,
                layout.rotary_emb,
                layout.modulation_mask,
                *(tensor for cache in caches for tensor in cache.get()),
                *((layout.attention_mask,) if layout.attention_mask is not None else ()),
            )
            key = (
                id(transformer),
                total,
                len(caches),
                layout.attention_mask is not None,
                tuple((tuple(value.shape), value.dtype, value.device) for value in inputs),
            )
            program = self._program
            if program is not None and (program.key != key or not program.plan.valid()):
                self.close()
                program = None
            reused = program is not None
            began = time.perf_counter()
            if program is None:
                try:
                    program = _Denoise(key, transformer, inputs, euler, caches, layout, total)
                except CapabilityError as error:
                    if error.code != "prepared_graph_ineligible":
                        raise
                    if "capture requires the component's weights to be resident" not in str(error):
                        tel.log("eager denoise", reason=str(error))
                        return None
                except torch.OutOfMemoryError:
                    pass  # No accepted work: try native block streaming at the same boundary.
                if program is None:
                    try:
                        program = _StreamedDenoise(
                            key, transformer, inputs, euler, caches, layout, total
                        )
                    except CapabilityError as error:
                        if error.code not in ("prepared_graph_ineligible", "prepared_load_unavailable"):
                            raise
                        # A not-local source is refused before submission. The ordinary
                        # eager loader can fetch it; accepted native work is never retried.
                        tel.log("eager denoise", reason=str(error))
                        return None
                    except torch.OutOfMemoryError:
                        tel.log(
                            "eager denoise", reason="native block preparation exceeded capacity"
                        )
                        return None
                self._program = program
            tel.metric(
                "request_path.denoise_capture_ms",
                0.0 if reused else (time.perf_counter() - began) * 1000,
            )
            # No fallback after submission: cancellation/device failures join before disposal.
            try:
                began = time.perf_counter()
                program.update(inputs)
                tel.metric(
                    "request_path.denoise_input_enqueue_ms", (time.perf_counter() - began) * 1000
                )
                tel.metric("request_path.denoise_reused", float(reused))
                launches = int(program.plan.facts()["host_graph_launches"])
                execution = ctx.plan(program.plan)
                observed = -1
                while True:
                    ctx.raise_if_cancelled()
                    progress = execution.poll()
                    completed = (
                        int(execution.facts()["sequence"]["completed_passes"])
                        if program.streaming
                        else progress.completed_steps
                    )
                    if completed != observed:
                        observed = completed
                        fraction = (1 + observed) / total
                        tel.progress(
                            fraction,
                            stage="generating image",
                            overall_fraction=0.1 + 0.8 * fraction,
                        )
                    if progress.done:
                        break
                    time.sleep(0.05)
                (result,) = execution.wait()
                if progress.canceled or observed != total - 1:
                    raise OutputError("prepared denoising stopped early", code="incomplete_denoise")
                tel.metric("request_path.denoise_native_steps", float(observed))
                tel.metric("request_path.denoise_streamed", float(program.streaming))
                tel.metric(
                    "request_path.denoise_host_launches",
                    float(int(execution.facts()["host_graph_launches"]) - launches),
                )
                tel.log(
                    "native denoise definition",
                    pid=os.getpid(),
                    plan=f"{id(program.plan):x}",
                    identity_scope="process-local",
                    reused=reused,
                    streamed=program.streaming,
                )
                # Execution.wait returns its own completed readout, not the plan's
                # mutable state. The next request cannot overwrite this VAE input.
                return result
            except BaseException:
                self.close()
                raise


class _Denoise:
    streaming = False

    def __init__(
        self,
        key: tuple[Any, ...],
        transformer: Any,
        inputs: tuple[torch.Tensor, ...],
        euler: _FlowEuler,
        caches: list[Any],
        layout: _Layout,
        total: int,
    ) -> None:
        self.key = key
        with torch.inference_mode(False), torch.no_grad():
            state = inputs[0].clone()
        slots = (state, *inputs[1:])
        rot, mask_t, attention_mask = (
            layout.rotary_emb,
            layout.modulation_mask,
            layout.attention_mask,
        )
        blocks = transformer.transformer_blocks

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

        self.plan = PreparedRepeat.capture(
            body,
            state=(state,),
            steps=total - 1,
            owners=(transformer, *slots[1:]),
            resident_only=True,
        )
        # The native definition owns these inputs. Weak references let ordinary Runtime
        # graph retirement free K/V/layout buffers immediately under pressure; this cache
        # keeps no second strong owner or text-encoder weight binding.
        # Explicit owners also cover inputs unused by a particular selected kernel.
        self._inputs = tuple(weakref.ref(value) for value in slots)

    def update(self, inputs: tuple[torch.Tensor, ...]) -> None:
        with torch.no_grad():
            for reference, value in zip(self._inputs, inputs, strict=True):
                target = reference()
                if target is None:
                    raise RuntimeError("prepared denoise inputs were retired")
                if target is not value:
                    target.copy_(value)


class _StreamedDenoise(_Denoise):
    streaming = True

    def __init__(self, key, transformer, inputs, euler, caches, layout, total):
        if not callable(getattr(PreparedSequence, "record", None)):
            raise CapabilityError(
                "native streamed sequence recording is unavailable",
                code="prepared_graph_ineligible",
            )
        if inputs[0].shape[0] != 1 or not transformer.config.causal_condition:
            raise CapabilityError(
                "streamed Qwen requires one causal cached sample", code="prepared_graph_ineligible"
            )
        self.key = key
        blocks = transformer.transformer_blocks
        if not blocks or len(caches) != len(blocks):
            raise CapabilityError(
                "streamed Qwen requires every layer cache", code="prepared_graph_ineligible"
            )
        with torch.inference_mode(False), torch.no_grad():
            state = inputs[0].clone()
            index = torch.zeros((1,), dtype=torch.int64, device=state.device)
            hidden = torch.zeros(
                (*state.shape[:2], transformer.inner_dim), dtype=state.dtype, device=state.device
            )
            temb = torch.zeros((2, transformer.inner_dim), dtype=state.dtype, device=state.device)
            modulation = torch.zeros(
                (2, 4 * transformer.inner_dim), dtype=state.dtype, device=state.device
            )
        slots = (state, *inputs[1:])
        rot, mask, attention = layout.rotary_emb, layout.modulation_mask, layout.attention_mask

        with PreparedSequence.record() as recording:
            # Explicit owners also retain immutable inputs unused by a selected native kernel.
            # All cross-stage state is allocated outside the shared graph scratch pool.
            def reset(_):
                index.zero_()

            stages = [
                recording.capture(
                    reset, state=(index,), steps=1, owners=(*slots, hidden, temb, modulation)
                )
            ]

            def enter(_):
                hidden.copy_(transformer.img_in(state))
                temb.copy_(transformer.time_text_embed(euler.timestep(index), hidden))
                modulation.copy_(transformer.modulation(temb))

            stages.append(
                recording.capture_stage(
                    enter, state=(hidden, temb, modulation), owner=transformer, paths=()
                )
            )
            for number, (block, cache) in enumerate(zip(blocks, caches, strict=True)):

                def body(_, block=block, cache=cache):
                    hidden.copy_(
                        block(
                            hidden_states=hidden,
                            modulation=modulation,
                            rotary_emb=rot,
                            attention_mask=attention,
                            target_token_mask=mask,
                            layer_cache=cache,
                            kv_cache_mode="cached",
                            cache_write_slice=None,
                            segments=None,
                            key_valid=None,
                        )
                    )

                stages.append(
                    recording.capture_stage(
                        body,
                        state=(hidden,),
                        owner=transformer,
                        paths=(f"transformer_blocks.{number}",),
                    )
                )

            def sample(_):
                normalized = transformer.norm_out(hidden, temb, mask)
                noise = transformer.proj_out(normalized)
                state.copy_((state.float() + euler.delta(index) * noise).to(state.dtype))
                index.add_(1)

            stages.append(
                recording.capture_stage(sample, state=(state, index), owner=transformer, paths=())
            )
            if any(not stage.valid() for stage in stages):
                raise CapabilityError(
                    "Qwen stage identity changed during preparation",
                    code="prepared_graph_ineligible",
                )
            self.plan = recording.finish(outputs=(state,), repeat=(1, len(blocks) + 3, total - 1))
        self._inputs = tuple(weakref.ref(value) for value in slots)
