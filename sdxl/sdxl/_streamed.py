"""SDXL Euler as native source stages with explicit persistent UNet skip state.

This adapter calls the package's existing Diffusers numerical modules. It covers the
plain SDXL text-time UNet, with no ControlNet, adapters, HiDiffusion or compiled wrappers.
Preparation is Python; a submitted interval advances/refills entirely in Runtime native
code. A surrounding native phase program may lend its sequence-owned scratch recorder.
"""

from __future__ import annotations

import time
from typing import Any
from contextlib import nullcontext

import torch
from cozy_runtime.author import CapabilityError, OutputError
from cozy_runtime.compute import PreparedRepeat, PreparedSequence
from diffusers import EulerDiscreteScheduler

from ._prepared import _Euler


def _refuse(detail: str) -> None:
    raise CapabilityError(detail, code="prepared_graph_ineligible")


class StreamedEuler:
    """One complete fixed-shape denoising interval, without resident weight ownership.

    Skip tensors and stage boundaries are ordinary persistent Torch allocations. Every
    down/up/mid stage declares its exact weight subtree plus the existing common group.
    Common weights contain time/add embeddings and the input/output convolutions. Source
    stages retain reservations, and acquire physical pins only for their actual consumers.
    No graph needs another stage's temporary activation storage or private parameter view.
    """

    def __init__(
        self,
        *,
        unet: Any,
        scheduler: Any,
        latents: torch.Tensor,
        prompt: torch.Tensor,
        pooled: torch.Tensor,
        time_ids: torch.Tensor,
        guidance: float,
        ordering: str = "events",
        _shared_pool: bool | None = None,
        _attach_sequence: bool = True,
        _latent_state: torch.Tensor | None = None,
        _borrowed_recording: Any = None,
    ) -> None:
        if _borrowed_recording is not None:
            if _attach_sequence or _shared_pool is False:
                _refuse("a borrowed scratch recording belongs to its surrounding sequence")
            _shared_pool = True
        elif _shared_pool is None:
            _shared_pool = _attach_sequence
        if _shared_pool and not _attach_sequence and _borrowed_recording is None:
            _refuse("a shared scratch recording must own its complete sequence")
        if _shared_pool and not callable(getattr(PreparedSequence, "record", None)):
            _refuse("Runtime lacks owned sequential graph pools")
        if not callable(getattr(PreparedRepeat, "capture_stage", None)):
            _refuse("Runtime lacks native model block stages")
        if (
            type(scheduler) is not EulerDiscreteScheduler
            or scheduler.config.prediction_type != "epsilon"
            or scheduler.step_index not in (None, 0)
            or scheduler.sigmas.device.type != "cpu"
            or scheduler.sigmas.dtype != torch.float32
            or not len(scheduler.timesteps)
            or len(scheduler.sigmas) != len(scheduler.timesteps) + 1
            or not latents.is_cuda
            or latents.ndim != 4
            or latents.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            _refuse("streamed SDXL needs an unstarted epsilon Euler interval")
        if (
            unet.training
            or bool(getattr(unet, "_cozy_hidiffusion_active", False))
            or unet.config.addition_embed_type != "text_time"
            or unet.class_embedding is not None
            or unet.encoder_hid_proj is not None
            or unet.mid_block is None
            or len(unet.down_blocks) != len(unet.up_blocks)
            or not unet.down_blocks
            or any(
                bool(getattr(block, "gradient_checkpointing", False))
                for block in (*unet.down_blocks, unet.mid_block, *unet.up_blocks)
            )
        ):
            _refuse("streamed SDXL requires the plain text-time UNet layout")
        down_count = sum(bool(block.downsamplers) for block in unet.down_blocks)
        divisor = 2**down_count
        if any(size % divisor for size in latents.shape[-2:]):
            _refuse("streamed SDXL initially requires integral downsample dimensions")
        if any(len(block.downsamplers or ()) > 1 for block in unet.down_blocks):
            _refuse("streamed SDXL expects at most one downsampler per block")
        if any(len(block.upsamplers or ()) > 1 for block in unet.up_blocks):
            _refuse("streamed SDXL expects at most one upsampler per block")
        if sum(bool(block.upsamplers) for block in unet.up_blocks) != down_count:
            _refuse("streamed SDXL requires matching up/down sampling")

        batch = latents.shape[0] * (2 if guidance > 1 else 1)
        if prompt.shape[0] != batch or pooled.shape[0] != batch or time_ids.shape[0] != batch:
            raise ValueError("conditioning batch does not match SDXL guidance")
        self.steps = len(scheduler.timesteps)
        self.calls: dict[str, int] = {}
        self.definitions: list[Any] = []
        self.capture_memory: list[dict[str, Any]] = []
        self.boundaries: list[torch.Tensor] = []
        self.plan = None
        with torch.inference_mode(False), torch.no_grad():
            if _latent_state is not None and (
                PreparedRepeat._layout(_latent_state)[:4] != PreparedRepeat._layout(latents)[:4]
                or _latent_state.is_inference()
            ):
                raise ValueError("shared pipeline latent must have matching non-inference geometry")
            self.latent = latents.clone() if _latent_state is None else _latent_state
            self.index = torch.zeros((1,), device=latents.device, dtype=torch.int64)
            euler = _Euler(scheduler, latents.device, 0)

            def allocate(shape):
                value = torch.zeros(shape, device=latents.device, dtype=latents.dtype)
                self.boundaries.append(value)
                return value

            height, width = latents.shape[-2:]
            hidden = allocate((batch, unet.conv_in.out_channels, height, width))
            embedding = allocate((batch, unet.time_embedding.linear_2.out_features))

        owns_recording = _shared_pool and _borrowed_recording is None
        recording = (
            PreparedSequence.record() if owns_recording else _borrowed_recording
        )
        self.shared_pool = _shared_pool

        def capture(name, operation, state, paths=None):
            self.calls[name] = 0

            def body(_index):
                self.calls[name] += 1
                operation()

            began = time.perf_counter()
            if paths is None:
                plan = (PreparedRepeat if recording is None else recording).capture(
                    body, state=state, steps=1
                )
            else:
                plan = (PreparedRepeat if recording is None else recording).capture_stage(
                    body, state=state, owner=unet, paths=paths, load_ordering=ordering
                )
            self.definitions.append(plan)
            self.capture_memory.append(
                {
                    "stage": name,
                    "paths": paths,
                    "capture_ms": (time.perf_counter() - began) * 1000,
                    "torch_allocated_bytes": int(torch.cuda.memory_allocated(latents.device)),
                    "torch_reserved_bytes": int(torch.cuda.memory_reserved(latents.device)),
                    "torch_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(latents.device)
                    ),
                    "torch_peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(latents.device)
                    ),
                }
            )
            return plan

        try:
            with recording if owns_recording else nullcontext():
                capture("reset", lambda: self.index.zero_(), (self.index,))

                def project(input_hidden=hidden):
                    timestep = euler.timesteps.index_select(0, self.index)
                    inverse_scale = euler.inverse_scales.index_select(0, self.index).reshape(())
                    sample = torch.cat([self.latent] * 2) if guidance > 1 else self.latent
                    sample = (sample.float() * inverse_scale).to(sample.dtype)
                    if unet.config.center_input_sample:
                        sample = 2 * sample - 1.0
                    t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
                    emb = unet.time_embedding(t_emb, None)
                    augmented = unet.get_aug_embed(
                        emb=emb,
                        encoder_hidden_states=prompt,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                    )
                    emb = emb + augmented if augmented is not None else emb
                    if unet.time_embed_act is not None:
                        emb = unet.time_embed_act(emb)
                    embedding.copy_(emb)
                    input_hidden.copy_(unet.conv_in(sample))

                input_hidden = hidden
                capture("input", project, (input_hidden, embedding), ())
                skips = [input_hidden]
                for number, block in enumerate(unet.down_blocks):
                    incoming = hidden
                    channels = block.resnets[-1].out_channels
                    with torch.inference_mode(False):
                        outgoing = [
                            allocate((batch, channels, height, width)) for _ in block.resnets
                        ]
                        if block.downsamplers:
                            height, width = height // 2, width // 2
                            outgoing.append(allocate((batch, channels, height, width)))
                    outputs = tuple(outgoing)

                    def down(block=block, incoming=incoming, outputs=outputs):
                        if getattr(block, "has_cross_attention", False):
                            result, residuals = block(
                                hidden_states=incoming,
                                temb=embedding,
                                encoder_hidden_states=prompt,
                                attention_mask=None,
                                cross_attention_kwargs=None,
                                encoder_attention_mask=None,
                            )
                        else:
                            result, residuals = block(hidden_states=incoming, temb=embedding)
                        if len(residuals) != len(outputs):
                            _refuse("SDXL down block changed its skip layout")
                        for slot, value in zip(outputs, residuals, strict=True):
                            slot.copy_(value)
                        # Diffusers' final residual is the same output used by the next block.
                        if result.shape != outputs[-1].shape:
                            _refuse("SDXL down output differs from its final skip layout")

                    capture(f"down{number}", down, outputs, (f"down_blocks.{number}",))
                    skips.extend(outputs)
                    hidden = outputs[-1]

                incoming = hidden
                with torch.inference_mode(False):
                    hidden = allocate(tuple(incoming.shape))
                mid_output = hidden

                def middle(incoming=incoming):
                    block = unet.mid_block
                    if getattr(block, "has_cross_attention", False):
                        result = block(
                            incoming,
                            embedding,
                            encoder_hidden_states=prompt,
                            attention_mask=None,
                            cross_attention_kwargs=None,
                            encoder_attention_mask=None,
                        )
                    else:
                        result = block(incoming, embedding)
                    mid_output.copy_(result)

                capture("mid", middle, (mid_output,), ("mid_block",))
                for number, block in enumerate(unet.up_blocks):
                    incoming = hidden
                    count = len(block.resnets)
                    residuals = tuple(skips[-count:])
                    del skips[-count:]
                    if len(residuals) != count:
                        _refuse("SDXL up block has insufficient persistent skips")
                    channels = block.resnets[-1].out_channels
                    if block.upsamplers:
                        height, width = height * 2, width * 2
                    with torch.inference_mode(False):
                        hidden = allocate((batch, channels, height, width))
                    outgoing = hidden

                    def up(
                        block=block,
                        incoming=incoming,
                        residuals=residuals,
                        outgoing=outgoing,
                    ):
                        arguments = {
                            "hidden_states": incoming,
                            "temb": embedding,
                            "res_hidden_states_tuple": residuals,
                            "upsample_size": None,
                        }
                        if getattr(block, "has_cross_attention", False):
                            arguments.update(
                                encoder_hidden_states=prompt,
                                cross_attention_kwargs=None,
                                attention_mask=None,
                                encoder_attention_mask=None,
                            )
                        outgoing.copy_(block(**arguments))

                    capture(f"up{number}", up, (outgoing,), (f"up_blocks.{number}",))
                if skips or (height, width) != tuple(latents.shape[-2:]):
                    _refuse("SDXL skip topology did not close at the original latent geometry")
                final_hidden = hidden

                def sample():
                    value = final_hidden
                    if unet.conv_norm_out is not None:
                        value = unet.conv_act(unet.conv_norm_out(value))
                    noise = unet.conv_out(value)
                    if guidance > 1:
                        uncond, cond = noise.chunk(2)
                        noise = uncond + guidance * (cond - uncond)
                    sigma = euler.sigmas.index_select(0, self.index).reshape(())
                    inverse_sigma = euler.inverse_sigmas.index_select(0, self.index).reshape(())
                    delta = euler.deltas.index_select(0, self.index).reshape(())
                    value = self.latent.float()
                    original = value - (sigma * noise.float()).to(noise.dtype)
                    derivative = (value - original) * inverse_sigma
                    self.latent.copy_((value + derivative * delta).to(noise.dtype))
                    self.index.add_(1)

                capture("sampler", sample, (self.latent, self.index), ())
                if _attach_sequence:
                    self.plan = (
                        PreparedSequence(
                            self.definitions,
                            outputs=(self.latent,),
                            repeat=(1, len(self.definitions), self.steps),
                        )
                        if recording is None
                        else recording.finish(
                            outputs=(self.latent,), repeat=(1, len(self.definitions), self.steps)
                        )
                    )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.plan is not None:
            self.plan.close()
        for definition in self.definitions:
            definition.close()

    def facts(self) -> dict[str, Any]:
        return {
            "shared_pool": self.shared_pool,
            "steps": self.steps,
            "stages": len(self.definitions),
            "boundary_bytes": sum(
                value.numel() * value.element_size() for value in self.boundaries
            ),
            "python_body_calls": dict(self.calls),
            "capture_memory": list(self.capture_memory),
            "scope": "plain SDXL denoising interval; TE and VAE remain separate phases",
        }


def stream_euler(*, ctx: Any, tel: Any, **parameters: Any) -> torch.Tensor | None:
    """Try the bounded native interval; only preparation refusal permits eager fallback."""
    try:
        program = StreamedEuler(**parameters)
    except CapabilityError as error:
        if error.code not in ("prepared_graph_ineligible", "prepared_load_unavailable"):
            raise
        tel.log("eager denoise", reason=str(error))
        return None
    except torch.OutOfMemoryError:
        tel.log("eager denoise", reason="streamed graph preparation exceeded capacity")
        return None
    try:
        execution = ctx.plan(program.plan)
        observed = -1
        while True:
            ctx.raise_if_cancelled()
            progress = execution.poll()
            completed = execution.facts()["sequence"]["completed_passes"]
            if completed != observed:
                observed = completed
                fraction = completed / program.steps
                tel.progress(fraction, stage="denoise", overall_fraction=0.10 + 0.80 * fraction)
            if progress.done:
                break
            time.sleep(0.05)
        (result,) = execution.wait()
        facts = execution.facts()
        if facts["sequence"]["canceled"] or facts["sequence"]["completed_passes"] != program.steps:
            raise OutputError("streamed denoising stopped early", code="incomplete_denoise")
        tel.metric("request_path.denoise_native_steps", float(program.steps))
        tel.metric(
            "request_path.denoise_native_load_bytes",
            float(sum(row["read_bytes"] for row in facts["stage_inputs"])),
        )
        tel.metric("request_path.denoise_native_launches", float(facts["host_graph_launches"]))
        tel.metric("request_path.denoise_shared_pool", float(program.shared_pool))
        tel.log("native streamed denoise", **program.facts())
        return result
    finally:
        # Checked close joins every accepted source/graph before the VAE can use capacity.
        program.close()
