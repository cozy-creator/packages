"""The cached Qwen interval as ordered block loads and GPU graphs.

The existing eager extract pass supplies each layer's prefix K/V. The remaining
steps use the installed model blocks and unchanged FlowMatch arithmetic; native
Runtime advances the block range and the explicit GPU timestep.
"""

from __future__ import annotations

import weakref

import torch
from cozy_runtime.author import CapabilityError
from cozy_runtime.compute import PreparedSequence

from ._prepared import _Denoise


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
