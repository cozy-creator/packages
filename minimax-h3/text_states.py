"""`encode_text`: the conditioning H3 computes for prompts, from any lane, as a file.

What `fl2va` and `fl2va_turbo` hand the DiT for a text-only prompt is the text encoder's
pre-norm `hidden_states[50]`. This returns it, and any earlier hidden state, so two lanes
(bf16 and fp8, say) can be compared on the encoder alone, without a sampler re-rolling the
picture. It runs the lane's real `condition_text`: its stored weights, kernels and activation
quantization.
"""

from __future__ import annotations

from typing import Annotated, Any

import msgspec
import torch
from cozy_runtime.author import FileBound, Context, File, Outputs, Telemetry
from safetensors.torch import save

from minimax_h3.model import H3Model
from minimax_h3.official import frames_for
from story import encoder_prompt

#: H3 conditions on hidden_states[50] of its 50 retained decoder layers.
FINAL = 50


class EncodeTextInput(msgspec.Struct, kw_only=True):
    """Prompts exactly as `fl2va` receives them, and the hidden states to return, 1-50; empty
    means 50 alone, the state H3 conditions on."""

    prompts: Annotated[list[str], msgspec.Meta(min_length=1, max_length=64)]
    hidden_states: list[Annotated[int, msgspec.Meta(ge=1, le=FINAL)]] = []


class EncodeTextOutput(msgspec.Struct):
    """`states` is safetensors: `{i}.hidden_states.{k}` bf16 `[tokens, 5120]` and
    `{i}.token_ids` int64 per prompt `i`."""

    states: Annotated[
        File, FileBound(max_bytes=1 << 30, media_types=("application/octet-stream",))
    ]
    tokens: list[int]


class _Capture:
    """Forward hooks on the conditioner's decoder layers and on its token input."""

    def __init__(self, conditioner: Any, wanted: list[int]) -> None:
        self.states: dict[int, torch.Tensor] = {}
        self.token_ids: torch.Tensor | None = None
        layers = conditioner.model.language_model.layers
        self.handles = [layers[k - 1].register_forward_hook(self._keep(k)) for k in wanted]
        self.handles.append(
            conditioner.model.register_forward_pre_hook(self._ids, with_kwargs=True)
        )

    def _keep(self, index: int) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            self.states[index] = value.detach()[0].to("cpu", torch.bfloat16)

        return hook

    def _ids(self, _module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        self.token_ids = kwargs["input_ids"].detach()[0].to("cpu", torch.int64)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def encode_text(
    ctx: Context, payload: EncodeTextInput, model: H3Model, out: Outputs, tel: Telemetry
) -> EncodeTextOutput:
    ctx.plan("text_encoder")
    conditioner = model.pipe.components["text_encoder"]
    wanted = sorted(set(payload.hidden_states)) or [FINAL]
    tensors: dict[str, torch.Tensor] = {}
    tokens: list[int] = []
    for index, prompt in enumerate(payload.prompts):
        ctx.raise_if_cancelled()
        view = model.for_request(ctx, seed=0)
        state = model.pipe.start_fl2va(
            prompt=encoder_prompt("fl2va", prompt),
            first_frame=None,
            last_frame=None,
            generator=model.pipe.generator(view.generator),
            steps=model.pipe.plans["fl2va"].schedule().transformer_evaluations,
            frames=frames_for(5),
            task="fl2va",
        )
        capture = _Capture(conditioner, wanted)
        try:
            model.condition_text("fl2va", state)
        finally:
            capture.close()
        if capture.token_ids is None or set(capture.states) != set(wanted):
            raise RuntimeError(
                "the text encoder ran outside this process (a multi-GPU group places it on "
                "another GPU); run encode_text on one GPU"
            )
        tensors[f"{index}.token_ids"] = capture.token_ids
        for k, value in capture.states.items():
            tensors[f"{index}.hidden_states.{k}"] = value.contiguous()
        tokens.append(int(capture.token_ids.numel()))
    data = save(tensors, metadata={"prompts": str(len(payload.prompts))})
    return EncodeTextOutput(out.save_bytes(data, media_type="application/octet-stream"), tokens)
