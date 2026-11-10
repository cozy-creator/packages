"""Exact 50-layer Qwen3-VL conditioner used by MiniMax-H3."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


_SOURCE_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
_SOURCE_LAYERS = 64
_RETAINED_LAYERS = 50


def release_position_cache(conditioner: Qwen3VLForConditionalGeneration) -> RemovableHandle:
    """Release Qwen's autoregressive position cache after every forward of `conditioner`.

    Diffusers requests hidden states with ``use_cache=False``. Qwen3VLModel still stores
    ``rope_deltas`` on its first image input; no later H3 call consumes it. Its one int64
    allocation kept 512 CUDA bytes after the second long-form shot, correctly tripping
    Runtime's strict ledger and rebuilding the entire group. A hook rather than a caller's
    scope, so it runs wherever the conditioner runs, including a group rank that hosts it,
    and on a failing forward too.
    """
    return conditioner.model.register_forward_hook(_forget_rope_deltas, always_call=True)


def _forget_rope_deltas(module: torch.nn.Module, _args: tuple[object, ...], _output: object) -> None:
    module.rope_deltas = None  # type: ignore[assignment]  # Qwen3VLModel's cache attribute, not a tensor slot


class FinalHiddenState:
    """The conditioner as Diffusers' H3 text block calls it, computing one hidden state.

    The block reads ``hidden_states[50]`` of a stack truncated to exactly 50 layers with an
    identity final norm, and transformers ties the last captured state to
    ``last_hidden_state``: the same tensor. Asking for all 51 kept ~1.5 GB of activations
    alive and, for a conditioner hosted on another rank, sent them all back.
    """

    def __init__(self, conditioner: Qwen3VLForConditionalGeneration, device: torch.device) -> None:
        self._conditioner = conditioner
        self._device = device

    def __getattr__(self, name: str) -> object:
        return getattr(self._conditioner, name)

    @property
    def device(self) -> torch.device:
        """Where the call's inputs and result live: the conditioner's device, or the model's
        (the Runtime's) when another GPU of the group hosts it and the local copy is a shell
        without bytes."""
        device = self._conditioner.device
        return self._device if device.type == "meta" else device

    def model(self, *, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        outputs = self._conditioner.model(input_ids=input_ids, **{**kwargs, "output_hidden_states": False})
        return SimpleNamespace(hidden_states={_RETAINED_LAYERS: outputs.last_hidden_state})


class PresentationLength(FinalHiddenState):
    """The conditioner as the text block calls it, running nothing: the block's own
    tokenization of the presentation is the whole answer an early length reader needs, so the
    call returns a shape on `meta` and no weight is touched (h3a-087)."""

    _hf_hook = None

    @property
    def device(self) -> torch.device:
        return torch.device("meta")

    def model(self, *, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            hidden_states={_RETAINED_LAYERS: torch.empty((*input_ids.shape, 1), device="meta")}
        )


def text_conditioner_config() -> dict[str, object]:
    """The closed artifact extension, shared by fixtures and production publishers."""
    return {
        "source_architecture": _SOURCE_ARCHITECTURE,
        "retained_decoder_layers": _RETAINED_LAYERS,
        "conditioning_hidden_state": _RETAINED_LAYERS,
        "output": "pre_norm",
        "language_model_head": False,
    }


def build_text_conditioner(config: Mapping[str, object]) -> Qwen3VLForConditionalGeneration:
    """Build the upstream surface, then remove computation after hidden state 50."""
    source: dict[str, Any] = dict(config)  # type: ignore[explicit-any]  # **config into Transformers' typed keywords
    extension = source.pop("cozy_h3", None)
    if extension != text_conditioner_config():
        raise ValueError(
            "artifact config 'text_encoder.cozy_h3' does not declare the exact H3 "
            "50-layer pre-norm conditioner",
        )
    if source.get("architectures") != [_SOURCE_ARCHITECTURE]:
        raise ValueError(
            "text conditioner source architecture is not Qwen3VLForConditionalGeneration",
        )
    text = source.get("text_config")
    if not isinstance(text, Mapping) or text.get("num_hidden_layers") != _SOURCE_LAYERS:
        raise ValueError(
            "text conditioner source must declare the exact 64-layer Qwen architecture",
        )

    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**source))  # type: ignore[no-untyped-call]  # Transformers' __init__ is untyped
    language: Qwen3VLTextModel = model.model.language_model
    if len(language.layers) != _SOURCE_LAYERS:
        raise ValueError(
            "constructed Qwen language stack disagrees with its source config",
        )
    language.layers = torch.nn.ModuleList(list(language.layers[:_RETAINED_LAYERS]))
    language.norm = torch.nn.Identity()  # type: ignore[assignment]  # the final norm, removed
    model.lm_head = torch.nn.Identity()  # type: ignore[assignment]  # the head, never built
    # Match from_pretrained(dtype=BF16): weights are BF16, while config-derived
    # rotary buffers retain their constructor precision. A whole-model .to(BF16)
    # rounds both text and vision frequencies before their FP32 forward math.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    model.eval()  # type: ignore[no-untyped-call]  # Transformers' eval is untyped
    _validate_census(model, language)
    release_position_cache(model)
    return model


def _validate_census(model: Qwen3VLForConditionalGeneration, language: Qwen3VLTextModel) -> None:
    """The three structural facts the truncation is FOR.

    Not a tensor census. Counting state_dict entries (902 total, 351 vision, 551
    language) and sweeping every dtype pinned a number that a transformers patch bump can
    change without changing behaviour — one rotary buffer becoming persistent would make
    every H3 worker refuse to load.
    """
    bad = [
        name
        for name, ok in (
            ("retained_decoder_layers", len(language.layers) == _RETAINED_LAYERS),
            ("output", isinstance(language.norm, torch.nn.Identity)),
            ("language_model_head", isinstance(model.lm_head, torch.nn.Identity)),
        )
        if not ok
    ]
    if bad:
        raise ValueError(
            "H3 text conditioner is not the truncated 50-layer pre-norm headless stack",
        )
