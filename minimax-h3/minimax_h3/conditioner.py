"""Exact 50-layer Qwen3-VL conditioner used by MiniMax-H3."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from cozy_runtime.author import ConformanceError

_SOURCE_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
_SOURCE_LAYERS = 64
_RETAINED_LAYERS = 50


def release_position_cache(conditioner: Any) -> Any:
    """Release Qwen's autoregressive position cache after every forward of `conditioner`.

    Diffusers requests hidden states with ``use_cache=False``. Qwen3VLModel still stores
    ``rope_deltas`` on its first image input; no later H3 call consumes it. Its one int64
    allocation kept 512 CUDA bytes after the second long-form shot, correctly tripping
    Runtime's strict ledger and rebuilding the entire group. A hook rather than a caller's
    scope, so it runs wherever the conditioner runs, including a group rank that hosts it,
    and on a failing forward too.
    """
    return conditioner.model.register_forward_hook(_forget_rope_deltas, always_call=True)


def _forget_rope_deltas(module: Any, _args: Any, _output: Any) -> None:
    module.rope_deltas = None


class FinalHiddenState:
    """The conditioner as Diffusers' H3 text block calls it, computing one hidden state.

    The block reads ``hidden_states[50]`` of a stack truncated to exactly 50 layers with an
    identity final norm, and transformers ties the last captured state to
    ``last_hidden_state``: the same tensor. Asking for all 51 kept ~1.5 GB of activations
    alive and, for a conditioner hosted on another rank, sent them all back.
    """

    def __init__(self, conditioner: Any, device: torch.device) -> None:
        self._conditioner = conditioner
        self._device = device

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conditioner, name)

    @property
    def device(self) -> Any:
        """Where the call's inputs and result live: the conditioner's device, or the model's
        (the Runtime's) when another GPU of the group hosts it and the local copy is a shell
        without bytes."""
        device = self._conditioner.device
        return self._device if device.type == "meta" else device

    def model(self, **kwargs: Any) -> Any:
        outputs = self._conditioner.model(**{**kwargs, "output_hidden_states": False})
        return SimpleNamespace(hidden_states={_RETAINED_LAYERS: outputs.last_hidden_state})


class PresentationLength(FinalHiddenState):
    """The conditioner as the text block calls it, running nothing: the block's own
    tokenization of the presentation is the whole answer an early length reader needs, so the
    call returns a shape on `meta` and no weight is touched (h3a-087)."""

    _hf_hook = None

    @property
    def device(self) -> Any:
        return torch.device("meta")

    def model(self, **kwargs: Any) -> Any:
        ids = kwargs["input_ids"]
        return SimpleNamespace(
            hidden_states={_RETAINED_LAYERS: torch.empty((*ids.shape, 1), device="meta")}
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


def build_text_conditioner(config: Mapping[str, object]) -> Any:
    """Build the upstream surface, then remove computation after hidden state 50."""
    source = dict(config)
    extension = source.pop("cozy_h3", None)
    if extension != text_conditioner_config():
        raise ConformanceError(
            "artifact config 'text_encoder.cozy_h3' does not declare the exact H3 "
            "50-layer pre-norm conditioner",
            code="artifact_config",
            fields=["text_encoder", "cozy_h3"],
        )
    if source.get("architectures") != [_SOURCE_ARCHITECTURE]:
        raise ConformanceError(
            "text conditioner source architecture is not Qwen3VLForConditionalGeneration",
            code="artifact_config",
            fields=["text_encoder", "architectures"],
        )
    text = source.get("text_config")
    if not isinstance(text, Mapping) or text.get("num_hidden_layers") != _SOURCE_LAYERS:
        raise ConformanceError(
            "text conditioner source must declare the exact 64-layer Qwen architecture",
            code="artifact_config",
            fields=["text_encoder", "text_config", "num_hidden_layers"],
        )

    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**source))
    language = cast(Any, model.model.language_model)
    if len(language.layers) != _SOURCE_LAYERS:
        raise ConformanceError(
            "constructed Qwen language stack disagrees with its source config",
            code="artifact_config",
            fields=["text_encoder", "text_config", "num_hidden_layers"],
        )
    language.layers = torch.nn.ModuleList(list(language.layers[:_RETAINED_LAYERS]))
    language.norm = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    # Match from_pretrained(dtype=BF16): weights are BF16, while config-derived
    # rotary buffers retain their constructor precision. A whole-model .to(BF16)
    # rounds both text and vision frequencies before their FP32 forward math.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    model.eval()
    _validate_census(model, language, torch)
    release_position_cache(model)
    return model


def _validate_census(model: Any, language: Any, torch: Any) -> None:
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
        raise ConformanceError(
            "H3 text conditioner is not the truncated 50-layer pre-norm headless stack",
            code="artifact_config",
            fields=["text_encoder", *bad],
        )
