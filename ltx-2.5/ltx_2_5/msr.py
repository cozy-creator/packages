"""Licon MSR V2 for LTX-2.5 (`LiconStudio/LTX-2.5-Multiple-Subject-Reference@aa19b48f`, Apache-2.0, mirrored as
`liconstudio/multiple-subject-reference`): up to five pictures and up to two of their voices kept in the video. A
LoRA on both stages, and two learned slot embeddings (Fourier MLPs) that TensorFS keeps beside its factors as PEFT's
`modules_to_save`: each picture is a still of 33 frames, encoded at the stage's size with its slot's embedding added,
appended as clean tokens at pic1-based negative frame offsets; each voice is clean audio tokens with its slot's
embedding, right-aligned in its picture's 5-second window before the clip. Ported from ComfyUI-LTX2.5-MSR v1.2.2
(nodes.py, Apache-2.0)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
from cozy_runtime.accel import lora, sequence_parallel
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from PIL import Image
from torch import nn
from torch.nn import functional

from . import adapter
from .model import Subjects

PICTURES = "reference_slot_embedding"
R = TypeVar("R")
VOICES = "reference_audio_slot_embedding"


class SlotEmbedding(nn.Module):
    """A slot's learned identity from its number: sin/cos features of `slot / 16`, then a two-layer SiLU MLP, in F32
    (nodes.py `_slot_embedding`)."""

    def __init__(self, frequencies: int, hidden: int, dim: int) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.zeros(frequencies), requires_grad=False)
        self.net = nn.Sequential(nn.Linear(1 + 2 * frequencies, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, slot: int) -> torch.Tensor:
        frequencies = self.frequencies.float()
        scaled = torch.tensor(slot / 16.0, device=frequencies.device)
        phases = scaled * frequencies
        features = torch.cat((scaled.reshape(1), torch.sin(phases), torch.cos(phases)))
        first, last = self.net[0], self.net[2]
        assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
        hidden = functional.silu(functional.linear(features, first.weight.float(), first.bias.float()))
        return functional.linear(hidden, last.weight.float(), last.bias.float())


class Graph(adapter.Graph):
    def saved(self, config: Config) -> dict[str, nn.Module]:
        made: dict[str, nn.Module] = {}
        for name in (PICTURES, VOICES):
            key = f"{adapter.COMPONENT}.{name}"
            frequencies, = config.tensor_shape(f"{key}.frequencies", default=(0,))
            hidden, _ = config.tensor_shape(f"{key}.net.0.weight", default=(0, 0))
            dim, _ = config.tensor_shape(f"{key}.net.2.weight", default=(0, 0))
            if frequencies:
                made[name] = SlotEmbedding(frequencies, hidden, dim)
        return made

    def number(self, key: str, default: float) -> float:
        return float(str(self.settings.get(key, default)))


@sequence_parallel(degrees=(2, 4, 8))
class MSR(Model[Graph]):
    """Replicated beside the sequence-sharded DiT, on whose sites its LoRA arms for both stages."""

    serving = Serving(encoded_leaves=False, min_bits=16)
    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)

    @uses_components(adapter.COMPONENT)
    def subjects(self, pictures: Sequence[Image.Image], voices: Sequence[tuple[int, torch.Tensor]]) -> Subjects:
        """The pictures in slot order, and the voices as (their picture's slot, from 1; latents), with their slots'
        embeddings."""
        components = self.graph.components[adapter.COMPONENT]
        to_picture, to_voice = components.get_submodule(PICTURES), components.get_submodule(VOICES)
        with torch.inference_mode():
            return Subjects(
                [(picture, to_picture(slot).cpu()) for slot, picture in enumerate(pictures, start=1)],
                [(slot, latents, to_voice(slot).cpu()) for slot, latents in voices],
                slot_s=self.graph.number("reference_audio_slot_duration_seconds", 5.0),
                margin_s=self.graph.number("reference_audio_end_margin_seconds", 0.04))

    @uses_components(adapter.COMPONENT)
    def applied(self, sites: lora.Sites, body: Callable[[], R]) -> R:
        """`body` with the LoRA armed on the DiT's `sites` (its factors carry no alpha: scale 1)."""
        with sites.applied(self.graph.factors):
            return body()
