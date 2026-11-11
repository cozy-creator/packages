"""RelayVSR's two components and the calls a stream makes of them; Runtime owns weights and devices."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from cozy_runtime import rng
from cozy_runtime.accel import hold_dtype
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import Config, Loader, Model, uses_components

from . import kernels
from .flash import Condition, FlashVSR, History
from .wan import CHANNELS, Ring, WanVSR

#: Frames are padded to this (upstream's `prepare_lr`: Wan's latent and patch grid at 1/4 scale).
STRIDE = 8
#: TensorFS's `relayvsr` table's `config.json`.
ARCHITECTURE = "RelayVSR"


class Components(TypedDict):
    wan: WanVSR
    flash: FlashVSR


class Graph:
    components: Components

    def __init__(self, config: Config) -> None:
        if config.mapping().get("architecture") != ARCHITECTURE:
            raise ValueError(f"this checkpoint is {config.mapping().get('architecture')!r}, and RelayVSR runs the {ARCHITECTURE!r} one: leave `model` at its default")
        self.components = {"wan": WanVSR().eval().to(torch.bfloat16), "flash": FlashVSR().eval()}


def build(config: Config) -> Graph:
    return Graph(config)


def frame_tensor(rgb: bytes, width: int, height: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """An RGB8 frame as (1, 3, h, w) in [-1, 1], in `dtype` as upstream scales it, padded by
    replicating its edges to a multiple of `STRIDE`."""
    pixels = torch.frombuffer(bytearray(rgb), dtype=torch.uint8).view(height, width, 3)
    x = pixels.to(device, non_blocking=True).permute(2, 0, 1)[None].to(dtype).div(127.5).sub(1.0)
    return F.pad(x, (0, (-width) % STRIDE, 0, (-height) % STRIDE), mode="replicate")


class RelayModel(Model[Graph]):
    """Each call declares the component it uses: the 5B keyframe model and the 32M decoder are
    placed on their own, and the keyframes' K/V are Runtime-held state (`Ring`). The calls reach
    the components through their methods, never a root's forward: they read and write the stream's
    state, which a CUDA graph of a root would not see."""

    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(build)

    def execution_plan(self, plan: Plan) -> str:
        return kernels.install(plan)

    def hold(self) -> None:
        """The decoder serves in bf16 on CUDA, as upstream's default; elsewhere its stored F32.
        Called before any of its scopes opens: the Runtime casts the bytes when it next fills them."""
        flash = self.graph.components["flash"]
        if torch.device(str(self.device)).type == "cuda":
            hold_dtype(flash, torch.bfloat16)

    @property
    def dtype(self) -> torch.dtype:
        """The decoder's serving dtype (`hold`), which frames take."""
        return torch.bfloat16 if torch.device(str(self.device)).type == "cuda" else torch.float32

    @uses_components("wan")
    def keyframe(self, frame: torch.Tensor, ring: Ring, generator: torch.Generator) -> torch.Tensor:
        """A padded frame's keyframe latent (1, 48, h/4, w/4): Wan's one step from noise, given the
        frame bilinearly upsampled 4x and the keyframes before it."""
        wan = self.graph.components["wan"]
        device = torch.device(str(self.device))
        upsampled = F.interpolate(frame.to(device, torch.bfloat16), scale_factor=4, mode="bilinear", align_corners=False)
        _, _, h, w = upsampled.shape
        noise = rng.noise((frame.shape[0], CHANNELS, h // 16, w // 16), generator, device=device)
        with torch.inference_mode():
            return wan.keyframe(upsampled, noise, ring)

    @uses_components("flash")
    def condition(self, latent: torch.Tensor) -> Condition:
        """A keyframe latent primed into the K/V its windows attend to."""
        with torch.inference_mode():
            return self.graph.components["flash"].condition(latent.to(self.dtype))

    @uses_components("flash")
    def decode(self, frame: torch.Tensor, start: Condition, end: Condition, history: History) -> tuple[torch.Tensor, History]:
        """A padded frame's 4x frame in [-1, 1], between the keyframes `start` and `end`."""
        device = torch.device(str(self.device))
        with torch.inference_mode():
            return self.graph.components["flash"].decode(frame.to(device, self.dtype), start, end, history)
