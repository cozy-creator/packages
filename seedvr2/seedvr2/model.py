"""SeedVR2's upstream graph; Runtime owns all learned tensor loading and devices."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageOps
from cozy_runtime import rng
from cozy_runtime.author import Config, Cut, Loader, Model, ModelAssets, Serving, Telemetry, split, uses_components

from .color import Correction, correct
from .frames import extend
from .upstream.infer import VideoDiffusionInfer
from .upstream.models.dit.nablocks.mmsr_block import GROUP
from .upstream.models.dit.nadit import NaDiT
from .upstream.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

#: Upstream's configs (ByteDance-Seed/SeedVR e4de8c24) as the checkpoint carries them: the 7B's,
#: and the VAE's it inherits (`__inherit__`).
MAIN = "configs_7b/main.yaml"
VAE_CONFIG = "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
#: SeedVR2's one-step restoration: upstream's `inference_seedvr2_7b.py` runs main.yaml's diffusion
#: with these (`generation_loop`'s defaults).
INFERENCE = {"timesteps": {"sampling": {"steps": 1}}, "cfg": {"scale": 1.0, "rescale": 0.0}}

ARCHITECTURE = "SeedVR2-7B"
#: DiT tokens (2x2 latent px a frame) restored together at most: whole frames, 16 at 1080p, 9 at
#: 1440p, 4 at 2160p. Window attention spans a piece, so the pieces are part of the result and
#: come from the clip's shape alone (`split`'s `most`), never from free memory. Inside a piece,
#: attention runs by window groups and the MLP by tokens (both exact), so a piece fits 8 GB.
PIECE = 131072
#: A piece's activation peak, in 2-byte elements: during its MLP, the block's input, the
#: attention's residual sum and the MLP's input and output (4 DIM a token) beside one `GROUP`'s
#: projection and GELU (2 EXPAND DIM a token). `split`'s `holds` counts it in windows of the
#: piece's fp16 latent, `most` elements each.
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}


def _arithmetic(text: str) -> object:
    """Upstream's `eval` resolver over what its configs write (`6 * 3072`, `36 * [(4,3,3)]`): literals
    and + - * //, never code."""

    def value(node: ast.expr) -> object:
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](value(node.left), value(node.right))
        return ast.literal_eval(node)

    return value(ast.parse(text, mode="eval").body)


OmegaConf.register_new_resolver("eval", _arithmetic, replace=True)


def _plain(node: object) -> object:
    """OmegaConf's lists as the tuples upstream's code writes; mappings as dicts."""
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items() if k != "__object__"}
    if isinstance(node, list):
        return tuple(_plain(v) for v in node)
    return node


@dataclass(frozen=True)
class Settings:
    """What upstream's configs set: the DiT's and VAE's arguments, the VAE's latent scale and causal
    slicing, and the diffusion the restoration runs."""

    dit: dict[str, object]
    vae: dict[str, object]
    scale: float
    split_size: int
    memory_device: str
    diffusion: dict[str, object]

    @property
    def peak(self) -> int:
        """`PEAK` at this DiT's width and MLP expansion."""
        dim, expand = cast(int, self.dit["vid_dim"]), cast(int, self.dit["expand_ratio"])
        return int((4 * PIECE + 2 * expand * GROUP) * dim)


def settings(assets: ModelAssets) -> Settings:
    """The checkpoint's upstream configs, resolved as upstream's `load_config` does."""
    main = cast(DictConfig, OmegaConf.create(assets.read(MAIN).decode()))
    vae = cast(DictConfig, main.vae)
    model = cast(DictConfig, vae.model)
    inherited = model.pop("__inherit__")
    if inherited != VAE_CONFIG:
        raise ValueError(f"{MAIN}'s VAE inherits {inherited!r}, not {VAE_CONFIG!r}")
    vae.model = OmegaConf.merge(OmegaConf.create(assets.read(VAE_CONFIG).decode()), model)
    diffusion = OmegaConf.merge(main.diffusion, INFERENCE)
    held = cast(dict[str, object], OmegaConf.to_container(main, resolve=True))
    run = cast(dict[str, object], OmegaConf.to_container(diffusion, resolve=True))
    dit = cast(dict[str, object], cast(dict[str, object], held["dit"])["model"])
    upstream = cast(dict[str, object], held["vae"])
    timesteps = cast(dict[str, object], run["timesteps"])
    slicing = cast(dict[str, object], upstream["slicing"])
    return Settings(
        dit=cast(dict[str, object], _plain(dit)),
        vae=cast(dict[str, object], _plain(upstream["model"])),
        scale=float(cast(float, upstream["scaling_factor"])),
        split_size=cast(int, slicing["split_size"]),
        memory_device=cast(str, slicing["memory_device"]),
        diffusion={
            "schedule": run["schedule"],
            "sampler": run["sampler"],
            "timesteps": {"sampling": timesteps["sampling"], "transform": timesteps["transform"]},
            "cfg": run["cfg"],
        },
    )


class Conditioning(torch.nn.Module):
    positive: torch.Tensor
    negative: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("positive", torch.empty(58, 5120, dtype=torch.bfloat16))
        self.register_buffer("negative", torch.empty(64, 5120, dtype=torch.bfloat16))


class Encoded(Protocol):
    @property
    def latent(self) -> torch.Tensor: ...


class Decoded(Protocol):
    @property
    def sample(self) -> torch.Tensor: ...


class VideoVae(Protocol):
    """What the restoration calls of upstream's VAE wrapper (vendored, not type-checked)."""

    def encode(self, x: torch.Tensor, generator: torch.Generator) -> Encoded: ...

    def decode(self, z: torch.Tensor) -> Decoded: ...

    def set_causal_slicing(self, *, split_size: int | None, memory_device: str) -> None: ...


class Schedule(Protocol):
    def forward(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...


class Runner(Protocol):
    """What the restoration calls of upstream's `VideoDiffusionInfer` (vendored, not type-checked)."""

    dit: torch.nn.Module
    schedule: Schedule

    def configure_diffusion(self, device: torch.device) -> None: ...

    def timestep_transform(self, timesteps: torch.Tensor, latents_shapes: torch.Tensor) -> torch.Tensor: ...

    def get_condition(self, latent: torch.Tensor, latent_blur: torch.Tensor, task: str) -> torch.Tensor: ...

    def inference(
        self,
        noises: list[torch.Tensor],
        conditions: list[torch.Tensor],
        texts_pos: list[torch.Tensor],
        texts_neg: list[torch.Tensor],
    ) -> list[torch.Tensor]: ...


class Modules(TypedDict):
    dit: torch.nn.Module
    vae: VideoVae


class Components(Modules):
    conditioning: Conditioning


def modules(held: Settings) -> Modules:
    dit: torch.nn.Module = NaDiT(**held.dit).eval().half()
    vae: VideoVae = VideoAutoencoderKLWrapper(**held.vae).eval().half()
    return {"dit": dit, "vae": vae}


class Graph:
    components: Components
    settings: Settings

    def __init__(self, config: Config, held: Settings) -> None:
        if config.mapping().get("architecture") != ARCHITECTURE:
            raise ValueError(f"this checkpoint is {config.mapping().get('architecture')!r}, and SeedVR2 runs the {ARCHITECTURE!r} one: leave `model` at its default")
        self.settings = held
        self.components = {**modules(held), "conditioning": Conditioning()}


class RestorationModel(Model[Graph]):
    """Each phase declares the components it uses, so the Runtime places each on its own:
    the VAE's activations take the room the transformer's weights held while it ran."""

    serving = Serving(unloaded=("dit.blocks.*.attn.rope.rope.freqs",))

    graph: Graph

    def load(self, loader: Loader) -> None:
        held = settings(loader.assets)

        def build(config: Config) -> Graph:
            return Graph(config, held)

        self.graph = loader.construct(build)

    @uses_components("vae")
    def encode(self, pixels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """A window's pixels, (c, t, h, w) bytes or [-1, 1] in host memory, to its scaled latent,
        (t, h, w, c) on the device, sampled from `generator`. The VAE moves one slice of frames
        at a time."""
        latent = _vae(self.graph).encode(pixels[None], generator).latent[0]
        if latent.ndim == 3:  # upstream squeezes a single frame's time axis
            latent = latent.unsqueeze(1)
        return latent.movedim(0, -1).to(str(self.device)) * self.graph.settings.scale

    @uses_components("dit", "conditioning")
    def denoise(
        self, latent: torch.Tensor, latent_noise: float, generator: torch.Generator
    ) -> torch.Tensor:
        """The one-step restoration of a window's latent, conditioned on itself, from
        `generator`'s noise, in pieces along time from its shape alone."""
        runner = _runner(self.graph, torch.device(str(self.device)))
        conditioning = self.graph.components["conditioning"]

        def draw() -> torch.Tensor:
            return rng.noise(latent.shape, generator, dtype=latent.dtype, device=latent.device)

        noise = draw()
        if latent_noise:
            # Upstream's `_add_noise`, including its (h, w, c) shape for the shift.
            latent = runner.schedule.forward(
                latent,
                noise * 0.1 + draw() * 0.05,
                runner.timestep_transform(
                    torch.tensor([1000.0 * latent_noise], device=latent.device),
                    torch.tensor(latent.shape[1:], device=latent.device)[None],
                ),
            )
        condition = runner.get_condition(noise, latent_blur=latent, task="sr")

        def dit(frames: torch.Tensor) -> torch.Tensor:
            x, c = frames.split([noise.shape[-1], condition.shape[-1]], dim=-1)
            with torch.autocast(frames.device.type, dtype=torch.bfloat16):
                return runner.inference([x], [c], [conditioning.positive], [conditioning.negative])[0]

        x = torch.cat([noise, condition], dim=-1)
        most = PIECE * 4 * x.shape[-1]
        return split(dit, x, Cut(0), most=most, holds=-(-self.graph.settings.peak // most))

    @uses_components("vae")
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """A restored latent to pixels, (c, t, h, w) in about [-1, 1] in host memory, one slice
        of frames at a time, in the VAE's own fp16: bf16 autocast ran its norms in F32, 1.26x
        slower and 10 dB further from the F32 decode."""
        z = (latent / self.graph.settings.scale).movedim(-1, 0)[None].to("cpu", torch.float16)
        return _vae(self.graph).decode(z).sample[0]


def _runner(graph: Graph, device: torch.device) -> Runner:
    """Upstream's inference driver over the constructed components (configuration only)."""
    held = graph.settings
    factors = {k: held.vae[k] for k in ("temporal_downsample_factor", "spatial_downsample_factor")}
    cfg = OmegaConf.create({"vae": {"model": factors}, "diffusion": held.diffusion})
    runner: Runner = VideoDiffusionInfer(cfg)
    runner.dit = graph.components["dit"]
    runner.configure_diffusion(device)
    return runner


def _vae(graph: Graph) -> VideoVae:
    vae = graph.components["vae"]
    # main.yaml's slicing: one latent frame a step of the causal scan; the Runtime cuts each op's rows.
    vae.set_causal_slicing(split_size=graph.settings.split_size, memory_device=graph.settings.memory_device)
    return vae


def restore(
    model: RestorationModel,
    frames: list[Image.Image],
    width: int,
    height: int,
    seed: int,
    tel: Telemetry,
    *,
    correction: Correction,
    input_noise: float,
    latent_noise: float,
) -> list[Image.Image]:
    """One window: encode, denoise and decode, each its own component scope."""
    if not frames:
        raise ValueError("SeedVR2 restoration needs at least one frame")
    padded = extend(frames, (1 - len(frames)) % 4)
    canvas = ((width + 15) // 16 * 16, (height + 15) // 16 * 16)
    left, top = (canvas[0] - width) // 2, (canvas[1] - height) // 2
    border = (left, top, canvas[0] - width - left, canvas[1] - height - top)
    # (t, h, w, 3) bytes in host memory: the VAE takes them a slice at a time.
    source = torch.from_numpy(
        np.stack(
            [
                np.asarray(ImageOps.expand(f.resize((width, height), Image.Resampling.LANCZOS), border))
                for f in padded
            ]
        )
    )
    crop = (slice(top, top + height), slice(left, left + width))
    device = torch.device(str(model.device))
    # Every draw is the CPU stream's: a seed restores the same window on any device.
    generator = rng.generator(seed)
    with torch.inference_mode():
        pixels = source.permute(3, 0, 1, 2).contiguous()
        if input_noise:
            pixels = _signed(pixels, torch.float32)
            pixels = pixels.add_(rng.noise(pixels.shape, generator).mul_(0.025 * input_noise)).half()
        with tel.stage("SeedVR2 VAE encode"):
            latent = model.encode(pixels, generator)
        del pixels
        with tel.stage("SeedVR2 denoise"):
            latent = model.denoise(latent, latent_noise, generator)
        with tel.stage("SeedVR2 VAE decode"):
            sample = model.decode(latent)
        if sample.ndim == 3:
            sample = sample.unsqueeze(1)
        restored = []
        for index in range(len(frames)):
            frame = sample[:, index, crop[0], crop[1]].to(device).float().clamp(-1, 1)[None]
            reference = _signed(source[index, crop[0], crop[1]].to(device), torch.float32)
            frame = correct(frame, reference.permute(2, 0, 1)[None], correction)
            # As torchvision's `to_pil_image` rounds a float frame: down.
            image = frame[0].clamp(-1, 1).add(1).div(2).mul(255).to(torch.uint8)
            restored.append(Image.fromarray(image.permute(1, 2, 0).cpu().numpy()))
    return restored


def _signed(image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bytes to [-1, 1]."""
    return image.to(dtype).div(127.5).sub(1)
