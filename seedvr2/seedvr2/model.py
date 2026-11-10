"""SeedVR2's upstream graph; Runtime owns all learned tensor loading and devices."""

from __future__ import annotations

from typing import Any, cast
from collections.abc import Callable


import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from cozy_runtime import rng
from cozy_runtime.author import Config, Cut, Loader, Model, Serving, Telemetry, split, uses_components

from .color import Correction, correct
from .frames import extend
from .upstream.infer import VideoDiffusionInfer
from .upstream.models.dit.nablocks.mmsr_block import GROUP
from .upstream.models.dit.nadit import NaDiT
from .upstream.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

# SeedVR's configs_7b/main.yaml: separate text/video weights throughout.
DIM, EXPAND = 3072, 4
DIT = dict(
    vid_in_channels=33,
    vid_out_channels=16,
    vid_dim=DIM,
    txt_in_dim=5120,
    txt_dim=DIM,
    emb_dim=18432,
    heads=24,
    head_dim=128,
    expand_ratio=EXPAND,
    norm="rms",
    norm_eps=1e-5,
    ada="single",
    qk_bias=False,
    qk_norm="rms",
    qk_rope=True,
    patch_size=(1, 2, 2),
    num_layers=36,
    mlp_type="normal",
    shared_mlp=False,
    shared_qkv=False,
    block_type=["mmdit_sr"] * 36,
    window=[(4, 3, 3)] * 36,
    window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 18,
)
VAE = dict(
    act_fn="silu",
    block_out_channels=(128, 256, 512, 512),
    down_block_types=("DownEncoderBlock3D",) * 4,
    up_block_types=("UpDecoderBlock3D",) * 4,
    in_channels=3,
    out_channels=3,
    latent_channels=16,
    layers_per_block=2,
    norm_num_groups=32,
    slicing_sample_min_size=4,
    temporal_scale_num=2,
    inflation_mode="pad",
    spatial_downsample_factor=8,
    temporal_downsample_factor=4,
    use_quant_conv=False,
    use_post_quant_conv=False,
    freeze_encoder=False,
)

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
PEAK = (4 * PIECE + 2 * EXPAND * GROUP) * DIM
#: SeedVR's configs_7b: the VAE's latent scaling factor (no shift).
SCALE = 0.9152


class Conditioning(torch.nn.Module):
    positive: torch.Tensor
    negative: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("positive", torch.empty(58, 5120, dtype=torch.bfloat16))
        self.register_buffer("negative", torch.empty(64, 5120, dtype=torch.bfloat16))


def modules() -> dict[str, torch.nn.Module]:
    return {
        "dit": cast(Callable[..., torch.nn.Module], NaDiT)(**DIT).eval().half(),
        "vae": cast(Callable[..., torch.nn.Module], VideoAutoencoderKLWrapper)(**VAE).eval().half(),
    }


class Graph:
    def __init__(self, config: Config) -> None:
        if config.mapping().get("architecture") != ARCHITECTURE:
            raise ValueError(f"this checkpoint is {config.mapping().get('architecture')!r}, and SeedVR2 runs the {ARCHITECTURE!r} one: leave `model` at its default")
        self.components = {**modules(), "conditioning": Conditioning()}


def build(config: Config) -> Graph:
    return Graph(config)


class RestorationModel(Model[Graph]):
    """Each phase declares the components it uses, so the Runtime places each on its own:
    the VAE's activations take the room the transformer's weights held while it ran."""

    serving = Serving(unloaded=("dit.blocks.*.attn.rope.rope.freqs",))

    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(build)

    @uses_components("vae")
    def encode(self, pixels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """A window's pixels, (c, t, h, w) bytes or [-1, 1] in host memory, to its scaled latent,
        (t, h, w, c) on the device, sampled from `generator`. The VAE moves one slice of frames
        at a time."""
        latent = _vae(self.graph).encode(pixels[None], generator).latent[0]
        return latent.movedim(0, -1).to(self.device) * SCALE

    @uses_components("dit", "conditioning")
    def denoise(
        self, latent: torch.Tensor, latent_noise: float, generator: torch.Generator
    ) -> torch.Tensor:
        """The one-step restoration of a window's latent, conditioned on itself, from
        `generator`'s noise, in pieces along time from its shape alone."""
        runner = _runner(self.graph, self.device)
        conditioning = cast(Conditioning, self.graph.components["conditioning"])

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
        return split(dit, x, Cut(0), most=most, holds=-(-PEAK // most))

    @uses_components("vae")
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """A restored latent to pixels, (c, t, h, w) in about [-1, 1] in host memory, one slice
        of frames at a time, in the VAE's own fp16: bf16 autocast ran its norms in F32, 1.26x
        slower and 10 dB further from the F32 decode."""
        z = (latent / SCALE).movedim(-1, 0)[None].to("cpu", torch.float16)
        return _vae(self.graph).decode(z).sample[0]


def _runner(graph: Graph, device: torch.device) -> Any:
    """Upstream's inference driver over the constructed components (configuration only)."""
    cfg = OmegaConf.create(
        {
            "vae": {"model": {"temporal_downsample_factor": 4, "spatial_downsample_factor": 8}},
            "diffusion": {
                "schedule": {"type": "lerp", "T": 1000.0},
                "sampler": {"type": "euler", "prediction_type": "v_lerp"},
                "timesteps": {
                    "sampling": {"type": "uniform_trailing", "steps": 1},
                    "transform": True,
                },
                "cfg": {"scale": 1.0, "rescale": 0.0},
            },
        }
    )
    runner = cast(Any, VideoDiffusionInfer(cfg))
    runner.dit = graph.components["dit"]
    runner.configure_diffusion(device)
    return runner


def _vae(graph: Graph) -> Any:
    vae = cast(Any, graph.components["vae"])
    # One latent frame a step of the causal scan, its finest; the Runtime cuts each op's rows.
    vae.set_causal_slicing(split_size=4, memory_device="same")
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
    device = model.device
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
