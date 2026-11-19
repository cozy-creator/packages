"""A dedicated upscale network, spandrel's architecture with its construction and weights the Runtime's.

The network is built from what its checkpoint stores, as spandrel's own loader reads a state dict (spandrel 0.4.2
`architectures/DAT/__init__.py`), here from the shapes the Runtime's `Config` answers for keys asked by name (`tensor_shape`):
it lists none (tracker xs-215, packages 13), so spandrel's detection, which walks the keys, can't run yet, and DAT, the
architecture of every default (`upscale_models.NETWORKS`), is the one built. spandrel's DAT reads its window from its
`rpe_biases` values; here it comes from two buffers' shapes and is checked against those values at the first pass.
"""

from __future__ import annotations

import math

import torch
from cozy_runtime.author import Config, Loader, Model, Serving, UnsupportedInput, uses_components
from PIL import Image
from spandrel.architectures.DAT import DAT
from torch import nn
from torch.nn import functional as F

#: An uploaded network's one component (a lone safetensors file uploaded as `file.safetensors:model`).
COMPONENT = "model"
#: A tile's largest side in input pixels and its overlap, as ComfyUI's Upscale Image (using Model) tiles: a tile's working
#: set, not the picture's, sets the memory a pass needs. Tiles are even (`_tiles`), whole multiples of DAT's window `ALIGN`.
TILE, OVERLAP, ALIGN = 512, 32, 32
#: Unknown: a config with no tensor shapes (config-only construction).
_UNKNOWN = (-1,)


def _shape(config: Config, key: str) -> tuple[int, ...] | None:
    """`key`'s stored shape, or None when the checkpoint has no such tensor."""
    try:
        shape = config.tensor_shape(f"{COMPONENT}.{key}", default=_UNKNOWN)
    except Exception as exc:  # the Runtime's ConformanceError, which `cozy_runtime.author` doesn't export
        if getattr(exc, "code", None) == "config_tensor_missing":
            return None
        raise
    if shape == _UNKNOWN:
        raise UnsupportedInput("an upscale network is built from its checkpoint's tensors", code="artifact_config")
    return shape


def _count(config: Config, key: str) -> int:
    """How many of `key.format(n)` the checkpoint stores, n = 0, 1, …"""
    n = 0
    while _shape(config, key.format(n)) is not None:
        n += 1
    return n


def _window(config: Config) -> tuple[int, int]:
    """DAT's split size (h, w), h ≤ w: `rpe_biases` holds (2h - 1)(2w - 1) offsets and `relative_position_index` is
    hw square, so h + w and hw are known and h, w are the roots of x² - (h + w)x + hw. Which is first is in the
    offsets' values, not their shapes: every published DAT runs the shorter first ([8, 16], [8, 32]); `check` holds it."""
    attn = "layers.0.blocks.0.attn.attns.0"
    offsets, index = _shape(config, f"{attn}.rpe_biases"), _shape(config, f"{attn}.relative_position_index")
    if offsets is None or index is None:
        raise UnsupportedInput("a DAT network without position biases isn't read here", code="artifact_config")
    product, total = index[0], (4 * index[0] + 1 - offsets[0]) // 2
    root = math.isqrt(total * total - 4 * product)
    h, w = (total - root) // 2, (total + root) // 2
    if h * w != product or (2 * h - 1) * (2 * w - 1) != offsets[0]:
        raise UnsupportedInput(f"DAT's window from {offsets} and {index} isn't whole", code="artifact_config")
    return h, w


class NetworkGraph:
    """The checkpoint's DAT, with spandrel's `DATArch.load` hyperparameters read from its shapes."""

    def __init__(self, config: Config) -> None:
        first = _shape(config, "conv_first.weight")
        if first is None or _shape(config, "before_RG.1.weight") is None:
            raise UnsupportedInput(
                "not a DAT network: the checkpoint has no conv_first and before_RG (DAT is the one architecture read here)",
                code="artifact_config",
            )
        embed_dim, in_chans = first[0], first[1]
        layers = _count(config, "layers.{}.blocks.0.norm1.weight")
        depth = [_count(config, f"layers.{i}.blocks.{{}}.norm1.weight") for i in range(layers)]
        heads = []
        for i in range(layers):
            if depth[i] >= 2:
                heads.append(_require(config, f"layers.{i}.blocks.1.attn.temperature")[0])
            else:
                heads.append(_require(config, f"layers.{i}.blocks.0.attn.attns.0.pos.pos3.2.weight")[0] * 2)
        direct = _shape(config, "conv_last.weight") is None
        if direct:
            up = _require(config, "upsample.0.weight")
            scale = math.isqrt(up[0] // in_chans)
        else:
            scale = 1
            for i in range(0, 10, 2):
                if (step := _shape(config, f"upsample.{i}.weight")) is None:
                    break
                scale *= math.isqrt(step[0] // step[1])
        mask = _shape(config, "layers.0.blocks.2.attn.attn_mask_0")
        self.window = _window(config)
        self.network = DAT(
            img_size=math.isqrt(mask[0] * mask[1]) if mask else 64,
            in_chans=in_chans,
            embed_dim=embed_dim,
            split_size=list(self.window),
            depth=depth,
            num_heads=heads,
            expansion_factor=_require(config, "layers.0.blocks.0.ffn.fc1.weight")[0] / embed_dim,
            qkv_bias=_shape(config, "layers.0.blocks.0.attn.qkv.bias") is not None,
            upscale=scale,
            img_range=1.0,
            resi_connection="1conv" if _shape(config, "conv_after_body.weight") is not None else "3conv",
            upsampler="pixelshuffledirect" if direct else "pixelshuffle",
        ).eval()
        self.scale = scale
        self.components: dict[str, nn.Module] = {COMPONENT: self.network}


def _require(config: Config, key: str) -> tuple[int, ...]:
    if (shape := _shape(config, key)) is None:
        raise UnsupportedInput(f"the DAT network has no {key}", code="artifact_config")
    return shape


class UpscaleNetwork(Model[NetworkGraph]):
    """One pass multiplies a picture's sides by the network's `scale`, in overlapping tiles blended linearly."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: NetworkGraph

    @property
    def scale(self) -> int:
        return self.graph.scale

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(NetworkGraph)

    def _check(self) -> None:
        """The window `_window` chose against the stored offsets' last row, (h - 1, w - 1): a transposed one would
        cut every attention window across the grain it was trained on."""
        attn = self.graph.network.get_submodule("layers.0.blocks.0.attn.attns.0")
        last = tuple(int(v) + 1 for v in attn.get_buffer("rpe_biases")[-1].tolist())
        if last != self.graph.window:
            raise UnsupportedInput(
                f"this DAT network's window is {list(last)}, which isn't read here (only the shorter side first)",
                code="artifact_config",
            )

    @uses_components(COMPONENT)
    def upscale(self, picture: Image.Image) -> Image.Image:
        """`picture` (RGB) at `scale` times its sides."""
        self._check()
        network = self.graph.network
        weight = next(network.parameters())
        device, dtype = weight.device, weight.dtype
        pixels = torch.frombuffer(bytearray(picture.tobytes()), dtype=torch.uint8)
        x = pixels.view(picture.height, picture.width, 3).permute(2, 0, 1).unsqueeze(0)
        out = _tiled(network, x.to(device=device, dtype=dtype) / 255, self.scale)
        rgb = out.clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).contiguous().cpu()
        return Image.frombytes("RGB", (rgb.shape[1], rgb.shape[0]), rgb.numpy().tobytes())


@torch.inference_mode()
def _tiled(network: nn.Module, x: torch.Tensor, scale: int) -> torch.Tensor:
    """`network(x)` over tiles of at most `TILE` input pixels a side that overlap by `OVERLAP`, each blended with a
    linear ramp across the overlap; a side under 16 pixels (DAT's least) is reflected out first and cropped after."""
    _, _, height, width = x.shape
    pad_h, pad_w = max(0, 16 - height), max(0, 16 - width)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    _, _, h, w = x.shape
    if h <= TILE and w <= TILE:
        out: torch.Tensor = network(x).float()
    else:
        out = torch.zeros(1, 3, h * scale, w * scale, device=x.device)
        total = torch.zeros(1, 1, h * scale, w * scale, device=x.device)
        (rows, th), (cols, tw) = _tiles(h), _tiles(w)
        for top in rows:
            for left in cols:
                tile = x[:, :, top : top + th, left : left + tw]
                made: torch.Tensor = network(tile).float()
                ramp = _ramp(made.shape[2], made.shape[3], top > 0, left > 0, top + th < h, left + tw < w, scale)
                ramp = ramp.to(made.device)
                out[:, :, top * scale : top * scale + made.shape[2], left * scale : left * scale + made.shape[3]] += made * ramp
                total[:, :, top * scale : top * scale + made.shape[2], left * scale : left * scale + made.shape[3]] += ramp
        out /= total
    return out[:, :, : height * scale, : width * scale]


def _tiles(side: int) -> tuple[list[int], int]:
    """The fewest tiles of at most `TILE` that cover `side` overlapping by `OVERLAP`, all one size (a multiple of `ALIGN`), their
    origins spread evenly, the last flush with the end. A 1024 px side takes three of 384 (1152 px of work), not 512s at 0, 480
    and 512 (1536 px: 2.25 times the picture's pixels at 2x, se-151)."""
    if side <= TILE:
        return [0], side
    count = -(-(side - OVERLAP) // (TILE - OVERLAP))
    span = -(-(side + (count - 1) * OVERLAP) // count)  # each tile's share, rounded up
    size = min(TILE, -(-span // ALIGN) * ALIGN)
    step = (side - size) / (count - 1)
    return [round(i * step) for i in range(count)], size


def _ramp(h: int, w: int, top: bool, left: bool, bottom: bool, right: bool, scale: int) -> torch.Tensor:
    """A tile's blend weight: 1 inside, rising linearly across each overlap it shares with a neighbour."""
    edge = OVERLAP * scale
    rows, cols = torch.ones(h), torch.ones(w)
    rise = torch.linspace(1 / (edge + 1), edge / (edge + 1), edge)
    if top:
        rows[:edge] = torch.minimum(rows[:edge], rise)
    if bottom:
        rows[-edge:] = torch.minimum(rows[-edge:], rise.flip(0))
    if left:
        cols[:edge] = torch.minimum(cols[:edge], rise)
    if right:
        cols[-edge:] = torch.minimum(cols[-edge:], rise.flip(0))
    return (rows[:, None] * cols[None, :]).view(1, 1, h, w)
