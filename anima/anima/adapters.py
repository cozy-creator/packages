"""Anima's adapters (se-127): LuciferTC's SigLIP2 IP-Adapter for mentioned pictures, Claquasse's pose control-LoRA
for poses and kohya's ControlNet-LLLite for the other control maps. Each is its author's file uploaded as it is,
configured by its own safetensors metadata.

The IP-Adapter follows its ComfyUI node (LuciferTC9527/ComfyUI-Anima_IP-Adapter `nodes.py`) and InvokeAI's port
(`backend/anima/ip_adapter.py`): SigLIP2's tokens of each picture, letterboxed on black at 512², scaled by its strength;
per block, attention from the block's own cross-attention query to `ip_k_proj`/`ip_v_proj` of those tokens, gated by
`adaln_ip` of the timestep embedding and added after the whole block, on the conditional rows only. Its bundled
LoRA applies on the cross-attention q/k/v/out only, as both references apply it.

LLLite follows kohya's `control_net_lllite_anima.py` (v2): the control map at the output's pixel size through
`conditioning1`, one token per DiT token; each target linear's input gains `up(silu(film(mid([cond + depth, silu(down
(x))]))))` times its strength, on every row.

The pose control follows Claquasse's ComfyUI node (`anima_control_lora`): the skeleton's VAE latent, unnormalized
as ComfyUI's `VAEEncode` gives it, through a zero-init `ControlEmbedder` (2×2 latent patches, 64 → 2048) added to
the patch embedding times its strength, on every row; its rank-16 LoRA on every DiT linear and the LLM adapter's
at 1, as `LoraLoaderModelOnly` loads it, armed on the model's `lora.Sites` while the pose steers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import TypeVar

import torch
import torch.nn.functional as F
from cozy_runtime.accel import lora
from cozy_runtime.author import Config, Loader, Model, Serving, UnsupportedInput, uses_components
from PIL import Image as PILImage
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel
from transformers.initialization import no_init_weights

T = TypeVar("T")
#: An as-is upload's one component.
COMPONENT = "model"
#: Anima Base's DiT: 28 blocks of width 2048, 16 heads of 128; its LLM adapter's blocks.
BLOCKS, WIDTH, HEADS = 28, 2048, 16
ADAPTER_BLOCKS = 6
#: The SigLIP2 input the IP-Adapter was trained on: a 512² letterbox.
SIGLIP_SIZE = 512


# ---------------------------------------------------------------- SigLIP2


class _Vision(nn.Module):
    """SigLIP2's vision tower under the name `SiglipModel` stores it by (`vision_model.`): its tokens."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.vision_model = SiglipVisionModel(config)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.vision_model(pixel_values=pixels, interpolate_pos_encoding=True).last_hidden_state
        return out


class EncoderGraph:
    def __init__(self, config: Config) -> None:
        vision = SiglipVisionConfig(**config.mapping()["vision_config"])
        with no_init_weights():
            encoder = _Vision(vision)
        stored = config.tensor_dtype(f"{COMPONENT}.vision_model.embeddings.patch_embedding.weight", default="f32")
        self.components: dict[str, nn.Module] = {
            COMPONENT: encoder.to(torch.float32 if stored == "f32" else torch.bfloat16).eval()
        }


def letterbox(picture: PILImage.Image, size: int = SIGLIP_SIZE) -> torch.Tensor:
    """The reference's preprocessing: the longest side to `size` (bilinear), centred on black, mean and std 0.5
    (not SiglipImageProcessor, which stretches). [3, size, size] f32."""
    picture = picture.convert("RGB")
    ratio = size / max(picture.size)
    width, height = max(1, round(picture.width * ratio)), max(1, round(picture.height * ratio))
    square = PILImage.new("RGB", (size, size))
    square.paste(picture.resize((width, height), PILImage.Resampling.BILINEAR), ((size - width) // 2, (size - height) // 2))
    pixels = torch.frombuffer(bytearray(square.tobytes()), dtype=torch.uint8).view(size, size, 3)
    return pixels.permute(2, 0, 1).float().div(127.5).sub(1.0)


class ImageEncoder(Model[EncoderGraph]):
    """google/siglip2-base-patch16-512's vision tower: 1024 tokens of 768 per picture."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: EncoderGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(EncoderGraph)

    @uses_components(COMPONENT)
    def tokens(self, pictures: Sequence[PILImage.Image]) -> torch.Tensor:
        """Each picture's `last_hidden_state`, [n, 1024, 768]."""
        encoder = self.graph.components[COMPONENT]
        weight = next(encoder.parameters())
        pixels = torch.stack([letterbox(p) for p in pictures]).to(torch.device(str(self.device)), weight.dtype)
        out: torch.Tensor = encoder(pixels)
        return out


# ---------------------------------------------------------------- IP-Adapter


class _IPBlock(nn.Module):
    def __init__(self, tokens: int, inner: int) -> None:
        super().__init__()
        self.ip_k_proj = nn.Linear(tokens, inner)
        self.ip_v_proj = nn.Linear(tokens, inner)
        self.adaln_ip = nn.Sequential(nn.SiLU(), nn.Linear(WIDTH, WIDTH))


class _Factors(nn.Module):
    """A PEFT site's `lora_A` [r, in] and `lora_B` [out, r]."""

    def __init__(self, inputs: int, outputs: int, rank: int) -> None:
        super().__init__()
        self.lora_A = nn.Linear(inputs, rank, bias=False)
        self.lora_B = nn.Linear(rank, outputs, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return out.to(x.dtype)


class _CrossLoRA(nn.Module):
    def __init__(self, text: int, rank: int) -> None:
        super().__init__()
        self.q_proj = _Factors(WIDTH, WIDTH, rank)
        self.k_proj = _Factors(text, WIDTH, rank)
        self.v_proj = _Factors(text, WIDTH, rank)
        self.output_proj = _Factors(WIDTH, WIDTH, rank)


class _Scoped(nn.Module):
    """A module holding one child, so the tree spells PEFT's `lora.base_model.model.…` names."""

    def __init__(self, name: str, child: nn.Module) -> None:
        super().__init__()
        self.add_module(name, child)


class IPGraph:
    def __init__(self, config: Config) -> None:
        said = config.mapping()
        for flag in ("ip_norm_keys", "ip_inject_before_mlp"):
            if str(said.get(flag, "False")).lower() == "true":
                raise UnsupportedInput(f"this IP-Adapter was trained with {flag}, a variant Anima does not run",
                                       code="ip_adapter_variant")
        inner, tokens = config.tensor_shape(f"{COMPONENT}.blocks.0.ip_k_proj.weight", default=(WIDTH, 768))
        config.tensor_shape(f"{COMPONENT}.blocks.{BLOCKS - 1}.ip_k_proj.weight", default=(inner, tokens))
        rank, text = config.tensor_shape(
            f"{COMPONENT}.lora.base_model.model.blocks.0.cross_attn.k_proj.lora_A.weight", default=(32, 1024)
        )
        root = nn.Module()
        self.blocks = [_IPBlock(tokens, inner) for _ in range(BLOCKS)]
        root.blocks = nn.ModuleList(self.blocks)
        self.crosses = [_CrossLoRA(text, rank) for _ in range(BLOCKS)]
        crosses = nn.ModuleList(_Scoped("cross_attn", cross) for cross in self.crosses)
        root.lora = _Scoped("base_model", _Scoped("model", _Scoped("blocks", crosses)))
        #: alpha over rank, as PEFT scales; the references apply alpha = rank.
        self.scale = float(said.get("lora_alpha", rank)) / rank
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(torch.bfloat16)}


class IPAdapter(Model[IPGraph]):
    """LuciferTC's Anima IP-Adapter: per-block image keys and values, a gate, and its cross-attention LoRA."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: IPGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(IPGraph)

    @uses_components(COMPONENT)
    def armed(self, body: Callable[[], T]) -> T:
        """`body` with this adapter's weights held: the denoise reads them inside the DiT's calls."""
        return body()


@dataclass(frozen=True)
class References:
    """One denoise's pictures for the IP-Adapter: their tokens, each scaled by its strength and concatenated."""

    adapter: IPGraph
    tokens: torch.Tensor  # [1, n·1024, 768]

    def cross(self, index: int) -> _CrossLoRA:
        return self.adapter.crosses[index]

    def term(self, index: int, query: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Block `index`'s image attention for `query` [b, heads, s, 128] (the block's normalized cross-attention
        query), gated by `adaln_ip` of the timestep embedding: [b, s, 2048]."""
        block = self.adapter.blocks[index]
        tokens = self.tokens.to(query.dtype)
        key = block.ip_k_proj(tokens).unflatten(2, (HEADS, -1)).transpose(1, 2)
        value = block.ip_v_proj(tokens).unflatten(2, (HEADS, -1)).transpose(1, 2)
        batch = query.shape[0]
        attended = F.scaled_dot_product_attention(query, key.expand(batch, -1, -1, -1), value.expand(batch, -1, -1, -1))
        gate = block.adaln_ip(timestep.to(query.dtype))
        if gate.ndim == 2:
            gate = gate.unsqueeze(1)
        out: torch.Tensor = attended.transpose(1, 2).flatten(2) * gate
        return out


# ---------------------------------------------------------------- ControlNet-LLLite

#: kohya's atomic targets: each names the linears of a block whose input it shifts.
TARGETS = {
    "self_attn_q_pre": ("self_attn_q_proj",),
    "self_attn_kv_pre": ("self_attn_k_proj", "self_attn_v_proj"),
    "cross_attn_q_pre": ("cross_attn_q_proj",),
    "mlp_fc1_pre": ("mlp_layer1",),
}
_PRESETS = {
    "self_attn_q": "self_attn_q_pre",
    "self_attn_qkv": "self_attn_q_pre,self_attn_kv_pre",
    "self_attn_qkv_cross_q": "self_attn_q_pre,self_attn_kv_pre,cross_attn_q_pre",
}


def _groups(channels: int) -> nn.GroupNorm:
    groups = 8
    while groups > 1 and channels % groups:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1, self.conv1 = _groups(channels), nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2, self.conv2 = _groups(channels), nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x + self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))
        return out


class _ASPP(nn.Module):
    def __init__(self, channels: int, dilations: Sequence[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 1) if d == 1 else nn.Conv2d(channels, channels, 3, padding=d, dilation=d),
                _groups(channels), nn.SiLU(),
            )
            for d in dilations
        )
        self.global_conv = nn.Sequential(nn.Conv2d(channels, channels, 1), _groups(channels), nn.SiLU())
        self.proj = nn.Sequential(nn.Conv2d(channels * (len(dilations) + 1), channels, 1), _groups(channels), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.interpolate(self.global_conv(x.mean((2, 3), keepdim=True)), size=x.shape[-2:], mode="bilinear")
        out: torch.Tensor = self.proj(torch.cat([*(branch(x) for branch in self.branches), pooled], dim=1))
        return out


class _Conditioning(nn.Module):
    """kohya's `conditioning1`: the map at 16 px per token, [b, tokens, cond_emb_dim]."""

    def __init__(self, inputs: int, width: int, out: int, resblocks: int, dilations: Sequence[int]) -> None:
        super().__init__()
        half = width // 2
        self.conv1, self.norm1 = nn.Conv2d(inputs, half, 4, stride=4), _groups(half)
        self.conv2, self.norm2 = nn.Conv2d(half, half, 3, padding=1), _groups(half)
        self.conv3, self.norm3 = nn.Conv2d(half, width, 4, stride=4), _groups(width)
        self.resblocks = nn.ModuleList(_ResBlock(width) for _ in range(resblocks))
        self.aspp = _ASPP(width, dilations) if dilations else None
        self.proj = nn.Conv2d(width, out, 1)
        self.out_norm = nn.LayerNorm(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for block in self.resblocks:
            h = block(h)
        if self.aspp is not None:
            h = self.aspp(h)
        out: torch.Tensor = self.out_norm(self.proj(h).flatten(2).transpose(1, 2))
        return out


class _Shift(nn.Module):
    """One target linear's LLLite: the shift its input gains."""

    def __init__(self, width: int, cond: int, mlp: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, mlp)
        self.mid = nn.Linear(mlp + cond, mlp)
        self.cond_to_film = nn.Linear(cond, 2 * mlp)
        self.up = nn.Linear(mlp, width)
        self.depth_embed = nn.Parameter(torch.empty(cond))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        local = (cond + self.depth_embed).to(self.down.weight.dtype)
        h = F.silu(self.down(x.to(self.down.weight.dtype)))
        gamma, beta = self.cond_to_film(local).chunk(2, dim=-1)
        mid = F.silu(self.mid(torch.cat([local.expand(h.shape[0], -1, -1), h], dim=-1)) * (1 + gamma) + beta)
        out: torch.Tensor = self.up(mid)
        return out


def _metadata(said: dict[str, object], key: str, default: str) -> str:
    return str(said.get(f"lllite.{key}", default))


class LLLiteGraph:
    def __init__(self, config: Config) -> None:
        said = config.mapping()
        if "lllite.version" in said and _metadata(said, "version", "2") != "2":
            raise UnsupportedInput("this ControlNet-LLLite is not kohya's v2 format", code="lllite_version")
        spec = _metadata(said, "target_atomics", _metadata(said, "target_layers", "self_attn_q_pre"))
        atomics = [a.strip() for a in _PRESETS.get(spec, spec).split(",") if a.strip()]
        if unknown := [a for a in atomics if a not in TARGETS]:
            raise UnsupportedInput(f"ControlNet-LLLite targets {', '.join(unknown)} are not Anima's", code="lllite_targets")
        cond, mlp = int(_metadata(said, "cond_emb_dim", "32")), int(_metadata(said, "mlp_dim", "64"))
        dilations: tuple[int, ...] = ()
        if _metadata(said, "use_aspp", "false").lower() == "true":
            dilations = tuple(int(d) for d in _metadata(said, "aspp_dilations", "1,2,4,8").split(",") if d.strip())
        #: 4: inpaint, RGB and a mask; kohya masks the RGB too (`inpaint_masked_input`).
        self.channels = int(_metadata(said, "cond_in_channels", "3"))
        self.masked = _metadata(said, "inpaint_masked_input", "true" if self.channels == 4 else "false").lower() == "true"
        trunk = _Conditioning(self.channels, int(_metadata(said, "cond_dim", "64")), cond,
                              int(_metadata(said, "cond_resblocks", "1")), dilations)
        root = nn.Module()
        root.lllite_conditioning1 = self.trunk = trunk
        #: (block, target) -> its shift, by the target names the DiT's block forward asks for.
        self.shifts: dict[tuple[int, str], _Shift] = {}
        for index in range(BLOCKS):
            for atomic in atomics:
                for target in TARGETS[atomic]:
                    shift = _Shift(WIDTH, cond, mlp)
                    root.add_module(f"lllite_dit_blocks_{index}_{target}", shift)
                    self.shifts[(index, target)] = shift
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(torch.bfloat16)}


class LLLite(Model[LLLiteGraph]):
    """A kohya v2 ControlNet-LLLite: its conditioning trunk and the shifts of its target linears."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: LLLiteGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(LLLiteGraph)

    @uses_components(COMPONENT)
    def armed(self, body: Callable[[], T]) -> T:
        """`body` with this adapter's weights held: the denoise reads them inside the DiT's calls."""
        return body()

    @uses_components(COMPONENT)
    def condition(self, control: torch.Tensor) -> torch.Tensor:
        """`control` [1, channels, H, W] in [-1, 1] at the output's pixel size: one token per DiT token."""
        trunk = self.graph.trunk
        out: torch.Tensor = trunk(control.to(torch.device(str(self.device)), next(trunk.parameters()).dtype))
        return out


@dataclass(frozen=True)
class Shifts:
    """One control map's LLLite for a denoise: its trunk's tokens and its strength."""

    adapter: LLLiteGraph
    cond: torch.Tensor  # [1, tokens, cond_emb_dim]
    strength: float

    def shift(self, index: int, target: str, x: torch.Tensor) -> torch.Tensor:
        found = self.adapter.shifts.get((index, target))
        if found is None or x.shape[1] != self.cond.shape[1]:
            return x
        out: torch.Tensor = x + (found(x, self.cond) * self.strength).to(x.dtype)
        return out


# ---------------------------------------------------------------- Claquasse's pose control-LoRA


def _pose_sites() -> dict[str, tuple[str, str]]:
    """Each LoRA site in the file's names (ComfyUI's, under `diffusion_model.`): the component and the linear it
    adapts there, by Diffusers' names (TensorFS's `anima.single_file`)."""
    sites: dict[str, tuple[str, str]] = {}
    for n in range(BLOCKS):
        native, ours = f"blocks.{n}", f"transformer_blocks.{n}"
        for attention, attn in (("self_attn", "attn1"), ("cross_attn", "attn2")):
            for p in "qkv":
                sites[f"{native}.{attention}.{p}_proj"] = ("transformer", f"{ours}.{attn}.to_{p}")
            sites[f"{native}.{attention}.output_proj"] = ("transformer", f"{ours}.{attn}.to_out.0")
        for part, norm in (("self_attn", "norm1"), ("cross_attn", "norm2"), ("mlp", "norm3")):
            for i in (1, 2):
                sites[f"{native}.adaln_modulation_{part}.{i}"] = ("transformer", f"{ours}.{norm}.linear_{i}")
        sites[f"{native}.mlp.layer1"] = ("transformer", f"{ours}.ff.net.0.proj")
        sites[f"{native}.mlp.layer2"] = ("transformer", f"{ours}.ff.net.2")
    for n in range(ADAPTER_BLOCKS):
        block = f"blocks.{n}"
        for attention in ("self_attn", "cross_attn"):
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                sites[f"llm_adapter.{block}.{attention}.{p}"] = ("text_conditioner", f"{block}.{attention}.{p}")
        for i in (0, 2):
            sites[f"llm_adapter.{block}.mlp.{i}"] = ("text_conditioner", f"{block}.mlp.{i}")
    return sites


def _put(root: nn.Module, path: str, module: nn.Module) -> None:
    parent = root
    for name in path.split(".")[:-1]:
        if not hasattr(parent, name):
            parent.add_module(name, nn.Module())
        parent = parent.get_submodule(name)
    parent.add_module(path.rsplit(".", 1)[-1], module)


#: The stored dtypes a pose file's tensors keep (its embedder f32, its LoRA bf16).
_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}


class PoseGraph:
    def __init__(self, config: Config) -> None:
        file = f"{COMPONENT}.diffusion_model"
        root = nn.Module()
        key = f"{file}.control_embedder.proj.weight"
        outputs, inputs = config.tensor_shape(key, default=(WIDTH, 64))
        self.embedder = nn.Linear(inputs, outputs, dtype=_DTYPES[config.tensor_dtype(key, default="f32")])
        _put(root, "diffusion_model.control_embedder.proj", self.embedder)
        #: Each component's factors by its linear's path there, as `lora.Sites` arms them.
        self.sites: dict[str, dict[str, lora.Pair]] = {"transformer": {}, "text_conditioner": {}}
        for native, (component, path) in _pose_sites().items():
            down, up = f"{file}.{native}.lora_A.weight", f"{file}.{native}.lora_B.weight"
            rank, fan_in = config.tensor_shape(down, default=(16, WIDTH))
            fan_out, _ = config.tensor_shape(up, default=(WIDTH, rank))
            # No stored alpha: ComfyUI's loader then scales by 1.
            pair = lora.Pair(fan_in, fan_out, rank).to(_DTYPES[config.tensor_dtype(down, default="bf16")])
            _put(root, f"diffusion_model.{native}", pair)
            self.sites[component][path] = pair
        self.components: dict[str, nn.Module] = {COMPONENT: root}


class PoseControl(Model[PoseGraph]):
    """Claquasse's Anima pose control-LoRA: its control embedder and its LoRA's factors, kept at their 16 bits."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: PoseGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(PoseGraph)

    @uses_components(COMPONENT)
    def tokens(self, latent: torch.Tensor) -> torch.Tensor:
        """The skeleton's latent [1, 16, 1, h, w], unnormalized, as tokens the patch embedding gains: [1, 1, h/2,
        w/2, 2048], its 2×2 patches flattened channel first, as Cosmos' patch embedding flattens."""
        b, c, t, h, w = latent.shape
        patches = latent.reshape(b, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).flatten(4)
        weight = self.graph.embedder.weight
        out: torch.Tensor = self.graph.embedder(patches.to(weight.device, weight.dtype))
        return out

    @uses_components(COMPONENT)
    def applied(self, sites: Mapping[str, lora.Sites], body: Callable[[], T]) -> T:
        """`body` with this LoRA armed on Anima's components (`sites`, by component)."""
        with ExitStack() as armed:
            for component, factors in self.graph.sites.items():
                armed.enter_context(sites[component].applied(factors))
            return body()


# ---------------------------------------------------------------- one denoise's adapters


@dataclass
class Adapters:
    """What a denoise's DiT calls add: the references, the control maps' shifts, the pose's tokens on the patch
    embedding, and which rows of the call are the conditional ones (the IP term goes there only). The DiT's call
    takes it as an argument, so the Runtime runs those calls as written, never from a recorded graph."""

    references: References | None = None
    shifts: list[Shifts] = field(default_factory=list)
    embedded: torch.Tensor | None = None  # [1, 1, h/2, w/2, 2048]
    rows: tuple[bool, ...] = (True,)

    def at(self, rows: tuple[bool, ...]) -> Adapters:
        return replace(self, rows=rows)

    def shift(self, index: int, target: str, x: torch.Tensor) -> torch.Tensor:
        for shifts in self.shifts:
            x = shifts.shift(index, target, x)
        return x
