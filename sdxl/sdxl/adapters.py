"""SDXL's picture adapters, each its author's file uploaded as it is (se-175):

- xinsir's ControlNet Union ProMax (`controlnet-union-sdxl-1.0`, `diffusion_pytorch_model_promax`): one ControlNet for
  every control kind, each map steering through its mode's task embedding. Diffusers' `ControlNetUnionModel`, called
  as its `StableDiffusionXLControlNetUnionPipeline` calls it: the maps in 0..1 at the output's size, on both CFG rows.
- h94's IP-Adapter Plus (`sdxl_models/ip-adapter-plus_sdxl_vit-h`) over LAION's OpenCLIP ViT-H/14, the encoder h94
  ships as `models/image_encoder` (the same tensors): each picture's penultimate ViT-H tokens through the adapter's
  Perceiver resampler into 16 tokens, which every UNet cross-attention reads beside the text through Diffusers'
  `IPAdapterAttnProcessor2_0`, scaled by the request's strength. The unconditional rows read a black picture's tokens,
  as Diffusers' pipelines do.
- h94's IP-Adapter-FaceID-PlusV2 (`ip-adapter-faceid-plusv2_sdxl`) for a picture whose subject is a face (`faces`), as
  its authors' `IPAdapterFaceIDPlusXL` and Diffusers' loader run it: the face's ArcFace embedding through Diffusers'
  `IPAdapterFaceIDPlusImageProjection` into 4 tokens, its perceiver reading the ViT-H tokens of the face aligned at 224²,
  the shortcut on (PlusV2) at 1; the unconditional rows a zero embedding and a black picture's tokens. Its rank-128 LoRA
  on every UNet attention's q, k, v and out, at 1, armed on the UNet's `lora.Sites` while the face steers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import torch
import torch.nn.functional as F
from cozy_runtime.accel import lora
from cozy_runtime.author import Config, Loader, Model, Serving, UnsupportedInput, uses_components
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel
from diffusers.models.embeddings import IPAdapterFaceIDPlusImageProjection
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
from transformers import initialization as transformer_init

if TYPE_CHECKING:
    from diffusers.models.attention_processor import Attention
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from PIL import Image as PILImage

T = TypeVar("T")
#: An as-is upload's one component.
COMPONENT = "model"


def _to[M: nn.Module](module: M, dtype: torch.dtype) -> M:
    """`module.to(dtype)`, typed: Diffusers' `to` is `functools.wraps`'d, which the checkers bind without `self`."""
    return module.to(dtype)


def _stored(config: Config, key: str) -> bool:
    """Whether the checkpoint stores `key`: the Runtime refuses a missing one (`config_tensor_missing`)."""
    try:
        return bool(config.tensor_dtype(key, default=""))
    except Exception as exc:  # the Runtime's ConformanceError, which `cozy_runtime.author` doesn't export
        if getattr(exc, "code", None) == "config_tensor_missing":
            return False
        raise


# ---------------------------------------------------------------- ControlNet Union


#: Each control kind's mode in the union ControlNet (xinsir's model card; Diffusers' union pipelines): 0 openpose,
#: 1 depth, 2 hed/pidi/scribble/ted, 3 canny/lineart/mlsd, 7 repaint (ProMax only, the inpaint).
MODES = {"pose": 0, "depth": 1, "sketch": 2, "lineart": 3, "canny": 3, "inpaint": 7}


class ControlGraph:
    def __init__(self, config: Config) -> None:
        mapping = dict(config.mapping())
        # The mode count is the task embedding's rows: 6 in the first release, 8 in ProMax (its own config says 8).
        self.modes, _ = config.tensor_shape(f"{COMPONENT}.task_embedding", default=(8, 320))
        mapping["num_control_type"] = self.modes
        with transformer_init.no_init_weights():
            net = cast(ControlNetUnionModel, ControlNetUnionModel.from_config(mapping))
        self.components: dict[str, nn.Module] = {COMPONENT: _to(net, torch.float16).eval()}


class ControlNet(Model[ControlGraph]):
    """xinsir's SDXL ControlNet Union: one model, a mode per kind."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: ControlGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(ControlGraph)

    @uses_components(COMPONENT)
    def armed(self, body: Callable[[], T]) -> T:
        """`body` with this ControlNet's weights held: each denoising step calls it (`Controls`)."""
        return body()


@dataclass(frozen=True)
class Controls:
    """One denoise's control maps: each [1, 3, H, W] in 0..1 at the output's size, its mode and strength."""

    graph: ControlGraph
    maps: list[torch.Tensor]
    modes: list[int]
    strengths: list[float]

    def residuals(
        self, sample: torch.Tensor, timestep: torch.Tensor, prompt: torch.Tensor, pooled: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """The residuals the UNet adds for `sample`'s rows (both CFG rows when there are two)."""
        net = cast(ControlNetUnionModel, self.graph.components[COMPONENT])
        rows = sample.shape[0]
        types = torch.zeros(self.graph.modes, device=sample.device, dtype=sample.dtype)
        types[self.modes] = 1
        down, mid = net(
            sample, timestep, encoder_hidden_states=prompt,
            controlnet_cond=[m.to(sample.device, sample.dtype).expand(rows, -1, -1, -1) for m in self.maps],
            control_type=types.expand(rows, -1), control_type_idx=self.modes, conditioning_scale=self.strengths,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}, return_dict=False,
        )  # fmt: skip
        return list(down), mid


# ---------------------------------------------------------------- the image encoder


class _Vision(nn.Module):
    """CLIP's vision tower under the name `CLIPModel` stores it by (`vision_model.`): its penultimate hidden states."""

    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        self.vision_model = CLIPVisionModel(config)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.vision_model(pixel_values=pixels, output_hidden_states=True).hidden_states[-2]
        return out


class EncoderGraph:
    def __init__(self, config: Config) -> None:
        vision = CLIPVisionConfig(**config.mapping()["vision_config"])
        with transformer_init.no_init_weights():
            encoder = _Vision(vision)
        # Built in its stored dtype (an as-is upload is never cast): LAION's is f32.
        stored = config.tensor_dtype(f"{COMPONENT}.vision_model.embeddings.patch_embedding.weight", default="f32")
        dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[stored]
        self.components: dict[str, nn.Module] = {COMPONENT: encoder.to(dtype).eval()}
        #: Diffusers' `load_ip_adapter`: CLIP's preprocessing at the encoder's own size.
        self.processor = CLIPImageProcessor(size=vision.image_size, crop_size=vision.image_size)


class ImageEncoder(Model[EncoderGraph]):
    """LAION's OpenCLIP ViT-H/14 vision tower (its CLIPModel upload; the text tower is never built)."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: EncoderGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(EncoderGraph)

    @uses_components(COMPONENT)
    def tokens(self, pictures: Sequence[PILImage.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Each picture's penultimate hidden states, and a black picture's (the unconditional), as Diffusers'
        `encode_image` takes them for IP-Adapter Plus: [n, 257, 1280] each."""
        encoder = self.graph.components[COMPONENT]
        weight = next(encoder.parameters())
        pixels = self.graph.processor(images=[p.convert("RGB") for p in pictures], return_tensors="pt").pixel_values
        pixels = pixels.to(weight.device, weight.dtype)
        both: torch.Tensor = encoder(torch.cat([pixels, torch.zeros_like(pixels)]))
        return both[: len(pictures)], both[len(pictures) :]


# ---------------------------------------------------------------- IP-Adapter Plus


class _Perceiver(nn.Module):
    """One resampler layer's attention, in h94's file layout (`layers.N.0`)."""

    def __init__(self, width: int, inner: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.to_q = nn.Linear(width, inner, bias=False)
        self.to_kv = nn.Linear(width, 2 * inner, bias=False)
        self.to_out = nn.Linear(inner, width, bias=False)


class _Projection(nn.Module):
    """IP-Adapter Plus's Perceiver resampler, as h94's file stores it (`image_proj.`)."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        key = f"{COMPONENT}.image_proj"
        _, queries, width = config.tensor_shape(f"{key}.latents", default=(1, 16, 1280))
        _, embeds = config.tensor_shape(f"{key}.proj_in.weight", default=(width, 1280))
        outputs, _ = config.tensor_shape(f"{key}.proj_out.weight", default=(2048, width))
        inner, _ = config.tensor_shape(f"{key}.layers.0.0.to_q.weight", default=(width, width))
        hidden, _ = config.tensor_shape(f"{key}.layers.0.1.1.weight", default=(4 * width, width))
        depth = 1
        while _stored(config, f"{key}.layers.{depth}.0.to_q.weight"):
            depth += 1
        #: 64 per head (tencent-ailab's `Resampler`; Diffusers' conversion reads the heads off `to_q` the same way).
        self.heads = inner // 64
        self.latents = nn.Parameter(torch.empty(1, queries, width))
        self.proj_in = nn.Linear(embeds, width)
        self.attentions = [_Perceiver(width, inner) for _ in range(depth)]
        self.feeds = [
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden, bias=False), nn.GELU(),
                          nn.Linear(hidden, width, bias=False))
            for _ in range(depth)
        ]  # fmt: skip
        self.layers = nn.ModuleList(nn.ModuleList(pair) for pair in zip(self.attentions, self.feeds, strict=True))
        self.proj_out = nn.Linear(width, outputs)
        self.norm_out = nn.LayerNorm(outputs)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """[n, 257, 1280] ViT-H tokens to [n, 16, 2048]: Diffusers' `IPAdapterPlusImageProjection`, whose blocks hold
        `to_kv` as two linears and so read Diffusers' converted names, not the file's."""
        latents = self.latents.repeat(embeds.shape[0], 1, 1)
        x = self.proj_in(embeds)
        for attention, feed in zip(self.attentions, self.feeds, strict=True):
            tokens = attention.norm1(x)
            queries = attention.norm2(latents)
            key, value = attention.to_kv(torch.cat([tokens, queries], dim=1)).chunk(2, dim=-1)
            q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (attention.to_q(queries), key, value))
            attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
            latents = attention.to_out(attended) + latents
            latents = feed(latents) + latents
        out: torch.Tensor = self.norm_out(self.proj_out(latents))
        return out


class _Site(nn.Module):
    """One cross-attention's image key and value (`ip_adapter.N`)."""

    def __init__(self, inputs: int, width: int) -> None:
        super().__init__()
        self.to_k_ip = nn.Linear(inputs, width, bias=False)
        self.to_v_ip = nn.Linear(inputs, width, bias=False)


def _sites(config: Config, prefix: str) -> dict[str, _Site]:
    """Each cross-attention's image key and value under `prefix`, in h94's numbering: 1, 3, 5, … over the UNet's
    cross-attentions in `attn_processors` order (Diffusers' `_convert_ip_adapter_attn_to_diffusers`)."""
    ids: list[int] = []
    while _stored(config, f"{prefix}.{2 * len(ids) + 1}.to_k_ip.weight"):
        ids.append(2 * len(ids) + 1)
    sites = {}
    for n in ids:
        width, inputs = config.tensor_shape(f"{prefix}.{n}.to_k_ip.weight", default=(1280, 2048))
        sites[str(n)] = _Site(inputs, width)
    return sites


class IPGraph:
    def __init__(self, config: Config) -> None:
        root = nn.Module()
        root.image_proj = self.projection = _Projection(config)
        sites = _sites(config, f"{COMPONENT}.ip_adapter")
        root.ip_adapter = nn.ModuleDict(sites)
        self.sites = list(sites.values())
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(torch.float16).eval()}


class IPAdapter(Model[IPGraph]):
    """h94's IP-Adapter Plus for SDXL: a Perceiver resampler and every cross-attention's image key and value."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: IPGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(IPGraph)

    @uses_components(COMPONENT)
    def tokens(self, embeds: torch.Tensor, empty: torch.Tensor, guided: bool) -> torch.Tensor:
        """The pictures' tokens side by side, [1, n·16, 2048], under the black picture's when CFG runs."""
        projection = self.graph.projection
        weight = projection.latents
        cond: torch.Tensor = projection(embeds.to(weight.device, weight.dtype)).flatten(0, 1).unsqueeze(0)
        if not guided:
            return cond
        uncond: torch.Tensor = projection(empty.to(weight.device, weight.dtype)).flatten(0, 1).unsqueeze(0)
        return torch.cat([uncond, cond])

    @uses_components(COMPONENT)
    def armed(self, body: Callable[[], T]) -> T:
        """`body` with this adapter's weights held: each denoising step reads its sites (`steering`)."""
        return body()


# ---------------------------------------------------------------- IP-Adapter-FaceID-PlusV2


#: FaceID-PlusV2's components, its file converted as TensorFS's `ipadapter.sdxl` lays out Plus (`scripts/faceid_ingest.py`):
#: `image_proj` is Diffusers' `IPAdapterFaceIDPlusImageProjection`'s state, `ip_adapter` each cross-attention's image key
#: and value.
PROJECTION = "image_proj"
SITES = "ip_adapter"


class _FaceProjection(IPAdapterFaceIDPlusImageProjection):
    """Diffusers' projection with the face's CLIP tokens an argument, not state set before the call: the Runtime replays
    a component's call by its arguments."""

    def forward(self, id_embeds: torch.Tensor, clip_embeds: torch.Tensor | None = None) -> torch.Tensor:
        self.clip_embeds = clip_embeds  # type: ignore[assignment]  # Diffusers types it None, which a pipeline then sets
        try:
            out: torch.Tensor = super().forward(id_embeds)
        finally:
            self.clip_embeds = None
        return out


class FaceIDGraph:
    def __init__(self, config: Config) -> None:
        # Its arguments off its tensors, as Diffusers' converter reads them off h94's file; the rest its defaults (4 blocks,
        # 64 per head, 4 tokens). Defaults: h94's SDXL file, for a construction from config alone.
        embed, hidden = config.tensor_shape(f"{PROJECTION}.proj_in.weight", default=(2048, 1280))
        output, _ = config.tensor_shape(f"{PROJECTION}.proj_out.weight", default=(2048, embed))
        inner, _ = config.tensor_shape(f"{PROJECTION}.layers.0.attn.to_q.weight", default=(embed, embed))
        _, identity = config.tensor_shape(f"{PROJECTION}.proj.net.0.proj.weight", default=(1024, 512))
        self.projection = _FaceProjection(embed_dims=embed, output_dims=output, hidden_dims=hidden, heads=inner // 64,
                                          id_embeddings_dim=identity)  # fmt: skip
        # PlusV2: the ID tokens plus the perceiver's, at h94's `s_scale` 1.
        self.projection.shortcut, self.projection.shortcut_scale = True, 1.0
        sites = _sites(config, SITES)
        self.sites = list(sites.values())
        self.components: dict[str, nn.Module] = {
            PROJECTION: _to(self.projection, torch.float16).eval(),
            SITES: _to(nn.ModuleDict(sites), torch.float16).eval(),
        }


class FaceIDAdapter(Model[FaceIDGraph]):
    """h94's IP-Adapter-FaceID-PlusV2 for SDXL: a face's identity and its aligned CLIP tokens into 4 tokens, and every
    cross-attention's image key and value."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: FaceIDGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(FaceIDGraph)

    @uses_components(PROJECTION)
    def tokens(self, identities: torch.Tensor, embeds: torch.Tensor, empty: torch.Tensor, guided: bool) -> torch.Tensor:
        """Each face's tokens side by side, [1, n·4, 2048], under a zero identity's over the black picture's when CFG
        runs: `identities` [n, 512] (ArcFace's, normalized), `embeds` and `empty` [n, 257, 1280] (ViT-H's)."""
        projection = self.graph.projection
        weight = projection.norm.weight

        def tokens(ids: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = projection(ids.to(weight.device, weight.dtype),
                                           clip.to(weight.device, weight.dtype).unsqueeze(1))  # fmt: skip
            return out.flatten(0, 1).unsqueeze(0)

        cond = tokens(identities, embeds)
        return torch.cat([tokens(torch.zeros_like(identities), empty), cond]) if guided else cond

    @uses_components(SITES)
    def armed(self, body: Callable[[], T]) -> T:
        """`body` with each cross-attention's image key and value held (`steering`)."""
        return body()


#: The FaceID LoRA's one component: TensorFS's `sdxl.lora`, PEFT factors under the UNet's module names.
LORA = "unet"


def attention_linears(unet: UNet2DConditionModel) -> list[str]:
    """Every UNet attention's q, k, v and out projections: where FaceID's LoRA arms (`lora.Sites`)."""
    return [f"{name.removesuffix('.processor')}.{linear}" for name in unet.attn_processors
            for linear in ("to_q", "to_k", "to_v", "to_out.0")]  # fmt: skip


class FaceLoRAGraph:
    def __init__(self, config: Config) -> None:
        adapter = config.mapping()  # the one component's PEFT adapter config
        root = nn.Module()
        self.factors: dict[str, lora.Pair] = {}
        dtype = torch.float16
        for target in adapter["target_modules"]:
            path = str(target).removeprefix(f"{LORA}.")
            key = f"{LORA}.{path}"
            down = config.tensor_shape(f"{key}.lora_A.weight", default=())
            up = config.tensor_shape(f"{key}.lora_B.weight", default=())
            if len(down) != 2 or len(up) != 2:
                raise ValueError(f"FaceID's LoRA has no stored factors for {path}")
            (rank, inputs), (outputs, _) = down, up
            # Alpha: the config's `lora_alpha` (h94's is the rank at every module), else r.
            factors = lora.Pair(inputs, outputs, rank, float(adapter.get("lora_alpha", rank)) / rank)
            dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[
                config.tensor_dtype(f"{key}.lora_A.weight", default="f16")]
            parent = root
            for name in path.split(".")[:-1]:
                if not hasattr(parent, name):
                    parent.add_module(name, nn.Module())
                parent = getattr(parent, name)
            parent.add_module(path.rsplit(".", 1)[-1], factors)
            self.factors[path] = factors
        self.components: dict[str, nn.Module] = {LORA: root.to(dtype)}


class FaceLoRA(Model[FaceLoRAGraph]):
    """FaceID-PlusV2's rank-128 LoRA over the UNet's attentions, kept at its stored 16 bits."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: FaceLoRAGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(FaceLoRAGraph)

    @uses_components(LORA)
    def applied(self, sites: lora.Sites, body: Callable[[], T]) -> T:
        """`body` with this LoRA armed on the UNet (`sites`): the Runtime bakes its factors into the weights it fills."""
        with sites.applied(self.graph.factors):
            return body()


# ---------------------------------------------------------------- steering


@dataclass(frozen=True)
class Steer:
    """One adapter's part of a denoise: its tokens [rows, n·q, 2048] (the unconditional row first when CFG runs), its
    image key and value at each UNet cross-attention, and its scale."""

    tokens: torch.Tensor
    sites: Sequence[_Site]
    scale: float


@dataclass(frozen=True)
class References:
    """One denoise's picture tokens, each adapter's. The UNet's call takes it as a cross-attention argument, so the
    Runtime runs those calls as written, never from a recorded graph."""

    tokens: list[torch.Tensor]


class _Steered:
    """A UNet cross-attention's processor while pictures steer: Diffusers' IP-Adapter processor over the text and the
    call's `References`, one adapter after another. Not a module, so the UNet's tree is unchanged."""

    def __init__(self, processor: IPAdapterAttnProcessor2_0) -> None:
        self.processor = processor

    def __call__(
        self, attn: Attention, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None, ip_hidden_states: References | None = None,
    ) -> torch.Tensor:
        assert ip_hidden_states is not None and encoder_hidden_states is not None
        # Diffusers types `encoder_hidden_states` one tensor; its body takes the (text, pictures) pair.
        pair = (encoder_hidden_states, ip_hidden_states.tokens)
        out: torch.Tensor = self.processor(attn, hidden_states, pair, attention_mask)  # type: ignore[arg-type]
        return out


@contextmanager
def steering(unet: UNet2DConditionModel, parts: Sequence[Steer]) -> Iterator[References]:
    """Diffusers' IP-Adapter processor on each of `unet`'s cross-attentions, over every part's key, value and scale (its
    loader's multi-adapter form); each site's own processor comes back after."""
    names = [name for name in unet.attn_processors if not name.endswith("attn1.processor")]
    for part in parts:
        if len(part.sites) != len(names):
            raise UnsupportedInput(f"this IP-Adapter adapts {len(part.sites)} cross-attentions and the UNet has "
                                   f"{len(names)}", code="ip_adapter_unet")  # fmt: skip
    held: list[tuple[Attention, object]] = []
    try:
        for i, name in enumerate(names):
            attn = cast("Attention", unet.get_submodule(name.removesuffix(".processor")))
            first = parts[0].sites[i].to_k_ip
            with torch.device("meta"):
                processor = IPAdapterAttnProcessor2_0(  # type: ignore[no-untyped-call]  # Diffusers' __init__ is unannotated
                    first.out_features, first.in_features, num_tokens=tuple(p.tokens.shape[1] for p in parts),
                    scale=[p.scale for p in parts],
                )
            processor.to_k_ip = nn.ModuleList([p.sites[i].to_k_ip for p in parts])
            processor.to_v_ip = nn.ModuleList([p.sites[i].to_v_ip for p in parts])
            held.append((attn, attn.processor))
            attn.set_processor(_Steered(processor))  # type: ignore[arg-type]  # Diffusers types processors by its own classes
        yield References([part.tokens for part in parts])
    finally:
        for attn, original in held:
            attn.set_processor(original)  # type: ignore[arg-type]
