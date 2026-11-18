"""SDXL's picture adapters, each its author's file uploaded as it is (se-175):

- xinsir's ControlNet Union ProMax (`controlnet-union-sdxl-1.0`, `diffusion_pytorch_model_promax`): one ControlNet for
  every control kind, each map steering through its mode's task embedding. Diffusers' `ControlNetUnionModel`, called
  as its `StableDiffusionXLControlNetUnionPipeline` calls it: the maps in 0..1 at the output's size, on both CFG rows.
- h94's IP-Adapter Plus (`sdxl_models/ip-adapter-plus_sdxl_vit-h`) over LAION's OpenCLIP ViT-H/14, the encoder h94
  ships as `models/image_encoder` (the same tensors): each picture's penultimate ViT-H tokens through the adapter's
  Perceiver resampler into 16 tokens, which every UNet cross-attention reads beside the text through Diffusers'
  `IPAdapterAttnProcessor2_0`, scaled by the request's strength. The unconditional rows read a black picture's tokens,
  as Diffusers' pipelines do.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import torch
import torch.nn.functional as F
from cozy_runtime.author import Config, Loader, Model, Serving, UnsupportedInput, uses_components
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel
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


class IPGraph:
    def __init__(self, config: Config) -> None:
        root = nn.Module()
        root.image_proj = self.projection = _Projection(config)
        #: h94's numbering: 1, 3, 5, … over the UNet's cross-attentions in `attn_processors` order (Diffusers'
        #: `_convert_ip_adapter_attn_to_diffusers`).
        ids: list[int] = []
        while _stored(config, f"{COMPONENT}.ip_adapter.{2 * len(ids) + 1}.to_k_ip.weight"):
            ids.append(2 * len(ids) + 1)
        sites = {}
        for n in ids:
            width, inputs = config.tensor_shape(f"{COMPONENT}.ip_adapter.{n}.to_k_ip.weight", default=(1280, 2048))
            sites[str(n)] = _Site(inputs, width)
        root.ip_adapter = nn.ModuleDict(sites)
        queries = self.projection.latents.shape[1]
        #: Diffusers' processor per site, its image key and value this file's linears (outside the component tree,
        #: which spells the file).
        self.processors: list[IPAdapterAttnProcessor2_0] = []
        for site in sites.values():
            assert isinstance(site, _Site)
            with torch.device("meta"):
                processor = IPAdapterAttnProcessor2_0(  # type: ignore[no-untyped-call]  # Diffusers' __init__ is unannotated
                    site.to_k_ip.out_features, site.to_k_ip.in_features, num_tokens=(queries,)
                )
            processor.to_k_ip, processor.to_v_ip = nn.ModuleList([site.to_k_ip]), nn.ModuleList([site.to_v_ip])
            self.processors.append(processor)
        self.components: dict[str, nn.Module] = {COMPONENT: root.to(torch.float16).eval()}


@dataclass(frozen=True)
class References:
    """One denoise's picture tokens, [rows, n·16, 2048]: the unconditional row first when CFG runs. The UNet's call
    takes it as a cross-attention argument, so the Runtime runs those calls as written, never from a recorded graph."""

    tokens: torch.Tensor


class _Steered:
    """A UNet cross-attention's processor while pictures steer: Diffusers' IP-Adapter processor over the text and the
    call's `References`. Not a module, so the UNet's tree is unchanged."""

    def __init__(self, processor: IPAdapterAttnProcessor2_0) -> None:
        self.processor = processor

    def __call__(
        self, attn: Attention, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None, ip_hidden_states: References | None = None,
    ) -> torch.Tensor:
        assert ip_hidden_states is not None and encoder_hidden_states is not None
        # Diffusers types `encoder_hidden_states` one tensor; its body takes the (text, pictures) pair.
        pair = (encoder_hidden_states, [ip_hidden_states.tokens])
        out: torch.Tensor = self.processor(attn, hidden_states, pair, attention_mask)  # type: ignore[arg-type]
        return out


class IPAdapter(Model[IPGraph]):
    """h94's IP-Adapter Plus for SDXL: a Perceiver resampler and every cross-attention's image key and value."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: IPGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(IPGraph)

    @uses_components(COMPONENT)
    def tokens(self, embeds: torch.Tensor, empty: torch.Tensor, guided: bool) -> References:
        """The pictures' tokens side by side, [1, n·16, 2048], under the black picture's when CFG runs."""
        projection = self.graph.projection
        weight = projection.latents
        cond = projection(embeds.to(weight.device, weight.dtype)).flatten(0, 1).unsqueeze(0)
        if not guided:
            return References(cond)
        uncond = projection(empty.to(weight.device, weight.dtype)).flatten(0, 1).unsqueeze(0)
        return References(torch.cat([uncond, cond]))

    @uses_components(COMPONENT)
    def armed(self, unet: UNet2DConditionModel, strength: float, body: Callable[[], T]) -> T:
        """`body` with this adapter's weights held and its processors on `unet`'s cross-attentions at `strength`;
        each site's own processor comes back after."""
        with self._steering(unet, strength):
            return body()

    @contextmanager
    def _steering(self, unet: UNet2DConditionModel, strength: float) -> Iterator[None]:
        names = [name for name in unet.attn_processors if not name.endswith("attn1.processor")]
        if len(names) != len(self.graph.processors):
            raise UnsupportedInput(f"this IP-Adapter adapts {len(self.graph.processors)} cross-attentions and the "
                                   f"UNet has {len(names)}", code="ip_adapter_unet")  # fmt: skip
        held: list[tuple[Attention, object]] = []
        try:
            for name, processor in zip(names, self.graph.processors, strict=True):
                attn = cast("Attention", unet.get_submodule(name.removesuffix(".processor")))
                held.append((attn, attn.processor))
                processor.scale = [strength]
                attn.set_processor(_Steered(processor))  # type: ignore[arg-type]  # Diffusers types processors by its own classes
            yield
        finally:
            for attn, original in held:
                attn.set_processor(original)  # type: ignore[arg-type]

