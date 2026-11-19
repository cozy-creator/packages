"""se-008 — the SDXL launch package: `generate`, text to image, four components.

The second launch family, and the ENGINELESS-PATH REFERENCE (`cozy-runtime-sdxl.md`): one
`Model` class with author-written `load`, public methods each declaring the
heavyweight components it may touch, one entrypoint. Runtime owns placement, offload and
pinning; the DPM++ 2M Karras steps are a plain loop over Diffusers' scheduler.

What it adds over cr-008b's corpus fixture, which it is otherwise faithful to:

  * THE SHARED IMAGE REQUEST (cr-219). Any W:H at any size, on SDXL's 64 px latent stride:
    1 MP is base SDXL (its training buckets), above it HiDiffusion, to 8 MP (16 MP at 1:1, the
    4096² flagship). Outside that range the request is held, with a `clamp` adjustment, never
    refused. Steps and guidance stay `ModelDefault`; a deployment's visible `Clamp` still
    lowers them in the adjustments envelope.
  * VALUE-PLANE CFG GATING. `guidance <= 1.0` skips the negative branch entirely — the
    turbo/lightning execution shape follows from the RESOLVED VALUE alone, with zero
    adapter awareness in this file. When the adapter runtime (cr-010) lands, a turbo
    recipe supplies that value as an omitted-field default and not one line here changes.
  * AN OUTPUT-INTEGRITY FLOOR. A decode that produced NaNs, or a flat field with no
    picture in it, is a failure this package reports rather than a PNG it publishes.

`FP8`, `INT8` and `W4A8` declare this family's quantization recipes; `cozy_runtime` owns the
encodings and runs them.

Code states CAPABILITY; bindings state SELECTION. Nothing here names a repo, release,
checkpoint or revision — `package.toml` and the deploy binding do.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, cast

import msgspec
import torch
import torch.nn.functional as F
from cozy_runtime import rng
from cozy_runtime.accel import lora
from cozy_runtime.author import (
    AdjustmentRow,
    AdjustmentRows,
    Adjustments,
    App,
    Config,
    Context,
    EncodingRule,
    ImageFrame,
    Loader,
    Model,
    ModelAssets,
    OutputError,
    QuantizationPlan,
    Recipe,
    Seed,
    Serving,
    Telemetry,
    preview,
    uses_components,
)
from cozy_runtime.fields import MENTION, Advanced, AspectRatio, FileBound, Image, Preflight, Prompt, Ratios
from cozy_runtime.media import ImageFile
from PIL import Image as PILImage
from cozy_runtime.image import (
    Control,
    Guidance,
    ImageRequest,
    ImageResult,
    MEGAPIXEL,
    NegativePrompt,
    Sizes,
    Steps,
    held,
)
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from hidiffusion import apply_hidiffusion, remove_hidiffusion
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from transformers import initialization as transformer_init

from ._numerics import decode_latents, encode_tokens, guided, hold_decode_dtype
from .adapters import (
    MODES,
    ControlNet,
    Controls,
    FaceIDAdapter,
    FaceLoRA,
    ImageEncoder,
    IPAdapter,
    References,
    Steer,
    attention_linears,
    steering,
)
from .faces import FaceDetector, FaceRecognizer, aligned, crop, readable, subject

if TYPE_CHECKING:
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.attention import BasicTransformerBlock
    from diffusers.utils.outputs import BaseOutput
    from torch import nn
    from torch.utils.hooks import RemovableHandle

app = App()


#: Pixel counts: SDXL's 1 MP training size up to HiDiffusion's 8 MP class, and its 4096² flagship at
#: 1:1 only (its 4096-class recipe needs both latent sides at 512); on its 64 px buckets, at most the
#: 1:4-4:1 its training buckets reach (512x2048) and 4096 px sides.
_SIZES = Sizes((1.0, 8.0), (64, 4096), 64,
               "SDXL's 1 MP training size to HiDiffusion's 8 MP (16 MP at 1:1)", widest=4.0)
_SQUARE = dataclasses.replace(_SIZES, megapixels=(1.0, 16.0))
#: Above this, a request rides HiDiffusion unless it opts out: base SDXL duplicates subjects past
#: its training size.
_NATIVE = 1.5


#: The control maps xinsir's union ControlNet has a mode for (`adapters.MODES`); `inpaint` needs ProMax's repaint mode.
Kind = Literal["depth", "pose", "lineart", "sketch", "canny", "inpaint"]
#: The pictures the IP-Adapter takes at once (InvokeAI's cap, Anima's): past them the last mentioned go, with a warning.
MAX_PICTURES = 4
#: `GenerateInput.reference_strength`'s default.
REFERENCE_STRENGTH = 0.4
#: FaceID-PlusV2's scale at the default `reference_strength`, which scales both adapters alike: its authors' 1
#: (`IPAdapterFaceIDPlusXL`), the face A/B's winning arm (se-175: ArcFace 0.65 against Plus's 0.42, prompts followed).
FACE_SCALE = 1.0


class GenerateInput(ImageRequest):
    #: Its words, and the pictures it mentions: each a reference for an IP-Adapter (`_said`, `_references`).
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text
    #: Image to image: start from this picture, `strength` of the schedule redrawing it.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    #: How much of the picture is redrawn: 1 is from noise.
    strength: Annotated[float, msgspec.Meta(gt=0.0, le=1.0), Advanced()] = 0.9
    #: Control maps, each through its mode of the union ControlNet; `inpaint`'s is the picture with the area to redraw
    #: transparent.
    control: list[Control[Kind]] = msgspec.field(default_factory=list)
    #: How strongly the mentioned pictures steer, Plus's scale (FaceID's `FACE_SCALE` times it over the default). 0.4: the
    #: words set the scene while the picture gives its subject and style. At tencent-ailab's 0.5 and above, IP-Adapter
    #: Plus redraws the picture, its backdrop too, past the words; at 0.3 the subject fades (se-175, SE175-41..56, 61..72).
    reference_strength: Annotated[float, msgspec.Meta(ge=0.0, le=2.0), Advanced()] = REFERENCE_STRENGTH
    steps: Steps = None
    guidance: Guidance = None
    #: `_negative_prompt`, with HiDiffusion's recipe after it.
    negative_prompt: NegativePrompt = None
    #: `resolve_generate`: None runs HiDiffusion where base SDXL cannot go, above 1 MP. An explicit
    #: choice always runs, and the run warns where it is known to hurt.
    hidiffusion: bool | None = None


# ------------------------------------------------------------------ the model


class HiDiffusionInfo(TypedDict):
    """The state `apply_hidiffusion` shares across a patched UNet and its blocks (`info`): the
    size, cleared between requests, is set by HiDiffusion's UNet hook before any block runs."""

    size: tuple[int, int] | None
    upsample_size: tuple[int, int] | None
    hooks: list[RemovableHandle]


class HiDiffusionUNet(UNet2DConditionModel):
    """Select and reset the request's denoising implementation before step zero.

    HiDiffusion's blocks count steps on themselves, and its window blocks draw their shift
    from torch's global RNG. So before every call each counting block is set to the request's
    step (`cozy_step`), and each window block (`_windowed`) takes its shift from the request's
    seed, step and its own index: a block runs the same however often it runs."""

    _base_num_upsamplers: int
    _hidiffusion_active: bool
    _num_timesteps: int
    name_or_path: str
    info: HiDiffusionInfo
    cozy_step = 0
    _hidiffusion_seed = 0

    def begin_denoise_request(self, steps: int, hidiffusion: bool, seed: int) -> None:
        if steps <= 0:
            raise ValueError("HiDiffusion requires a positive timestep count")
        active = bool(getattr(self, "_hidiffusion_active", False))
        if hidiffusion != active:
            if hidiffusion:
                self.num_upsamplers = self._base_num_upsamplers
                apply_hidiffusion(self)
                window = 0
                for block in [m for m in self.modules() if hasattr(m, "_parent")]:
                    hook: Callable[[nn.Module, tuple[object, ...]], None]
                    if hasattr(block, "max_timestep"):
                        hook = self._count_step
                    else:
                        block.__class__ = _windowed(block._parent)
                        hook, window = functools.partial(self._draw_shift, window), window + 1
                    self.info["hooks"].append(block.register_forward_pre_hook(hook))
            else:
                remove_hidiffusion(self)
                self.num_upsamplers = self._base_num_upsamplers
            self._hidiffusion_active = hidiffusion
        if not hidiffusion:
            return
        self._num_timesteps, self._hidiffusion_seed = steps, seed
        self.info["size"] = self.info["upsample_size"] = None
        for module in self.modules():
            if hasattr(module, "max_timestep"):
                module.max_timestep = steps  # type: ignore[assignment]  # HiDiffusion's block attribute

    def _count_step(self, block: nn.Module, *_: object) -> None:
        block.timestep = self.cozy_step  # type: ignore[assignment]  # HiDiffusion's block attribute

    def _draw_shift(self, index: int, block: nn.Module, *_: object) -> None:
        """HiDiffusion's `torch.rand(1)`, from the request's stream at this step and window."""
        draw = (self._hidiffusion_seed * 1_000_003 + self.cozy_step * 1_009 + index) % 2**63
        block.cozy_draw = float(torch.rand(1, generator=rng.generator(draw)))  # type: ignore[assignment]  # `Windowed`'s


@functools.cache
def _windowed(parent: type[BasicTransformerBlock]) -> type[BasicTransformerBlock]:
    """HiDiffusion's MSW-MSA block (`make_diffusers_transformer_block`) over SDXL's layer-norm
    `BasicTransformerBlock`, its window shift from `cozy_draw`, not torch's global RNG.
    `remove_hidiffusion` restores `_parent`."""

    class Windowed(parent):  # type: ignore[valid-type,misc]  # a subclass of the block class HiDiffusion patched
        _parent = parent
        cozy_draw = 0.0
        info: HiDiffusionInfo

        def forward(  # type: ignore[explicit-any]  # cross_attention_kwargs: Diffusers' attention-processor kwargs
            self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None,
            timestep: torch.Tensor | None = None,
            cross_attention_kwargs: dict[str, Any] | None = None,
            class_labels: torch.Tensor | None = None, added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        ) -> torch.Tensor:
            kwargs = dict(cross_attention_kwargs or {})
            kwargs.pop("gligen", None)
            _, tokens, _ = hidden_states.shape
            height, width = self.info["size"]  # type: ignore[misc]  # set by HiDiffusion's UNet hook before blocks run
            ratio = round((height * width / tokens) ** 0.5)
            rows, columns = math.ceil(height / ratio), math.ceil(width / ratio)
            size = (math.ceil(rows / 2), math.ceil(columns / 2))
            quarter = max(0, math.ceil(self.cozy_draw * 4) - 1)
            shift = (size[0] // 4 * quarter, size[1] // 4 * quarter)
            windows = _partition(self.norm1(hidden_states), size, shift, rows, columns)
            context = encoder_hidden_states if self.only_cross_attention else None
            attended: torch.Tensor = self.attn1(
                windows, encoder_hidden_states=context, attention_mask=attention_mask, **kwargs
            )
            hidden_states = _unpartition(attended, size, shift, rows, columns) + hidden_states
            # SDXL's blocks all cross-attend: their attn2, norm2 and norm3 are set.
            crossed: torch.Tensor = self.attn2(  # pyright: ignore[reportOptionalCall]
                self.norm2(hidden_states),  # pyright: ignore[reportOptionalCall]
                encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask, **kwargs,
            )
            hidden_states = crossed + hidden_states
            fed: torch.Tensor = self.ff(self.norm3(hidden_states))  # pyright: ignore[reportOptionalCall]
            return fed + hidden_states

    return Windowed


def _partition(
    x: torch.Tensor, size: tuple[int, int], shift: tuple[int, int], rows: int, columns: int
) -> torch.Tensor:
    """HiDiffusion's `window_partition`: the map rolled by `shift`, as 2x2 windows of `size`."""
    batch, _, channels = x.shape
    x = x.view(batch, rows, columns, channels)
    if rows % 2 or columns % 2:
        x = F.interpolate(x.permute(0, 3, 1, 2).contiguous(), size=(size[0] * 2, size[1] * 2), mode="bicubic")
        x = x.permute(0, 2, 3, 1).contiguous()
    if shift[0] > 0:
        x = torch.roll(x, shifts=(-shift[0], -shift[1]), dims=(1, 2))
    x = x.view(batch, 2, size[0], 2, size[1], channels).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, size[0] * size[1], channels)


def _unpartition(
    windows: torch.Tensor, size: tuple[int, int], shift: tuple[int, int], rows: int, columns: int
) -> torch.Tensor:
    """HiDiffusion's `window_reverse`: the windows back as the map, rolled back."""
    channels = windows.shape[-1]
    batch = windows.shape[0] // 4
    x = windows.view(batch, 2, 2, size[0], size[1], channels).permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(batch, size[0] * 2, size[1] * 2, channels)
    if shift[0] > 0:
        x = torch.roll(x, shifts=(shift[0], shift[1]), dims=(1, 2))
    if rows % 2 or columns % 2:
        x = F.interpolate(x.permute(0, 3, 1, 2).contiguous(), size=(rows, columns), mode="bicubic")
        x = x.permute(0, 2, 3, 1).contiguous()
    return x.view(batch, rows * columns, channels)


class SdxlComponents(TypedDict):
    text_encoder: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection
    unet: HiDiffusionUNet
    vae: AutoencoderKL


#: DPM++ 2M Karras, ComfyUI's and A1111's usual SDXL sampler; second order, as `generative-models`' DPMPP2M.
DPMPP_2M_KARRAS: dict[str, object] = {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def _half[M: nn.Module](module: M) -> M:
    """`module.to(torch.float16)`, typed: Diffusers' and Transformers' `to` is `functools.wraps`'d,
    which the checkers bind without `self`."""
    return module.to(torch.float16)


class SdxlPipeline:
    """The four constructed component roots, built from the artifact's immutable config.

        text_encoder     196 destinations   0.229 GiB   CLIPTextModel (ViT-L text tower)
        text_encoder_2   517 destinations   1.294 GiB   CLIPTextModelWithProjection (bigG)
        unet            1680 destinations   4.782 GiB   UNet2DConditionModel
        vae              248 destinations   0.156 GiB   AutoencoderKL
                        ────────────────────────────
                        2641 destinations   6.461 GiB

    Construction declares fp16 logical destinations. Plain weights fill those
    destinations; the runtime may replace supported UNet linear leaves with native
    encoded implementations when the checkpoint carries FP8 weights.
    """

    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        text_encoder = dict(mapping["text_encoder"])
        text_encoder_2 = dict(mapping["text_encoder_2"])
        for clip_config in (text_encoder, text_encoder_2):
            clip_config["initializer_factor"] = float(clip_config["initializer_factor"])
        with transformer_init.no_init_weights():
            self.components: SdxlComponents = {
                "text_encoder": _half(CLIPTextModel(CLIPTextConfig(**text_encoder))),
                "text_encoder_2": _half(CLIPTextModelWithProjection(CLIPTextConfig(**text_encoder_2))),
                "unet": _half(construct(HiDiffusionUNet, mapping["unet"])),
                "vae": _half(construct(AutoencoderKL, mapping["vae"])),
            }
        unet = self.components["unet"]
        # HiDiffusion recognizes this constructed SDXL UNet by its module keys. Keep the
        # package free of a catalog/model identifier and let that structural check decide.
        unet.name_or_path = ""
        unet._base_num_upsamplers = unet.num_upsamplers
        unet._hidiffusion_active = False
        #: Where FaceID's LoRA arms while a face steers: every UNet attention's q, k, v and out.
        self.sites = lora.Sites(unet, attention_linears(unet))
        self.scheduler_config: dict[str, object] = dict(mapping["scheduler"])
        self.vae_scale: float = float(mapping["vae"]["scaling_factor"])


def warmup() -> None:
    """Runtime's off-lock import hook (cr-104). Every import and class is at module scope (se-041),
    so `import sdxl` already paid the cost."""


def build_pipeline(config: Config) -> SdxlPipeline:
    return SdxlPipeline(config)


class SdxlModel(Model[SdxlPipeline]):
    """Admits by pure topology satisfaction: no structural twins, so no stamp keyword.

    Runtime may replace supported linear leaves with encoded implementations. The
    model invokes those leaves through their forward methods and leaves storage and
    execution routing to Runtime. Plain checkpoints remain supported.

    THREE component sets rather than one coarse whole-pipeline set. The coarse form is
    legal and simpler, and on this card it is not servable: 6.461 GiB of weights declared
    resident at once beside a 1024px VAE decode does not fit 8 GiB, and the answer to that
    is the runtime's staging ladder — which has nothing to stage if every method declares
    everything.
    """

    #: Caller LoRAs on the UNet and both text towers, at any strength. Ingest normalizes
    #: kohya and Diffusers SDXL LoRAs to PEFT keys; the Runtime bakes them into the weights,
    #: or computes them per step when a request says `lora_mode: per_step`.
    serving = Serving(loras=("unet", "text_encoder", "text_encoder_2"))

    pipe: SdxlPipeline
    tokenizers: tuple[CLIPTokenizer, CLIPTokenizer]

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(build_pipeline)
        # Checkpoint vocabularies override the standard SDXL CLIP vocabulary. Both
        # sources are read-only and the parsed tokenizers survive between requests.
        self.tokenizers = (
            _tokenizer(loader.assets, "tokenizer"),
            _tokenizer(loader.assets, "tokenizer_2"),
        )

    @uses_components("text_encoder", "text_encoder_2", "unet", "vae")
    def generate(
        self,
        tokens: tuple[torch.Tensor, torch.Tensor],
        negative_tokens: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        hidiffusion: bool,
        seed: int,
        progress: Callable[[int, int], None],
        start: torch.Tensor | None = None,
        strength: float = 1.0,
        mask: torch.Tensor | None = None,
        controls: Controls | None = None,
        references: References | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Text encoding, every step and the decode: DPM++ 2M Karras (Diffusers'
        `DPMSolverMultistepScheduler`), or with HiDiffusion its recipe's `DDIMScheduler` at eta 1. HiDiffusion's blocks choose by step
        (RAU-Net's switch, each window's shift). Returns the image, conditioning and latents. With `start` (pixels
        [1, 3, H, W] in -1..1), Diffusers' img2img: its latents noised to the last `strength` of the steps, which run.
        With `mask` too ([1, 1, H, W], 1 redraws), Diffusers' inpaint for a 4-channel UNet: after each step the
        picture's latents, noised to the next step, are put back outside it. `controls` and `references` steer every
        step (`guided`). Guidance above 1 runs CFG against `negative_tokens`, or without them against zeros: SDXL's
        unconditional, which its training dropped the text to (generative-models' `force_uc_zero_embeddings`,
        Diffusers' `force_zeros_for_empty_prompt`)."""
        components = self.pipe.components
        first, second = components["text_encoder"], components["text_encoder_2"]
        unet, vae = components["unet"], components["vae"]
        device = torch.device(str(self.device))
        generator = rng.generator(seed)
        scheduler: DDIMScheduler | DPMSolverMultistepScheduler
        advance: Callable[[torch.Tensor, int, torch.Tensor], BaseOutput | tuple[torch.Tensor, ...]]
        if hidiffusion:
            ddim = construct(DDIMScheduler, self.pipe.scheduler_config)
            ddim.set_timesteps(steps, device=device)
            scheduler, advance = ddim, functools.partial(ddim.step, return_dict=False, eta=1.0, generator=generator)
        else:
            dpm = construct(DPMSolverMultistepScheduler, {**self.pipe.scheduler_config, **DPMPP_2M_KARRAS})
            dpm.set_timesteps(steps, device=device)
            scheduler, advance = dpm, functools.partial(dpm.step, return_dict=False)
            dpm.set_begin_index(0)
        unet.begin_denoise_request(steps, hidiffusion, seed)
        batch = 2 if guidance > 1.0 else 1
        dtype, scale = next(vae.parameters()).dtype, self.pipe.vae_scale
        latents = rng.noise((1, 4, height // 8, width // 8), generator, dtype=torch.float16, device=device)
        first_step = 0
        kept: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if start is None:
            latents = latents * scheduler.init_noise_sigma
        else:
            first_step = steps - max(1, min(steps, int(steps * strength)))
            if isinstance(scheduler, DPMSolverMultistepScheduler):
                scheduler.set_begin_index(first_step)
            with torch.no_grad():
                clean = vae.encode(start.to(device=device, dtype=next(vae.parameters()).dtype)).latent_dist.mode()
            clean = (clean.float() * self.pipe.vae_scale).to(torch.float16)
            start_time = cast(torch.IntTensor, scheduler.timesteps[first_step : first_step + 1])  # Diffusers types it so
            if mask is not None:
                area = F.interpolate(mask.to(device, torch.float32), size=clean.shape[-2:], mode="area").to(torch.float16)
                kept = clean, latents, area
            latents = scheduler.add_noise(clean, latents, start_time)
        # SDXL's micro-conditioning: (original_h, original_w, crop_top, crop_left, target_h,
        # target_w). The bucket IS the original size — these fine-tunes were trained on it.
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]] * batch, device=device, dtype=torch.float16
        )
        with torch.no_grad():
            prompt, pooled = encode_tokens(first, second, *(t.to(device) for t in tokens))
            if guidance > 1.0:
                if negative_tokens is None:
                    negative, negative_pool = torch.zeros_like(prompt), torch.zeros_like(pooled)
                else:
                    negative, negative_pool = encode_tokens(first, second, *(t.to(device) for t in negative_tokens))
                prompt, pooled = torch.cat([negative, prompt]), torch.cat([negative_pool, pooled])
            prompt, pooled = prompt.to(torch.float16), pooled.to(torch.float16)
            # The schedulers read their own tables by the host timestep; the UNet takes the device one.
            for i, t in enumerate(scheduler.timesteps.tolist()):
                if i < first_step:
                    continue
                if hidiffusion:  # its hooks read the step; the plain UNet reads nothing of it
                    unet.cozy_step = i
                noise = guided(unet, scheduler.scale_model_input(latents, t), scheduler.timesteps[i], prompt,
                               pooled, time_ids, guidance, controls=controls, references=references)
                preview(lambda: (latents - (1 - scheduler.alphas_cumprod[t]) ** 0.5 * noise) / scheduler.alphas_cumprod[t] ** 0.5, "sdxl")
                latents = advance(noise, t, latents)[0]
                if kept is not None:
                    clean, drawn, area = kept
                    if i + 1 < steps:
                        upcoming = cast(torch.IntTensor, scheduler.timesteps[i + 1 : i + 2])
                        clean = scheduler.add_noise(clean, drawn, upcoming)
                    latents = (1 - area) * clean + area * latents
                progress(i - first_step, steps - first_step)
            image = decode_latents(vae, latents.to(dtype=dtype), scale)
        return image, prompt, pooled, latents


# ------------------------------------------------------------------ the handler


def _finite(value: torch.Tensor) -> float:
    """The tensor's largest magnitude, with NaN neutralized so the metric is spellable.

    A metric that cannot be serialized is a metric that is not there: the observation emit
    boundary refuses a non-finite value, so the NaN FRACTION is its own number and this one
    stays a real magnitude.
    """
    return round(float(torch.nan_to_num(value, 0.0, 0.0, 0.0).abs().max()), 4)


def _tokenizer(assets: ModelAssets, name: str) -> CLIPTokenizer:
    """Read the complete tokenizer pair from the checkpoint's verified model assets."""
    pad = {"tokenizer": "<|endoftext|>", "tokenizer_2": "!"}[name]
    vocab_bytes = assets.read(f"{name}/vocab.json")
    merges_bytes = assets.read(f"{name}/merges.txt")
    vocab = json.loads(vocab_bytes)
    merges = [
        tuple(line.split(" "))
        for line in merges_bytes.decode().splitlines()
        if line and not line.startswith("#version")
    ]
    return CLIPTokenizer(  # Transformers types merges as strings; Tokenizers' BPE takes these pairs.
        vocab=vocab, merges=merges, errors="replace", pad_token=pad, model_max_length=77  # type: ignore[arg-type]
    )


def _tokenize(tokenizers: tuple[CLIPTokenizer, CLIPTokenizer], prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = tokenizers
    return _ids(first, prompt), _ids(second, prompt)


def _ids(tok: CLIPTokenizer, prompt: str) -> torch.Tensor:
    ids: torch.Tensor = tok(
        prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return ids


#: THE NaN CHECK reads the FLOAT decode, before quantization, and that is the whole of
#: what a red arm taught here: `torch.isnan` on a `clamp(0,1).to(uint8)` tensor is
#: constant False, because a uint8 cannot be NaN — the very quantization that makes an
#: image encodable is what erases the evidence of a diverged decode.
#:
#: Pixel SPREAD is reported as telemetry, not refused. A flat field is a real failure
#: mode (a wrong VAE scaling factor, an unfilled decoder) but it is also a real REQUEST:
#: "solid white background", "minimalist pure black poster", "flat pastel swatch". A
#: refusal here destroyed a correct, completed, billed generation.
def _integrity(image: torch.Tensor, pixels: torch.Tensor, tel: Telemetry) -> None:
    nan_fraction = float(torch.isnan(image).float().mean())
    tel.metric("image_nan_fraction", round(nan_fraction, 6))
    if nan_fraction > 0.0:
        raise OutputError(
            f"the decode produced NaN over {nan_fraction:.4%} of the image and this "
            "package does not publish it: try another seed, and if it repeats another lane "
            "with `--lane bf16`",
            code="output_integrity",
        )
    tel.metric("image_spread", round(float(pixels.to(torch.float32).std()), 4))


#: HiDiffusion's README SDXL negative prompt, part of the recipe it publishes.
_HIDIFFUSION_NEGATIVE = (
    "blurry, ugly, duplicate, poorly drawn face, deformed, mosaic, artifacts, bad limbs"
)


def _negative_prompt(requested: str, hidiffusion: bool) -> str:
    """With HiDiffusion, the request's negative prompt followed by its recipe's: DDIM alone
    still hatched 2 of 5 seeds at 4 MP that the full recipe left clean."""
    if not hidiffusion:
        return requested
    return f"{requested}, {_HIDIFFUSION_NEGATIVE}" if requested else _HIDIFFUSION_NEGATIVE


class GenerateFacts(msgspec.Struct, frozen=True):
    width: int
    height: int
    resolved_steps: int
    guidance: float
    hidiffusion_applied: bool
    negative_prompt: str
    adjustments: tuple[AdjustmentRow, ...]


def _resolve(payload: GenerateInput, first: tuple[int, int] | None) -> GenerateFacts:
    """The request's size (a missing aspect ratio or size is `first`'s, the picture it starts from or follows), steps,
    guidance and HiDiffusion choice. Control maps run base SDXL: HiDiffusion's RAU-Net reshapes the UNet's down path,
    which the ControlNet's residuals add into."""
    adj = AdjustmentRows()
    target = payload.target(first, megapixels=1.0)
    sizes = _SQUARE if target.ratio[0] == target.ratio[1] else _SIZES
    width, height = sizes.fit(adj, target.ratio, sizes.pixels(adj, target))
    above_native = width * height > _NATIVE * MEGAPIXEL
    controlled = any(c.strength > 0 for c in payload.control)
    hidiffusion_applied = above_native if payload.hidiffusion is None else payload.hidiffusion
    if controlled and hidiffusion_applied:
        if payload.hidiffusion:
            adj.warn("hidiffusion", "off with control maps: the ControlNet steers base SDXL")
        hidiffusion_applied = False
    # DPM++ 2M Karras at 30 (se-121); HiDiffusion's recipe is 30 DDIM steps.
    default_steps, default_guidance = (30, 7.5) if hidiffusion_applied else (30, 7.0)
    steps = held(adj, "steps", payload.steps or default_steps, 1, 50)
    guidance = held(adj, "guidance", default_guidance if payload.guidance is None else payload.guidance, 0.0, 20.0)
    negative_prompt = _negative_prompt(payload.negative_prompt or "", hidiffusion_applied)
    if above_native and not hidiffusion_applied:
        adj.warn("hidiffusion", "HiDiffusion off above 1 MP: expect duplicated or tiled subjects")
    if hidiffusion_applied and not above_native:
        adj.warn("hidiffusion", "HiDiffusion at 1 MP composes at half resolution: expect simpler scenes")
    return GenerateFacts(
        width, height, steps, guidance, hidiffusion_applied, negative_prompt, tuple(adj.rows)
    )


def resolve_generate(payload: GenerateInput) -> GenerateFacts:
    """`_resolve` before any picture is read: the run resolves again from its first picture's size."""
    return _resolve(payload, None)


def _said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[Image]]:
    """SDXL's words and the pictures for the IP-Adapter. A mention reads as its picture's description, else as
    nothing: the IP-Adapter carries the picture, and upstream prompts never name it (Anima's `_said`)."""
    keys = list(prompt.references)
    if not keys:
        return prompt.text, []
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"SDXL's IP-Adapter takes {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
    words = MENTION.sub(lambda found: prompt.references[found[1]].description.strip()
                        if found[1] in prompt.references else found[0], prompt.text)
    return " ".join(words.split()), [prompt.references[key] for key in keys[:MAX_PICTURES]]


def _pixels(picture: PILImage.Image) -> torch.Tensor:
    """[1, C, H, W] in 0..1."""
    pixels = torch.frombuffer(bytearray(picture.tobytes()), dtype=torch.uint8)
    return pixels.view(picture.height, picture.width, -1).permute(2, 0, 1).unsqueeze(0).float() / 255


def _maps(controls: list[Control[Kind]], width: int, height: int, modes: int,
          adj: Adjustments) -> tuple[list[tuple[Control[Kind], torch.Tensor]], torch.Tensor | None]:
    """Each control map at the output's size in 0..1, [1, 3, H, W], as the union pipeline's processor gives it
    (bicubic); `inpaint`'s blacked out where it is transparent (Diffusers' ProMax example), and that area, [1, 1, H, W]
    (1 redraws), the first inpaint map's."""
    maps: list[tuple[Control[Kind], torch.Tensor]] = []
    area = None
    for control in controls:
        if control.strength <= 0:
            continue
        if MODES[control.kind] >= modes:
            adj.warn("control", f"this ControlNet has no {control.kind} mode: that map is ignored")
            continue
        if control.kind != "inpaint":
            picture = control.image.read().resize((width, height), PILImage.Resampling.BICUBIC)
            maps.append((control, _pixels(picture)))
            continue
        pixels = _pixels(control.image.read(mode="RGBA").resize((width, height), PILImage.Resampling.BICUBIC))
        redraw = (pixels[:, 3:] < 0.5).float()
        if area is None:
            area = redraw
        else:
            adj.warn("control", "SDXL inpaints one area at a time: the first inpaint map's is redrawn")
        maps.append((control, pixels[:, :3] * (1 - redraw)))
    return maps, area


#: An inpaint's edge, in pixels: the area grows this far, then fades out over about as far again, so the new picture
#: meets the kept one in a ring around the area (Anima's `_feathered`: InvokeAI's gradient mask; A1111 blurs its mask,
#: ComfyUI grows it). A hard edge left a bright ring where the kept latents met the new ones.
_GROW, _FADE = 16, 8


def _feathered(mask: torch.Tensor) -> torch.Tensor:
    """`mask` [1, 1, H, W] (1 redraws) grown by `_GROW` px and faded by three box blurs of `_FADE` px."""
    grown = F.max_pool2d(mask, (1, 2 * _GROW + 1), stride=1, padding=(0, _GROW))
    soft = F.max_pool2d(grown, (2 * _GROW + 1, 1), stride=1, padding=(_GROW, 0))
    for _ in range(3):
        soft = F.avg_pool2d(soft, (1, 2 * _FADE + 1), stride=1, padding=(0, _FADE), count_include_pad=False)
        soft = F.avg_pool2d(soft, (2 * _FADE + 1, 1), stride=1, padding=(_FADE, 0), count_include_pad=False)
    return torch.maximum(mask, soft)


def _start(payload: GenerateInput, width: int, height: int,
           area: torch.Tensor | None) -> tuple[torch.Tensor | None, float]:
    """The picture to start from in -1..1, [1, 3, H, W], and its strength: `image`, else an inpaint's own picture,
    redrawn whole inside its area."""
    picture = payload.image.read() if payload.image is not None else None
    strength = payload.strength
    if picture is None and area is not None:
        picture = next(c.image.read() for c in payload.control if c.kind == "inpaint" and c.strength > 0)
        strength = 1.0
    if picture is None:
        return None, strength
    canvas = picture.resize((width, height), PILImage.Resampling.LANCZOS)
    return _pixels(canvas) * 2 - 1, strength


#: Which requests need each adapter (cl-1103, `_generate`): the ControlNet with a map at a strength above 0, the image
#: encoder, IP-Adapter Plus and the face detector with a picture in the prompt at a reference_strength above 0.
_MAPPED = "request.control.exists(c, c.strength > 0)"
_STEERED = 'request.reference_strength > 0 && request.prompt.references.exists(r, r.type == "image")'
_BOX_STEERED = 'request.reference_strength > 0 && request.instructions.references.exists(r, r.type == "image")'
_NEEDS = {"controlnet": _MAPPED, "image_encoder": _STEERED, "ip_adapter": _STEERED, "faces": _STEERED}
_BOX_NEEDS = {"controlnet": _MAPPED, "image_encoder": _BOX_STEERED, "ip_adapter": _BOX_STEERED, "faces": _BOX_STEERED}
#: ArcFace, FaceID-PlusV2 and its LoRA: what only the run learns, when a picture's subject is a face (`_references`).
_ON_DEMAND = ("face_identity", "face_adapter", "face_lora")
_DEFAULTS = {"model": "stabilityai/sdxl@1.0.0"}
#: The adapters, each its author's file as uploaded (se-175), fetched only when a request uses it.
_ADAPTERS = {
    "model": "stabilityai/sdxl@1.0.0",
    "controlnet": "xinsir/controlnet-union-sdxl-promax@0.1.0",
    "image_encoder": "laion/clip-vit-h-14-laion2b-s32b-b79k@0.1.0",
    "ip_adapter": "h94/ip-adapter-plus-sdxl-vit-h@0.1.0",
    "faces": "deepinsight/buffalo-l@0.1.0",
    "face_identity": "deepinsight/buffalo-l@0.1.0",
    "face_adapter": "h94/ip-adapter-faceid-plusv2-sdxl@0.1.0",
    "face_lora": "h94/ip-adapter-faceid-plusv2-sdxl-lora@0.1.0",
}


@dataclasses.dataclass(frozen=True)
class _Adapters:
    """The models a denoise's references may use."""

    image_encoder: ImageEncoder
    ip_adapter: IPAdapter
    faces: FaceDetector
    face_identity: FaceRecognizer
    face_adapter: FaceIDAdapter
    face_lora: FaceLoRA


def _references(ctx: Context, pictures: list[PILImage.Image], strength: float, guided: bool,
                adapters: _Adapters) -> tuple[list[Steer], int]:  # fmt: skip
    """The denoise's parts, and how many pictures are faces: IP-Adapter Plus over every picture whole (its hair, outfit,
    style and scene), and FaceID-PlusV2 besides for a picture whose subject is a face ArcFace can read (`faces.subject`,
    `faces.readable`), each at its own scale on the same cross-attentions (Diffusers' multi-IP-Adapter)."""
    found = [subject(adapters.faces.find(picture), *picture.size) for picture in pictures]
    found = [face if face is not None and readable(face) else None for face in found]
    portraits = [(picture, face) for picture, face in zip(pictures, found, strict=True) if face is not None]
    if portraits:
        for slot in _ON_DEMAND:
            ctx.need(slot)
    parts = []
    if pictures:
        embeds, empty = adapters.image_encoder.tokens(pictures)
        tokens = adapters.ip_adapter.tokens(embeds, empty, guided)
        parts.append(Steer(tokens, adapters.ip_adapter.graph.sites, strength))
    if portraits:
        identities = adapters.face_identity.identities([aligned(picture, face, 112) for picture, face in portraits])
        # The face aligned at 224² is the CLIP picture (h94's PlusV2 example: `norm_crop(…, image_size=224)`).
        embeds, empty = adapters.image_encoder.tokens([crop(picture, face, 224) for picture, face in portraits])
        tokens = adapters.face_adapter.tokens(identities, embeds, empty, guided)
        parts.append(Steer(tokens, adapters.face_adapter.graph.sites, FACE_SCALE * strength / REFERENCE_STRENGTH))
    return parts, len(portraits)


def _generate(ctx: Context, payload: GenerateInput, facts: GenerateFacts, model: SdxlModel, controlnet: ControlNet,
              adapters: _Adapters, tel: Telemetry, adj: Adjustments) -> ImageResult:
    hold_decode_dtype(model.pipe.components["vae"], tel)
    prompt, pictures = _said(payload.prompt, adj)
    first = payload.image or next((c.image for c in payload.control if c.strength > 0), None)
    if first is not None:
        facts = _resolve(payload, first.read().size)
    adj.extend(facts.adjustments)
    seed = payload.drawn_seed()
    width, height = facts.width, facts.height
    steps, guidance = facts.resolved_steps, facts.guidance
    hidiffusion_applied, negative_prompt = facts.hidiffusion_applied, facts.negative_prompt
    # The value plane, and the only branch in this file that reads a request number: above
    # 1.0 the negative branch is worth its second forward pass, at or below it is not.
    classifier_free = guidance > 1.0

    with tel.stage("tokenize"):
        ids, ids_2 = _tokenize(model.tokenizers, prompt)
        # No negative words: SDXL's zero unconditional (`SdxlModel.generate`), not CLIP's reading of "".
        negative = _tokenize(model.tokenizers, negative_prompt) if classifier_free and negative_prompt.strip() else None

    controls = None
    area = None
    if any(c.strength > 0 for c in payload.control):
        with tel.stage("reading pictures"):
            maps, area = _maps(payload.control, width, height, controlnet.graph.modes, adj)
        if maps:
            controls = Controls(controlnet.graph, [m for _, m in maps], [MODES[c.kind] for c, _ in maps],
                                [c.strength for c, _ in maps])  # fmt: skip
            tel.log("control", kinds=", ".join(c.kind for c, _ in maps))
    start, strength = _start(payload, width, height, area)
    if area is not None:
        area = _feathered(area)
    parts: list[Steer] = []
    faced = 0
    if pictures and payload.reference_strength > 0:
        with tel.stage("encoding references"):
            parts, faced = _references(ctx, [p.file.read() for p in pictures], payload.reference_strength,
                                       classifier_free, adapters)  # fmt: skip

    def body(references: References | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return model.generate(
            (ids, ids_2), negative, width=width, height=height, steps=steps, guidance=guidance,
            hidiffusion=hidiffusion_applied, seed=seed, progress=ctx.progress, start=start, strength=strength,
            mask=area, controls=controls, references=references,
        )

    def steered() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with steering(model.pipe.components["unet"], parts) as references:
            return body(references)

    run: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = steered if parts else body
    if controls is not None:
        run = functools.partial(controlnet.armed, run)
    if parts:
        run = functools.partial(adapters.ip_adapter.armed, run)
    if faced:
        run = functools.partial(adapters.face_adapter.armed, run)
        run = functools.partial(adapters.face_lora.applied, model.pipe.sites, run)
    with tel.stage("generate"):
        image, encoded, pooled, latents = run()
    if area is not None and start is not None:
        # Around an inpaint the picture's own pixels, faded into the new ones: no VAE round trip shows there.
        kept = area.to(image.device, torch.float32)
        image = kept * image.float() + (1 - kept) * start.to(image.device, torch.float32)
    tel.metric("prompt_absmax", _finite(encoded))
    tel.metric("pooled_absmax", _finite(pooled))
    tel.metric("latent_absmax", _finite(latents))
    tel.metric("image_absmax", _finite(image))
    # F32 and rounded, as Diffusers' `VaeImageProcessor`: bf16 then truncation darkened every pixel.
    pixels = ((image[0].float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    pixels = pixels.permute(1, 2, 0).contiguous()
    _integrity(image, pixels, tel)

    rgb = bytes(pixels.cpu().numpy().tobytes())
    decoded_h, decoded_w = int(pixels.shape[0]), int(pixels.shape[1])
    tel.metric("decoded_pixels", float(decoded_w * decoded_h))
    with tel.stage("encode_webp"):
        asset = ImageFile.from_image(ImageFrame(decoded_w, decoded_h, rgb), format="webp")
    tel.log("sampled", sampler="ddim eta=1.0" if hidiffusion_applied else "dpmpp_2m_karras", steps=steps,
            guidance=guidance, classifier_free=classifier_free, hidiffusion=hidiffusion_applied,
            image_to_image=start is not None, references=len(pictures) if parts else 0, faces=faced)
    return ImageResult(asset, decoded_w, decoded_h, seed, prompt)


@app.entrypoint(defaults=_ADAPTERS, needs=_NEEDS, on_demand=_ON_DEMAND, memoize=True, preflight=resolve_generate)
def generate(
    ctx: Context,
    payload: GenerateInput,
    facts: Preflight[GenerateFacts],
    model: SdxlModel,
    controlnet: ControlNet,
    image_encoder: ImageEncoder,
    ip_adapter: IPAdapter,
    faces: FaceDetector,
    face_identity: FaceRecognizer,
    face_adapter: FaceIDAdapter,
    face_lora: FaceLoRA,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """SDXL 1.0 image, as a WebP: from words, steered by the pictures they mention and by control maps, or from a
    start picture.

    Tokenize, encode, denoise, decode, encode the WebP."""
    adapters = _Adapters(image_encoder, ip_adapter, faces, face_identity, face_adapter, face_lora)
    return _generate(ctx, payload, facts, model, controlnet, adapters, tel, adj)


class ImageBox(msgspec.Struct, kw_only=True):
    #: `generate`'s prompt, and the pictures it mentions for the IP-Adapter (`_said`).
    instructions: Prompt[Image] = Prompt()
    #: `generate`: start from this picture (image to image).
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    #: `generate`: how much of the start picture is redrawn.
    strength: Annotated[float, msgspec.Meta(gt=0.0, le=1.0), Advanced()] = 0.9
    #: `generate`: control maps, each through its mode of the union ControlNet.
    control: list[Control[Kind]] = msgspec.field(default_factory=list)
    #: `generate`: how strongly the mentioned pictures steer.
    reference_strength: Annotated[float, msgspec.Meta(ge=0.0, le=2.0), Advanced()] = REFERENCE_STRENGTH
    #: `resolve_generate`, held to SDXL's 1:4 through 4:1 buckets.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 4, max=4)] | None = None
    #: `resolve_generate`: base SDXL's 1 MP, HiDiffusion's 2048² and 8 MP classes, and its 4096² flagship at 1:1 only
    #: (this module's docstring, `_SIZES`).
    megapixels: Literal[1, 4, 8, 16] = 1
    #: `_negative_prompt`.
    negative_prompt: str | None = None
    seed: Seed | None = None


def _request(box: ImageBox) -> GenerateInput:
    """`generate`'s request, HiDiffusion as `resolve_generate` chooses it: above 1 MP."""
    return GenerateInput(prompt=box.instructions, aspect_ratio=box.aspect_ratio, megapixels=box.megapixels,
                         seed=box.seed, negative_prompt=box.negative_prompt, image=box.image, strength=box.strength,
                         control=box.control, reference_strength=box.reference_strength)  # fmt: skip


def resolve_image(payload: ImageBox) -> GenerateFacts:
    return resolve_generate(_request(payload))


@app.tool(label="Image", named="image", defaults=_ADAPTERS, needs=_BOX_NEEDS, on_demand=_ON_DEMAND, memoize=True,
          preflight=resolve_image)
def image(ctx: Context, payload: ImageBox, facts: Preflight[GenerateFacts], model: SdxlModel, controlnet: ControlNet,
          image_encoder: ImageEncoder, ip_adapter: IPAdapter, faces: FaceDetector, face_identity: FaceRecognizer,
          face_adapter: FaceIDAdapter, face_lora: FaceLoRA, tel: Telemetry, adj: Adjustments) -> ImageResult:  # fmt: skip
    """SDXL on a canvas: `generate` from the box. Mentioned pictures steer it, a start picture makes it an edit."""
    adapters = _Adapters(image_encoder, ip_adapter, faces, face_identity, face_adapter, face_lora)
    return _generate(ctx, _request(payload), facts, model, controlnet, adapters, tel, adj)


#: What SDXL is told of a picture it upscales, and kept from.
UPSCALE_PROMPT = "a high quality, highly detailed, sharp picture"
UPSCALE_NEGATIVE = "blurry, lowres, jpeg artifacts, noise, out of focus"


class UpscaleInput(msgspec.Struct, kw_only=True):
    #: `upscale`: the picture, the box's own version.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
    #: `upscale`: the picture's ratio at this size (`_SIZES.fit`), up to 8 MP.
    megapixels: Literal[2, 4, 8] = 8
    #: `SdxlModel.generate`'s `strength`: the share of the 30 steps run over the picture (se-123).
    denoise: Annotated[float, msgspec.Meta(gt=0, le=1), Advanced()] = 0.2
    seed: Seed | None = None


@app.tool(label="Upscale", named="image", defaults=_DEFAULTS, memoize=True)
def upscale(ctx: Context, payload: UpscaleInput, model: SdxlModel, tel: Telemetry,
            adj: Adjustments) -> ImageResult:
    """The picture larger: Lanczos to the size asked, then SDXL img2img over the whole canvas (DPM++ 2M Karras)."""
    hold_decode_dtype(model.pipe.components["vae"], tel)
    with tel.stage("reading picture"):
        picture = payload.image.read()
    width, height = _SIZES.fit(adj, picture.size, payload.megapixels * MEGAPIXEL)
    canvas = picture.resize((width, height), PILImage.Resampling.LANCZOS)
    pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).view(height, width, 3)
    start = (pixels.permute(2, 0, 1)[None].float() / 127.5 - 1.0)
    seed = payload.seed if payload.seed is not None else ImageRequest(prompt="").drawn_seed()
    with tel.stage("tokenize"):
        tokens = _tokenize(model.tokenizers, UPSCALE_PROMPT)
        negative = _tokenize(model.tokenizers, UPSCALE_NEGATIVE)
    with tel.stage("generate"):
        image, *_ = model.generate(tokens, negative, width=width, height=height, steps=30, guidance=7.0,
                                   hidiffusion=False, seed=seed, progress=ctx.progress, start=start,
                                   strength=payload.denoise)  # fmt: skip
    rgb = ((image[0].float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    _integrity(image, rgb, tel)
    with tel.stage("saving image"):
        asset = ImageFile.from_image(ImageFrame(width, height, bytes(rgb.cpu().numpy().tobytes())), format="png")
    tel.log("upscale", source=f"{picture.width}x{picture.height}", width=width, height=height, denoise=payload.denoise)
    return ImageResult(asset, width, height, seed, UPSCALE_PROMPT)


# ------------------------------------------------------------------ quantized lanes

# The SDXL recipes: which weights each lane encodes; every other tensor inherits unchanged.
# `cozy model quantize <model> --lane int8|fp8|w4a8 --package <this>` runs the one for that lane.
# `cozy_runtime.derive` owns the encodings and the round-trip tripwire.

#: The fp8 lane: the UNet's block-aligned rank-2 weights, row-wise.
FP8 = Recipe(
    lane="fp8", model=SdxlModel, plan=QuantizationPlan(encoding="fp8.rowwise", components=("unet",))
)

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores): every transformer-block linear of
#: the UNet whose input width is a multiple of 256 (the rotation group), and every linear of both
#: CLIP encoders. The 640-channel levels' self-attention, cross-attention queries, GEGLU input and
#: projections (input width 640), the ResNet convolutions, time embeddings and the VAE keep their
#: source bytes.
_UNET_1280 = ("down_blocks.2", "mid_block", "up_blocks.0")
INT8 = Recipe(
    lane="int8",
    model=SdxlModel,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("unet", "text_encoder", "text_encoder_2"),
        include=(
            *(
                f"unet.{level}.attentions.*.{key}"
                for level in _UNET_1280
                for key in (
                    "transformer_blocks.*.attn[12].to_[qkv].weight",
                    "transformer_blocks.*.attn[12].to_out.0.weight",
                    "transformer_blocks.*.ff.net.*weight",
                    "proj_[io]*.weight",
                )
            ),
            "unet.*_blocks.1.attentions.*.transformer_blocks.*.attn2.to_[kv].weight",
            "unet.*_blocks.1.attentions.*.transformer_blocks.*.ff.net.2.weight",
            "text_encoder.text_model.encoder.layers.*.self_attn.*_proj.weight",
            "text_encoder.text_model.encoder.layers.*.mlp.fc[12].weight",
            "text_encoder_2.text_model.encoder.layers.*.self_attn.*_proj.weight",
            "text_encoder_2.text_model.encoder.layers.*.mlp.fc[12].weight",
        ),
        output_precision="preserve",
    ),
)

#: The 4-bit lane before Blackwell (xs-062): the int8 lane's UNet linears in W4A8
#: (`w4a8.convrot`, multiplied on the int8 tensor cores: Ampere, Ada, Hopper); both CLIP
#: encoders stay int8.
W4A8 = Recipe(
    lane="w4a8",
    model=SdxlModel,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule("unet.*", "w4a8.convrot"),),
        output_precision="preserve",
    ),
)
