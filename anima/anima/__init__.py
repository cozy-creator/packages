"""Anima through Diffusers' maintained modular pipeline: text to image, image to image, inpaint, mentioned
pictures through LuciferTC's IP-Adapter and control maps through kohya's ControlNet-LLLite (se-127)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Annotated, Literal

import msgspec
import torch
import torch.nn.functional as F
from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    ImageFrame,
    Outputs,
    QuantizationPlan,
    Recipe,
    Seed,
    Telemetry,
)
from cozy_runtime.fields import MENTION, Advanced, AspectRatio, FileBound, Image, Label, Prompt, Ratios
from cozy_runtime.image import Control, Guidance, ImageRequest, ImageResult, NegativePrompt, Sizes, Steps, held
from cozy_runtime.media import ImageFile
from PIL import Image as PILImage

from .adapters import Adapters, ImageEncoder, IPAdapter, LLLite, References, Shifts
from .model import _SAVE, AnimaModel, Start, _Phases

app = App()


#: Anima's native resolution classes (se-024): the ~1 MP training set (1024²) to the 1536 class
#: (1536², 2.25 MiP) the model is meant to run at, this package's default. Denoise attention is full
#: (no windowing), so time and VRAM scale roughly linearly with area. The Cosmos rotary table holds
#: 128 patches of 16 px on an axis.
SIZES = Sizes((1.0, 2.25), (32, 128 * 16), 32, "Anima's 1 MP training size to its 1536 class")
#: Steps (default, range) and CFG scale (default, range): the model card's.
_STEPS, _GUIDANCE = (30, 8, 50), (4.5, 1.0, 10.0)


#: The official Anima model card's negative prompt, verbatim (se-026). Owner-labelled bank
#: 2026-09-02: with seed, steps, guidance and geometry fixed, THIS string is the quality
#: knob — Anima's score-bucket conditioning acts through the negative, so pushing away
#: score_1/score_2/score_3 (plus jpeg artifacts, chromatic aberration, artist name — a
#: literal tag that suppresses signature text) moves output off the model's flat dated
#: aesthetic. The negative is the ISOLATED cause: the labelled loser carried `score_7, safe`
#: in its positive prompt and still lost, and one labelled winner carried no positive quality
#: prefix at all. The card's positive quality tags are prompt text, the caller's to write (the
#: README gives them). Changing this string changes every render that omits the field.
CARD_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, "
    "artist name, blurry, jpeg artifacts, chromatic aberration"
)


#: The control maps Anima has an adapter for (canny waits for one).
Kind = Literal["depth", "pose", "lineart", "sketch", "inpaint"]
#: The pictures the IP-Adapter takes at once (InvokeAI's cap): past them the last mentioned go, with a warning.
MAX_PICTURES = 4


class GenerateInput(ImageRequest):
    #: Its words, and the pictures it mentions: each a reference for the IP-Adapter (`_said`).
    prompt: Prompt[Image] = Prompt()  # type: ignore[assignment]  # a prompt with pictures, in place of the base's text
    #: Image to image: start from this picture, `strength` of the schedule redrawing it.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    #: How much of the picture is redrawn: 1 is from noise.
    strength: Annotated[float, msgspec.Meta(gt=0.0, le=1.0), Advanced()] = 0.6
    #: Control maps, each through its kind's ControlNet-LLLite; `inpaint`'s is the picture with the area to
    #: redraw transparent.
    control: list[Control[Kind]] = msgspec.field(default_factory=list)
    #: How strongly the mentioned pictures steer, the IP-Adapter's token scale.
    reference_strength: Annotated[float, msgspec.Meta(ge=0.0, le=2.0), Advanced()] = 1.0
    steps: Steps = None
    guidance: Guidance = None
    #: `generate`: None is the card's (`CARD_NEGATIVE`).
    negative_prompt: NegativePrompt = None
    #: CFG interval, `AnimaModel.render`'s guider (cr-086 arm 1, Kynkäänniemi et al., NeurIPS
    #: 2024): guidance helps only in a middle band of the noise schedule, so the uncond forward is
    #: skipped outside [start, stop) of the step fraction. (0, 1) is full-range CFG. The banked default is
    #: ON (se-026, 2026-09-03): at the default 1536 class it cut the denoise loop from 205 s
    #: to 156 s on an sm89 4070, and the same-seed image was at least as good — the paper's
    #: own claim is that the omitted band costs nothing. Subtractive and bounded: every block
    #: still runs on every step, so its failure mode is uniform rather than structural.
    cfg_interval_start: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.15
    cfg_interval_stop: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.7
    #: First-block cache (cr-086 arm 2, FBCache, `_apply_first_block_cache`): when the first
    #: transformer block's residual moves less than this threshold between steps, the remaining
    #: 27 blocks are skipped and the cached tail residual is reused. Cond and uncond passes keep separate
    #: cache states, so a request with it on runs CFG as two batch-1 forwards.
    #:
    #: DEFAULT OFF, deliberately (se-026, 2026-09-03). Measured at 0.075 rather than assumed:
    #: it extrapolates 44% of forwards, and against the same seed it consistently smooths
    #: faces and coarsens fine texture — the two places the first block is a bad proxy for
    #: the other 27. It also costs VRAM, because those cached residuals stay live: ALONE at
    #: the default 1536 class it refuses `device_shortfall` on an 8 GiB card, reproducibly.
    #: It survives only when `cfg_interval` happens to be narrowing the live cache states,
    #: and a default that works only because another default masks it is not a default.
    first_block_cache: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.0


def warmup() -> None:
    """Runtime's off-lock import hook (cr-104). Every import is at module scope (se-041), so
    `import anima` already paid the cost and nothing is left to defer."""


#: The adapters, each its author's file as uploaded (se-127), fetched only when a request uses it.
_DEFAULTS = {
    "model": "circlestone-labs/anima@1.0.0",
    "image_encoder": "google/siglip2-base-patch16-512@0.1.0",
    "ip_adapter": "lucifertc/ip-adapter@0.1.0",
    "depth": "kohya-ss/lllite-depth@0.1.0",
    "pose": "kohya-ss/lllite-pose@0.1.0",
    "lineart": "kohya-ss/lllite-any-test-like@0.1.0",
    "sketch": "kohya-ss/lllite-any-test-like@0.1.0",
    "inpaint": "kohya-ss/lllite-inpainting@0.1.0",
}
_ON_DEMAND = ("image_encoder", "ip_adapter", "depth", "pose", "lineart", "sketch", "inpaint")


def _said(prompt: Prompt[Image], adj: Adjustments) -> tuple[str, list[Image]]:
    """Anima's words and the pictures for the IP-Adapter. A mention reads as its picture's description, else as
    nothing: the IP-Adapter carries the picture, and upstream prompts never name it."""
    keys = list(prompt.references)
    if len(keys) > MAX_PICTURES:
        adj.warn("prompt", f"Anima's IP-Adapter takes {MAX_PICTURES} pictures, so it goes without "
                           f"{', '.join(f'<{key}>' for key in keys[MAX_PICTURES:])}")
    words = MENTION.sub(lambda found: prompt.references[found[1]].description.strip()
                        if found[1] in prompt.references else found[0], prompt.text)
    return " ".join(words.split()), [prompt.references[key] for key in keys[:MAX_PICTURES]]


def _map(control: Control[Kind], width: int, height: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """A control map at the output's size in [-1, 1], [1, 3, H, W] (kohya's node: bicubic); for `inpaint`, the RGB
    blacked out where it is transparent with the mask as a 4th channel, and that mask, [1, 1, H, W] (1 redraws)."""
    picture = control.image.read(mode="RGBA" if control.kind == "inpaint" else "RGB")
    pixels = torch.frombuffer(bytearray(picture.tobytes()), dtype=torch.uint8)
    pixels = pixels.view(picture.height, picture.width, -1).permute(2, 0, 1).unsqueeze(0).float() / 255
    rgb = F.interpolate(pixels[:, :3], size=(height, width), mode="bicubic").clamp(0, 1) * 2 - 1
    if control.kind != "inpaint":
        return rgb, None
    mask = (F.interpolate(pixels[:, 3:], size=(height, width), mode="nearest") < 0.5).float()
    return torch.cat([rgb * (1 - mask), mask * 2 - 1], dim=1), mask


def _start(payload: GenerateInput, width: int, height: int, adj: Adjustments) -> Start | None:
    """Image to image from `image`, or an inpaint from the first `inpaint` map (its picture where `image` is
    absent, fully redrawn inside the mask)."""
    inpaints = [c for c in payload.control if c.kind == "inpaint"]
    if len(inpaints) > 1:
        adj.warn("control", "Anima inpaints one area at a time: the first inpaint map is used")
    picture = None if payload.image is None else payload.image.read().resize((width, height), PILImage.Resampling.LANCZOS)
    if not inpaints:
        return None if picture is None else Start(picture, payload.strength)
    _, mask = _map(inpaints[0], width, height)
    assert mask is not None
    if picture is None:
        picture = inpaints[0].image.read().resize((width, height), PILImage.Resampling.LANCZOS)
    latent = F.interpolate(mask, size=(height // 8, width // 8), mode="area")
    return Start(picture, payload.strength if payload.image is not None else 1.0, latent.unsqueeze(2))


def _generate(
    ctx: Context,
    payload: GenerateInput,
    model: AnimaModel,
    encoder: ImageEncoder,
    ip_adapter: IPAdapter,
    lllites: dict[str, LLLite],
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    model.hold_vae(tel)
    prompt, pictures = _said(payload.prompt, adj)
    first = payload.image or next((c.image for c in payload.control), None)
    width, height = payload.size(adj, None if first is None else first.read().size, SIZES,
                                 megapixels=SIZES.megapixels[1])
    steps = held(adj, "steps", payload.steps or _STEPS[0], *_STEPS[1:])
    guidance = held(adj, "guidance", _GUIDANCE[0] if payload.guidance is None else payload.guidance,
                    *_GUIDANCE[1:])
    negative = CARD_NEGATIVE if payload.negative_prompt is None else payload.negative_prompt
    seed = payload.drawn_seed()
    interval = (payload.cfg_interval_start, payload.cfg_interval_stop)
    if interval[0] > interval[1]:
        adj.warn("cfg_interval_start", f"{interval[0]} is past cfg_interval_stop {interval[1]}: they run swapped")
        interval = interval[::-1]
    # Both CFG conditions in one forward: short of memory, the Runtime cuts its rows. FBCache
    # keeps a cache per condition, so it runs them apart.
    guided = guidance > 1.0 and interval[0] < interval[1]
    batched = guided and payload.first_block_cache <= 0.0
    start = _start(payload, width, height, adj)
    tokens = None
    if pictures and payload.reference_strength > 0:
        with tel.stage("encoding references"):
            tokens = encoder.tokens([p.file.read() for p in pictures])
        tokens = (tokens.float() * payload.reference_strength).to(torch.bfloat16).flatten(0, 1).unsqueeze(0)
    controls = [c for c in payload.control if c.strength > 0]
    if controls:
        tel.log("control", kinds=", ".join(c.kind for c in controls))
    with _Phases(tel, ctx.progress) as phases:

        def render(adapters: Adapters | None) -> torch.Tensor:
            return model.render(prompt, negative, width, height, steps, guidance, interval,
                                payload.first_block_cache, seed, phases, batched, start=start, adapters=adapters)

        def armed() -> torch.Tensor:
            """Each adapter holds its weights through the denoise; inside, the tensors its calls read."""
            if not controls and tokens is None:
                return render(None)
            shifts = []
            for control in controls:
                lllite = lllites[control.kind]
                pixels, _ = _map(control, width, height)
                if pixels.shape[1] != lllite.graph.channels:
                    adj.warn("control", f"{control.kind}'s adapter reads {lllite.graph.channels} channels, not "
                                        f"{pixels.shape[1]}: that map is ignored")
                    continue
                shifts.append(Shifts(lllite.graph, lllite.condition(pixels), control.strength))
            references = None if tokens is None else References(ip_adapter.graph, tokens.to(torch.device(str(model.device))))
            return render(Adapters(references, shifts))

        body: Callable[[], torch.Tensor] = armed
        for used in {id(lllites[c.kind]): lllites[c.kind] for c in controls}.values():
            body = partial(used.armed, body)
        if tokens is not None:
            body = partial(ip_adapter.armed, body)
        images = body()
        # Still `decoding`: the pipeline left that bracket open and the host copy below is
        # the tail of the same work.
        image = images[0]
        # F32 and rounded, as Diffusers' `VaeImageProcessor` makes PIL bytes: bf16 then
        # truncation darkened every pixel.
        pixels = (image.float().clamp(0, 1) * 255).round().to("cpu", dtype=torch.uint8)
        if pixels.ndim == 3 and pixels.shape[0] == 3:
            pixels = pixels.permute(1, 2, 0)
        pixels = pixels.contiguous()
        rgb = bytes(pixels.numpy().tobytes())
        phases.enter(_SAVE)
        asset = out.save_image(ImageFrame(width, height, rgb), format="webp")
    return ImageResult(asset, width, height, seed, prompt)


@app.entrypoint(defaults=_DEFAULTS, on_demand=_ON_DEMAND, memoize=True)
def generate(ctx: Context, payload: GenerateInput, model: AnimaModel, image_encoder: ImageEncoder,
             ip_adapter: IPAdapter, depth: LLLite, pose: LLLite, lineart: LLLite, sketch: LLLite, inpaint: LLLite,
             out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:  # fmt: skip
    """One native-resolution Anima image: from words, steered by the pictures they mention and by control maps,
    or from a start picture."""
    lllites = {"depth": depth, "pose": pose, "lineart": lineart, "sketch": sketch, "inpaint": inpaint}
    return _generate(ctx, payload, model, image_encoder, ip_adapter, lllites, out, tel, adj)


# ------------------------------------------------------------------ quantized lanes

# The Anima recipes: which weights each lane encodes; every other tensor inherits unchanged.
# `cozy model quantize <model> --lane int8|fp8 --package <this>` runs the one for that lane.
# `cozy_runtime.derive` owns the encodings and the round-trip tripwire.

#: The fp8 lane: the DiT's block-aligned rank-2 weights, row-wise.
FP8 = Recipe(
    lane="fp8",
    model=AnimaModel,
    plan=QuantizationPlan(encoding="fp8.rowwise", components=("transformer",)),
)

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores): every attention and MLP linear of
#: the 28 DiT blocks and of the Qwen3 text encoder's 28 layers. The AdaLN modulation, embeddings,
#: the text conditioner and the VAE keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=AnimaModel,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            "transformer.transformer_blocks.*.attn[12].to_[qkv].weight",
            "transformer.transformer_blocks.*.attn[12].to_out.0.weight",
            "transformer.transformer_blocks.*.ff.net.*weight",
            "text_encoder.layers.*.self_attn.[qkvo]_proj.weight",
            "text_encoder.layers.*.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)


class ImageBox(msgspec.Struct, kw_only=True):
    #: `generate`'s prompt, and the pictures it mentions for the IP-Adapter (`_said`).
    instructions: Prompt[Image] = Prompt()
    #: `generate`: start from this picture (image to image).
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    #: `generate`: how much of the start picture is redrawn.
    strength: Annotated[float, msgspec.Meta(gt=0.0, le=1.0), Advanced()] = 0.6
    #: `generate`: control maps, each through its kind's ControlNet-LLLite.
    control: list[Control[Kind]] = msgspec.field(default_factory=list)
    #: `generate`: how strongly the mentioned pictures steer.
    reference_strength: Annotated[float, msgspec.Meta(ge=0.0, le=2.0), Advanced()] = 1.0
    #: `generate`, through `ImageRequest.size`; held to 1:16 through 16:1.
    aspect_ratio: Annotated[AspectRatio, Ratios(min=1 / 16, max=16)] | None = None
    #: `generate`, through `SIZES.pixels`: Anima's 1 MP training size and its 1536² class (README, Sizes; `SIZES`).
    megapixels: Annotated[Literal["1", "2.25"], Label("", options={"1": "1 MP", "2.25": "2.25 MP"})] = "2.25"
    #: `generate`: unset, the model card's.
    negative_prompt: str | None = None
    seed: Seed | None = None


@app.tool(label="Image", named="image", defaults=_DEFAULTS, on_demand=_ON_DEMAND, memoize=True)
def image(ctx: Context, payload: ImageBox, model: AnimaModel, image_encoder: ImageEncoder, ip_adapter: IPAdapter,
          depth: LLLite, pose: LLLite, lineart: LLLite, sketch: LLLite, inpaint: LLLite, out: Outputs, tel: Telemetry,
          adj: Adjustments) -> ImageResult:  # fmt: skip
    """Anima on a canvas: `generate` from the box. Mentioned pictures steer it, a start picture makes it an edit."""
    request = GenerateInput(prompt=payload.instructions, aspect_ratio=payload.aspect_ratio, megapixels=float(payload.megapixels),
                            seed=payload.seed, negative_prompt=payload.negative_prompt, image=payload.image,
                            strength=payload.strength, control=payload.control,
                            reference_strength=payload.reference_strength)  # fmt: skip
    lllites = {"depth": depth, "pose": pose, "lineart": lineart, "sketch": sketch, "inpaint": inpaint}
    return _generate(ctx, request, model, image_encoder, ip_adapter, lllites, out, tel, adj)
