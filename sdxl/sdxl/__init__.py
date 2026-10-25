"""se-008 — the SDXL launch package: `generate`, text to image, four components.

The second launch family, and the ENGINELESS-PATH REFERENCE (`cozy-runtime-sdxl.md`): one
`Model` class with author-written `load`, three public methods each declaring the
heavyweight components it may touch, one entrypoint. Runtime owns placement, offload and
pinning. An explicitly prepared resident variant can run text encoding, all Euler steps
and VAE decoding in one native submission. Its input buffers survive prompt/seed changes;
Runtime retires the definition before reclaiming its weights. The existing native UNet
interval and streamed eager path remain available when the full pipeline cannot be resident.

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
from pathlib import Path
from typing import Annotated, Any

import msgspec
import torch
from cozy_runtime import rng
from cozy_runtime.author import (
    MEGAPIXEL,
    AdapterCompatibility,
    AdjustmentRow,
    AdjustmentRows,
    Adjustments,
    App,
    Context,
    EncodingRule,
    ImageFrame,
    ImageRequest,
    ImageResult,
    Loader,
    Memoize,
    Model,
    OutputError,
    Outputs,
    Preflight,
    QuantizationPlan,
    Recipe,
    ReferenceImages,
    Sizes,
    Telemetry,
    held,
    uses_components,
)
from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from hidiffusion import (  # type: ignore[import-not-found,import-untyped]
    apply_hidiffusion,
    remove_hidiffusion,
)
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from transformers import initialization as transformer_init

from ._prepared import ResidentPipeline, decode_latents, encode_tokens, repeat_euler

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


class GenerateInput(ImageRequest):
    #: SDXL makes images from text only.
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=0)] = msgspec.field(
        default_factory=list
    )
    #: None runs HiDiffusion where base SDXL cannot go: above 1 MP. An explicit choice always
    #: runs, and the run warns where it is known to hurt.
    hidiffusion: bool | None = None


# ------------------------------------------------------------------ the model


def _hidiffusion_unet_type() -> type[Any]:
    """The patched UNet owner, imported only when Runtime constructs the model."""

    class HiDiffusionUNet(UNet2DConditionModel):
        """Select and reset the request's denoising implementation before step zero.

        HiDiffusion's blocks count steps on themselves, and its window blocks draw their shift
        from torch's CPU RNG. A block that runs again (the Runtime's out-of-memory recovery, a
        band) would advance both. So before every call each counting block is set to the
        request's step (`cozy_step`), and each window block draws from the request's seed, step
        and its own index: a block runs the same however often it runs."""

        _cozy_base_num_upsamplers: int
        _cozy_hidiffusion_active: bool
        cozy_step = 0
        _cozy_seed = 0

        def begin_denoise_request(self, steps: int, hidiffusion: bool, seed: int) -> None:
            if steps <= 0:
                raise ValueError("HiDiffusion requires a positive timestep count")
            active = bool(getattr(self, "_cozy_hidiffusion_active", False))
            if hidiffusion != active:
                if hidiffusion:
                    self.num_upsamplers = self._cozy_base_num_upsamplers
                    apply_hidiffusion(self)
                    window = 0
                    for block in [m for m in self.modules() if hasattr(m, "_parent")]:
                        if hasattr(block, "max_timestep"):
                            hook: Any = self._cozy_count
                        else:
                            hook, window = functools.partial(self._cozy_draw, window), window + 1
                        self.info["hooks"].append(block.register_forward_pre_hook(hook))
                else:
                    remove_hidiffusion(self)
                    self.num_upsamplers = self._cozy_base_num_upsamplers
                self._cozy_hidiffusion_active = hidiffusion
            if not hidiffusion:
                return
            self._num_timesteps, self._cozy_seed = steps, seed
            self.info["size"] = self.info["upsample_size"] = None
            for module in self.modules():  # type: ignore[attr-defined]
                if hasattr(module, "max_timestep"):
                    module.max_timestep = steps

        def _cozy_count(self, block: Any, *_: Any) -> None:
            block.timestep = self.cozy_step

        def _cozy_draw(self, index: int, *_: Any) -> None:
            draw = (self._cozy_seed * 1_000_003 + self.cozy_step * 1_009 + index) % 2**63
            torch.default_generator.manual_seed(draw)  # the CPU RNG `torch.rand(1)` reads

    return HiDiffusionUNet


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

    def __init__(self, config: Any) -> None:
        mapping = config.mapping()
        text_encoder = dict(mapping["text_encoder"])
        text_encoder_2 = dict(mapping["text_encoder_2"])
        for clip_config in (text_encoder, text_encoder_2):
            clip_config["initializer_factor"] = float(clip_config["initializer_factor"])
        unet_type = _hidiffusion_unet_type()
        with transformer_init.no_init_weights():
            self.components: dict[str, Any] = {
                "text_encoder": CLIPTextModel(CLIPTextConfig(**text_encoder)).to(
                    torch.float16  # type: ignore[arg-type]
                ),
                "text_encoder_2": CLIPTextModelWithProjection(CLIPTextConfig(**text_encoder_2)).to(
                    torch.float16
                ),  # type: ignore[arg-type]
                "unet": unet_type.from_config(mapping["unet"]).to(torch.float16),
                "vae": AutoencoderKL.from_config(mapping["vae"]).to(  # type: ignore[no-untyped-call]
                    torch.float16
                ),
            }
        unet = self.components["unet"]
        # HiDiffusion recognizes this constructed SDXL UNet by its module keys. Keep the
        # package free of a catalog/model identifier and let that structural check decide.
        unet.name_or_path = ""
        unet._cozy_base_num_upsamplers = unet.num_upsamplers
        unet._cozy_hidiffusion_active = False
        self.scheduler_config: dict[str, Any] = dict(mapping["scheduler"])
        self.vae_scale: float = float(mapping["vae"]["scaling_factor"])
        self.prepared = ResidentPipeline()


def warmup() -> None:
    """Runtime's off-lock import hook (cr-104). Every import is at module scope (se-041), so
    `import sdxl` already paid the cost; the HiDiffusion subclass body left here is free."""
    _hidiffusion_unet_type()


def build_pipeline(config: Any) -> SdxlPipeline:
    return SdxlPipeline(config)


class SdxlModel(Model[SdxlPipeline], encoded_leaves="accept"):
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
    __adapter_compatibility__ = (
        AdapterCompatibility(
            "lora", "", ("unet", "text_encoder", "text_encoder_2"), -math.inf, math.inf
        ),
    )

    pipe: SdxlPipeline
    tokenizers: tuple[Any, Any]

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(SdxlPipeline, factory=build_pipeline)
        # Checkpoint vocabularies override the standard SDXL CLIP vocabulary. Both
        # sources are read-only and the parsed tokenizers survive between requests.
        self.tokenizers = (
            _tokenizer(loader.assets, "tokenizer"),
            _tokenizer(loader.assets, "tokenizer_2"),
        )

    @uses_components("text_encoder", "text_encoder_2", memoize=True)
    def encode(self, ids: Any, ids_2: Any) -> tuple[Any, Any]:
        """SDXL's two-tower conditioning. ONE method, BOTH encoders — a composite operation
        declares its components together rather than opening two scopes.

        IT PLACES ITS OWN INPUTS (#534c). Token ids arrive on the host, and the only code
        entitled to say where they go is code that holds a lease on the component they are
        going to: it follows the encoder's own weights to wherever the runtime already put
        them. The caller used to do this with a hard-coded `cuda:0`, which was an author
        naming a device — the one live instance se-001's record left open — and it made this
        package unservable on any envelope the runtime did not place at ordinal zero.
        """
        first_encoder = self.pipe.components["text_encoder"]
        second_encoder = self.pipe.components["text_encoder_2"]
        with torch.inference_mode():
            ids = ids.to(first_encoder.get_input_embeddings().weight.device)
            ids_2 = ids_2.to(second_encoder.get_input_embeddings().weight.device)
            return encode_tokens(first_encoder, second_encoder, ids, ids_2)

    @uses_components("unet")
    def denoise(
        self,
        latents: Any,
        timestep: Any,
        prompt: Any,
        text_embeds: Any,
        time_ids: Any,
        step: int,
        total_steps: int,
        hidiffusion: bool,
        seed: int,
    ) -> Any:
        unet = self.pipe.components["unet"]
        if step == 0:
            unet.begin_denoise_request(total_steps, hidiffusion, seed)
        unet.cozy_step = step
        with torch.inference_mode():
            return unet(
                latents,
                timestep,
                encoder_hidden_states=prompt,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample

    @uses_components("vae")
    def decode(self, latents: Any) -> Any:
        vae = self.pipe.components["vae"]
        original_dtype = next(vae.parameters()).dtype
        upcast = bool(getattr(vae.config, "force_upcast", False))
        if upcast:
            # The SDXL VAE overflows fp16. bf16 has fp32's range: 48 dB PSNR to fp32 on the
            # 1024px fox (run 1642), ~0.5 s faster on an RTX 4070 Laptop.
            native_bf16 = latents.is_cuda and torch.cuda.is_bf16_supported(
                including_emulation=False
            )
            wide = torch.bfloat16 if native_bf16 else torch.float32
            vae.to(dtype=wide)
            latents = latents.to(dtype=wide)
        try:
            with torch.inference_mode():
                return decode_latents(vae, latents, self.pipe.vae_scale)
        finally:
            if upcast:
                vae.to(dtype=original_dtype)


# ------------------------------------------------------------------ the handler


def _finite(torch: Any, value: Any) -> float:
    """The tensor's largest magnitude, with NaN neutralized so the metric is spellable.

    A metric that cannot be serialized is a metric that is not there: the observation emit
    boundary refuses a non-finite value, so the NaN FRACTION is its own number and this one
    stays a real magnitude.
    """
    return round(float(torch.nan_to_num(value, 0.0, 0.0, 0.0).abs().max()), 4)


def _tokenizer(assets: Any, name: str) -> Any:
    """Use checkpoint overrides, or bundled CLIP data when the whole pair is absent.

    A partial override still fails through ModelAssets; mixing vocabularies and merges
    from different sources would silently change the model's tokenization. No download
    or temporary file is needed, including during read-only admission.
    """
    pad = {"tokenizer": "<|endoftext|>", "tokenizer_2": "!"}[name]
    vocab_name, merges_name = f"{name}/vocab.json", f"{name}/merges.txt"
    names = assets.names()
    if vocab_name in names or merges_name in names:
        vocab_bytes = assets.read(vocab_name)
        merges_bytes = assets.read(merges_name)
    else:
        root = Path(__file__).parent
        vocab_bytes = (root / "clip_vocab.json").read_bytes()
        merges_bytes = (root / "clip_merges.txt").read_bytes()
    vocab = json.loads(vocab_bytes)
    merges = [
        tuple(line.split(" "))
        for line in merges_bytes.decode().splitlines()
        if line and not line.startswith("#version")
    ]
    return CLIPTokenizer(
        vocab=vocab, merges=merges, errors="replace", pad_token=pad, model_max_length=77
    )


def _tokenize(tokenizers: tuple[Any, Any], prompt: str) -> tuple[Any, Any]:
    first, second = tokenizers
    return _ids(first, prompt), _ids(second, prompt)


def _ids(tok: Any, prompt: str) -> Any:
    return tok(
        prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


#: THE NaN CHECK reads the FLOAT decode, before quantization, and that is the whole of
#: what a red arm taught here: `torch.isnan` on a `clamp(0,1).to(uint8)` tensor is
#: constant False, because a uint8 cannot be NaN — the very quantization that makes an
#: image encodable is what erases the evidence of a diverged decode.
#:
#: Pixel SPREAD is reported as telemetry, not refused. A flat field is a real failure
#: mode (a wrong VAE scaling factor, an unfilled decoder) but it is also a real REQUEST:
#: "solid white background", "minimalist pure black poster", "flat pastel swatch". A
#: refusal here destroyed a correct, completed, billed generation.
def _integrity(torch: Any, image: Any, pixels: Any, tel: Telemetry) -> None:
    nan_fraction = float(torch.isnan(image).float().mean())
    tel.metric("image_nan_fraction", round(nan_fraction, 6))
    if nan_fraction > 0.0:
        raise OutputError(
            f"the decode produced NaN over {nan_fraction:.4%} of the image and this "
            "package does not publish it: a non-finite decode is a failed generation, "
            "not a picture with artefacts",
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


def _sample(
    model: SdxlModel,
    tokens: tuple[Any, Any],
    negative_tokens: tuple[Any, Any] | None,
    *,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    hidiffusion_applied: bool,
    seed: int,
    ctx: Context,
    tel: Telemetry,
) -> Any:
    """The existing streamed/eager path; preparation refusal reaches it before submission."""
    ids, ids_2 = tokens
    classifier_free = guidance > 1.0
    if classifier_free:
        assert negative_tokens is not None
        neg, neg_2 = negative_tokens
    with tel.stage("encode", overall_range=(0.02, 0.10)):
        prompt, pooled = model.encode(ids, ids_2)
        if classifier_free:
            negative, neg_pooled = model.encode(neg, neg_2)
    # THE DEVICE ENVELOPE, READ RATHER THAN NAMED (#534c). The conditioning came back from
    # the encoders the runtime placed, so it already carries where this request runs; the
    # latents and the schedule follow it. This is data flow across the boundary an author
    # owns, and it is the whole of what the deleted `torch.device("cuda", 0)` was doing.
    device = prompt.device
    tel.metric("prompt_absmax", _finite(torch, prompt))
    tel.metric("pooled_absmax", _finite(torch, pooled))

    # HiDiffusion's recipe samples with DDIM at eta=1.0: fresh noise every step, from the request's
    # generator. Without HiDiffusion, deterministic Euler.
    sampler = DDIMScheduler if hidiffusion_applied else EulerDiscreteScheduler
    scheduler = sampler.from_config(model.pipe.scheduler_config)  # type: ignore[no-untyped-call]
    scheduler.set_timesteps(steps, device=device)
    generator = rng.generator(seed)
    step = (
        functools.partial(scheduler.step, eta=1.0, generator=generator)
        if hidiffusion_applied
        else scheduler.step
    )
    shape = (1, 4, height // 8, width // 8)
    latents = rng.noise(shape, generator, dtype=torch.float16, device=device) * scheduler.init_noise_sigma
    # SDXL's micro-conditioning: (original_h, original_w, crop_top, crop_left, target_h,
    # target_w). The bucket IS the original size — these fine-tunes were trained on it.
    time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], device=device, dtype=torch.float16
    )
    if classifier_free:
        batch_prompt = torch.cat([negative, prompt])
        batch_pooled = torch.cat([neg_pooled, pooled])
        batch_ids = torch.cat([time_ids, time_ids])
    else:
        batch_prompt, batch_pooled, batch_ids = prompt, pooled, time_ids

    on_step = tel.step_callback(steps, stage="denoise", overall_range=(0.10, 0.90))
    # HiDiffusion's window blocks seed torch's CPU RNG from the request (`_cozy_draw`): the
    # RNG state outside the loop is left as it was.
    with torch.random.fork_rng(devices=[]), tel.stage("denoise", overall_range=(0.10, 0.90)):
        for index, timestep in enumerate(scheduler.timesteps):
            ctx.raise_if_cancelled()
            batch = torch.cat([latents] * 2) if classifier_free else latents
            model_input = scheduler.scale_model_input(batch, timestep)
            noise = model.denoise(
                model_input,
                timestep,
                batch_prompt,
                batch_pooled,
                batch_ids,
                index,
                steps,
                hidiffusion_applied,
                seed,
            )
            if classifier_free:
                uncond, cond = noise.chunk(2)
                noise = uncond + guidance * (cond - uncond)
            if index == 0:
                tel.metric("noise_absmax_step0", _finite(torch, noise))
            latents = step(noise, timestep, latents).prev_sample
            on_step(index)
            if index == 0 and not hidiffusion_applied:
                prepared = repeat_euler(
                    unet=model.pipe.components["unet"],
                    scheduler=scheduler,
                    latents=latents,
                    prompt=batch_prompt,
                    pooled=batch_pooled,
                    time_ids=batch_ids,
                    guidance=guidance,
                    ctx=ctx,
                    tel=tel,
                )
                if prepared is not None:
                    latents = prepared
                    break
    tel.metric("latent_absmax", _finite(torch, latents))

    with tel.stage("decode", overall_range=(0.90, 0.98)):
        image = model.decode(latents)
    return image


class GenerateFacts(msgspec.Struct, frozen=True):
    width: int
    height: int
    resolved_steps: int
    guidance: float
    hidiffusion_applied: bool
    negative_prompt: str
    adjustments: tuple[AdjustmentRow, ...]


def resolve_generate(payload: GenerateInput) -> GenerateFacts:
    """Resolve one variant identically for admitted preparation and ordinary inference."""
    adj = AdjustmentRows()
    target = payload.target(None, megapixels=1.0)
    sizes = _SQUARE if target.ratio[0] == target.ratio[1] else _SIZES
    width, height = sizes.fit(adj, target.ratio, sizes.pixels(adj, target))
    above_native = width * height > _NATIVE * MEGAPIXEL
    hidiffusion_applied = above_native if payload.hidiffusion is None else payload.hidiffusion
    default_steps, default_guidance = (30, 7.5) if hidiffusion_applied else (20, 7.0)
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


def prepare_generate(
    ctx: Context,
    facts: Preflight[GenerateFacts],
    model: SdxlModel,
    tel: Telemetry,
    adj: Adjustments,
) -> bool:
    """Prepare one known resident variant in Runtime's admitted GPU warm turn."""
    adj.extend(facts.adjustments)
    if facts.hidiffusion_applied:
        return False
    ctx.plan({"text_encoder", "text_encoder_2"}, "unet", "vae")
    # Real tokenizer IDs initialize fixed-shape slots; request values replace them later.
    # Preparation does not draw a seed or submit an image-generation request.
    tokens = _tokenize(model.tokenizers, "")
    return model.pipe.prepared.prepare(
        components=model.pipe.components,
        scheduler_config=model.pipe.scheduler_config,
        vae_scale=model.pipe.vae_scale,
        tokens=tokens,
        negative_tokens=tokens if facts.guidance > 1.0 else None,
        width=facts.width,
        height=facts.height,
        steps=facts.resolved_steps,
        guidance=facts.guidance,
        ctx=ctx,
        tel=tel,
    )


@app.entrypoint(
    defaults={"model": "sdxl@1.0.0"},
    memoize=Memoize(requires=["seed"]),
    preflight=resolve_generate,
    prepare=prepare_generate,
)
def generate(
    ctx: Context,
    payload: GenerateInput,
    facts: Preflight[GenerateFacts],
    model: SdxlModel,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageResult:
    """One text-to-image generation: tokenize, encode, denoise, decode, encode a WebP."""
    ctx.plan({"text_encoder", "text_encoder_2"}, "unet", "vae")
    payload.ignored(adj, "prompt_upsampling", reason="SDXL has no prompt upsampler")
    adj.extend(facts.adjustments)
    seed = payload.drawn_seed()
    width, height = facts.width, facts.height
    steps, guidance = facts.resolved_steps, facts.guidance
    hidiffusion_applied, negative_prompt = facts.hidiffusion_applied, facts.negative_prompt
    # The value plane, and the only branch in this file that reads a request number: above
    # 1.0 the negative branch is worth its second forward pass, at or below it is not.
    classifier_free = guidance > 1.0

    with tel.stage("tokenize", overall_range=(0.00, 0.02)):
        ids, ids_2 = _tokenize(model.tokenizers, payload.prompt)
        if classifier_free:
            neg, neg_2 = _tokenize(model.tokenizers, negative_prompt)

    tokens = (ids, ids_2)
    negative_tokens = (neg, neg_2) if classifier_free else None
    if hidiffusion_applied:
        model.pipe.prepared.close()  # HiDiffusion mutates the UNet implementation.
        image = None
    else:
        image = model.pipe.prepared.run(
            components=model.pipe.components,
            scheduler_config=model.pipe.scheduler_config,
            vae_scale=model.pipe.vae_scale,
            tokens=tokens,
            negative_tokens=negative_tokens,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            ctx=ctx,
            tel=tel,
        )
    if image is None:
        image = _sample(
            model, tokens, negative_tokens,
            width=width, height=height, steps=steps, guidance=guidance,
            hidiffusion_applied=hidiffusion_applied, seed=seed, ctx=ctx, tel=tel,
        )
    tel.metric("image_absmax", _finite(torch, image))
    pixels = ((image / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    _integrity(torch, image, pixels, tel)

    rgb = bytes(pixels.cpu().numpy().tobytes())
    decoded_h, decoded_w = int(pixels.shape[0]), int(pixels.shape[1])
    tel.metric("decoded_pixels", float(decoded_w * decoded_h))
    with tel.stage("encode_webp", overall_range=(0.98, 1.00)):
        asset = out.save_image(ImageFrame(decoded_w, decoded_h, rgb), format="webp")
    tel.log("sampled", sampler="ddim eta=1.0" if hidiffusion_applied else "euler", steps=steps,
            guidance=guidance, classifier_free=classifier_free, hidiffusion=hidiffusion_applied)
    return ImageResult(asset, decoded_w, decoded_h, seed, payload.prompt)


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
