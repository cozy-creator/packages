"""FLUX.3 Image Klein, base and distilled, component-staged over Diffusers' modules.

The modules are Diffusers' (`Flux3Transformer2DModel`, `AutoencoderKLFlux2`, the Qwen3-VL prompt
encoding of its Flux3 modular pipeline); reference preparation and sampling follow BFL's reference
code (`black-forest-labs/flux-3-image-klein`, `flux3/pipeline.py`, `flux3/sampling.py`). The
transformer's forward is the package's: every block is one parallel block over one token sequence,
as in BFL's code, which Runtime fuses (`BLOCK`). Runtime constructs and fills the modules and owns
their residency. A caller's LoRAs adapt the DiT through Runtime's PEFT layers (`serving.loras`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import NamedTuple, TypedDict, cast

import numpy as np
import torch
import torch.nn.functional as F
from cozy_runtime import rng, vlm
from cozy_runtime.accel import fusion, hold_dtype
from cozy_runtime.author import (
    Config,
    Loader,
    Model,
    OutputError,
    Serving,
    Telemetry,
    memoize,
    preview,
    retained,
    uses_components,
)
from cozy_runtime.accel.kernels import Plan
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux3 import (
    Flux3ModeTransformerBlock,
    Flux3Transformer2DModel,
    Flux3SingleTransformerBlock,
)
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from PIL import Image, ImageOps
from transformers import (
    AutoModelForImageTextToText,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from transformers import initialization as transformer_init

#: BFL's constant timestep shift, the image RoPE scale and Qwen3-VL's conditioning layers.
ALPHA = 16.0
IMAGE_ROPE_SCALE = 0.5
TEXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
PAD_MULTIPLE = 80
MAX_TOKENS = 4096
#: Pixels per latent token: the VAE's 8x, then 2x2 packing.
PATCH = 16
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
COMPONENTS = ("transformer", "text_encoder", "vae")
#: The VAE's dtype on a GPU, as ComfyUI runs FLUX.2's VAE: its activations fit fp16 (dtype audit M1).
VAE_DTYPE = torch.float16
#: A token sequence's rotary tables, one tensor an axis (`pos_embed`).
Rope = tuple[torch.Tensor, ...]


def _conditioning_only(encoder: Qwen3VLForConditionalGeneration) -> None:
    """Klein reads hidden states 4..32 of Qwen3-VL's text stack: the vision tower, the last four
    layers, the final norm and the head are never run, so they are not built (their stored bytes
    are skipped). With the norm an identity, hidden state 32 stays the pre-norm output."""
    language = encoder.model.language_model
    language.layers = torch.nn.ModuleList(list(language.layers)[: max(TEXT_LAYERS)])
    language.norm = torch.nn.Identity()
    encoder.lm_head = torch.nn.Identity()
    del encoder.model.visual


class Rows(NamedTuple):
    """A block call's adaLN: each mode's (shift, scale, gate) row and each token's row index."""

    table: torch.Tensor
    index: torch.Tensor  # type: ignore[assignment]  # the row index, in place of tuple.index


class KleinBlock(Flux3ModeTransformerBlock):
    """Every Klein block, per-mode and joint: parallel attention and SwiGLU under one adaLN, over
    one token sequence. The joint blocks take [image, text, references] as that sequence (BFL's
    layout) instead of Diffusers' two streams; each token reads its mode's row."""

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor, temb_mod: Rows, image_rotary_emb: Rope
    ) -> torch.Tensor:
        shift, scale, gate = temb_mod.table[temb_mod.index].chunk(3, dim=-1)
        normed = (1 + scale) * self.norm(hidden_states) + shift
        attended = self.attn(hidden_states=normed, image_rotary_emb=image_rotary_emb)
        out: torch.Tensor = hidden_states + gate * (attended + self.ff(normed))
        return out


def _rows(block: torch.nn.Module, arguments: Mapping[str, object]) -> fusion.Modulation:
    rows = cast(Rows, arguments["temb_mod"])  # the block's adaLN rows
    shift, scale, gate = rows.table.chunk(3, dim=-1)
    return fusion.Modulation(rows.index, scale, gate, shift_msa=shift)


#: Klein's block as Runtime fuses it: one LayerNorm modulated by (shift, scale), q/k RMSNorm and
#: RoPE in fp32 (repeated cos/sin pairs), `linear_in` = [gate | up], one gate on attention + MLP.
BLOCK = fusion.DiTBlock(
    KleinBlock,
    width=3072,
    head_dim=128,
    modulation=_rows,
    norms=("norm",),
    norm="layer",
    attention="attn",
    arguments=(),
    mlp=fusion.GatedMLP("ff", gate="linear_in", down="linear_out"),
    rope=fusion.Rope("image_rotary_emb", interleaved=True, rounding="fp32"),
    shift=True,
)


class Transformer(Flux3Transformer2DModel):
    """`Flux3Transformer2DModel` whose forward is one step past the conditions: BFL's layout, the image's
    early blocks, then the joint blocks over [image, text, references]."""

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor, conditions: list[torch.Tensor], cond_rows: torch.Tensor,
        image_rope: Rope, joints: list[tuple[Rope, tuple[int, ...]]],
    ) -> list[torch.Tensor]:
        """The velocity of `x` for each of `conditions` (`_conditions`: a prompt's tokens after their
        early blocks, `cond_rows` their modes' rows), with that prompt's joint rope and mode sizes."""
        act = F.silu(temb)
        zero = torch.zeros(1, dtype=torch.long, device=x.device)
        velocities = []
        for cond, (rope, sizes) in zip(conditions, joints, strict=False):
            image = _early(self.mode_blocks["image"], self.x_embedders["image"](x),
                           self.early_stream_modulations["image"], act, image_rope, zero)
            index = torch.cat([torch.full((size,), mode, device=x.device) for mode, size in enumerate(sizes)])
            rows = Rows(torch.cat([self.single_stream_modulations["image"](act), cond_rows]), index)
            tokens = torch.cat([image, cond], dim=1)
            for block in self.single_transformer_blocks:
                tokens = block(tokens, rows, rope)
            velocities.append(self.proj_out(self.norm_out(tokens[:, : image.shape[1]], temb[:, None])))
        return velocities


class KleinGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = construct(Flux3Transformer2DModel, config.section("transformer").mapping())
            text_encoder: Qwen3VLForConditionalGeneration = AutoModelForImageTextToText.from_config(  # type: ignore[no-untyped-call]  # Transformers' auto classes are unannotated
                Qwen3VLConfig(**config.section("text_encoder").mapping()), dtype=torch.bfloat16
            )
            vae = construct(AutoencoderKLFlux2, config.section("vae").mapping())
        _conditioning_only(text_encoder)
        transformer.__class__ = Transformer
        for block in transformer.modules():
            if type(block) in (Flux3ModeTransformerBlock, Flux3SingleTransformerBlock):
                block.__class__ = KleinBlock
        self.components: Components = {
            "transformer": _to(cast(Transformer, transformer), torch.bfloat16),  # its class, set above
            "text_encoder": text_encoder,
            "vae": _to(vae, torch.float32),
        }


class Components(TypedDict):
    transformer: Transformer
    text_encoder: Qwen3VLForConditionalGeneration
    vae: AutoencoderKLFlux2


def construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast(M, cls.from_config(config))


def _to[M: torch.nn.Module](module: M, dtype: torch.dtype) -> M:
    """`module.to(dtype=dtype)`, typed: Diffusers' `to` is `functools.wraps`'d, which the checkers bind without `self`."""
    return module.to(dtype=dtype)


def _ceil16(value: int) -> int:
    return -(-value // PATCH) * PATCH


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(fitted size, canvas) of a lone reference: never upscaled, the canvas (the fit rounded up to
    16) at most 4 MiP with sides 64-4096."""
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = (min(MAX_SIDE, max(MIN_SIDE, _ceil16(fit[0]))), min(MAX_SIDE, max(MIN_SIDE, _ceil16(fit[1]))))
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """A lone reference, reflect-padded to its canvas; with the fitted image's box when padded."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    pixels = np.asarray(image)
    padded = Image.fromarray(
        np.pad(pixels, padding, mode="reflect") if min(image.size) > PATCH
        else np.pad(pixels, padding, mode="symmetric")
    )
    return padded, (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    """One of several references: under 1 MP, centre-cropped to the 16-pixel grid."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = width // PATCH * PATCH, height // PATCH * PATCH
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def image_ids(rows: int, columns: int, layer: int = 0) -> torch.Tensor:
    """(t, h, w, l) position ids of a token grid; references take l = -1, -2, ..."""
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(rows), torch.arange(columns), torch.tensor([layer])
    ).float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def text_ids(length: int) -> torch.Tensor:
    return torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length)).float()


def schedule(steps: int) -> list[float]:
    t = np.linspace(1.0, 0.0, steps + 1)
    return [float(v) for v in ALPHA * t / (1.0 + (ALPHA - 1.0) * t)]


def _bn(vae: AutoencoderKLFlux2, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like)  # type: ignore[union-attr]  # tracked: it keeps its statistics
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config["batch_norm_eps"]).to(like)  # type: ignore[union-attr]  # tracked
    return mean, std


def _embed(tf: Transformer, embedder: torch.nn.Module, now: torch.Tensor) -> torch.Tensor:
    """Diffusers' embedding of the bf16 time `now` [1], scaled in bf16 as it does."""
    out: torch.Tensor = embedder(tf.time_proj(now * 1000).to(torch.bfloat16))
    return out


def _joint(*ropes: Rope | None) -> tuple[Rope, tuple[int, ...]]:
    """A joint sequence's rotary tables, and each mode's token count, in order."""
    present = [rope for rope in ropes if rope is not None]
    return tuple(torch.cat(axis) for axis in zip(*present, strict=True)), tuple(len(rope[0]) for rope in present)


def _early(blocks: torch.nn.Module, tokens: torch.Tensor, modulation: torch.nn.Module, act: torch.Tensor, rope: Rope,
           zero: torch.Tensor) -> torch.Tensor:
    rows = Rows(modulation(act), zero)
    for block in cast(torch.nn.ModuleList, blocks):  # a mode's blocks
        tokens = block(tokens, rows, rope)
    return tokens


def _conditions(
    tf: Transformer, texts: list[torch.Tensor], text_ropes: list[Rope], refs: torch.Tensor | None, ref_rope: Rope | None,
    act: torch.Tensor, zero: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Each text's [text, references] tokens after their early blocks, and the joint rows of
    those modes. `act` is their (t = 0) embedding, activated."""
    modes = ["txt"]
    ref = None
    if refs is not None:
        modes.append("image_cond")
        ref = _early(tf.mode_blocks["image_cond"], tf.x_embedders["image_cond"](refs),
                     tf.early_stream_modulations["image_cond"], act, ref_rope, zero)  # type: ignore[arg-type]  # with references, their rope
    tokens = []
    for text, rope in zip(texts, text_ropes, strict=True):
        txt = _early(tf.txt_mode_blocks, tf.context_embedder(text), tf.early_stream_modulations["txt"],
                     act, rope, zero)
        tokens.append(txt if ref is None else torch.cat([txt, ref], dim=1))
    return tokens, torch.cat([tf.single_stream_modulations[mode](act) for mode in modes])


class Klein(Model[KleinGraph]):
    """One Klein checkpoint: the distilled flow map (no guidance) or the base model (CFG)."""

    #: Attention as BFL's int8 release (`comfy_attention.config`): the image and joint blocks at 8 bits; the
    #: text and reference early blocks keep exact attention. Caller LoRAs on the DiT, at any strength: the
    #: Runtime bakes them into the weights.
    serving = Serving(
        attention_bits={"transformer.mode_blocks.image.": 8, "transformer.single_transformer_blocks.": 8},
        loras=("transformer",),
    )

    graph: KleinGraph
    processor: Qwen3VLProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(KleinGraph)
        self.processor = cast(Qwen3VLProcessor, vlm.processor(loader.assets, "text_encoder/"))  # Qwen3-VL's own

    @property
    def flow_map(self) -> bool:
        return bool(self.graph.components["transformer"].config["flow_map"])

    @memoize
    @uses_components("text_encoder")
    def encode(self, prompt: str) -> torch.Tensor:
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            embeds: torch.Tensor = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                encoder,
                self.processor,  # type: ignore[arg-type]  # Diffusers annotates the AutoProcessor factory
                prompt,
                device=torch.device(str(self.device)),
                max_sequence_length=MAX_TOKENS,
                pad_multiple=PAD_MULTIPLE,
                hidden_states_layers=TEXT_LAYERS,
            )
        return embeds.cpu()

    def hold_vae(self, tel: Telemetry) -> None:
        """Serve the VAE at `VAE_DTYPE` on a GPU from now on, before any VAE scope opens: the Runtime
        casts the stored float32 bytes when it next fills them. A CPU keeps float32."""
        vae = self.graph.components["vae"]
        if torch.device(str(self.device)).type == "cuda" and next(vae.parameters()).dtype != VAE_DTYPE:
            hold_dtype(vae, VAE_DTYPE)
            tel.log("vae representation", dtype=str(VAE_DTYPE).removeprefix("torch."))

    @memoize
    @uses_components("vae")
    def encode_references(self, references: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized packed latents of the prepared references as one token sequence, with ids."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        tokens, ids = [], []
        with torch.inference_mode():
            for index, reference in enumerate(references):
                pixels = torch.from_numpy(np.array(reference)).permute(2, 0, 1).float() / 127.5 - 1.0
                latents = vae.encode(pixels[None].to(device=device, dtype=vae.dtype)).latent_dist.mode().float()
                b, c, h, w = latents.shape
                latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
                latents = latents.reshape(b, c * 4, h // 2, w // 2)
                mean, std = _bn(vae, latents)
                latents = (latents - mean) / std
                tokens.append(latents.flatten(2).transpose(1, 2).to(torch.bfloat16).cpu())
                ids.append(image_ids(h // 2, w // 2, -(1 + index)))
        return torch.cat(tokens, dim=1), torch.cat(ids)

    def execution_plan(self, plan: Plan) -> str:
        return fusion.install(plan, BLOCK, self.graph.components["transformer"])

    @uses_components("transformer")
    def denoise(
        self,
        embeds: torch.Tensor,
        negative: torch.Tensor | None,
        references: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float | None,
        seed: int,
        progress: Callable[[int, int], None],
    ) -> torch.Tensor:
        """BFL's Euler sampler: a flow map jumps to the next timestep; the base model takes CFG.

        Text and references sit at t = 0, so the base model runs their early blocks once per
        request and the Runtime holds them (`retained`: on the card while there is room, else
        refilled from pinned host memory); a flow map adds each step's target time to every mode,
        so it runs them per step.
        """
        tf = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        timesteps = schedule(steps)
        with torch.no_grad():
            noise = rng.noise((rows * columns, tf.config["in_channels"]), seed)
            x = noise[None].to(torch.bfloat16).float().to(device)  # bf16 values, an fp32 state across steps
            texts = [embeds] if negative is None else [negative, embeds]
            texts = [t.to(device=device, dtype=torch.bfloat16) for t in texts]
            refs = None if references is None else references[0].to(device)
            zero = torch.zeros(1, dtype=torch.long, device=device)
            image_rope = tf.pos_embed(image_ids(rows, columns).to(device))
            text_ropes = [tf.pos_embed(text_ids(t.shape[1]).to(device)) for t in texts]
            ref_rope = None if references is None else tf.pos_embed(references[1].to(device))
            joints = [_joint(image_rope, rope, ref_rope) for rope in text_ropes]
            at_zero = _embed(tf, tf.timestep_embedder, torch.zeros(1, dtype=torch.bfloat16, device=device))
            times = torch.tensor(timesteps, dtype=torch.bfloat16, device=device)
            with ExitStack() as stack:
                held = None
                if not self.flow_map:
                    tokens, cond_rows = _conditions(tf, texts, text_ropes, refs, ref_rope, F.silu(at_zero), zero)
                    # The conditions die with the loop.
                    store = stack.enter_context(retained(sum(t.nbytes for t in tokens) + cond_rows.nbytes))
                    held = store.put(*tokens, cond_rows)
                    del tokens, cond_rows
                for i, (now, then) in enumerate(itertools.pairwise(timesteps)):
                    temb = _embed(tf, tf.timestep_embedder, times[i : i + 1])
                    if held is not None:
                        *tokens, cond_rows = held.get()
                    else:
                        target = _embed(tf, tf.target_timestep_embedder, times[i + 1 : i + 2])  # type: ignore[arg-type]  # a flow map's own
                        temb = temb + target
                        tokens, cond_rows = _conditions(tf, texts, text_ropes, refs, ref_rope,
                                                        F.silu(at_zero + target), zero)
                    image = x.to(torch.bfloat16)
                    if len(tokens) == 2 and tokens[0].shape == tokens[1].shape:
                        # CFG texts of one length share the joint rope and rows: batch 2.
                        (both,) = tf(image.repeat(2, 1, 1), temb, [torch.cat(tokens)], cond_rows, image_rope, joints)
                        velocities = list(both.float().split(1))
                    else:
                        velocities = [v.float() for v in tf(image, temb, list(tokens), cond_rows, image_rope, joints)]
                    if guidance is None:
                        (velocity,) = velocities
                    else:
                        unconditional, conditional = velocities
                        velocity = unconditional + guidance * (conditional - unconditional)
                    preview(lambda: x - now * velocity, "flux2", grid=(rows, columns))
                    x = x + (then - now) * velocity
                    progress(i, steps)
        return x.cpu()

    @uses_components("vae")
    def decode(self, tokens: torch.Tensor, *, width: int, height: int) -> Image.Image:
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        with torch.inference_mode():
            latents = tokens.to(device=device, dtype=torch.float32).view(1, rows, columns, -1).permute(0, 3, 1, 2)
            mean, std = _bn(vae, latents)
            latents = latents * std + mean
            b, c, h, w = latents.shape
            latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0][0]
            if not torch.isfinite(image).all():
                raise OutputError(
                    "Klein's decoder produced non-finite pixels, so there is no picture: try another seed, "
                    "and if it repeats another lane with `--lane bf16`",
                    code="output_integrity",
                )
            pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)
