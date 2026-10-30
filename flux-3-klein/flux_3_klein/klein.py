"""FLUX.3 Image Klein, base and distilled, component-staged over Diffusers' modules.

The modules are Diffusers' (`Flux3Transformer2DModel`, `AutoencoderKLFlux2`, the Qwen3-VL prompt
encoding of its Flux3 modular pipeline); reference preparation and sampling follow BFL's reference
code (`black-forest-labs/flux-3-image-klein`, `flux3/pipeline.py`, `flux3/sampling.py`). The
transformer's forward is the package's: every block is one parallel block over one token sequence,
as in BFL's code, which Runtime fuses (`BLOCK`). Runtime constructs and fills the modules and owns
their residency.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from cozy_runtime import fusion, rng, vlm
from cozy_runtime.author import (
    Config,
    Loader,
    Model,
    Telemetry,
    hold_dtype,
    uses_components,
)
from cozy_runtime.author.kernels import Plan
from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
from diffusers.models.transformers.transformer_flux3 import (
    Flux3ModeTransformerBlock,
    Flux3SingleTransformerBlock,
)
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from PIL import Image, ImageOps
from transformers import (
    AutoModelForImageTextToText,
    Qwen3VLConfig,
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


def _conditioning_only(encoder: Any) -> None:
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
    index: torch.Tensor


class KleinBlock(Flux3ModeTransformerBlock):
    """Every Klein block, per-mode and joint: parallel attention and SwiGLU under one adaLN, over
    one token sequence. The joint blocks take [image, text, references] as that sequence (BFL's
    layout) instead of Diffusers' two streams; each token reads its mode's row."""

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor, temb_mod: Rows, image_rotary_emb: Any
    ) -> torch.Tensor:
        shift, scale, gate = temb_mod.table[temb_mod.index].chunk(3, dim=-1)
        normed = (1 + scale) * self.norm(hidden_states) + shift
        attended = self.attn(hidden_states=normed, image_rotary_emb=image_rotary_emb)
        return hidden_states + gate * (attended + self.ff(normed))


def _rows(block: KleinBlock, arguments: Mapping[str, Any]) -> fusion.Modulation:
    rows: Rows = arguments["temb_mod"]
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


class KleinGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = Flux3Transformer2DModel.from_config(config.section("transformer").mapping())
            text_encoder = AutoModelForImageTextToText.from_config(
                Qwen3VLConfig(**config.section("text_encoder").mapping()), dtype=torch.bfloat16
            )
            vae = AutoencoderKLFlux2.from_config(config.section("vae").mapping())
        _conditioning_only(text_encoder)
        for block in transformer.modules():
            if type(block) in (Flux3ModeTransformerBlock, Flux3SingleTransformerBlock):
                block.__class__ = KleinBlock
        self.components: dict[str, Any] = {
            "transformer": transformer.to(dtype=torch.bfloat16),
            "text_encoder": text_encoder,
            "vae": vae.to(dtype=torch.float32),
        }


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


def _bn(vae: Any, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(like)
    return mean, std


def _embed(tf: Any, embedder: Any, t: float, device: torch.device) -> torch.Tensor:
    """Diffusers' embedding of time `t`, scaled in bf16 as it does."""
    now = torch.full((1,), t, device=device, dtype=torch.bfloat16)
    return embedder(tf.time_proj(now * 1000).to(torch.bfloat16))


def _joint(*ropes: Any) -> tuple[Any, torch.Tensor]:
    """A joint sequence's rotary tables, and each token's row: the index of its mode."""
    present = [rope for rope in ropes if rope is not None]
    cos = present[0][0]
    lengths = torch.tensor([len(rope[0]) for rope in present], device=cos.device)
    modes = torch.arange(len(present), device=cos.device)
    return tuple(torch.cat(axis) for axis in zip(*present, strict=True)), modes.repeat_interleave(lengths)


def _early(blocks: Any, tokens: torch.Tensor, modulation: Any, act: torch.Tensor, rope: Any,
           zero: torch.Tensor) -> torch.Tensor:
    rows = Rows(modulation(act), zero)
    for block in blocks:
        tokens = block(tokens, rows, rope)
    return tokens


def _conditions(
    tf: Any, texts: list[torch.Tensor], text_ropes: list[Any], refs: torch.Tensor | None, ref_rope: Any,
    act: torch.Tensor, zero: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Each text's [text, references] tokens after their early blocks, and the joint rows of
    those modes. `act` is their (t = 0) embedding, activated."""
    modes = ["txt"]
    ref = None
    if refs is not None:
        modes.append("image_cond")
        ref = _early(tf.mode_blocks["image_cond"], tf.x_embedders["image_cond"](refs),
                     tf.early_stream_modulations["image_cond"], act, ref_rope, zero)
    tokens = []
    for text, rope in zip(texts, text_ropes, strict=True):
        txt = _early(tf.txt_mode_blocks, tf.context_embedder(text), tf.early_stream_modulations["txt"],
                     act, rope, zero)
        tokens.append(txt if ref is None else torch.cat([txt, ref], dim=1))
    return tokens, torch.cat([tf.single_stream_modulations[mode](act) for mode in modes])


def _velocity(tf: Any, x: torch.Tensor, temb: torch.Tensor, cond: torch.Tensor, cond_rows: torch.Tensor,
              image_rope: Any, joint: tuple[Any, torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    """One transformer call: the image's early blocks, then the joint blocks over [image, cond]."""
    act = F.silu(temb)
    image = _early(tf.mode_blocks["image"], tf.x_embedders["image"](x), tf.early_stream_modulations["image"],
                   act, image_rope, zero)
    rope, index = joint
    rows = Rows(torch.cat([tf.single_stream_modulations["image"](act), cond_rows]), index)
    tokens = torch.cat([image, cond], dim=1)
    for block in tf.single_transformer_blocks:
        tokens = block(tokens, rows, rope)
    return tf.proj_out(tf.norm_out(tokens[:, : image.shape[1]], temb[:, None]))


class Klein(
    Model[KleinGraph],
    encoded_leaves="accept",
    # As BFL's int8 release (`comfy_attention.config`): the image and joint blocks' attention at 8
    # bits; the text and reference early blocks keep exact attention.
    attention_bits={
        "transformer.mode_blocks.image.": 8,
        "transformer.single_transformer_blocks.": 8,
    },
):
    """One Klein checkpoint: the distilled flow map (no guidance) or the base model (CFG)."""

    graph: KleinGraph
    processor: Qwen3VLProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(KleinGraph, factory=KleinGraph)
        self.processor = vlm.processor(loader.assets, "text_encoder/")

    @property
    def flow_map(self) -> bool:
        return bool(self.graph.components["transformer"].config.flow_map)

    @uses_components("text_encoder", memoize=True, memo_dependencies=(vlm.processor, _conditioning_only))
    def encode(self, prompt: str) -> torch.Tensor:
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            embeds = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                encoder,
                self.processor,
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

    @uses_components("vae", memoize=True)
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
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
    ) -> torch.Tensor:
        """BFL's Euler sampler: a flow map jumps to the next timestep; the base model takes CFG.

        Text and references sit at t = 0, so the base model runs their early blocks once per
        request; a flow map adds each step's target time to every mode, so it runs them per step.
        """
        tf = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        noise = rng.noise((rows * columns, tf.config.in_channels), seed)
        x = noise.to(torch.bfloat16).float().to(device)[None]  # bf16 values, an fp32 state across steps
        texts = [embeds] if negative is None else [negative, embeds]
        texts = [t.to(device=device, dtype=torch.bfloat16) for t in texts]
        refs = None if references is None else references[0].to(device)
        timesteps = schedule(steps)
        with torch.inference_mode():
            zero = torch.zeros(1, dtype=torch.long, device=device)
            image_rope = tf.pos_embed(image_ids(rows, columns).to(device))
            text_ropes = [tf.pos_embed(text_ids(t.shape[1]).to(device)) for t in texts]
            ref_rope = None if references is None else tf.pos_embed(references[1].to(device))
            joints = [_joint(image_rope, rope, ref_rope) for rope in text_ropes]
            at_zero = _embed(tf, tf.timestep_embedder, 0.0, device)
            fixed = None
            if not self.flow_map:
                fixed = _conditions(tf, texts, text_ropes, refs, ref_rope, F.silu(at_zero), zero)
            for step in range(steps):
                cancel()
                t, nxt = timesteps[step], timesteps[step + 1]
                temb = _embed(tf, tf.timestep_embedder, t, device)
                conditions = fixed
                if conditions is None:
                    target = _embed(tf, tf.target_timestep_embedder, nxt, device)
                    temb = temb + target
                    act = F.silu(at_zero + target)
                    conditions = _conditions(tf, texts, text_ropes, refs, ref_rope, act, zero)
                tokens, cond_rows = conditions
                image = x.to(torch.bfloat16)
                if len(tokens) == 2 and tokens[0].shape == tokens[1].shape:
                    # CFG texts of one length share the joint rope and rows: one call, batch 2.
                    both = _velocity(tf, image.repeat(2, 1, 1), temb, torch.cat(tokens), cond_rows,
                                     image_rope, joints[0], zero)
                    velocities = list(both.float().split(1))
                else:
                    velocities = [
                        _velocity(tf, image, temb, cond, cond_rows, image_rope, joint, zero).float()
                        for cond, joint in zip(tokens, joints, strict=True)
                    ]
                if guidance is None:
                    (velocity,) = velocities
                else:
                    unconditional, conditional = velocities
                    velocity = unconditional + guidance * (conditional - unconditional)
                x = x + (nxt - t) * velocity
                on_step(step)
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
                raise ValueError("Klein's decoder produced non-finite pixels")
            pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)
