"""MiniMax-H3-Image's lanes and its prompt expander's.

The image model: `bf16-full` is MiniMax's Diffusers upload as it is (`transformer_diffusers`, `vae_diffusers`).
`prepare` writes `bf16` from it: the DiT's time embedder and AdaLN projections replaced by exact tables of the rows the
6 steps read (`model.table_keys`), and the VAE split into `vae_encoder` (H3's encoder, float32, byte-identical) and
`vae_decoder` (bf16 as Diffusers loads it, its norms and layer scales float32). `int8` (the default), `w4a8`, `nvfp4`
and `fp8` are recipes over `bf16` (`cozy model quantize <bf16 checkpoint> --lane int8 --package minimax-h3-image`).

The prompt expander: `nvfp4` is MiniMax's ModelOpt NVFP4 upload as it is. `expander_bf16` writes `bf16` from it, every
NVFP4 linear decoded (its FP4 values times their block and global scales, rounded once to bf16), and, with
`abliterate`, the refusal direction projected out of every residual writer. `int8`, `w4a8` and `nvfp4` are recipes
over a `bf16`. Its int8 can be no better than the NVFP4 it decodes: the FP4 grid is the source.
"""

from __future__ import annotations

import math
import msgspec
import torch
from cozy_runtime.author import (
    Context,
    EncodingRule,
    ModelArtifact,
    QuantizationPlan,
    Recipe,
    Telemetry,
    UnsupportedInput,
    Weights,
    WeightsOutput,
    canonical_json,
)
from cozy_runtime.vlm import COMPONENT
from cozy_runtime.weights_store import Checkpoint, Derivation, Output, read_into
from torch.nn import functional as F

from .application import app
from .expander import H3ImagePE
from .model import TABLES, H3Image, table_keys
from .transformer import build_transformer, compute_dtypes, parse_tables
from .vae import ImageDecoder, ImageEncoder, kept_fp32

MAX_PREPARED_BYTES = 4 << 30
#: The DiT's modulation shape: hidden, blocks, timestep features, time embedder hidden and output widths.
HIDDEN, BLOCKS, FREQ_DIM, TIME_HIDDEN, TIME_DIM = 5376, 50, 256, 5376, 2688
_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}
_TORCH = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def _removed() -> set[str]:
    """The modulation weights the tables replace."""
    keys = {f"time_embedder.linear_{n}.{p}" for n in (1, 2) for p in ("weight", "bias")}
    keys |= {f"transformer_blocks.{i}.adaln_proj.linear.{p}" for i in range(BLOCKS) for p in ("weight", "bias")}
    return keys | {"norm_out.linear.weight", "norm_out.linear.bias"}


def _plain(source: Checkpoint, component: str, key: str) -> torch.Tensor:
    """A plain tensor of `source`, on the CPU in its stored dtype."""
    row = source.components[component].tensors[key]
    value = torch.empty(tuple(row.shape), dtype=_TORCH[row.dtype.lower()])
    flat = value.reshape(-1).view(torch.uint8)
    step = 32 << 20
    for start in range(0, flat.numel(), step):
        read_into(source, component, key, "", start, memoryview(flat[start : start + step].numpy()))
    return value


def _put(out: Output, component: str, key: str, value: torch.Tensor) -> int:
    data = value.detach().to("cpu").contiguous()
    raw = memoryview(data.view(torch.uint8).numpy()).cast("B")
    out.put(component, key, _NAMES[data.dtype], tuple(data.shape), raw)
    return len(raw)


def _features(timesteps: torch.Tensor) -> torch.Tensor:
    """Diffusers' `Timesteps(FREQ_DIM, flip_sin_to_cos=True, downscale_freq_shift=0)`: cosines, then sines."""
    half = FREQ_DIM // 2
    exponent = -math.log(10_000) * torch.arange(0, half, dtype=torch.float32) / half
    phase = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


@torch.inference_mode()
def _write_tables(ctx: Context, source: Checkpoint, dit: str, out: Output) -> int:
    """Each block's `adaln_proj.table` and `norm_out.table` in Diffusers' operation order: the time embedder in
    float32, SiLU to bf16, each projection's rows at the labelled `(timestep, modality)` pairs, accumulated in float32
    and rounded once to bf16 (as a CUDA bf16 GEMM does). The bytes written."""
    timesteps, block_rows = parse_tables(table_keys())
    t = torch.tensor(timesteps, dtype=torch.float32)
    w1, b1 = (_plain(source, dit, f"time_embedder.linear_1.{p}").float() for p in ("weight", "bias"))
    w2, b2 = (_plain(source, dit, f"time_embedder.linear_2.{p}").float() for p in ("weight", "bias"))
    activated = F.silu(F.linear(F.silu(F.linear(_features(t), w1, b1)), w2, b2)).to(torch.bfloat16).float()
    sparse = torch.tensor([row * 3 + tag for row, tag in block_rows], dtype=torch.int64)
    written = 0
    for index in range(BLOCKS):
        ctx.raise_if_cancelled()
        key = f"transformer_blocks.{index}.adaln_proj.table"
        if not out.has("transformer", key):
            prefix = f"transformer_blocks.{index}.adaln_proj.linear"
            weight, bias = (_plain(source, dit, f"{prefix}.{p}").float() for p in ("weight", "bias"))
            rows = F.linear(activated, weight, bias).to(torch.bfloat16).reshape(-1, 6, HIDDEN).index_select(0, sparse)
            written += _put(out, "transformer", key, rows)
            del weight, bias, rows
        ctx.progress(index, BLOCKS + 1)
    if not out.has("transformer", "norm_out.table"):
        weight, bias = (_plain(source, dit, f"norm_out.linear.{p}").float() for p in ("weight", "bias"))
        rows = F.linear(activated, weight, bias).to(torch.bfloat16).reshape(len(timesteps), 2, HIDDEN)
        written += _put(out, "transformer", "norm_out.table", rows)
    ctx.progress(BLOCKS, BLOCKS + 1)
    return written


def _named(source: Checkpoint, *names: str) -> str:
    for name in names:
        if name in source.components and source.components[name].tensors:
            return name
    raise UnsupportedInput(
        f"this checkpoint has {sorted(source.components)}, not MiniMax-H3-Image's {names[0]}: upload "
        "hf://MiniMaxAI/MiniMax-H3-Image:transformer_diffusers,vae_diffusers as it is (`bf16-full`)",
        code="minimax_h3_image_source",
    )


def _order(module: torch.nn.Module) -> list[str]:
    return list(module.state_dict().keys())


async def prepare(ctx: Context, *, source: Weights, tel: Telemetry) -> ModelArtifact:
    """MiniMax-H3-Image's `bf16` lane from its `bf16-full` upload: exact AdaLN tables for the 6 steps, the VAE as an
    encoder and a decoder."""
    src = ctx.tensorfs_source(source)
    dit, vae = _named(src, "transformer_diffusers", "transformer"), _named(src, "vae_diffusers", "vae")
    present = src.components[dit].tensors
    section = dict(src.components[dit].config)
    shape = tuple(section.get(n) for n in ("hidden_size", "num_layers", "freq_dim", "time_embed_hidden_dim", "time_embed_dim"))
    if shape != (HIDDEN, BLOCKS, FREQ_DIM, TIME_HIDDEN, TIME_DIM) or TABLES in section or not _removed() <= set(present):
        raise UnsupportedInput(f"this DiT ({shape}) is not MiniMax-H3-Image's as uploaded", code="minimax_h3_image_source")
    vae_config = {k: v for k, v in dict(src.components[vae].config).items() if not k.startswith("_")}
    dit_config = {**{k: v for k, v in section.items() if not k.startswith("_")}, TABLES: table_keys()}
    with torch.device("meta"):
        dit_order = _order(compute_dtypes(build_transformer(dit_config, parse_tables(table_keys()))))
        encoder_order, decoder_order = _order(ImageEncoder(vae_config)), _order(ImageDecoder(vae_config))
    tables = {k for k in dit_order if k.endswith(".table")}
    if set(dit_order) != (set(present) - _removed()) | tables:
        raise UnsupportedInput("this DiT's tensors are not MiniMax-H3-Image's", code="minimax_h3_image_source")
    vae_keys = set(src.components[vae].tensors)
    if set(encoder_order) | set(decoder_order) != vae_keys:
        raise UnsupportedInput("this VAE's tensors are not MiniMax-H3-Image's", code="minimax_h3_image_source")
    rows = [("transformer", k) for k in dit_order] + [("vae_encoder", k) for k in encoder_order]
    rows += [("vae_decoder", k) for k in decoder_order]
    converted = 0
    with ctx.output("bf16", Derivation(base=None, sources=(src,), order=rows)) as out:
        configs = {"transformer": dit_config, "vae_encoder": vae_config, "vae_decoder": vae_config}
        for name, config in canonical_json.decode(canonical_json.encode(configs)).items():
            out.config(name, config)
        for name, held in src.components.items():  # the release's LICENSE and NOTICE travel with the DiT
            for asset in held.assets:
                if asset.upper().startswith(("LICENSE", "NOTICE")):
                    out.component("transformer").asset(asset, held.asset(asset))
        with tel.stage("copying"):
            for done, (component, key) in enumerate(rows, 1):
                if out.has(component, key) or key in tables:
                    continue
                if component == "transformer":
                    out.copy(component, key, src, from_component=dit)
                elif component == "vae_decoder" and not kept_fp32(key) and src.components[vae].tensors[key].dtype.lower() != "bf16":
                    converted += _put(out, component, key, _plain(src, vae, key).to(torch.bfloat16))
                else:
                    out.copy(component, key, src, from_component=vae)
                ctx.progress(done - 1, len(rows))
        with tel.stage("AdaLN tables"):
            written = _write_tables(ctx, src, dit, out)
        artifact = ctx.adopt_model(out)
    tel.log("minimax-h3-image bf16", tables=len(tables), table_bytes=written, decoder_bf16_bytes=converted)
    return artifact


app.entrypoint(prepare, name="prepare", accelerator=False,
               weights=(WeightsOutput("bf16", max_new_bytes=MAX_PREPARED_BYTES),))  # fmt: skip


_BLOCKS = "transformer.transformer_blocks.*"
_REFINER = "transformer.token_refiner.refiner_blocks.*"
_DECODER = "vae_decoder.decoder.transformer_blocks.*"
#: The DiT's linears: every block's q/k/v/o and FFN, the token refiner's and the text embedder. The AdaLN tables,
#: input projection, output heads and norms keep their bytes.
_DIT = (
    f"{_BLOCKS}.attn.to_[qkv].weight", f"{_BLOCKS}.attn.to_out.0.weight", f"{_BLOCKS}.ff.net.0.proj.weight",
    f"{_BLOCKS}.ff.net.2.weight", f"{_REFINER}.attn.to_[qkv].weight", f"{_REFINER}.attn.to_out.0.weight",
    f"{_REFINER}.ff.net.0.proj.weight", f"{_REFINER}.ff.net.2.weight", "transformer.context_embedder.weight",
)  # fmt: skip
#: The VAE decoder's 24 blocks' linears; its pixel projection and the encoder stay.
_DECODING = (
    f"{_DECODER}.attn.to_[qkv].weight", f"{_DECODER}.attn.to_out.0.weight", f"{_DECODER}.ff.net.0.proj.weight",
    f"{_DECODER}.ff.net.2.weight",
)  # fmt: skip

#: W8A8 on the int8 tensor cores: the default lane (Paul, 11-16).
INT8 = Recipe(
    lane="int8",
    model=H3Image,
    plan=QuantizationPlan(encoding="int8.convrot", components=("transformer", "vae_decoder"),
                          include=(*_DIT, *_DECODING), output_precision="preserve"),
)  # fmt: skip
#: The DiT's blocks in W4A8 (4-bit codes, 8-bit activations), the refiner, text embedder and decoder int8.
W4A8 = Recipe(
    lane="w4a8",
    model=H3Image,
    plan=QuantizationPlan(encoding="int8.convrot", components=INT8.plan.components, include=INT8.plan.include,
                          encodings=(EncodingRule(_BLOCKS, "w4a8.convrot"),), output_precision="preserve"),
)  # fmt: skip
#: The DiT's blocks NVFP4 (Blackwell's FP4 tensor cores), the rest int8.
NVFP4 = Recipe(
    lane="nvfp4",
    model=H3Image,
    plan=QuantizationPlan(encoding="int8.convrot", components=INT8.plan.components, include=INT8.plan.include,
                          encodings=(EncodingRule(_BLOCKS, "nvfp4"),), output_precision="preserve"),
)  # fmt: skip

_LM = "model.model.language_model.layers.*"
#: The expander's language layers: attention, gated-delta-net and MLP projections. The vision tower, embeddings,
#: decay/gate projections, convolutions, norms and LM head keep their bytes.
_EXPANDING = (
    f"{_LM}.linear_attn.in_proj_[qz]*.weight", f"{_LM}.linear_attn.out_proj.weight",
    f"{_LM}.self_attn.[qkvo]_proj.weight", f"{_LM}.mlp.*_proj.weight",
)  # fmt: skip
EXPANDER_INT8 = Recipe(
    lane="int8",
    model=H3ImagePE,
    plan=QuantizationPlan(encoding="int8.convrot", components=(COMPONENT,), include=_EXPANDING,
                          output_precision="preserve"),
)  # fmt: skip
#: The 4-bit lane before Blackwell: every MLP W4A8, the rest int8 (qwen3.6's).
EXPANDER_W4A8 = Recipe(
    lane="w4a8",
    model=H3ImagePE,
    plan=QuantizationPlan(encoding="int8.convrot", components=(COMPONENT,), include=_EXPANDING,
                          encodings=(EncodingRule(f"{_LM}.mlp.*", "w4a8.convrot"),), output_precision="preserve"),
)  # fmt: skip


#: E2M1's 16 codes, low nibble first (ModelOpt's `nibble_order: low_first`).
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _part(source: Checkpoint, key: str, role: str, nbytes: int) -> torch.Tensor:
    data = torch.empty(nbytes, dtype=torch.uint8)
    step = 32 << 20
    for start in range(0, nbytes, step):
        read_into(source, COMPONENT, key, role, start, memoryview(data[start : start + step].numpy()))
    return data


def _nvfp4(source: Checkpoint, key: str) -> torch.Tensor:
    """An NVFP4 weight's values in float32: each FP4 code times its 16-element block's E4M3 scale times the tensor's
    global scale (`multiply`), the scales row-major."""
    row = source.components[COMPONENT].tensors[key]
    encoding = row.encoding
    params = dict(encoding.params) if encoding is not None else {}
    if encoding is None or encoding.scheme != "nvfp4" or params.get("scale_layout", "row_major") != "row_major":
        raise UnsupportedInput(f"{key} is not a row-major NVFP4 weight", code="expander_source")
    rows, cols = (int(n) for n in row.shape)
    codes = _part(source, key, "weight", rows * cols // 2)
    low, high = codes & 0x0F, codes >> 4
    if params.get("nibble_order", "low_first") != "low_first":
        low, high = high, low
    values = _E2M1[torch.stack((low, high), dim=-1).reshape(rows, cols).long()]
    scales = _part(source, key, "scale", rows * cols // 16).view(torch.float8_e4m3fn).float().reshape(rows, cols // 16)
    glob = _part(source, key, "scale_2", 4).view(torch.float32)[0]
    if params.get("global_scale", "multiply") != "multiply":
        glob = 1.0 / glob
    return (values.reshape(rows, cols // 16, 16) * scales[..., None] * glob).reshape(rows, cols)


#: Residual writers: each writes the stream the refusal direction lives in.
_WRITERS = ("linear_attn.out_proj.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight")
_EMBED = "model.language_model.embed_tokens.weight"


async def expander_bf16(ctx: Context, *, source: Weights, tel: Telemetry) -> ModelArtifact:
    """The expander's `bf16` lane from its NVFP4 upload: each NVFP4 weight decoded and rounded once to bf16; every
    other tensor is the upload's."""
    src = ctx.tensorfs_source(source)
    tensors = src.components[COMPONENT].tensors
    order = list(tensors)
    decoded = 0
    with ctx.output("bf16", Derivation(base=src, order=[(COMPONENT, k) for k in order])) as out:
        with tel.stage("decoding"):
            for done, key in enumerate(order, 1):
                ctx.raise_if_cancelled()
                if tensors[key].encoding is not None and not out.has(COMPONENT, key):
                    _put(out, COMPONENT, key, _nvfp4(src, key).to(torch.bfloat16))
                    decoded += 1
                ctx.progress(done - 1, len(order))
        artifact = ctx.adopt_model(out)
    tel.log("expander bf16", decoded=decoded)
    return artifact


app.entrypoint(expander_bf16, name="expander_bf16", accelerator=False,
               weights=(WeightsOutput("bf16", max_new_bytes=64 << 30),))  # fmt: skip


class Abliteration(msgspec.Struct, kw_only=True):
    #: Requests the expander refuses or tones down, and ones it expands as asked: the refusal direction is the
    #: difference of their mean residuals.
    refused: list[str]
    expanded: list[str]
    #: The hidden state the direction is read at; None: the one of widest separation in the middle 30-80% of the
    #: stack.
    layer: int | None = None


async def abliterate(ctx: Context, payload: Abliteration, expander: H3ImagePE, *, source: Weights,
                     tel: Telemetry) -> ModelArtifact:  # fmt: skip
    """The expander's `bf16` lane (`source`) abliterated, as FailSpy's and Arditi et al.'s abliteration does: the
    refusal direction `r` (`expander`, any lane, reads each request's residual at its last prompt token) projected
    out of every residual writer (`W - r rᵀW`) and of the embeddings (`E - (E r) rᵀ`). Every other tensor is the
    source's."""
    src = ctx.tensorfs_source(source)
    tensors = src.components[COMPONENT].tensors
    if any(row.encoding is not None for row in tensors.values()):
        raise UnsupportedInput("abliterate a `bf16` lane (`expander_bf16`)", code="expander_source")
    with tel.stage("refusal direction"):
        refused, expanded = expander.residuals(payload.refused), expander.residuals(payload.expanded)
    difference = refused.mean(dim=1) - expanded.mean(dim=1)
    spread = torch.cat([refused, expanded], dim=1).std(dim=1).mean(dim=-1)
    separation = (difference.norm(dim=-1) / (spread * math.sqrt(difference.shape[-1]))).tolist()
    layer = payload.layer
    if layer is None:
        layer = max(range(int(0.3 * len(separation)), int(0.8 * len(separation))), key=lambda i: separation[i])
    r = difference[layer].float()
    r = r / r.norm()
    tel.log("refusal direction", layer=layer, separation=[round(s, 4) for s in separation])
    writers = [k for k in tensors if (k.startswith("model.language_model.layers.") and k.endswith(_WRITERS)) or k == _EMBED]
    with ctx.output("bf16", Derivation(base=src, order=[(COMPONENT, k) for k in tensors])) as out:
        with tel.stage("orthogonalizing"):
            for done, key in enumerate(writers, 1):
                ctx.raise_if_cancelled()
                if out.has(COMPONENT, key):
                    continue
                value = _plain(src, COMPONENT, key).float()
                if key == _EMBED:
                    value -= (value @ r)[:, None] * r[None, :]
                else:
                    value -= r[:, None] * (r @ value)[None, :]
                _put(out, COMPONENT, key, value.to(torch.bfloat16))
                ctx.progress(done - 1, len(writers))
        artifact = ctx.adopt_model(out)
    tel.log("abliterated", layer=layer, writers=len(writers))
    return artifact


app.entrypoint(abliterate, name="abliterate", weights=(WeightsOutput("bf16", max_new_bytes=24 << 30),))
