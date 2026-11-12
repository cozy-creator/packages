"""P-H3-Image's lanes. `bf16-full` is Pruna's upload as it is. `pruned` writes `bf16` from it: the DiT's time
embedder and AdaLN projections replaced by exact tables of the rows the 8 steps read (`model.table_keys`), and
the VAE's encoder, `quant_conv` and pixel projection stored float32, as upstream computes them. The quantized
lanes are recipes over `bf16` (`cozy model quantize <bf16 checkpoint> --lane int8 --package p-h3-image`).
"""

from __future__ import annotations

import json
import math
from importlib.resources import files

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
from cozy_runtime.weights_store import Checkpoint, Derivation, Output
from torch.nn import functional as F

from .application import app
from .model import TABLES, PH3Image, table_keys
from .transformer import parse_tables

#: The components a P-H3-Image checkpoint stores tensors for.
COMPONENTS = ("prompt_upsampler", "text_connector", "text_encoder", "transformer", "vae")
#: Upstream computes these in float32 from their bf16 bytes (`AutoencoderKLPH3Image._keep_in_fp32_modules`):
#: `bf16` stores them widened, exactly.
WIDENED = ("encoder.", "quant_conv.", "decoder.proj_out.")
MAX_PRUNED_BYTES = 6 << 30
#: The DiT's modulation shape: hidden, blocks, timestep features, time embedder hidden and output widths.
HIDDEN, BLOCKS, FREQ_DIM, TIME_HIDDEN, TIME_DIM = 5376, 50, 256, 5376, 2688
_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


def _removed() -> set[str]:
    """The modulation weights the tables replace."""
    keys = {f"time_embedder.linear_{n}.{p}" for n in (1, 2) for p in ("weight", "bias")}
    keys |= {f"transformer_blocks.{i}.adaln_proj.linear.{p}" for i in range(BLOCKS) for p in ("weight", "bias")}
    return keys | {"norm_out.linear.weight", "norm_out.linear.bias"}


def _read(source: Checkpoint, key: str, dtype: torch.dtype, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    value = torch.empty(shape, dtype=dtype)
    source.components["transformer"].tensors.read_into(key, memoryview(value.view(torch.uint8).numpy()))
    return value.to(device)


def _put(out: Output, component: str, key: str, value: torch.Tensor) -> int:
    data = value.detach().to("cpu").contiguous()
    raw = memoryview(data.view(torch.uint8).numpy()).cast("B")
    out.put(component, key, _NAMES[data.dtype], tuple(data.shape), raw)
    return len(raw)


def _features(timesteps: torch.Tensor) -> torch.Tensor:
    """Diffusers' `Timesteps(FREQ_DIM, True, 0)`: cosines, then sines."""
    half = FREQ_DIM // 2
    exponent = -math.log(10_000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    phase = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


@torch.inference_mode()
def _write_tables(ctx: Context, source: Checkpoint, out: Output) -> int:
    """Each block's `adaln_proj.table` and `norm_out.table`, in Diffusers 0.41's operation order: the time
    embedder in float32, SiLU to bf16, each projection's rows at the labelled `(timestep, modality)` pairs. One
    block's projection is resident at a time. The bytes written."""
    device = torch.device(str(ctx.device))
    timesteps, block_rows = parse_tables(table_keys())
    t = torch.tensor(timesteps, dtype=torch.float32, device=device)
    w1 = _read(source, "time_embedder.linear_1.weight", torch.float32, (TIME_HIDDEN, FREQ_DIM), device)
    b1 = _read(source, "time_embedder.linear_1.bias", torch.float32, (TIME_HIDDEN,), device)
    w2 = _read(source, "time_embedder.linear_2.weight", torch.float32, (TIME_DIM, TIME_HIDDEN), device)
    b2 = _read(source, "time_embedder.linear_2.bias", torch.float32, (TIME_DIM,), device)
    activated = F.silu(F.linear(F.silu(F.linear(_features(t), w1, b1)), w2, b2)).to(torch.bfloat16)
    del w1, b1, w2, b2
    sparse = torch.tensor([row * 3 + tag for row, tag in block_rows], dtype=torch.int64, device=device)
    written = 0
    for index in range(BLOCKS):
        ctx.raise_if_cancelled()
        key = f"transformer_blocks.{index}.adaln_proj.table"
        if not out.has("transformer", key):
            prefix = f"transformer_blocks.{index}.adaln_proj.linear"
            weight = _read(source, f"{prefix}.weight", torch.bfloat16, (18 * HIDDEN, TIME_DIM), device)
            bias = _read(source, f"{prefix}.bias", torch.bfloat16, (18 * HIDDEN,), device)
            rows = F.linear(activated, weight, bias).reshape(-1, 6, HIDDEN).index_select(0, sparse)
            written += _put(out, "transformer", key, rows)
            del weight, bias, rows
        ctx.progress(index, BLOCKS + 1)
    if not out.has("transformer", "norm_out.table"):
        weight = _read(source, "norm_out.linear.weight", torch.bfloat16, (2 * HIDDEN, TIME_DIM), device)
        bias = _read(source, "norm_out.linear.bias", torch.bfloat16, (2 * HIDDEN,), device)
        written += _put(out, "transformer", "norm_out.table", F.linear(activated, weight, bias).reshape(len(timesteps), 2, HIDDEN))
    ctx.progress(BLOCKS, BLOCKS + 1)
    return written


async def pruned(ctx: Context, *, source: Weights, tel: Telemetry) -> ModelArtifact:
    """P-H3-Image's `bf16` lane from its `bf16-full` upload: exact AdaLN tables for the 8 steps."""
    if ctx.device.type != "cuda":
        raise UnsupportedInput("AdaLN tables need a CUDA machine; run on a GPU rental", code="adaln_tables_need_cuda")
    src = ctx.tensorfs_source(source)
    present = {name: component.tensors for name, component in src.components.items()}
    held = sorted(name for name, tensors in present.items() if tensors)
    if held != sorted(COMPONENTS):
        raise UnsupportedInput(
            f"this checkpoint has {held}; P-H3-Image's holds {list(COMPONENTS)}: upload hf://PrunaAI/P-H3-Image "
            "as it is (`bf16-full`)",
            code="p_h3_image_source",
        )
    section = dict(src.components["transformer"].config)
    shape = tuple(section.get(name) for name in ("hidden_size", "num_layers", "freq_dim", "time_embed_hidden_dim", "time_embed_dim"))
    if shape != (HIDDEN, BLOCKS, FREQ_DIM, TIME_HIDDEN, TIME_DIM):
        raise UnsupportedInput(f"this DiT's modulation shape is {shape}, not P-H3-Image's", code="p_h3_image_source")
    if TABLES in section or not _removed() <= set(present["transformer"]):
        raise UnsupportedInput("this DiT holds no AdaLN projections: write `bf16` from `bf16-full`", code="p_h3_image_source")
    timesteps, block_rows = parse_tables(table_keys())
    tables = {f"transformer_blocks.{i}.adaln_proj.table": (len(block_rows), 6, HIDDEN) for i in range(BLOCKS)}
    tables["norm_out.table"] = (len(timesteps), 2, HIDDEN)
    order = json.loads(files(__package__).joinpath("assets", "transformer-order.json").read_bytes())
    if set(order) != (set(present["transformer"]) - _removed()) | set(tables):
        raise UnsupportedInput("this DiT is not P-H3-Image's", code="p_h3_image_source")
    rows = [(name, key) for name in present for key in (order if name == "transformer" else present[name])]
    widened = {key for key, row in present["vae"].items() if key.startswith(WIDENED) and row.dtype.lower() == "bf16"}
    with ctx.output("bf16", Derivation(base=None, sources=(src,), order=rows)) as out:
        document: dict[str, object] = {name: dict(component.config) for name, component in src.components.items()}
        document["transformer"] = {**section, TABLES: table_keys()}
        for name, config in canonical_json.decode(canonical_json.encode(document)).items():
            out.config(name, config)
        out.inherit(src)
        with tel.stage("copying"):
            for done, (component, key) in enumerate(rows, 1):
                if out.has(component, key) or (component == "transformer" and key in tables):
                    continue
                if component == "vae" and key in widened:
                    row = present["vae"][key]
                    value = torch.empty(tuple(row.shape), dtype=torch.bfloat16)
                    src.components["vae"].tensors.read_into(key, memoryview(value.view(torch.uint8).numpy()))
                    _put(out, "vae", key, value.float())
                else:
                    out.copy(component, key, src)
                ctx.progress(done - 1, len(rows))
        with tel.stage("AdaLN tables"):
            written = _write_tables(ctx, src, out)
        artifact = ctx.adopt_model(out)
    tel.log("p-h3-image bf16", tables=len(tables), table_bytes=sum(2 * math.prod(s) for s in tables.values()),
            widened=len(widened), written=written)  # fmt: skip
    return artifact


app.entrypoint(pruned, name="pruned", weights=(WeightsOutput("bf16", max_new_bytes=MAX_PRUNED_BYTES),))


_BLOCKS = "transformer.transformer_blocks.*"
_REFINER = "transformer.token_refiner.refiner_blocks.*"
_TEXT = "text_encoder.model.language_model.layers.*"
_DECODER = "vae.decoder.transformer_blocks.*"
_UPSAMPLER = "prompt_upsampler.model.language_model.layers.*"
#: The DiT's linears (every block's q/k/v/o and FFN, the token refiner's, the text embedder) and the text
#: encoder's language layers. The AdaLN tables, embeddings, vision towers, norms and pixel projections keep their
#: source bytes.
_DIT_TEXT = (
    f"{_BLOCKS}.attn.to_[qkv].weight", f"{_BLOCKS}.attn.to_out.0.weight", f"{_BLOCKS}.ff.net.0.proj.weight",
    f"{_BLOCKS}.ff.net.2.weight", f"{_REFINER}.attn.to_[qkv].weight", f"{_REFINER}.attn.to_out.0.weight",
    f"{_REFINER}.ff.net.0.proj.weight", f"{_REFINER}.ff.net.2.weight", "transformer.context_embedder.weight",
    f"{_TEXT}.self_attn.[qkvo]_proj.weight", f"{_TEXT}.mlp.*_proj.weight",
)  # fmt: skip
#: The upsampler's language layers and head: the bytes each decoded token reads.
_UPSAMPLING = (
    f"{_UPSAMPLER}.mlp.*_proj.weight", f"{_UPSAMPLER}.linear_attn.in_proj_qkv.weight",
    f"{_UPSAMPLER}.linear_attn.in_proj_z.weight", f"{_UPSAMPLER}.linear_attn.out_proj.weight",
    f"{_UPSAMPLER}.self_attn.[qkvo]_proj.weight", "prompt_upsampler.lm_head.weight",
)  # fmt: skip
#: The VAE decoder's 216 block linears; its pixel projection stays.
_DECODING = (
    f"{_DECODER}.attn.to_[qkv].weight", f"{_DECODER}.attn.to_out.0.weight", f"{_DECODER}.ff.net.0.proj.weight",
    f"{_DECODER}.ff.net.2.weight",
)  # fmt: skip

#: W8A8 on the int8 tensor cores: Pruna's ComfyUI int8 files are this encoding (`int8_convrot`).
INT8 = Recipe(
    lane="int8",
    model=PH3Image,
    plan=QuantizationPlan(
        encoding="int8.convrot", components=("prompt_upsampler", "text_encoder", "transformer", "vae"),
        include=(*_DIT_TEXT, *_UPSAMPLING, *_DECODING), output_precision="preserve",
    ),
)  # fmt: skip
#: The DiT, text encoder and upsampler row-wise FP8, the VAE as stored.
FP8 = Recipe(
    lane="fp8",
    model=PH3Image,
    plan=QuantizationPlan(
        encoding="fp8.rowwise", components=("prompt_upsampler", "text_encoder", "transformer"),
        include=(*_DIT_TEXT, *_UPSAMPLING), output_precision="preserve",
    ),
)  # fmt: skip
#: The DiT blocks NVFP4 (Blackwell's FP4 tensor cores), the rest int8; the token refiner, text embedder and blocks
#: 0-2's `ff.net.2` (audio rows far below the image's amax) stay int8.
NVFP4 = Recipe(
    lane="nvfp4",
    model=PH3Image,
    plan=QuantizationPlan(
        encoding="int8.convrot", components=INT8.plan.components, include=INT8.plan.include,
        encodings=(
            EncodingRule("transformer.transformer_blocks.0.ff.net.2.weight", "int8.convrot"),
            EncodingRule("transformer.transformer_blocks.1.ff.net.2.weight", "int8.convrot"),
            EncodingRule("transformer.transformer_blocks.2.ff.net.2.weight", "int8.convrot"),
            EncodingRule(_BLOCKS, "nvfp4"),
        ),
        output_precision="preserve",
    ),
)  # fmt: skip
