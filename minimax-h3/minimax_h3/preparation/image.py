"""P-H3-Image's lanes (`minimax_h3/image.py`). `bf16-full` is the upload as it is. `p-h3-image-pruned` writes
`bf16` from it: the DiT's AdaLN projections replaced by tables at the one 8-step schedule (H3's producer), and
the VAE's encoder, `quant_conv` and pixel projection stored float32, as upstream computes them. The quantized
lanes are declared recipes over `bf16` (`cozy model quantize <bf16 commit> --lane int8 --package minimax-h3`)."""

from __future__ import annotations

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
from cozy_runtime.weights_store import Derivation

from .. import image as _image
from ..application import app
from .kernel import H3Topology, removed_keys, table_bytes, table_shapes
from .order import current_order
from .plans import parse_plan
from .source import Rows, dtype, open_source, put, read, rows, write_config, write_tables

#: The components a P-H3-Image checkpoint stores tensors for.
COMPONENTS = ("prompt_upsampler", "text_connector", "text_encoder", "transformer", "vae")
#: Upstream computes these in float32 from their bf16 bytes (`AutoencoderKLPH3Image._keep_in_fp32_modules`):
#: `bf16` stores them widened, exactly. H3's own VAE computes its encoder in f16; this one was stored bf16.
WIDENED = ("encoder.", "quant_conv.", "decoder.proj_out.")
MAX_PRUNED_BYTES = 6 << 30


def _require(present: Rows) -> None:
    held = sorted(name for name, tensors in present.items() if tensors)
    if held != sorted(COMPONENTS):
        raise UnsupportedInput(
            f"this checkpoint has {held}; P-H3-Image's holds {list(COMPONENTS)}: upload "
            "hf://PrunaAI/P-H3-Image as it is (`bf16-full`)",
            code="p_h3_image_source",
        )


async def p_h3_image_pruned(ctx: Context, *, source: Weights, tel: Telemetry) -> ModelArtifact:
    """P-H3-Image's AdaLN-pruned `bf16` lane from its `bf16-full` upload."""
    if ctx.device.type != "cuda":
        raise UnsupportedInput("AdaLN tables need a CUDA worker; run on a GPU rental", code="h3_tables_need_cuda")
    src = open_source(ctx, source)
    present = rows(src)
    _require(present)
    plan = parse_plan(_image.plan().canonical_bytes(), task="fl2va")
    section = dict(src.components["transformer"].config)
    topology = H3Topology.from_config(section)
    held = present["transformer"]
    removed = set(removed_keys(topology))
    if not removed <= set(held):
        raise UnsupportedInput(
            "this transformer lacks its AdaLN projections: write `bf16` from the `bf16-full` upload",
            code="p_h3_image_source",
        )
    tables = table_shapes(topology, plan)
    pruned = [key for component, key in current_order(files(__package__).joinpath("assets", "whole-order.json").read_bytes()).rows
              if component == "transformer"]  # fmt: skip
    if set(pruned) != (set(held) - removed) | set(tables):
        raise UnsupportedInput("this transformer is not H3's FL2VA architecture", code="p_h3_image_source")
    order = [(name, key) for name in present for key in (pruned if name == "transformer" else present[name])]
    document: dict[str, dict[str, object]] = {name: dict(component.config) for name, component in src.components.items()}
    document["transformer"] = {**section, "cozy_h3": {"task": "fl2va", "modulation": "adaln-pruned",
                                                      "table_keys": plan.table_keys}}  # fmt: skip
    widened = {key for key, row in present["vae"].items() if key.startswith(WIDENED) and dtype(row) == "bf16"}
    with ctx.output("bf16", Derivation(base=None, sources=(src,), order=order)) as out:
        write_config(out, canonical_json.encode(document))
        out.inherit(src)
        with tel.stage("bf16-copy"):
            for done, (component, key) in enumerate(order, 1):
                if out.has(component, key) or (component == "transformer" and key in tables):
                    continue
                if component == "vae" and key in widened:
                    row = present["vae"][key]
                    put(out, "vae", key, read(src, "vae", key, torch.bfloat16, tuple(row.shape)).float())
                else:
                    out.copy(component, key, src)
                ctx.progress(done - 1, len(order))
        with tel.stage("timestep-table"):
            got = write_tables(ctx, plan=plan, topology=topology, source=src, component="transformer", outputs=(out,),
                               progress=lambda done, total: ctx.progress(done - 1, total))  # fmt: skip
        artifact = ctx.adopt_model(out)
    tel.log("p-h3-image pruned", tables=len(tables), table_bytes=table_bytes(topology, plan), widened=len(widened),
            source_bytes=got[0], new_bytes=got[1], reused_tables=got[2])  # fmt: skip
    return artifact


app.entrypoint(
    p_h3_image_pruned,
    name="p-h3-image-pruned",
    weights=(WeightsOutput("bf16", max_new_bytes=MAX_PRUNED_BYTES),),
)


_BLOCKS = "transformer.transformer_blocks.*"
_REFINER = "transformer.token_refiner.refiner_blocks.*"
_TEXT = "text_encoder.model.language_model.layers.*"
_DECODER = "vae.decoder.transformer_blocks.*"
_UPSAMPLER = "prompt_upsampler.model.language_model.layers.*"
#: H3's reviewed DiT linears (its 313-row plan: every block's q/k/v/o and FFN, the token refiner's, the text
#: embedder) and the text encoder's language layers. The AdaLN tables, embeddings, vision towers, norms and
#: pixel projections keep their source bytes.
_DIT_TEXT = (
    f"{_BLOCKS}.attn.to_[qkv].weight", f"{_BLOCKS}.attn.to_out.0.weight", f"{_BLOCKS}.ff.net.0.proj.weight",
    f"{_BLOCKS}.ff.net.2.weight", f"{_REFINER}.attn.to_[qkv].weight", f"{_REFINER}.attn.to_out.0.weight",
    f"{_REFINER}.ff.net.0.proj.weight", f"{_REFINER}.ff.net.2.weight", "transformer.context_embedder.weight",
    f"{_TEXT}.self_attn.[qkvo]_proj.weight", f"{_TEXT}.mlp.*_proj.weight",
)  # fmt: skip
#: The upsampler's language layers and head: the bytes each decoded token reads (Klein's upsampler lanes).
_UPSAMPLING = (
    f"{_UPSAMPLER}.mlp.*_proj.weight", f"{_UPSAMPLER}.linear_attn.in_proj_qkv.weight",
    f"{_UPSAMPLER}.linear_attn.in_proj_z.weight", f"{_UPSAMPLER}.linear_attn.out_proj.weight",
    f"{_UPSAMPLER}.self_attn.[qkvo]_proj.weight", "prompt_upsampler.lm_head.weight",
)  # fmt: skip
#: The VAE decoder's 216 block linears, as H3's int8 and nvfp4 lanes take them; its pixel projection stays.
_DECODING = (
    f"{_DECODER}.attn.to_[qkv].weight", f"{_DECODER}.attn.to_out.0.weight", f"{_DECODER}.ff.net.0.proj.weight",
    f"{_DECODER}.ff.net.2.weight",
)  # fmt: skip

#: W8A8 on the int8 tensor cores: Pruna's ComfyUI int8 files are this encoding (`int8_convrot`).
IMAGE_INT8 = Recipe(
    lane="int8",
    model=_image.PH3Image,
    plan=QuantizationPlan(
        encoding="int8.convrot", components=("prompt_upsampler", "text_encoder", "transformer", "vae"),
        include=(*_DIT_TEXT, *_UPSAMPLING, *_DECODING), output_precision="preserve",
    ),
)  # fmt: skip
#: The DiT, text encoder and upsampler row-wise FP8, the VAE as stored: H3's fp8 lane.
IMAGE_FP8 = Recipe(
    lane="fp8",
    model=_image.PH3Image,
    plan=QuantizationPlan(
        encoding="fp8.rowwise", components=("prompt_upsampler", "text_encoder", "transformer"),
        include=(*_DIT_TEXT, *_UPSAMPLING), output_precision="preserve",
    ),
)  # fmt: skip
#: The DiT blocks NVFP4 (Blackwell's FP4 tensor cores), the rest int8: H3's nvfp4 lane, whose token refiner,
#: text embedder and blocks 0-2's `ff.net.2` (audio rows far below the video's amax) stay int8.
IMAGE_NVFP4 = Recipe(
    lane="nvfp4",
    model=_image.PH3Image,
    plan=QuantizationPlan(
        encoding="int8.convrot", components=IMAGE_INT8.plan.components, include=IMAGE_INT8.plan.include,
        encodings=(
            *(EncodingRule(f"transformer.transformer_blocks.{block}.ff.net.2.weight", "int8.convrot") for block in range(3)),
            EncodingRule(_BLOCKS, "nvfp4"),
        ),
        output_precision="preserve",
    ),
)  # fmt: skip
