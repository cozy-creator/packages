"""PAN2's curve form (se-140), written from the full form: `vec(t)`, the time embedding's SiLU, is a smooth curve in
4096 dimensions; its first 192 singular directions hold it, so each modulation `linear` takes those 192 coordinates
(`W @ B`) and a table of them over 4097 points of t in [0, 1] replaces the time embedder (Comfy-Org's `pan2_*` files are
this form). `lanes.curve` writes `bf16` so; the int8 recipe quantizes it as it does the full form."""

from __future__ import annotations

import math

import torch
from cozy_runtime.author import Context, ModelArtifact, Telemetry, UnsupportedInput, Weights, WeightsOutput, canonical_json
from cozy_runtime.weights_store import Checkpoint, Derivation, Output
from torch.nn import functional as F

from . import app
from .model import CURVE

GRID, DIM = 4097, 192
HIDDEN, FREQ = 4096, 256
_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}
_TORCH = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _read(source: Checkpoint, key: str, device: torch.device) -> torch.Tensor:
    row = source.components["transformer"].tensors[key]
    value = torch.empty(tuple(row.shape), dtype=_TORCH[row.dtype.upper()])
    source.components["transformer"].tensors.read_into(key, memoryview(value.view(torch.uint8).numpy()))
    return value.to(device, torch.float32)


def _put(out: Output, key: str, value: torch.Tensor) -> None:
    data = value.detach().to("cpu", torch.float32).contiguous()
    out.put("transformer", key, _NAMES[data.dtype], tuple(data.shape), memoryview(data.view(torch.uint8).numpy()).cast("B"))


def _features(timesteps: torch.Tensor) -> torch.Tensor:
    """Diffusers' `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)`: cosines, then sines."""
    half = FREQ // 2
    exponent = -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    phase = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


def _vec(source: Checkpoint, timesteps: torch.Tensor) -> torch.Tensor:
    """`SiLU(time_embedder(time_proj(t)))`, the full form's modulation input, in float32."""
    device = timesteps.device
    hidden = F.silu(F.linear(_features(timesteps), _read(source, "time_embedder.linear_1.weight", device),
                             _read(source, "time_embedder.linear_1.bias", device)))
    return F.silu(F.linear(hidden, _read(source, "time_embedder.linear_2.weight", device),
                           _read(source, "time_embedder.linear_2.bias", device)))


async def curve(ctx: Context, *, source: Weights, tel: Telemetry) -> ModelArtifact:
    """PAN2's `bf16` in the curve form, from the full form (`ifm/pan2`'s bf16): the time-embedding curve's first 192
    singular directions `B`, its coordinates at 4097 points of t as `adaln_t_table`, and each modulation weight `W @ B`
    in float32. Every other tensor is copied as it is."""
    src = ctx.tensorfs_source(source)
    present = {name: component.tensors for name, component in src.components.items()}
    section = dict(src.components["transformer"].config)
    if CURVE[0] in section or "time_embedder.linear_1.weight" not in present.get("transformer", {}):
        raise UnsupportedInput("this PAN2 is not in the full form: write the curve from ifm/pan2's bf16", code="pan2_source")
    blocks = int(section["num_layers"])
    modulation = [f"transformer_blocks.{i}.norm1.linear.weight" for i in range(blocks)] + ["norm_out.linear.weight"]
    removed = {key for key in present["transformer"] if key.startswith("time_embedder.")}
    rows = [(name, key) for name in present for key in present[name] if not (name == "transformer" and key in removed)]
    rows.append(("transformer", "adaln_t_table"))
    device = torch.device(str(ctx.device))
    with torch.inference_mode():
        grid = torch.linspace(0, 1000, GRID, dtype=torch.float32, device=device)
        curve_ = _vec(src, grid)
        _, _, vh = torch.linalg.svd(curve_, full_matrices=False)
        basis = vh[:DIM].T.contiguous()  # [4096, 192]
        table = curve_ @ basis
        lost = float((curve_ - table @ basis.T).norm() / curve_.norm())
        # between grid points the form lerps: the worst modulation error there, against the full form
        midway = (grid[:-1] + grid[1:]) / 2
        exact, lerped = _vec(src, midway), (table[:-1] + table[1:]) / 2 @ basis.T
    with ctx.output("bf16", Derivation(base=None, sources=(src,), order=rows)) as out:
        document: dict[str, object] = {name: dict(component.config) for name, component in src.components.items()}
        document["transformer"] = {**section, CURVE[0]: GRID, CURVE[1]: DIM}
        for name, config in canonical_json.decode(canonical_json.encode(document)).items():
            out.config(name, config)
        out.inherit(src)
        worst = 0.0
        with tel.stage("copying"), torch.inference_mode():
            for done, (component, key) in enumerate(rows, 1):
                if out.has(component, key) or key == "adaln_t_table" or (component == "transformer" and key in modulation):
                    continue
                out.copy(component, key, src)
                ctx.progress(done - 1, len(rows))
        with tel.stage("curve"), torch.inference_mode():
            _put(out, "adaln_t_table", table)
            for key in modulation:
                ctx.raise_if_cancelled()
                weight = _read(src, key, device)
                full = exact @ weight.T
                worst = max(worst, float((full - lerped @ weight.T).norm() / full.norm()))
                _put(out, key, weight @ basis)
        artifact = ctx.adopt_model(out)
    tel.log("pan2 curve", grid=GRID, dim=DIM, rank_loss=round(lost, 9), worst_modulation_error=round(worst, 9),
            removed=len(removed))  # fmt: skip
    return artifact


app.entrypoint(curve, name="curve", accelerator=True, weights=(WeightsOutput("bf16", max_new_bytes=2 << 30),))
