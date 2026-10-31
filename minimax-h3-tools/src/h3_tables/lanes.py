"""The H3 lane catalogue: what each lane does to each of the five components.

A lane is a NAME plus, per component, what this producer does to it. A component the lane
does not name is copied unchanged (its large parts keep their objects), so every lane that
leaves the conditioner and the two VAEs alone shares their stored objects. The catalogue is
CODE, not a request field: output slots are decorator-time facts, and every lane mints a
MiniMax H3 Model Derivative under §I.11(i) of the community licence, a reviewed act.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import msgspec
import numpy as np
from cozy_runtime.author import Context, Telemetry, UnsupportedInput
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.quantization import (
    BF16_ROUND_TRIP_BOUND,
    ArtifactQuantizationPlan,
    QuantizationTensor,
    prepare_source_quantization,
)
from cozy_runtime.weights_store import Checkpoint, Output

from .model_config import DIT_COMPONENT
from .source import Row, dtype, plain

CastDtype = Literal["bf16", "f16"]
#: Which of a component's float32 rows a cast covers: `component` is every plain one,
#: `vae_operands` the video VAE's serving destination only (`NORMALISED_COMPONENTS`).
CastScope = Literal["component", "vae_operands"]
Encoding = Literal["fp8.rowwise", "int8.convrot", "nvfp4", "w4a8.convrot"]
Modulation = Literal["full", "adaln-pruned"]

#: The DiTs encode through the package-owned reviewed plan (`quantization.dit_plan`), never
#: the structural selector: their AdaLN table rows are rank-2 bf16 `.weight`-less tensors a
#: shape rule would still be tempted by, and the plan is the reviewed set.
DIT_COMPONENTS = frozenset(DIT_COMPONENT.values())

#: What each encoded lane of the vocabulary (`bf16 fp16 int8 fp8 nvfp4 w4a8`) stores.
LANE_ENCODINGS: Mapping[str, Encoding] = {
    "int8": "int8.convrot",
    "fp8": "fp8.rowwise",
    "nvfp4": "nvfp4",
    "w4a8": "w4a8.convrot",
}

#: Components no lane may treat. Refused BY NAME: the structural selector would take the
#: audio VAE's six rank-2 `pre_block` ENCODER linears, encode the conditioning path and
#: leave the BigVGAN decoder (rank-3 weight_norm pairs) alone. Upstream pins the component
#: to float32 (`_keep_in_fp32_modules`; bf16 decodes about 20 dB quieter) and decodes it in
#: plain float32 with no autocast, so it must not move at any width.
REFUSED_COMPONENTS: Mapping[str, str] = {
    "audio_vae": (
        "the audio VAE is not representable: 637 of its 1,087 rows are rank-3, including the "
        "BigVGAN decoder's 344 weight_norm weight_g/weight_v parameters, and the only six "
        "rank-2 float weights it has are the pre_block ENCODER attention/MLP linears, so "
        "treating it would encode the audio conditioning path and leave the decoder alone"
    ),
}

#: Author-declared ceilings on the new bytes ONE treatment of a component may write: they size
#: the declared output slots, a decorator-time fact. Each is above the component's plain size.
MAX_DIT_BYTES = 32 << 30
COMPONENT_MAX_NEW_BYTES: Mapping[str, int] = {
    "transformer": MAX_DIT_BYTES,
    "transformer_ref": MAX_DIT_BYTES,
    "text_encoder": 56 << 30,
    "vae": 12 << 30,
}


@dataclass(frozen=True, slots=True)
class Treatment:
    """What one lane does to one component.

    ``cast`` rewrites every plain float32 tensor (in ``cast_scope``) at the named dtype; a
    16-bit tensor is copied untouched (#695). ``encode`` replaces the component's selected
    weights with the named scheme; ``keep`` names exact keys the encoding must NOT select, and
    ``encode_as`` exact selected keys that take another scheme instead.
    A treatment that changes nothing is spelled by leaving the component out of the lane.
    """

    cast: CastDtype | None = None
    encode: Encoding | None = None
    keep: tuple[str, ...] = ()
    cast_scope: CastScope = "component"
    encode_as: tuple[tuple[str, Encoding], ...] = ()

    def __post_init__(self) -> None:
        if self.cast is None and self.encode is None:
            raise ValueError("a treatment casts, encodes, or both; absence means inherit")
        if self.cast_scope != "component" and self.cast is None:
            raise ValueError("a cast scope without a cast names nothing")
        if self.keep and self.encode is None:
            raise ValueError("a keep list only excludes keys from an encoding")
        if len(set(self.keep)) != len(self.keep):
            raise ValueError("a treatment keep list repeats a key")
        other = [key for key, _ in self.encode_as]
        if other and self.encode is None:
            raise ValueError("encode_as only redirects keys of an encoding")
        if len(set(other)) != len(other) or set(other) & set(self.keep):
            raise ValueError("encode_as repeats a key or names a kept one")
        if any(scheme not in LANE_ENCODINGS.values() for _, scheme in self.encode_as):
            raise ValueError(f"encode_as names a scheme outside {sorted(LANE_ENCODINGS.values())}")

    def describe(self) -> str:
        spelled = []
        if self.cast is not None:
            spelled.append(f"cast={self.cast}")
            if self.cast_scope != "component":
                spelled.append(f"scope={self.cast_scope}")
        if self.encode is not None:
            spelled.append(f"encode={self.encode}")
            if self.keep:
                spelled.append(f"keep={len(self.keep)}")
            for scheme in sorted({scheme for _, scheme in self.encode_as}):
                spelled.append(f"{scheme}={sum(s == scheme for _, s in self.encode_as)}")
        return " ".join(spelled)


@dataclass(frozen=True, slots=True)
class Lane:
    """One published checkpoint: a modulation plus the components this lane rewrites."""

    modulation: Modulation
    components: Mapping[str, Treatment] = field(default_factory=dict)


_DIT_FP8 = Treatment(encode="fp8.rowwise")


def _both_dits(treatment: Treatment) -> dict[str, Treatment]:
    return {component: treatment for component in sorted(DIT_COMPONENTS)}


#: The conditioner rows h3a-028 holds at bf16. The structural selector takes 441 rows of the
#: 902-row conditioner; the wanted ones are the 350 `layers.{0..49}` projections. The other 91
#: are the token embedding, the 27-block visual tower and its mergers. `blocks.N.mlp.linear_fc2
#: [1152, 4304]` is absent on purpose: 4304 % 32 == 16, so the selector never takes it.
TEXT_ENCODER_KEEP: tuple[str, ...] = (
    "model.language_model.embed_tokens.weight",
    "model.visual.pos_embed.weight",
    "model.visual.merger.linear_fc1.weight",
    "model.visual.merger.linear_fc2.weight",
    *(
        f"model.visual.deepstack_merger_list.{index}.linear_fc{leaf}.weight"
        for index in range(3)
        for leaf in (1, 2)
    ),
    *(
        f"model.visual.blocks.{index}.{leaf}.weight"
        for index in range(27)
        for leaf in ("attn.qkv", "attn.proj", "mlp.linear_fc1")
    ),
)

_TEXT_ENCODER_FP8 = Treatment(encode="fp8.rowwise", keep=TEXT_ENCODER_KEEP)

#: nvfp4's DiT keep list: the text-only `context_embedder` and token refiner stay bf16 (they run
#: on prompt rows, about 1 % of the DiT's FLOPs). Measured on H3 (cr-207, RTX PRO 4500): the
#: first and last blocks' W4A4 error sits in the middle blocks' range, so they are W4A4.
NVFP4_DIT_KEEP: tuple[str, ...] = (
    "context_embedder.weight",
    *(
        f"token_refiner.refiner_blocks.{block}.{leaf}.weight"
        for block in range(2)
        for leaf in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
    ),
)
#: nvfp4's DiT linears that serve int8.convrot instead: `ff.net.2` of blocks 0-2, whose audio rows
#: sit far below the video rows' amax, so one per-call activation scale gives 56-100 % of their
#: 16-element groups a subnormal E4M3 scale (block 1: W4A4 audio error 0.14-0.16, int8 0.01).
NVFP4_DIT_INT8: tuple[str, ...] = tuple(f"transformer_blocks.{block}.ff.net.2.weight" for block in range(3))
#: The int8 lane's video VAE: the decoder ViT's 216 block linears serve int8.convrot (on consumer
#: cards int8 runs at 4x the f16-with-f32-accumulate rate the f16 decode gets). Its pixel
#: projection `proj_out` stays f16; `proj_in` (24 columns) is never selected.
VAE_INT8_KEEP: tuple[str, ...] = ("decoder.proj_out.weight",)

#: The video VAE's stored form is not a lane choice, and the fill refuses a stored dtype that differs
#: from the destination, so every lane stores these rows f16, `bf16-full` included
#: (`official.py:_apply_video_vae_dtype` builds them so):
#: - the decoder's conv/linear operands, which `MiniMaxH3VideoDecodeStep`'s `torch.autocast(float16)`
#:   rounds to f16 on every forward whatever they are stored as (bit-exact, 10.4 -> 5.6 GB);
#: - every row of the encoder and `quant_conv`, which compute in plain float16 (dtype audit M1: the
#:   largest activation is 181, 362x inside float16's range; bf16 loses 9.2% on near-black: never bf16).
#: The decoder's norms, biases and `decoder.rope.inv_freq` stay f32.
DECODE_OPERAND_KEYS = frozenset({"post_quant_conv.weight"})
DECODE_OPERAND_PREFIX = "decoder."
ENCODE_PREFIXES = ("encoder.", "quant_conv.")


def vae_operand(key: str, shape: Sequence[int]) -> bool:
    """Is this video-VAE row stored f16: any encoder row, or a decode operand?"""
    return key.startswith(ENCODE_PREFIXES) or (
        len(shape) >= 2
        and key.rsplit(".", 1)[-1] == "weight"
        and (key.startswith(DECODE_OPERAND_PREFIX) or key in DECODE_OPERAND_KEYS)
    )


CAST_SCOPES: Mapping[str, Callable[[str, Sequence[int]], bool]] = {
    "component": lambda key, shape: True,
    "vae_operands": vae_operand,
}

#: Applied to EVERY lane, ahead of whatever that lane authors; one component, one treatment.
NORMALISED_COMPONENTS: Mapping[str, Treatment] = {
    "vae": Treatment(cast="f16", cast_scope="vae_operands"),
}


def lane_treatments(lane: Lane) -> Mapping[str, Treatment]:
    """Everything this lane does to each component: the normalisations, then its own."""
    return {**NORMALISED_COMPONENTS, **lane.components}


#: The reviewed catalogue. The quantized lanes encode both DiTs and the text conditioner's 350
#: decoder linears (`TEXT_ENCODER_KEEP` stays bf16): at fp8 the conditioner drifted 0.1 % in mean
#: cosine and kept 15 s A/B renders equivalent (progress/h3-te-fp8-20261009).
LANES: Mapping[str, Lane] = {
    "bf16-full": Lane("full"),
    "bf16": Lane("adaln-pruned"),
    "fp8": Lane(
        "adaln-pruned",
        {**_both_dits(_DIT_FP8), "text_encoder": _TEXT_ENCODER_FP8},
    ),
}


def validate_catalogue(lanes: Mapping[str, Lane] = LANES) -> None:
    """Refuse an unservable, unrepresentable or unbudgeted lane before anything imports."""
    if not lanes:
        raise ValueError("the H3 lane catalogue is empty")
    for name, lane in lanes.items():
        if lane.modulation == "full" and lane.components:
            raise ValueError(f"lane {name!r} is a FULL lane and AUTHORS no treatment")
        double = sorted(set(lane.components) & set(NORMALISED_COMPONENTS))
        if double:
            raise ValueError(
                f"lane {name!r} authors {double}, which the producer already normalises; "
                "one component carries one treatment"
            )
        for component, treatment in lane_treatments(lane).items():
            if component in REFUSED_COMPONENTS:
                raise ValueError(
                    f"lane {name!r} treats {component}: {REFUSED_COMPONENTS[component]}"
                )
            if component not in COMPONENT_MAX_NEW_BYTES:
                raise ValueError(f"lane {name!r} treats unknown component {component!r}")
            if treatment.encode is not None and treatment.encode not in LANE_ENCODINGS.values():
                raise ValueError(
                    f"lane {name!r} encodes {component} as {treatment.encode}, which no lane of "
                    f"the vocabulary stores: {dict(LANE_ENCODINGS)}"
                )
            if component in DIT_COMPONENTS and treatment.cast is not None:
                raise ValueError(
                    f"lane {name!r} casts {component}: the DiTs are bf16 at source and their "
                    "AdaLN rows are computed, never converted"
                )


validate_catalogue()


def lane_max_new_bytes(lane: Lane, *, full_bytes: int, pruned_bytes: int) -> int:
    """The declared new-byte ceiling of one lane's output slot."""
    base = full_bytes if lane.modulation == "full" else pruned_bytes
    return base + sum(COMPONENT_MAX_NEW_BYTES[component] for component in lane_treatments(lane))


@dataclass(frozen=True, slots=True)
class Selection:
    """One component's resolved work: which keys cast, which encode."""

    component: str
    treatment: Treatment
    cast: tuple[tuple[str, tuple[int, ...]], ...]
    plan: ArtifactQuantizationPlan | None

    @property
    def encoded(self) -> tuple[str, ...]:
        return () if self.plan is None else tuple(tensor.key for tensor in self.plan.tensors)

    @property
    def replaced(self) -> frozenset[str]:
        return frozenset(key for key, _ in self.cast) | frozenset(self.encoded)


def select(
    component: str,
    treatment: Treatment,
    source: Checkpoint,
    carried: Mapping[str, Row],
    *,
    dit_plan: Sequence[tuple[str, tuple[int, int]]] = (),
    allow_inert: bool = False,
) -> Selection:
    """Resolve one treatment against the rows a lane carries of `component`.

    `carried` excludes the rows the lane drops, so an encoding never resurrects one (the
    conditioner's `lm_head.weight` is rank-2 and block-aligned). ``allow_inert`` is for the
    normalisations only: a source whose video VAE is already at the serving destination has
    nothing left to do. A ``keep`` entry that excludes nothing REFUSES: it is a typo that
    would otherwise silently quantize, say, an embedding table.
    """
    if component in REFUSED_COMPONENTS:
        raise UnsupportedInput(
            f"{component} is refused: {REFUSED_COMPONENTS[component]}",
            code="h3_component_refused",
        )
    if not carried:
        raise UnsupportedInput(f"source has no component {component!r}", code="h3_component_absent")
    plan: ArtifactQuantizationPlan | None = None
    if treatment.encode is not None:
        if component in DIT_COMPONENTS:
            if not dit_plan:
                raise UnsupportedInput(
                    f"{component} encodes through the reviewed DiT plan, which was not supplied",
                    code="h3_plan_absent",
                )
            for key, shape in dit_plan:
                row = carried.get(key)
                if row is None or not plain(row, "bf16", shape):
                    raise UnsupportedInput(
                        f"{component}.{key} is not the reviewed plain bf16 {list(shape)} weight",
                        code="h3_plan_source",
                    )
            candidates = [QuantizationTensor(component, key, "bf16", shape) for key, shape in dit_plan]
            unmatched = "the reviewed plan never selects"
        else:
            structural = prepare_source_quantization(source, components=(component,))
            candidates = [tensor for tensor in structural.tensors if tensor.key in carried]
            unmatched = "the structural selection never picked; a keep entry that excludes nothing is a typo"
        kept = set(treatment.keep)
        missing = sorted(kept - {tensor.key for tensor in candidates})
        if missing:
            raise UnsupportedInput(
                f"{component} keep list names {missing}, which {unmatched}",
                code="h3_keep_unmatched",
            )
        selected = [tensor for tensor in candidates if tensor.key not in kept]
        if not selected:
            raise UnsupportedInput(
                f"{component} keep list excludes every selected weight", code="h3_keep_exhaustive"
            )
        scheme_of = dict(treatment.encode_as)
        missing = sorted(set(scheme_of) - {tensor.key for tensor in selected})
        if missing:
            raise UnsupportedInput(
                f"{component} encode_as names {missing}, which {unmatched}",
                code="h3_keep_unmatched",
            )
        selected = [
            msgspec.structs.replace(tensor, encoding=scheme_of[tensor.key])
            if tensor.key in scheme_of
            else tensor
            for tensor in selected
        ]
        plan = ArtifactQuantizationPlan(
            components=[component], order=[(component, key) for key in carried], tensors=selected
        )
    encoded = {tensor.key for tensor in plan.tensors} if plan is not None else set()
    cast: list[tuple[str, tuple[int, ...]]] = []
    if treatment.cast is not None:
        for key, row in carried.items():
            # Only plain float32 converts; a 16-bit tensor already has the target's width.
            if key in encoded or dtype(row) != "f32" or row.encoding is not None:
                continue
            if CAST_SCOPES[treatment.cast_scope](key, row.shape):
                cast.append((key, tuple(row.shape)))
    if not cast and plan is None and not allow_inert:
        raise UnsupportedInput(
            f"{component} treatment {treatment.describe()!r} changes nothing on this source",
            code="h3_treatment_inert",
        )
    return Selection(component, treatment, tuple(cast), plan)


@dataclass(frozen=True, slots=True)
class CastStats:
    """What one cast recorded: the tier-1 round-trip shape the encoders record."""

    converted_keys: int
    reused_keys: int
    source_bytes_read: int
    new_bytes_written: int
    worst_relative_frobenius: float | None


#: Half-ulp of each carrier's significand as a multiple of the tensor's amax. Overflow is not a
#: rounding error: an fp16 cast of a large outlier produces inf, and that refuses.
CAST_ROUND_TRIP_BOUND: Mapping[str, float] = {"bf16": BF16_ROUND_TRIP_BOUND, "f16": 2.0**-11}
MAX_CAST_TENSOR_BYTES = 1 << 30


def write_cast(
    source: Checkpoint, out: Output, ctx: Context, tel: Telemetry, *, selection: Selection
) -> CastStats:
    """Stream one component's float32 cut through RNE into the declared carrier, each tensor's
    round trip held to the carrier's bound and its relative Frobenius recorded."""
    target = selection.treatment.cast
    if target is None or not selection.cast:
        return CastStats(0, 0, 0, 0, None)
    component, carrier, bound = selection.component, target.upper(), CAST_ROUND_TRIP_BOUND[target]
    read = written = reused = 0
    worst = 0.0
    total = len(selection.cast)
    for done, (key, shape) in enumerate(selection.cast, 1):
        ctx.raise_if_cancelled()
        stage = f"cast-{target}-{component}"
        if out.has(component, key):
            reused += 1
            tel.progress(done / total, stage=stage)
            continue
        values, length = _read_f32(source, component, key, shape)
        with np.errstate(over="ignore"):
            raw = st.from_f32(values, carrier)
        restored = st.to_f32(raw, carrier, list(shape))
        if not bool(np.isfinite(restored).all()):
            raise UnsupportedInput(
                f"{component}.{key}: the {target} cast overflowed to a non-finite value "
                f"(source amax {float(np.max(np.abs(values))):.6g})",
                code="quantization_tripwire",
            )
        error = np.abs(values - restored)
        allowed = float(np.max(np.abs(values))) * bound
        if float(np.max(error)) > allowed:
            raise UnsupportedInput(
                f"{component}.{key}: {target} round-trip error {float(np.max(error)):.4g} "
                f"exceeds the representational bound {allowed:.4g}",
                code="quantization_tripwire",
            )
        norm = float(np.linalg.norm(values))
        relative = float(np.linalg.norm(error)) / norm if norm else 0.0
        worst = max(worst, relative)
        out.put(component, key, carrier, shape, raw)
        read += length
        written += len(raw)
        tel.log("cast tensor", level="info", dtype=target, component=component, key=key,
                relative_frobenius=relative, done=done, total=total)  # fmt: skip
        tel.progress(done / total, stage=stage)
    return CastStats(total - reused, reused, read, written, None if reused else worst)


def _read_f32(
    source: Checkpoint, component: str, key: str, shape: tuple[int, ...]
) -> tuple[np.ndarray, int]:
    length = math.prod(shape) * 4
    if length > MAX_CAST_TENSOR_BYTES:
        raise UnsupportedInput(
            f"source {component}.{key} is {length} bytes; the per-tensor cast bound is "
            f"{MAX_CAST_TENSOR_BYTES}",
            code="h3_cast_tensor_bound",
        )
    raw = bytearray(length)
    source.components[component].tensors.read_into(key, raw)
    values = st.to_f32(raw, "F32", list(shape))
    if not bool(np.isfinite(values).all()):
        raise UnsupportedInput(
            f"source {component}.{key} contains non-finite weights",
            code="h3_cast_source_nonfinite",
        )
    return values, length
