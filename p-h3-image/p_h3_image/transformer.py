"""P-H3-Image's DiT: Diffusers' `MiniMaxH3Transformer3DModel`, the class Pruna's checkpoint names, with its
modulation computed from the time embedder and AdaLN projections (`bf16-full`, as uploaded) or read from exact
tables at the 8-step schedule (`bf16` and the lanes over it), and its packing scoped to the forward.
"""

from __future__ import annotations

import inspect
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, NoReturn, cast

import torch
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Transformer3DModel,
)
from torch import nn

#: Rows the packing and the output heads take at a time (0.35 GB of bf16 rows): a full-length temporary there
#: sits outside the Runtime's managed sites.
_CHUNK = 1 << 15
#: Packed-row modality tags.
MODALITIES = ("video", "text", "audio")


def _refuse(message: str) -> NoReturn:
    raise ValueError(f"model modulation tables {message}")


def _timestep(value: object) -> float:
    if not isinstance(value, str):
        _refuse("need exact float32 hexadecimal timestep labels")
    try:
        number = float.fromhex(value)
        rounded = struct.unpack("<f", struct.pack("<f", number))[0]
    except (ValueError, OverflowError, TypeError):
        _refuse("contain an invalid timestep label")
    if not math.isfinite(number) or number != rounded or number.hex() != value:
        _refuse("need finite, canonical float32 timestep labels")
    return number


def parse_tables(value: object) -> tuple[tuple[float, ...], tuple[tuple[int, int], ...]]:
    """The timesteps and `(timestep row, modality tag)` block rows a `bf16` checkpoint's `adaln_tables` label."""
    if not isinstance(value, Mapping):
        _refuse("must include their final-normalization and block-modulation row labels")
    final, blocks = value.get("final_normalization"), value.get("block_modulation")
    if not isinstance(final, list) or not final or not isinstance(blocks, list) or not blocks:
        _refuse("must contain nonempty final-normalization and block-modulation label lists")
    timesteps: list[float] = []
    rows: dict[float, int] = {}
    for index, row in enumerate(final):
        if not isinstance(row, Mapping) or row.get("index") != index:
            _refuse("need final-normalization rows in index order starting at zero")
        timestep = _timestep(row.get("timestep"))
        if timestep in rows:
            _refuse("contain duplicate final-normalization timestep labels")
        rows[timestep] = index
        timesteps.append(timestep)
    keys: list[tuple[int, int]] = []
    for index, row in enumerate(blocks):
        if not isinstance(row, Mapping) or row.get("index") != index:
            _refuse("need block-modulation rows in index order starting at zero")
        tag, timestep = row.get("modality_tag"), _timestep(row.get("timestep"))
        if type(tag) is not int or tag not in range(len(MODALITIES)) or row.get("modality") != MODALITIES[tag]:
            _refuse("must label modalities as video=0, text=1, or audio=2")
        if timestep not in rows:
            _refuse("reference a timestep absent from their final-normalization rows")
        key = (rows[timestep], tag)
        if key in keys:
            _refuse("contain a duplicate timestep/modality pair")
        keys.append(key)
    return tuple(timesteps), tuple(keys)


class PH3ImageTransformer(MiniMaxH3Transformer3DModel):
    """The DiT with its modulation computed at any timestep (`bf16-full`). Its forward packs the rows a chunk at
    a time and takes the rotary tables and refined text rows its denoise computed once."""

    def __init__(self, **config: Any) -> None:  # type: ignore[explicit-any]  # **config into Diffusers' typed keywords
        super().__init__(**config)

    def forward(  # the denoise's own call: rotary and text rows computed once
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
        text: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diffusers 0.41's forward, the image and audio velocities, with the projected rows scoped to the
        packing (Diffusers keeps them to its end)."""
        sequence_length = position_ids.shape[0]
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, "
                f"got {list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )
        rotary_emb = self.rope(position_ids) if rotary is None else rotary
        text = self.text_rows(encoder_hidden_states) if text is None else text
        rows = text.new_zeros((text.shape[0], sequence_length, text.shape[-1]))
        rows.index_copy_(1, text_indices, text)
        del text
        dtype = get_parameter_dtype(self.proj_in)
        for start in range(0, int(video_indices.shape[0]), _CHUNK):
            part = slice(start, start + _CHUNK)
            video = self.proj_in(hidden_states[:, part].to(dtype)).to(rows.dtype)
            rows.index_copy_(1, video_indices[part], video)
            del video
        audio = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        rows.index_copy_(1, audio_indices, audio.to(rows.dtype))
        del audio
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        for block in self.transformer_blocks:
            rows = block(rows, temb, adaln_indices, rotary_emb)
        dtype = get_parameter_dtype(self.proj_out)

        def head(proj: nn.Module, indices: torch.Tensor) -> torch.Tensor:
            parts = []
            for part in indices.split(_CHUNK):
                normed = self.norm_out(rows.index_select(1, part), temb, timestep_indices[part])
                parts.append(proj(normed.to(dtype)))
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        return head(self.proj_out, video_indices), head(self.audio_proj_out, audio_indices)

    def text_rows(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """The refined text rows a forward packs."""
        dtype = get_parameter_dtype(self.context_embedder)
        return cast(torch.Tensor, self.token_refiner(self.context_embedder(encoder_hidden_states.to(dtype))))

    def require_rows(self, timestep: torch.Tensor, timestep_indices: torch.Tensor, token_tags: torch.Tensor) -> None:
        """Refuse a step whose rows the modulation cannot serve, before the denoise's first forward."""
        if bool(torch.any((token_tags < 0) | (token_tags > 2))):
            raise ValueError("packed rows contain a modality tag outside the canonical 0/1/2 set")


class _TimestepLookup(nn.Module):
    """Exact float32 timestep values to the checkpoint's labelled rows."""

    timesteps: torch.Tensor

    def __init__(self, timesteps: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("timesteps", torch.tensor(tuple(timesteps), dtype=torch.float32), persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """The rows, on the device: the denoise checked them (`rows`) before its loop."""
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        return (values == self.timesteps.reshape(1, -1)).to(dtype=torch.int64).argmax(dim=1)

    def rows(self, timestep: torch.Tensor) -> torch.Tensor:
        matches = timestep.to(dtype=torch.float32).reshape(-1, 1) == self.timesteps.reshape(1, -1)
        if not bool(torch.all(matches.sum(dim=1) == 1)):
            raise ValueError(
                f"model modulation tables do not cover the requested timesteps: "
                f"{timestep.detach().float().cpu().tolist()}"
            )
        return matches.to(dtype=torch.int64).argmax(dim=1)


class _TimestepRows(nn.Module):
    """The removed timestep MLP's place: the rows pass through."""

    def __init__(self) -> None:
        super().__init__()
        # The forward calls `get_parameter_dtype(time_embedder)`, which raises for a module with no tensor; a
        # floating anchor would cast the integer row ids. A zero-length int64 buffer keeps the upstream call.
        self.register_buffer("row_dtype", torch.empty(0, dtype=torch.int64), persistent=False)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return rows


class _BlockTable(nn.Module):
    """A block's exact modulation rows, in the dense local shape Diffusers' blocks consume."""

    table_index: torch.Tensor

    def __init__(self, *, hidden_size: int, timestep_count: int, keys: Sequence[tuple[int, int]]) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.table = nn.Parameter(torch.empty(len(keys), 6, hidden_size))
        index = torch.full((timestep_count, len(MODALITIES)), -1, dtype=torch.int64)
        for table_row, (timestep_row, modality_tag) in enumerate(keys):
            index[timestep_row, modality_tag] = table_row
        self.register_buffer("table_index", index, persistent=False)

    def forward(self, timestep_rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # `require_rows` proved every selected pair present; row zero only keeps Diffusers' dense indexing.
        present = self.table_index.index_select(0, timestep_rows).reshape(-1).clamp_min(0)
        return self.table.index_select(0, present).reshape(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class _OutputTable(nn.Module):
    """The final RMSNorm with exact shift/scale rows."""

    def __init__(self, norm: nn.Module, *, hidden_size: int, timestep_count: int) -> None:
        super().__init__()
        self.norm = norm
        self.table = nn.Parameter(torch.empty(timestep_count, 2, hidden_size))

    def forward(
        self, hidden_states: torch.Tensor, timestep_rows: torch.Tensor, timestep_indices: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.table.index_select(0, timestep_rows).unbind(dim=1)
        normed = self.norm(hidden_states)
        return cast(torch.Tensor, normed * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices))


class TabledPH3ImageTransformer(PH3ImageTransformer):
    """The DiT with the checkpoint's exact AdaLN tables in place of the time embedder and projections."""

    time_proj: _TimestepLookup  # type: ignore[assignment]
    time_embedder: _TimestepRows  # type: ignore[assignment]
    norm_out: _OutputTable  # type: ignore[assignment]
    _keep_in_fp32_modules: ClassVar[list[str]] = [  # type: ignore[assignment, misc]  # Diffusers declares it per instance; ModelMixin types it None
        *MiniMaxH3Transformer3DModel._keep_in_fp32_modules,
        "time_proj",
    ]

    def __init__(  # type: ignore[explicit-any]  # **config into Diffusers' typed keywords
        self, *, table_timesteps: Sequence[float], table_block_keys: Sequence[tuple[int, int]], **config: Any
    ) -> None:
        super().__init__(**config)
        hidden = int(self.config["hidden_size"])
        self.time_proj = _TimestepLookup(table_timesteps)
        self.time_embedder = _TimestepRows()
        for block in self.transformer_blocks:
            block.adaln_proj = _BlockTable(hidden_size=hidden, timestep_count=len(table_timesteps), keys=table_block_keys)
        self.norm_out = _OutputTable(self.norm_out.norm, hidden_size=hidden, timestep_count=len(table_timesteps))

    def require_rows(self, timestep: torch.Tensor, timestep_indices: torch.Tensor, token_tags: torch.Tensor) -> None:
        super().require_rows(timestep, timestep_indices, token_tags)
        table_index = cast(_BlockTable, self.transformer_blocks[0].adaln_proj).table_index
        rows = self.time_proj.rows(timestep)
        if not bool(torch.all(table_index[rows.index_select(0, timestep_indices), token_tags] >= 0)):
            raise ValueError("model modulation tables do not cover a requested timestep/modality")


def build_transformer(
    config: Mapping[str, object], tables: tuple[Sequence[float], Sequence[tuple[int, int]]] | None = None
) -> PH3ImageTransformer:
    """The DiT of a checkpoint's upstream config: computed modulation, or the tables `tables` labels."""
    parameters = inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters
    upstream = {name: value for name, value in config.items() if name in parameters}
    if tables is None:
        return PH3ImageTransformer(**upstream)
    timesteps, block_keys = tables
    return TabledPH3ImageTransformer(table_timesteps=timesteps, table_block_keys=block_keys, **upstream)


def compute_dtypes[M: MiniMaxH3Transformer3DModel](transformer: M) -> M:
    """Diffusers' mixed compute policy: each root in its final dtype directly (RoPE frequencies are
    config-derived buffers: bf16 then f32 cannot restore them)."""
    for name, component in transformer.named_children():
        component.to(dtype=torch.float32 if name in transformer._keep_in_fp32_modules else torch.bfloat16)
    return transformer.eval()
