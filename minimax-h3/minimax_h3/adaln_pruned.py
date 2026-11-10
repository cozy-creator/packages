"""The official Diffusers MiniMax-H3 DiT as H3 serves it: FULL, or with exact AdaLN tables.

Attention, feed-forward, RoPE, input/output projections and packed-row indexing stay upstream
code. Its modulation comes from one of two sources. FULL (`H3Transformer`) computes it from the
checkpoint's time embedder and AdaLN projections at any timestep. AdaLN-pruned
(`AdaLNPrunedMiniMaxH3Transformer`) reads rows a producer computed from those same weights, in the
same operation order, at the timesteps the checkpoint labels, and holds no projection.

An overlay (`overlay.py`: PDD-8 turbo, the Regenerate refine) attaches beside the DiT and is
served per forward: the forward that names its bank in `attention_kwargs`, or runs inside a
denoise loop that engaged it, reads the overlay's tables, heads and low-rank factors through the
same modules; every other forward reads the base modulation, FULL or tabled.
"""

from __future__ import annotations

import inspect
from functools import partial
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, cast

import torch
from diffusers.models._modeling_parallel import ContextParallelInput
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
)
from diffusers.utils.peft_utils import apply_lora_scale
from torch import nn
from torch.nn import functional as F

from cozy_runtime.accel import lora

from .overlay import (
    ATTENTION_KWARG,
    LORA_FAMILIES,
    OVERLAY_KWARG,
    Overlay,
    OverlayArming,
    OverlayTables,
    armed,
)


#: Rows the packing and the output heads take at a time (0.35 GB of bf16 rows). A full-length
#: temporary there sits outside the Runtime's managed sites, which alone reclaim cached
#: weights: the packing's third copy refused the 2160p 15 s refine on 24 and 48 GB (runs 20463
#: and 20453), the output heads' norm took several more.
_CHUNK = 1 << 15


_CHECKED: ContextVar[bool] = ContextVar("minimax_h3_checked_loop", default=False)


@contextmanager
def checked_loop() -> Iterator[None]:
    """A denoise that checked every step's rows before its loop: its forwards skip the check and
    keep the overlay it armed before the loop."""
    token = _CHECKED.set(True)
    try:
        yield
    finally:
        _CHECKED.reset(token)


class _Armed[S]:
    """One module's turbo source for the forward in flight; a plain object, so placing it
    on a module registers nothing and the DiT's state dict is what it was."""

    __slots__ = ("source", "step")

    def __init__(self, source: S, step: int = 0) -> None:
        self.source = source
        self.step = step


class _AdaLNPrunedTimestepLookup(nn.Module):
    """Map exact float32 timestep values to the checkpoint's labelled rows."""

    timesteps: torch.Tensor

    def __init__(self, timesteps: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer(
            "timesteps", torch.tensor(tuple(timesteps), dtype=torch.float32), persistent=False
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """The rows, on the device: the transformer checked them (`rows`) before its forward."""
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        return (values == self.timesteps.reshape(1, -1)).to(dtype=torch.int64).argmax(dim=1)

    def rows(self, timestep: torch.Tensor) -> torch.Tensor:
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        matches = values == self.timesteps.reshape(1, -1)
        counts = matches.sum(dim=1)
        if not bool(torch.all(counts == 1)):
            unknown = timestep.detach().float().cpu().tolist()
            raise ValueError(
                f"model modulation tables do not cover the requested timesteps: {unknown}; "
                "select a covered step count or regenerate the tables for this schedule",
            )
        return matches.to(dtype=torch.int64).argmax(dim=1)


class _AdaLNPrunedTimestepRows(nn.Module):
    """The removed timestep MLP's place: the rows pass through."""

    def __init__(self) -> None:
        super().__init__()
        # The forward calls `get_parameter_dtype(time_embedder)` before this identity. Its
        # helper raises `UnboundLocalError` for a module with no tensor, and a floating anchor
        # would cast these exact integer row ids. This zero-length int64 buffer is therefore
        # the smallest state that preserves the upstream call.
        self.register_buffer("row_dtype", torch.empty(0, dtype=torch.int64), persistent=False)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return rows


class _AdaLNPrunedBlockTable(nn.Module):
    """Sparse exact table with the dense local shape Diffusers blocks already consume."""

    table_index: torch.Tensor

    def __init__(
        self,
        *,
        hidden_size: int,
        timestep_count: int,
        keys: Sequence[tuple[int, int]],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.table = nn.Parameter(torch.empty(len(keys), 6, hidden_size))
        index = torch.full((timestep_count, 3), -1, dtype=torch.int64)
        for table_row, (timestep_row, modality_tag) in enumerate(keys):
            index[timestep_row, modality_tag] = table_row
        self.register_buffer("table_index", index, persistent=False)
        self._armed: _Armed[OverlayTables] | None = None

    def forward(self, timestep_rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self._armed is not None:
            return _overlay_block(self._armed, timestep_rows, self.hidden_size)
        rows = self.table.index_select(0, _present(self.table_index, timestep_rows))
        return _block_chunks(rows, self.hidden_size)


def _present(table_index: torch.Tensor, timestep_rows: torch.Tensor) -> torch.Tensor:
    """The table rows of these timestep rows' (row, modality) pairs. The transformer's `_arm`
    proves every selected pair is present. Missing cells are never observed by Diffusers'
    `adaln_indices`; row zero is only the placeholder that keeps its dense local indexing."""
    return table_index.index_select(0, timestep_rows).reshape(-1).clamp_min(0)


def _block_chunks(rows: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, ...]:
    """A block's six modulation chunks in the dense local shape Diffusers' blocks consume."""
    return rows.reshape(-1, 6 * hidden_size).chunk(6, dim=-1)


def _overlay_block(
    armed: _Armed[OverlayTables], timestep_rows: torch.Tensor, hidden_size: int
) -> tuple[torch.Tensor, ...]:
    """An armed overlay's block table, read through its own forward: a piece of its overlay."""
    source = armed.source
    return _block_chunks(source(_present(source.table_index, timestep_rows)), hidden_size)


def _modulate(normed: torch.Tensor, rows: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """The final norm's shift/scale rows applied per packed row, as Diffusers' `norm_out`."""
    shift, scale = rows.unbind(dim=1)
    return normed * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)


class _BlockModulation(MiniMaxH3AdaLayerNormModulation):
    """FULL: the official AdaLN projection, or an armed overlay's table. Diffusers' module with
    this class (`H3Transformer.__init__`): its parameters and their paths are the checkpoint's."""

    _armed: _Armed[OverlayTables] | None = None

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self._armed is not None:
            return _overlay_block(self._armed, temb, self.hidden_size)
        return super().forward(temb)


class _OutputModulation(MiniMaxH3AdaLayerNormOut):
    """FULL: the official final norm and its projection, or an armed overlay's table (as
    `_BlockModulation`, Diffusers' module with this class)."""

    _armed: _Armed[OverlayTables] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self._armed is None:
            return super().forward(hidden_states, temb, timestep_indices)
        return _modulate(self.norm(hidden_states), self._armed.source(temb), timestep_indices)


class _AdaLNPrunedOutputTable(nn.Module):
    """Official final RMSNorm with exact precomputed shift/scale rows."""

    def __init__(self, norm: nn.Module, *, hidden_size: int, timestep_count: int) -> None:
        super().__init__()
        self.norm = norm
        self.table = nn.Parameter(torch.empty(timestep_count, 2, hidden_size))
        self._armed: _Armed[OverlayTables] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_rows: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self._armed is None:
            rows = self.table.index_select(0, timestep_rows)
        else:  # an armed turbo table is read through its own forward (a piece of its overlay)
            rows = self._armed.source(timestep_rows)
        return _modulate(self.norm(hidden_states), rows, timestep_indices)


class _SelectableHead(nn.Linear):
    """The official output head, or the turbo evaluation's collapsed head when armed."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)
        self._armed: _Armed[nn.Module] | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._armed is None:
            return F.linear(hidden_states, self.weight, self.bias)
        return cast(torch.Tensor, self._armed.source(hidden_states, self._armed.step))


def _projection_base(module: object) -> object:
    """Public PEFT wrappers keep the model-owned selectable head as their base layer."""
    get_base = getattr(module, "get_base_layer", None)
    return get_base() if callable(get_base) else module


class H3Transformer(MiniMaxH3Transformer3DModel):
    """The official MiniMax-H3 transformer with FULL modulation, as H3's loops serve it: the
    forward's packing scoped, a loop's rotary and text rows taken, an overlay armed per forward."""

    # FULL's armable modulation modules, or the pruned class's tables (each `__init__`).
    time_proj: nn.Module  # type: ignore[assignment]
    time_embedder: nn.Module  # type: ignore[assignment]
    norm_out: _OutputModulation | _AdaLNPrunedOutputTable  # type: ignore[assignment]
    #: Diffusers' plan, the rotary tables split where each block reads them, not where `rope`
    #: returns them. A denoise loop computes them once, on the GPU that runs it, and its calls
    #: hand every GPU those tables (`rotary=`): each GPU must take its own rows of the whole
    #: document's, never the rows of the GPU that ran the loop.
    _cp_plan: ClassVar[dict[str, Any]] = {  # type: ignore[explicit-any, misc]  # Diffusers' context-parallel plan, declared per instance
        **{k: plan for k, plan in MiniMaxH3Transformer3DModel._cp_plan.items() if k != "rope"},
        "transformer_blocks.*": {
            **MiniMaxH3Transformer3DModel._cp_plan["transformer_blocks.*"],
            "rotary_emb": [ContextParallelInput(split_dim=0, expected_dims=2)] * 2,
        },
    }

    def __init__(self, **config: Any) -> None:  # type: ignore[explicit-any]  # **config into Diffusers' typed keywords
        super().__init__(**config)
        # Armable in place: no module is built or dropped, so the construction (its random draws
        # included) and the state dict are Diffusers' own.
        for block in self.transformer_blocks:
            block.adaln_proj.__class__ = _BlockModulation
        self.norm_out.__class__ = _OutputModulation
        for name in ("proj_out", "audio_proj_out"):
            head = cast(nn.Linear, getattr(self, name))
            setattr(self, name, _SelectableHead(head.in_features, head.out_features))
        self._arming: OverlayArming | None = None
        self._lora_hooks_installed = False

    @apply_lora_scale("attention_kwargs")  # type: ignore[untyped-decorator]
    def forward(  # type: ignore[misc]  # untyped after Diffusers' unannotated `apply_lora_scale`
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
        attention_kwargs: dict[str, object] | None = None,
        return_dict: bool = True,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
        text: torch.Tensor | None = None,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        """Diffusers 0.41's forward, the projected rows scoped to the packing: Diffusers keeps
        them to its end, the video's float32 rows (2.25 GB a GPU at H3 15 s, degree 2) through
        every block. It checks the rows and arms a turbo bank `attention_kwargs` names (`_arm`)
        and ends disarmed, in the forward; unless its denoise checked every step's rows ahead
        (`checked_loop`): a loop's step, which keeps the overlay its denoise armed before the loop
        and passes the `rotary` tables (`rope`) and `text` rows (`text_rows`) it computed before
        it."""
        looped = _CHECKED.get()
        if not looped:
            self._arm(timestep, timestep_indices, token_tags, attention_kwargs)
        del attention_kwargs  # `apply_lora_scale` consumed its scale
        try:
            sequence_length = position_ids.shape[0]
            if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
                raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
            if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
                raise ValueError(
                    "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching "
                    f"`position_ids`, got {list(token_tags.shape)} and {list(timestep_indices.shape)} "
                    f"for seq_len={sequence_length}."
                )

            rotary_emb = self.rope(position_ids) if rotary is None else rotary
            text = self.text_rows(encoder_hidden_states) if text is None else text
            # The packed rows are this forward's one full-length tensor outside the blocks: the
            # video goes in, and the heads come out, `_CHUNK` rows at a time.
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
            if self._arming is None:
                temb = self.time_proj(timestep)
                temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
            else:  # the overlay's own table rows, which its tables read in every block
                temb = self._arming.overlay.rows(timestep)
            adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    rows = self._gradient_checkpointing_func(block, rows, temb, adaln_indices, rotary_emb)  # type: ignore[misc]  # set by enable_gradient_checkpointing
                else:
                    rows = block(rows, temb, adaln_indices, rotary_emb)
            dtype = get_parameter_dtype(self.proj_out)
            if rows.shape[1] != sequence_length:
                # Across GPUs `rows` is this GPU's share: the `_cp_plan` split it at the first
                # block, splits `timestep_indices` at `norm_out` and gathers at the heads, so
                # sequence-wide indices only apply after the heads, and they cannot be chunked.
                normed = self.norm_out(rows, temb, timestep_indices).to(dtype)
                video_output = self.proj_out(normed).index_select(1, video_indices)
                audio_output = self.audio_proj_out(normed).index_select(1, audio_indices)
                if not return_dict:
                    return (video_output, audio_output)
                return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

            def head(proj: nn.Module, indices: torch.Tensor) -> torch.Tensor:
                parts = []
                for part in indices.split(_CHUNK):
                    normed = self.norm_out(rows.index_select(1, part), temb, timestep_indices[part])
                    parts.append(proj(normed.to(dtype)))
                return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

            video_output = head(self.proj_out, video_indices)
            audio_output = head(self.audio_proj_out, audio_indices)
            if not return_dict:
                return (video_output, audio_output)
            return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)
        finally:
            if not looped:
                self._release()

    def text_rows(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """The refined text rows a forward packs, under the overlay armed now."""
        dtype = get_parameter_dtype(self.context_embedder)
        return cast(
            torch.Tensor, self.token_refiner(self.context_embedder(encoder_hidden_states.to(dtype)))
        )

    def install_lora_consumers(self) -> None:
        """Install once during turbo-capable construction, before Runtime fill/fusion."""
        if self._lora_hooks_installed:
            return
        for path, site in self.named_modules():
            if path.startswith(
                ("token_refiner.refiner_blocks.", "transformer_blocks.")
            ) and path.endswith(LORA_FAMILIES):
                site.register_forward_hook(lora.Hook(partial(armed, self, path)))
        self._lora_hooks_installed = True

    def _arm(
        self,
        timestep: object,
        timestep_indices: object,
        token_tags: object,
        selector: object,
    ) -> None:
        """Check one forward's rows against the tables it reads: the turbo overlay's, engaged
        here, when `selector` (its `attention_kwargs`) names the bank; else the base's, if it
        has tables."""
        self._release()
        bank = selector.get(ATTENTION_KWARG) if isinstance(selector, Mapping) else None
        if not all(
            isinstance(value, torch.Tensor) for value in (timestep, timestep_indices, token_tags)
        ):
            if bank is not None:
                raise ValueError(
                    "a turbo forward needs typed timestep, timestep_indices and token_tags",
                )
            return  # Diffusers owns its ordinary argument diagnostics.
        assert isinstance(timestep, torch.Tensor)
        assert isinstance(timestep_indices, torch.Tensor)
        assert isinstance(token_tags, torch.Tensor)
        if bool(torch.any((token_tags < 0) | (token_tags > 2))):
            raise ValueError(
                "packed rows contain a modality tag outside the canonical 0/1/2 set",
            )
        covered: tuple[torch.Tensor, torch.Tensor] | None
        try:
            if bank is not None:
                overlay = selector.get(OVERLAY_KWARG) if isinstance(selector, Mapping) else None
                if (
                    not isinstance(overlay, Overlay)
                    or overlay.bank != bank
                    or not self._lora_hooks_installed
                ):
                    raise ValueError(
                        f"bank {bank!r} requires an overlay-capable base and that bank's "
                        "prepared overlay component",
                    )
                step = overlay.schedule.step(timestep) if overlay.heads else 0  # type: ignore[union-attr]  # `heads`: it has a schedule
                self._engage(overlay, step)
                covered = overlay.rows(timestep), _overlay_tables(overlay, 0).table_index
            else:
                covered = self._tables(timestep)
            if covered is None:
                return
            global_rows, table_index = covered
            present = table_index[global_rows.index_select(0, timestep_indices), token_tags] >= 0
            if not bool(torch.all(present)):
                raise ValueError(
                    "model modulation tables do not cover a requested reference timestep/modality; "
                    "regenerate the tables with coverage for this reference type",
                )
        except Exception:
            self._release()
            raise

    def _tables(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The base tables' global rows of `timestep` and their (row, modality) index. FULL
        modulation has no tables: it computes every timestep's."""
        return None

    def _engage(self, overlay: Overlay, step: int) -> None:
        self._arming = OverlayArming(overlay, step)
        for index, block in enumerate(self.transformer_blocks):
            _modulation(block)._armed = _Armed(_overlay_tables(overlay, index))
        self.norm_out._armed = _Armed(overlay.norm_out)
        for name in ("proj_out", "audio_proj_out") if overlay.heads else ():
            head = _projection_base(getattr(self, name))
            if not isinstance(head, _SelectableHead):
                raise ValueError(
                    f"{name} is {type(head).__name__}, which cannot serve a turbo head",
                )
            head._armed = _Armed(getattr(overlay, name), step)

    def _release(self) -> None:
        """Every forward starts disarmed, whatever the previous one left behind."""
        self._arming = None
        for block in self.transformer_blocks:
            _modulation(block)._armed = None
        self.norm_out._armed = None
        for name in ("proj_out", "audio_proj_out"):
            head = _projection_base(getattr(self, name))
            if isinstance(head, _SelectableHead):
                head._armed = None


def _modulation(block: nn.Module) -> _BlockModulation | _AdaLNPrunedBlockTable:
    """A block's modulation as `H3Transformer` installs it: FULL's projection or the table."""
    return cast("_BlockModulation | _AdaLNPrunedBlockTable", block.adaln_proj)


def _overlay_tables(overlay: Overlay, block: int) -> OverlayTables:
    """An overlay's block table (`Overlay.__init__` places one in every block)."""
    return cast(OverlayTables, overlay.transformer_blocks[block].adaln_proj)


class AdaLNPrunedMiniMaxH3Transformer(H3Transformer):
    """H3's DiT with its modulation the checkpoint's exact tables in place of the projections."""

    time_proj: _AdaLNPrunedTimestepLookup
    time_embedder: _AdaLNPrunedTimestepRows
    _keep_in_fp32_modules: ClassVar[list[str]] = [  # type: ignore[assignment, misc]  # Diffusers declares it per instance; ModelMixin types it None
        *MiniMaxH3Transformer3DModel._keep_in_fp32_modules,
        "time_proj",
    ]

    def __init__(  # type: ignore[explicit-any]  # **config into Diffusers' typed keywords
        self,
        *,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.time_proj = _AdaLNPrunedTimestepLookup(table_timesteps)
        self.time_embedder = _AdaLNPrunedTimestepRows()
        for block in self.transformer_blocks:
            block.adaln_proj = _AdaLNPrunedBlockTable(
                hidden_size=int(self.config["hidden_size"]),
                timestep_count=len(table_timesteps),
                keys=table_block_keys,
            )
        self.norm_out = _AdaLNPrunedOutputTable(
            self.norm_out.norm,
            hidden_size=int(self.config["hidden_size"]),
            timestep_count=len(table_timesteps),
        )

    def _tables(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = cast(_AdaLNPrunedBlockTable, self.transformer_blocks[0].adaln_proj)
        return self.time_proj.rows(timestep), first.table_index


def build_transformer(
    config: Mapping[str, object],
    tables: tuple[Sequence[float], Sequence[tuple[int, int]]] | None = None,
) -> H3Transformer:
    """The DiT of an artifact's upstream config: FULL, or AdaLN-pruned over `tables` (the
    timesteps and block rows its checkpoint labels)."""
    parameters = inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters
    upstream = {name: value for name, value in config.items() if name in parameters}
    if tables is None:
        return H3Transformer(**upstream)
    timesteps, block_keys = tables
    return AdaLNPrunedMiniMaxH3Transformer(
        table_timesteps=timesteps, table_block_keys=block_keys, **upstream
    )
