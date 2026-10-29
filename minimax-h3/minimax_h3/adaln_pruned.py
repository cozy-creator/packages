"""Exact-timestep AdaLN tables for the official Diffusers MiniMax-H3 DiT.

The extension deliberately inherits Diffusers' forward unchanged. It replaces only the
time/AdaLN modules whose checkpoint-specific outputs were precomputed at the timestep rows
labelled by the checkpoint before those dynamic modules were pruned. The remaining
transformer graph, including attention, feed-forward, RoPE, input/output projections,
and packed-row indexing, stays upstream code.

An overlay (`overlay.py`: PDD-8 turbo, the Regenerate refine) attaches beside the DiT and is
served per forward: the forward that names its bank in `attention_kwargs`, or runs inside a
denoise loop that engaged it, reads the overlay's tables, heads and low-rank factors through the
same modules; every other forward reads the base tables it always did.
"""

from __future__ import annotations

import inspect
from functools import partial
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, cast

import torch
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3TransformerOutput,
)
from diffusers.utils import apply_lora_scale
from torch import nn
from torch.nn import functional as F

from cozy_runtime import lora
from cozy_runtime.author import ConformanceError

from .fit import release_if_tight
from .overlay import (
    ATTENTION_KWARG,
    LORA_FAMILIES,
    OVERLAY_KWARG,
    Overlay,
    OverlayArming,
    armed,
)


class StepInvariants:
    """What one denoise computes once rather than every step: the rotary tables of its packed
    positions and its refined text rows. Each is kept while the same input tensors come back
    unmodified (the same objects at the same in-place version) under the same turbo overlay, so
    a layout change (a clip's rows leaving before the first sparse step) recomputes it.
    `checked`: the denoise checked every step's rows before its loop, so its forwards read no
    device value on the host (a `denoise_loop` step). `overlay`: the one overlay that loop's every
    forward is served under, engaged for each (it has no per-step heads)."""

    __slots__ = ("checked", "held", "overlay")

    def __init__(self, checked: bool = False, overlay: Overlay | None = None) -> None:
        self.held: dict[str, tuple[tuple[tuple[object, int], ...], Any]] = {}
        self.checked = checked
        self.overlay = overlay

    def get(self, name: str, inputs: tuple[object, ...], compute: Callable[[], Any]) -> Any:
        key = tuple((value, getattr(value, "_version", 0)) for value in inputs)
        held = self.held.get(name)
        if (
            held is not None
            and len(held[0]) == len(key)
            and all(
                old is new and old_version == new_version
                for (old, old_version), (new, new_version) in zip(held[0], key, strict=True)
            )
        ):
            return held[1]
        value = compute()
        self.held[name] = (key, value)
        return value


_STEP_INVARIANTS: ContextVar[StepInvariants | None] = ContextVar(
    "minimax_h3_step_invariants", default=None
)


@contextmanager
def step_invariants(
    *, checked: bool = False, overlay: Overlay | None = None
) -> Iterator[StepInvariants]:
    """One denoise loop's step invariants, dropped when it ends."""
    if overlay is not None and (overlay.heads or not checked):
        raise ConformanceError(
            "a denoise loop engages only a checked overlay without per-step heads",
            code="artifact_config",
        )
    memo = StepInvariants(checked, overlay)
    token = _STEP_INVARIANTS.set(memo)
    try:
        yield memo
    finally:
        _STEP_INVARIANTS.reset(token)
        memo.held.clear()


class _Armed:
    """One module's turbo source for the forward in flight; a plain object, so placing it
    on a module registers nothing and the DiT's state dict is what it was."""

    __slots__ = ("source", "step")

    def __init__(self, source: Any, step: int = 0) -> None:
        self.source = source
        self.step = step


class _AdaLNPrunedTimestepLookup(nn.Module):  # type: ignore[misc]
    """Map exact float32 timestep values to the checkpoint's labelled rows."""

    def __init__(self, timesteps: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer(
            "timesteps", torch.tensor(tuple(timesteps), dtype=torch.float32), persistent=False
        )
        self._armed: _Armed | None = None

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """The rows, on the device: the transformer checked them (`rows`) before its forward."""
        if self._armed is not None:
            return cast(torch.Tensor, self._armed.source.rows(timestep))
        return self._lookup(timestep)

    def _lookup(self, timestep: torch.Tensor) -> torch.Tensor:
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        return (values == self.timesteps.reshape(1, -1)).to(dtype=torch.int64).argmax(dim=1)

    def rows(self, timestep: torch.Tensor) -> torch.Tensor:
        if self._armed is not None:
            return cast(torch.Tensor, self._armed.source.rows(timestep))
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        matches = values == self.timesteps.reshape(1, -1)
        counts = matches.sum(dim=1)
        if not bool(torch.all(counts == 1)):
            unknown = timestep.detach().float().cpu().tolist()
            raise ConformanceError(
                f"model modulation tables do not cover the requested timesteps: {unknown}; "
                "select a covered step count or regenerate the tables for this schedule",
                code="artifact_config",
            )
        return matches.to(dtype=torch.int64).argmax(dim=1)


class _AdaLNPrunedTimestepRows(nn.Module):  # type: ignore[misc]
    """Keep inherited ``forward`` intact while replacing the removed timestep MLP."""

    def __init__(self) -> None:
        super().__init__()
        # The inherited Diffusers forward calls `get_parameter_dtype(time_embedder)` before
        # this identity. Its helper raises `UnboundLocalError` for a module with no tensor,
        # and a floating anchor would cast these exact integer row ids. This zero-length
        # int64 buffer is therefore the smallest state that preserves the upstream call.
        self.register_buffer("row_dtype", torch.empty(0, dtype=torch.int64), persistent=False)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return rows


class _AdaLNPrunedBlockTable(nn.Module):  # type: ignore[misc]
    """Sparse exact table with the dense local shape Diffusers blocks already consume."""

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
        self._armed: _Armed | None = None

    def forward(self, timestep_rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        source = self if self._armed is None else self._armed.source
        indices = source.table_index.index_select(0, timestep_rows).reshape(-1)
        # The transformer's pre-hook proves every selected pair is present. Missing cells
        # are never observed by Diffusers' existing `adaln_indices`; row zero is only the
        # placeholder needed to preserve that dense local indexing convention. An armed
        # turbo table is read through its own forward: it is a piece of its overlay.
        index = indices.clamp_min(0)
        rows = self.table.index_select(0, index) if self._armed is None else source(index)
        return cast(
            "tuple[torch.Tensor, ...]",
            rows.reshape(-1, 6 * self.hidden_size).chunk(6, dim=-1),
        )


class _AdaLNPrunedOutputTable(nn.Module):  # type: ignore[misc]
    """Official final RMSNorm with exact precomputed shift/scale rows."""

    def __init__(self, norm: nn.Module, *, hidden_size: int, timestep_count: int) -> None:
        super().__init__()
        self.norm = norm
        self.table = nn.Parameter(torch.empty(timestep_count, 2, hidden_size))
        self._armed: _Armed | None = None

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
        shift, scale = rows.unbind(dim=1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class _SelectableHead(nn.Linear):  # type: ignore[misc]
    """The official output head, or the turbo evaluation's collapsed head when armed."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)
        self._armed: _Armed | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._armed is None:
            return F.linear(hidden_states, self.weight, self.bias)
        return cast(torch.Tensor, self._armed.source(hidden_states, self._armed.step))


def _projection_base(module: object) -> object:
    """Public PEFT wrappers keep the model-owned selectable head as their base layer."""
    get_base = getattr(module, "get_base_layer", None)
    return get_base() if callable(get_base) else module


class AdaLNPrunedMiniMaxH3Transformer(MiniMaxH3Transformer3DModel):  # type: ignore[misc]
    """Official MiniMax-H3 transformer with only its exact modulation source replaced."""

    _keep_in_fp32_modules: ClassVar[list[str]] = [
        *MiniMaxH3Transformer3DModel._keep_in_fp32_modules,
        "time_proj",
    ]

    def __init__(
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
                hidden_size=int(self.config.hidden_size),
                timestep_count=len(table_timesteps),
                keys=table_block_keys,
            )
        norm_out = cast(Any, self.norm_out)  # type: ignore[has-type]
        self.norm_out = _AdaLNPrunedOutputTable(
            norm_out.norm,
            hidden_size=int(self.config.hidden_size),
            timestep_count=len(table_timesteps),
        )
        for name in ("proj_out", "audio_proj_out"):
            head = cast(nn.Linear, getattr(self, name))
            setattr(self, name, _SelectableHead(head.in_features, head.out_features))
        self._arming: OverlayArming | None = None
        self._lora_hooks_installed = False

    @apply_lora_scale("attention_kwargs")  # type: ignore[untyped-decorator]
    def forward(
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
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        """Diffusers 0.41's forward, the projected rows scoped to the packing: Diffusers keeps
        them to its end, the video's float32 rows (2.25 GB a GPU at H3 15 s, degree 2) through
        every block. It checks the rows and arms a turbo bank `attention_kwargs` names (`_arm`),
        unless its denoise checked every step's rows ahead (`StepInvariants.checked`), and ends
        disarmed: in the forward, as a package forward hook would keep it out of native loops."""
        memo = _STEP_INVARIANTS.get()
        if memo is None or not memo.checked:
            self._arm(timestep, timestep_indices, token_tags, attention_kwargs)
        elif memo.overlay is not None:
            self._engage(memo.overlay, 0)
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

            def refined() -> torch.Tensor:
                dtype = get_parameter_dtype(self.context_embedder)
                return cast(
                    torch.Tensor,
                    self.token_refiner(self.context_embedder(encoder_hidden_states.to(dtype))),
                )

            memo = _STEP_INVARIANTS.get()
            if memo is None:
                rotary_emb, text = self.rope(position_ids), refined()
            else:  # inside a denoise loop: once per layout, not once per step
                rotary_emb = memo.get("rope", (position_ids,), lambda: self.rope(position_ids))
                overlay = self._arming.overlay if self._arming is not None else None
                text = memo.get("text", (encoder_hidden_states, overlay), refined)
            rows = text.new_zeros((text.shape[0], sequence_length, text.shape[-1]))
            rows = rows.index_copy(1, text_indices, text)
            del text
            video = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
            rows = rows.index_copy(1, video_indices, video.to(rows.dtype))
            del video
            audio = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
            rows = rows.index_copy(1, audio_indices, audio.to(rows.dtype))
            del audio
            # The packing's float32 video rows (2.2 GB at 15 s) leave the cache before the blocks.
            release_if_tight(rows, self.transformer_blocks[0].attn)
            temb = self.time_proj(timestep)
            temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
            adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    rows = self._gradient_checkpointing_func(block, rows, temb, adaln_indices, rotary_emb)
                else:
                    rows = block(rows, temb, adaln_indices, rotary_emb)
            rows = self.norm_out(rows, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
            video_output = self.proj_out(rows).index_select(1, video_indices)
            audio_output = self.audio_proj_out(rows).index_select(1, audio_indices)
            if not return_dict:
                return (video_output, audio_output)
            return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)
        finally:
            self._release()

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
        timestep: Any,
        timestep_indices: Any,
        token_tags: Any,
        selector: Any,
    ) -> None:
        """Check one forward's rows against the tables it reads: the turbo overlay's, engaged
        here, when `selector` (its `attention_kwargs`) names the bank."""
        self._release()
        bank = selector.get(ATTENTION_KWARG) if isinstance(selector, Mapping) else None
        if not all(
            isinstance(value, torch.Tensor) for value in (timestep, timestep_indices, token_tags)
        ):
            if bank is not None:
                raise ConformanceError(
                    "a turbo forward needs typed timestep, timestep_indices and token_tags",
                    code="artifact_config",
                )
            return  # Diffusers owns its ordinary argument diagnostics.
        assert isinstance(timestep, torch.Tensor)
        assert isinstance(timestep_indices, torch.Tensor)
        assert isinstance(token_tags, torch.Tensor)
        if bool(torch.any((token_tags < 0) | (token_tags > 2))):
            raise ConformanceError(
                "packed rows contain a modality tag outside the canonical 0/1/2 set",
                code="artifact_config",
            )
        tables: Any = self
        try:
            if bank is not None:
                overlay = selector.get(OVERLAY_KWARG) if isinstance(selector, Mapping) else None
                if (
                    not isinstance(overlay, Overlay)
                    or overlay.bank != bank
                    or not self._lora_hooks_installed
                ):
                    raise ConformanceError(
                        f"bank {bank!r} requires an overlay-capable base and that bank's "
                        "prepared overlay component",
                        code="artifact_config",
                        fields=["attention_kwargs", ATTENTION_KWARG],
                    )
                step = overlay.schedule.step(timestep) if overlay.heads else 0
                self._engage(overlay, step)
                tables = overlay
            global_rows = self.time_proj.rows(timestep)
            row_timesteps = global_rows.index_select(0, timestep_indices)
            table_index = tables.transformer_blocks[0].adaln_proj.table_index
            present = table_index[row_timesteps, token_tags] >= 0
            if not bool(torch.all(present)):
                raise ConformanceError(
                    "model modulation tables do not cover a requested reference timestep/modality; "
                    "regenerate the tables with coverage for this reference type",
                    code="artifact_config",
                )
        except Exception:
            self._release()
            raise

    def _engage(self, overlay: Overlay, step: int) -> None:
        self._arming = OverlayArming(overlay, step)
        self.time_proj._armed = _Armed(overlay)
        for block, source in zip(self.transformer_blocks, overlay.transformer_blocks, strict=True):
            block.adaln_proj._armed = _Armed(source.adaln_proj)
        self.norm_out._armed = _Armed(overlay.norm_out)
        for name in ("proj_out", "audio_proj_out") if overlay.heads else ():
            head = _projection_base(getattr(self, name))
            if not isinstance(head, _SelectableHead):
                raise ConformanceError(
                    f"{name} is {type(head).__name__}, which cannot serve a turbo head",
                    code="artifact_config",
                )
            head._armed = _Armed(getattr(overlay, name), step)

    def _release(self) -> None:
        """Every forward starts disarmed, whatever the previous one left behind."""
        self._arming = None
        self.time_proj._armed = None
        for block in self.transformer_blocks:
            block.adaln_proj._armed = None
        self.norm_out._armed = None
        for name in ("proj_out", "audio_proj_out"):
            head = _projection_base(getattr(self, name))
            if isinstance(head, _SelectableHead):
                head._armed = None

    @classmethod
    def from_official_config(
        cls,
        config: Mapping[str, Any],
        *,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
    ) -> AdaLNPrunedMiniMaxH3Transformer:
        """Construct from the same upstream config accepted by the official class."""
        parameters = inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters
        kwargs = {name: value for name, value in config.items() if name in parameters}
        return cls(
            table_timesteps=table_timesteps,
            table_block_keys=table_block_keys,
            **kwargs,
        )
