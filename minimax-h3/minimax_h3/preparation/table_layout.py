"""The checkpoint's ordered AdaLN row labels, independent of a sampling preset.

The producer owns this reader; minimax-h3 imports it to build its AdaLN-pruned DiT
rather than vendoring a copy.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn


_MODALITIES = ("video", "text", "audio")


def _refuse(message: str) -> NoReturn:
    raise ValueError(
        f"model modulation tables {message}"
    )


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


def _modality_tag(row: Mapping[str, object]) -> int:
    """The row's modality from its tag, its name, or both when they agree."""
    tag, name = row.get("modality_tag"), row.get("modality")
    named = _MODALITIES.index(name) if name in _MODALITIES else None
    if type(tag) is int and tag in range(3) and named in (None, tag):
        return tag
    if tag is None and named is not None:
        return named
    _refuse("must label modalities as video=0, text=1, or audio=2")


@dataclass(frozen=True, slots=True)
class TableLayout:
    timesteps: tuple[float, ...]
    block_keys: tuple[tuple[int, int], ...]

    @classmethod
    def parse(cls, value: object) -> TableLayout:
        """Read the labels a writer of any version recorded; unknown keys are ignored."""
        if not isinstance(value, Mapping):
            _refuse("must include their final-normalization and block-modulation row labels")
        final = value.get("final_normalization")
        blocks = value.get("block_modulation")
        if not isinstance(final, list) or not final or not isinstance(blocks, list) or not blocks:
            _refuse("must contain nonempty final-normalization and block-modulation label lists")
        timesteps: list[float] = []
        rows: dict[float, int] = {}
        for index, row in enumerate(final):
            if not isinstance(row, Mapping) or row.get("index", index) != index:
                _refuse("need final-normalization rows in index order starting at zero")
            timestep = _timestep(row.get("timestep"))
            if timestep in rows:
                _refuse("contain duplicate final-normalization timestep labels")
            rows[timestep] = index
            timesteps.append(timestep)
        keys: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for index, row in enumerate(blocks):
            if not isinstance(row, Mapping) or row.get("index", index) != index:
                _refuse("need block-modulation rows in index order starting at zero")
            tag = _modality_tag(row)
            timestep = _timestep(row.get("timestep"))
            if timestep not in rows:
                _refuse("reference a timestep absent from their final-normalization rows")
            key = (rows[timestep], tag)
            if key in seen:
                _refuse("contain a duplicate timestep/modality pair")
            seen.add(key)
            keys.append(key)
        return cls(tuple(timesteps), tuple(keys))

    def require(self, video: Sequence[float], audio: Sequence[float]) -> None:
        """Check the requested schedule before its first transformer evaluation.

        Conditional rows are checked against the actual packed tokens by the transformer's
        existing pre-hook; a text-only call need not carry unused reference rows.
        """
        available = {(self.timesteps[row], tag) for row, tag in self.block_keys}
        for values, tags in ((video, (0, 1)), (audio, (2,))):
            for timestep in values:
                for tag in tags:
                    if (timestep, tag) not in available:
                        _refuse(
                            f"do not cover the requested schedule: missing {_MODALITIES[tag]} "
                            f"timestep {timestep:g}. Select a covered step count or regenerate "
                            "the checkpoint's modulation tables for this schedule."
                        )
