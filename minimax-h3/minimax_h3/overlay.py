"""Overlays over the AdaLN-pruned MiniMax-H3 DiT: PDD-8 turbo and the Regenerate refine.

An overlay is LoRA factors on six linear families, AdaLN tables at its plan's exact timesteps and,
for turbo, a bank of output heads. A forward names the bank it is served under; every other
forward reads the base.

Parallel Decoding Distillation (arXiv 2607.26004; `alibaba-pai/MiniMax-H3-Acc-LoRAs`) is a
rank-64 LoRA on seven linear families plus a bank of per-interval output heads, sampled on a
32-point grid four intervals at a time: eight transformer evaluations. On a pruned lane six of
the families are ordinary modules and apply here, as `y += (alpha/rank) * up(down(x))` over
the fp8 base; the seventh, `adaln_proj.linear`, has no module to attach to and is baked into
the turbo AdaLN tables by the producer. The head bank collapses offline to one head per
evaluation because the reference's per-step fusion is linear in the weights and its plan
depends only on the fixed grid.

Open-H3-Regenerate-2K (NVlabs) is one merged rank-576 LoRA on the same six families and LMS
Sharp's AdaLN curve, which its producer folds into the tables; it keeps the base heads.

The overlay is one component per task beside the DiT it patches. Nothing here changes the
DiT's own parameters or its destination names: the base functions see the same construction
with the overlay absent from every forward they run.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional as F

from cozy_runtime import lora
from cozy_runtime.author import ConformanceError

#: The forward-call selector: `attention_kwargs={ATTENTION_KWARG: TURBO_BANK}` names the bank
#: one DiT forward serves. Absent, the forward is the base's, whatever overlay is attached.
ATTENTION_KWARG = "cozy_h3_bank"
OVERLAY_KWARG = "cozy_h3_overlay"
TURBO_BANK = "turbo"
REGENERATE_BANK = "regenerate"
#: The families applied at inference, as the checkpoint names them under a block.
LORA_FAMILIES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
#: The family the producer bakes into the turbo tables; refused on the overlay by name.
TABLED_FAMILY = "adaln_proj.linear"
#: Rows per LoRA GEMM. Fixed, so a row's update never depends on how many rows a rank holds
#: (Ulysses), and large enough that kernel launches stop mattering (TensorD #26: 256-row
#: chunks cost 7.6% of a dense step on an RTX PRO 6000, one GEMM pair per site 3.3%).
_LORA_ROW_CHUNK = 4096
#: Fix the FP32 projection's GEMM shape across Ulysses sequence partitions.
_HEAD_ROW_CHUNK = 1024


class LoRAFactors(nn.Module):  # type: ignore[misc]
    """One site's rank-`r` factors over the original tensor operand."""

    def __init__(self, in_features: int, out_features: int, rank: int, scale: float) -> None:
        super().__init__()
        self.lora_down = nn.Parameter(torch.empty(rank, in_features, dtype=torch.bfloat16))
        self.lora_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch.bfloat16))
        self.scale = scale

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """Adds this site's update into `out` in fixed row chunks (`cozy_runtime.lora`)."""
        lora.accumulate(x, out, self.lora_down, self.lora_up, self.scale, chunk=_LORA_ROW_CHUNK)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self.lora_down, self.lora_up, float(self.scale)


class _Site(nn.Module):  # type: ignore[misc]
    """A named container whose children reproduce the checkpoint's module paths."""

    def __init__(self, **children: nn.Module) -> None:
        super().__init__()
        for name, child in children.items():
            self.add_module(name, child)


class OverlayTables(nn.Module):  # type: ignore[misc]
    """An overlay's exact rows for one family of tables: the same shape the base tables have."""

    def __init__(self, table: torch.Tensor, index: torch.Tensor | None = None) -> None:
        super().__init__()
        self.table = nn.Parameter(table)
        if index is not None:
            self.register_buffer("table_index", index, persistent=False)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """The table's `rows`, read inside its forward: the table is a piece of its overlay."""
        return self.table.index_select(0, rows)


class TurboHeads(nn.Module):  # type: ignore[misc]
    """One collapsed head per transformer evaluation, selected by the step being served."""

    def __init__(self, evaluations: int, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(evaluations, out_features, in_features, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(evaluations, out_features, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, step: int) -> torch.Tensor:
        inputs = hidden_states.reshape(-1, hidden_states.shape[-1])
        weight, bias = self.weight[step], self.bias[step]
        output = hidden_states.new_empty((*hidden_states.shape[:-1], weight.shape[0]))
        flat = output.view(-1, weight.shape[0])
        for start in range(0, int(inputs.shape[0]), _HEAD_ROW_CHUNK):
            rows = inputs[start : start + _HEAD_ROW_CHUNK].contiguous()
            count = rows.shape[0]
            # Changing M selects different FP32 reduction kernels even with TF32
            # disabled. Full tiles and tails must execute the same projection.
            if count < _HEAD_ROW_CHUNK:
                rows = F.pad(rows, (0, 0, 0, _HEAD_ROW_CHUNK - count))
            flat[start : start + count].copy_(F.linear(rows, weight, bias)[:count])
        return output


class TurboSchedule:
    """The exact float32 timesteps of the one turbo schedule, and the step each pair names."""

    def __init__(self, video_timesteps: Sequence[float], audio_timesteps: Sequence[float]) -> None:
        video, audio = tuple(video_timesteps), tuple(audio_timesteps)
        if len(video) != len(audio) or not video:
            raise ValueError("a turbo schedule pairs one audio timestep with every video one")
        # The step is read off the timesteps a forward presents, so the two grids may only
        # meet at the shared origin: any other shared value would make a step ambiguous.
        if set(video) & set(audio) != {0.0} or len(set(video)) != len(video):
            raise ValueError("turbo video and audio grids must share only the origin")
        self.video = video
        self.audio = audio
        self.evaluations = len(video)

    def step(self, timestep: torch.Tensor) -> int:
        present = set(timestep.detach().to(torch.float32).cpu().tolist())
        steps = [index for index, value in enumerate(self.video) if value in present]
        if len(steps) != 1 or self.audio[steps[0]] not in present:
            raise ConformanceError(
                f"timesteps {sorted(present)} do not name one turbo evaluation",
                code="artifact_config",
            )
        return steps[0]


class Overlay(nn.Module):  # type: ignore[misc]
    """Everything one overlay adds to its trunk DiT, under the DiT's own paths.

    State-dict keys are the checkpoint's: `transformer_blocks.{i}.attn.to_q.lora_down`,
    `token_refiner.refiner_blocks.{i}.ff.net.2.lora_up`, `transformer_blocks.{i}.adaln_proj.table`,
    `norm_out.table`, `proj_out.weight` — so a LoRA site's path here IS its path on the DiT.
    """

    present = True
    #: Its pieces: each site's factors, each table and each head is its own block, so a
    #: consumer binds the few MB it reads, never the whole overlay (~620 MB, cr-203 U9 (c)).
    _no_split_modules: ClassVar[list[str]] = ["LoRAFactors", "OverlayTables", "TurboHeads"]

    def __init__(
        self,
        *,
        hidden_size: int,
        inner_dim: int,
        ffn_dim: int,
        num_layers: int,
        num_refiner_layers: int,
        video_out: int,
        audio_out: int,
        rank: int,
        alpha: float,
        bank: str,
        schedule: TurboSchedule | None,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
        block_table_dtype: torch.dtype,
        final_table_dtype: torch.dtype,
    ) -> None:
        """`schedule` is the turbo bank's evaluation grid, which selects its heads; an overlay
        without one keeps the base heads."""
        super().__init__()
        self.bank = bank
        scale = alpha / rank

        def factors(in_features: int, out_features: int) -> LoRAFactors:
            return LoRAFactors(in_features, out_features, rank, scale)

        def block_sites() -> dict[str, nn.Module]:
            return {
                "attn": _Site(
                    to_q=factors(hidden_size, inner_dim),
                    to_k=factors(hidden_size, inner_dim),
                    to_v=factors(hidden_size, inner_dim),
                    to_out=nn.ModuleList([factors(inner_dim, hidden_size)]),
                ),
                "ff": _Site(
                    net=nn.ModuleList(
                        [
                            _Site(proj=factors(hidden_size, 2 * ffn_dim)),
                            nn.Identity(),
                            factors(ffn_dim, hidden_size),
                        ]
                    )
                ),
            }

        index = torch.full((len(table_timesteps), 3), -1, dtype=torch.int64)
        for table_row, (timestep_row, modality_tag) in enumerate(table_block_keys):
            index[timestep_row, modality_tag] = table_row
        self.schedule = schedule
        self.register_buffer(
            "timesteps", torch.tensor(tuple(table_timesteps), dtype=torch.float32), persistent=False
        )
        self.token_refiner = _Site(
            refiner_blocks=nn.ModuleList(
                [_Site(**block_sites()) for _ in range(num_refiner_layers)]
            )
        )
        self.transformer_blocks = nn.ModuleList(
            [
                _Site(
                    **block_sites(),
                    adaln_proj=OverlayTables(
                        torch.empty(len(table_block_keys), 6, hidden_size, dtype=block_table_dtype),
                        index,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = OverlayTables(
            torch.empty(len(table_timesteps), 2, hidden_size, dtype=final_table_dtype)
        )
        self.heads = schedule is not None
        if schedule is not None:
            self.proj_out = TurboHeads(schedule.evaluations, video_out, hidden_size)
            self.audio_proj_out = TurboHeads(schedule.evaluations, audio_out, hidden_size)

    @classmethod
    def from_official_config(
        cls,
        config: Mapping[str, Any],
        *,
        rank: int,
        alpha: float,
        bank: str,
        schedule: TurboSchedule | None,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
        block_table_dtype: torch.dtype,
        final_table_dtype: torch.dtype,
    ) -> Overlay:
        patch = 1
        for value in config["patch_size"]:
            patch *= int(value)
        return cls(
            hidden_size=int(config["hidden_size"]),
            inner_dim=int(config["num_attention_heads"]) * int(config["attention_head_dim"]),
            ffn_dim=int(config["ffn_dim"]),
            num_layers=int(config["num_layers"]),
            num_refiner_layers=int(config["num_refiner_layers"]),
            video_out=int(config["in_channels"]) * patch,
            audio_out=int(config["audio_in_channels"]),
            rank=rank,
            alpha=alpha,
            bank=bank,
            schedule=schedule,
            table_timesteps=table_timesteps,
            table_block_keys=table_block_keys,
            block_table_dtype=block_table_dtype,
            final_table_dtype=final_table_dtype,
        )

    def lora_sites(self) -> Iterator[tuple[str, LoRAFactors]]:
        for name, module in self.named_modules():
            if isinstance(module, LoRAFactors):
                yield name, module

    def rows(self, timestep: torch.Tensor) -> torch.Tensor:
        """Global turbo table rows of exact float32 timesteps; off-plan values refuse."""
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        matches = values == self.timesteps.reshape(1, -1)
        if not bool(torch.all(matches.sum(dim=1) == 1)):
            raise ConformanceError(
                f"timestep rows are not covered exactly by the {self.bank} plan: "
                f"{timestep.detach().float().cpu().tolist()}",
                code="artifact_config",
            )
        return matches.to(dtype=torch.int64).argmax(dim=1)


class OverlayArming:
    """What one DiT forward under an overlay's bank reads: set from the forward's own
    `attention_kwargs` (or its denoise loop), cleared when the forward ends, never request state."""

    __slots__ = ("overlay", "step")

    def __init__(self, overlay: Overlay, step: int) -> None:
        self.overlay = overlay
        self.step = step


def armed(owner: Any, path: str) -> LoRAFactors | None:
    """The armed overlay's factors at `path`, or None in a base forward."""
    arming = owner._arming
    if arming is None:
        return None
    return arming.overlay.get_submodule(path)  # type: ignore[no-any-return]
