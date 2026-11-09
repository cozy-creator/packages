"""HiDream-O1's three samplers as upstream's `models/pipeline.py` drives them
(HiDream-ai/HiDream-O1-Image, main 2c2d29f).

Each `step(i, flow, sample)` takes the flow `-v` and the state in f32 and returns the next state in
f32. `UniPC` is Wan's `FlowUniPCMultistepScheduler` (Alibaba Wan Team, Apache-2.0; HiDream's
`models/fm_solvers_unipc.py`) in the one configuration HiDream runs: bh2, order 2, x0 prediction,
first order at the last step, final sigma 0. `Flash` is HiDream's
`FlashFlowMatchEulerDiscreteScheduler` on the Dev model's fixed timesteps: each step predicts the
clean image and noises it again to the next sigma, with noise computed from the seed and the step
(`rng.counter_noise`). `Euler` is Diffusers' `FlowMatchEulerDiscreteScheduler` on the same timesteps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cozy_runtime import rng
from diffusers import FlowMatchEulerDiscreteScheduler

#: The Dev model's 28 timesteps (`DEFAULT_TIMESTEPS`).
DEV_TIMESTEPS = (
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
)  # fmt: skip


class Euler:
    """Diffusers' flow-match Euler over integer timesteps, sigma = t / 1000 then 0."""

    def __init__(self, device: torch.device, timesteps: tuple[int, ...] = DEV_TIMESTEPS) -> None:
        self.timesteps = list(timesteps)
        self.scheduler: Any = FlowMatchEulerDiscreteScheduler(shift=1.0)
        self.scheduler.set_timesteps(sigmas=[t / 1000.0 for t in timesteps], device=device)
        self.scheduler.set_begin_index(0)
        self.sigmas: torch.Tensor = self.scheduler.sigmas

    def step(self, i: int, flow: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.scheduler.step(flow, self.scheduler.timesteps[i], sample, return_dict=False)[0]
        return out


class Flash(Euler):
    """Predict x0, then noise it again: `sigma' * noise * s_noise + (1 - sigma') * x0`, the noise
    clipped at `clip` of its own standard deviation (0: unclipped). Each step's noise is
    `rng.counter_noise` of the seed and the step; upstream draws it from the initial noise's
    generator instead, so the same seed makes another image."""

    def __init__(self, device: torch.device, seed: int, s_noise: float, clip: float) -> None:
        super().__init__(device)
        self.seed, self.s_noise, self.clip = seed, s_noise, clip

    def step(self, i: int, flow: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        denoised = sample - flow * self.sigmas[i]
        noise = rng.counter_noise(flow.shape, self.seed, i, dtype=denoised.dtype, device=flow.device)
        if self.clip > 0:
            bound = self.clip * noise.std()
            noise = noise.clamp(min=-bound, max=bound)
        following = self.sigmas[i + 1]
        return following * noise * self.s_noise + (1.0 - following) * denoised


class UniPC:
    """UniPC-bh2 of order 2 predicting x0, its sigmas shifted twice as Wan's are. It keeps the last
    two predictions (`x0`) and the last corrected sample (`last`) from step to step."""

    def __init__(self, device: torch.device, steps: int, shift: float) -> None:
        base = torch.from_numpy(1.0 - np.linspace(1, 1 / 1000, 1000)[::-1].copy()).to(torch.float32)
        base = shift * base / (1 + (shift - 1) * base)
        sigmas = np.linspace(base[0].item(), base[-1].item(), steps + 1).copy()[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = torch.from_numpy(sigmas * 1000).to(torch.int64).tolist()
        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(np.float32)).to(device)
        #: Each step's predictor order; a step's corrector takes the order of the step before.
        self.orders = [min(2, steps - i, i + 1) for i in range(steps)]
        self.tables = self._tables(device)
        self.x0: list[torch.Tensor] = []
        self.last: torch.Tensor | None = None

    def _tables(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Every 0-dim scalar a step reads, by the same device operations as Wan's solver, a row
        per step: sigma, then each update's `sigma_t / sigma_s0`, `alpha_t·h·phi_1`, `alpha_t·B(h)`
        and `r_1`, the predictor's (row i: step i to i + 1) then the corrector's (row i: step i - 1
        to i, row 0 unused), then the corrector's two `rhos`."""
        sigmas, orders = self.sigmas, self.orders
        lambdas = [torch.log(1 - sigma) - torch.log(sigma) for sigma in sigmas]
        one = torch.ones((), device=device)
        rows: list[list[torch.Tensor]] = []
        for i, order in enumerate(orders):
            predictor = self._update(sigmas, lambdas, i + 1, i, order, device)[:4]
            if i:
                *corrector, rhos = self._update(sigmas, lambdas, i, i - 1, orders[i - 1], device)
            else:
                corrector, rhos = [one] * 4, one.new_tensor([0.5, 0.5])
            rows.append([sigmas[i], *predictor, *corrector, rhos[0], rhos[-1]])
        return tuple(torch.stack(column) for column in zip(*rows, strict=True))

    @staticmethod
    def _update(sigmas: torch.Tensor, lambdas: list[torch.Tensor], t: int, s0: int, order: int,
                device: torch.device) -> list[torch.Tensor]:
        """The scalars of the update from `s0` to `t` at `order`: `sigma_t / sigma_s0`,
        `alpha_t·h·phi_1`, `alpha_t·B(h)`, `r_1` (1 at order 1) and the corrector's `rhos`."""
        alpha_t = 1 - sigmas[t]
        h = lambdas[t] - lambdas[s0]
        rks = [(lambdas[s0 - k] - lambdas[s0]) / h for k in range(1, order)]
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial, rows, b = 1, [], []
        ranks = torch.tensor([*rks, 1.0], device=device)
        b_h = torch.expm1(hh)
        for k in range(1, len(ranks) + 1):
            rows.append(torch.pow(ranks, k - 1))
            b.append(h_phi_k * factorial / b_h)
            factorial *= k + 1
            h_phi_k = h_phi_k / hh - 1 / factorial
        rhos = torch.tensor([0.5], device=device) if order == 1 else torch.linalg.solve(
            torch.stack(rows), torch.tensor(b, device=device)
        )
        r1 = rks[0] if rks else torch.ones((), device=device)
        return [sigmas[t] / sigmas[s0], alpha_t * h_phi_1, alpha_t * b_h, r1, rhos]

    def step(self, i: int, flow: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        sigma, *scalars = (table[i] for table in self.tables)
        predictor, corrector, (rho_0, rho_last) = scalars[:4], scalars[4:8], scalars[8:]
        x0 = sample - sigma * flow
        if self.last is not None:
            ratio, phi, b_h, rk = corrector
            out = ratio * self.last - phi * self.x0[-1]
            residual: Any = 0
            if self.orders[i - 1] == 2:
                residual = rho_0 * ((self.x0[-2] - self.x0[-1]) / rk)
            sample = out - b_h * (residual + rho_last * (x0 - self.x0[-1]))
        self.x0, self.last = [*self.x0[-1:], x0], sample
        ratio, phi, b_h, rk = predictor
        out = ratio * sample - phi * x0
        if self.orders[i] == 2:
            out = out - b_h * (0.5 * ((self.x0[0] - x0) / rk))
        return out
