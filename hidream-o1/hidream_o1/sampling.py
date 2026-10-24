"""HiDream-O1's three samplers as upstream's `models/pipeline.py` drives them
(HiDream-ai/HiDream-O1-Image, main 2c2d29f).

Each takes the flow `-v` and the state in f32 and returns the next state in f32. `UniPC` is Wan's
`FlowUniPCMultistepScheduler` (Alibaba Wan Team, Apache-2.0; HiDream's `models/fm_solvers_unipc.py`)
in the one configuration HiDream runs: bh2, order 2, x0 prediction, first order at the last step,
final sigma 0. `Flash` is HiDream's `FlashFlowMatchEulerDiscreteScheduler` on the Dev model's fixed
timesteps: each step predicts the clean image and noises it again to the next sigma, with noise drawn
on the device. `Euler` is Diffusers' `FlowMatchEulerDiscreteScheduler.step` on the same timesteps.
"""

from __future__ import annotations

import numpy as np
import torch
from cozy_runtime import rng

#: The Dev model's 28 timesteps (`DEFAULT_TIMESTEPS`).
DEV_TIMESTEPS = (
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
)  # fmt: skip


class Euler:
    """Diffusers' flow-match Euler over integer timesteps, sigma = t / 1000 then 0."""

    def __init__(self, timesteps: tuple[int, ...] = DEV_TIMESTEPS) -> None:
        self.timesteps = list(timesteps)
        self.sigmas = torch.tensor([t / 1000.0 for t in timesteps] + [0.0])

    def step(self, i: int, flow: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        sigmas = self.sigmas.to(sample.device)
        return sample + (sigmas[i + 1] - sigmas[i]) * flow


class Flash(Euler):
    """Predict x0, then noise it again: `sigma' * noise * s_noise + (1 - sigma') * x0`, the noise
    clipped at `clip` of its own standard deviation (0: unclipped)."""

    def __init__(self, generator: torch.Generator, s_noise: float, clip: float) -> None:
        super().__init__()
        self.generator, self.s_noise, self.clip = generator, s_noise, clip

    def step(self, i: int, flow: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        sigmas = self.sigmas.to(sample.device)
        denoised = sample - flow * sigmas[i]
        noise = rng.noise(flow.shape, self.generator, dtype=denoised.dtype, device=flow.device)
        if self.clip > 0:
            bound = self.clip * noise.std().item()
            noise = noise.clamp(min=-bound, max=bound)
        following = sigmas[i + 1]
        return following * noise * self.s_noise + (1.0 - following) * denoised


class UniPC:
    """UniPC-bh2 of order 2 predicting x0, its sigmas shifted twice as Wan's are."""

    def __init__(self, steps: int, shift: float) -> None:
        base = torch.from_numpy(1.0 - np.linspace(1, 1 / 1000, 1000)[::-1].copy()).to(torch.float32)
        base = shift * base / (1 + (shift - 1) * base)
        sigmas = np.linspace(base[0].item(), base[-1].item(), steps + 1).copy()[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = torch.from_numpy(sigmas * 1000).to(torch.int64).tolist()
        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(np.float32))
        self.outputs: list[torch.Tensor] = []
        self.last: torch.Tensor | None = None
        self.order = 0

    def _lambda(self, i: int) -> torch.Tensor:
        sigma = self.sigmas[i]
        return torch.log(1 - sigma) - torch.log(sigma)

    def _coefficients(self, h: torch.Tensor, rks: list[torch.Tensor], device: torch.device) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """h·phi_1, B(h), and the R and b of UniPC's linear system."""
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
        return h_phi_1, b_h, torch.stack(rows), torch.tensor(b, device=device)

    def _update(self, i: int, x: torch.Tensor, order: int, corrected: torch.Tensor | None) -> torch.Tensor:
        """The predictor from step `i` to `i + 1` (`corrected` None), or the corrector of step `i`
        from `i - 1` with the new prediction `corrected`."""
        t, s0 = (i + 1, i) if corrected is None else (i, i - 1)
        sigma_t, sigma_s0 = self.sigmas[t], self.sigmas[s0]
        alpha_t = 1 - sigma_t
        lambda_s0 = self._lambda(s0)
        h = self._lambda(t) - lambda_s0
        m0 = self.outputs[-1]
        rks, d1s = [], []
        for k in range(1, order):
            rk = (self._lambda(s0 - k) - lambda_s0) / h
            rks.append(rk)
            d1s.append((self.outputs[-(k + 1)] - m0) / rk)
        h_phi_1, b_h, r, b = self._coefficients(h, rks, x.device)
        out = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if corrected is None:
            if d1s:
                out = out - alpha_t * b_h * (0.5 * d1s[0])
            return out
        rhos = torch.tensor([0.5], device=x.device) if order == 1 else torch.linalg.solve(r, b)
        residual = rhos[0] * d1s[0] if d1s else 0
        return out - alpha_t * b_h * (residual + rhos[-1] * (corrected - m0))

    def step(self, i: int, flow: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        self.sigmas = self.sigmas.to(sample.device)
        x0 = sample - self.sigmas[i] * flow
        if i > 0 and self.last is not None:
            sample = self._update(i, self.last, self.order, x0)
        self.outputs = [*self.outputs[-1:], x0]
        self.order = min(2, len(self.timesteps) - i, i + 1)
        self.last = sample
        return self._update(i, sample, self.order, None)
