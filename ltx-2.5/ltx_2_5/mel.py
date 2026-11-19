"""A soundtrack as LTX-2's audio VAE reads it: upstream's log-mel (ltx_core/model/audio_vae/ops.py `AudioProcessor`,
torchaudio's `MelSpectrogram` with Slaney scale and norm) and torchaudio's default resampler, in plain Torch: torchaudio
has no release for the Torch the machines serve (minimax-h3's `_resample` ports the same resampler)."""

from __future__ import annotations

import math

import torch


def resample(waveform: torch.Tensor, rate: int, target: int) -> torch.Tensor:
    """torchaudio's `functional.resample(waveform, rate, target)` with its defaults (sinc, Hann window, width 6)."""
    if rate == target:
        return waveform
    gcd = math.gcd(rate, target)
    rate, target = rate // gcd, target // gcd
    base = min(rate, target) * 0.99
    width = math.ceil(6 * rate / base)
    idx = torch.arange(-width, width + rate, dtype=torch.float64, device=waveform.device)[None, None] / rate
    t = torch.arange(0, -target, -1, dtype=torch.float32, device=waveform.device)[:, None, None] / target + idx
    t = (t * base).clamp_(-6, 6)
    window = torch.cos(t * math.pi / 6 / 2) ** 2
    t = t * math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), t.sin() / t) * window * (base / rate)
    channels, length = waveform.shape
    padded = torch.nn.functional.pad(waveform.float(), (width, width + rate))
    out = torch.nn.functional.conv1d(padded[:, None], kernel.float(), stride=rate)
    return out.transpose(1, 2).reshape(channels, -1)[:, : math.ceil(target * length / rate)]


def _hz_to_mel(hz: float) -> float:
    """Slaney's scale: linear to 1 kHz, logarithmic above."""
    if hz < 1000.0:
        return hz * 3.0 / 200.0
    return 15.0 + math.log(hz / 1000.0) / (math.log(6.4) / 27.0)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    linear = mels * 200.0 / 3.0
    return torch.where(mels >= 15.0, 1000.0 * torch.exp((math.log(6.4) / 27.0) * (mels - 15.0)), linear)


def filterbank(n_fft: int, mels: int, rate: int, device: torch.device) -> torch.Tensor:
    """torchaudio's `melscale_fbanks(n_fft // 2 + 1, 0, rate / 2, mels, rate, norm="slaney", mel_scale="slaney")`:
    [frequencies, mels]."""
    frequencies = torch.linspace(0, rate // 2, n_fft // 2 + 1, dtype=torch.float32, device=device)
    points = _mel_to_hz(torch.linspace(_hz_to_mel(0.0), _hz_to_mel(rate / 2.0), mels + 2, device=device))
    gaps = points[1:] - points[:-1]
    slopes = points[None, :] - frequencies[:, None]
    down = -slopes[:, :-2] / gaps[:-1]
    up = slopes[:, 2:] / gaps[1:]
    bank = torch.clamp(torch.minimum(down, up), min=0.0)
    return bank * (2.0 / (points[2 : mels + 2] - points[:mels]))[None, :]


def log_mel(waveform: torch.Tensor, rate: int, target: int, mels: int, hop: int, *, channels: int,
            n_fft: int = 1024) -> torch.Tensor:
    """[1, channels, frames, mels] from [channels, samples]: resampled to `target`, the magnitude of a centred,
    reflect-padded Hann STFT on the Slaney mel bank, log of at least 1e-5. A mono clip is both channels."""
    audio = resample(waveform.float(), rate, target)
    audio = audio.expand(channels, -1) if audio.shape[0] < channels else audio[:channels]
    spectrum = torch.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=torch.hann_window(n_fft, device=audio.device), center=True, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True).abs()
    mel = torch.matmul(spectrum.transpose(-1, -2), filterbank(n_fft, mels, target, audio.device))
    return torch.log(torch.clamp(mel, min=1e-5)).unsqueeze(0).contiguous()
