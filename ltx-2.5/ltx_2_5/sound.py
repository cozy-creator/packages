"""A soundtrack as LTX-2's audio VAE reads it: upstream's log-mel (ltx_core/model/audio_vae/ops.py `AudioProcessor`,
torchaudio's `MelSpectrogram` with Slaney mel scale and norm, magnitude, log of at least 1e-5), in plain torch."""

from __future__ import annotations

import math

import torch


def resample(waveform: torch.Tensor, rate: int, target: int) -> torch.Tensor:
    """[channels, samples] at `target` Hz: band-limited, by the spectrum (upstream resamples with torchaudio's sinc)."""
    if rate == target:
        return waveform
    count = max(1, round(waveform.shape[-1] * target / rate))
    spectrum = torch.fft.rfft(waveform.float(), dim=-1)
    bins = count // 2 + 1
    kept = spectrum[..., :bins] if spectrum.shape[-1] >= bins else torch.nn.functional.pad(spectrum, (0, bins - spectrum.shape[-1]))
    return torch.fft.irfft(kept, n=count, dim=-1) * (count / waveform.shape[-1])


def _mel(hz: torch.Tensor) -> torch.Tensor:
    linear = hz / (200.0 / 3)
    return torch.where(hz >= 1000.0, 15.0 + torch.log(hz.clamp(min=1e-10) / 1000.0) / (math.log(6.4) / 27.0), linear)


def _hz(mel: torch.Tensor) -> torch.Tensor:
    linear = mel * (200.0 / 3)
    return torch.where(mel >= 15.0, 1000.0 * torch.exp((math.log(6.4) / 27.0) * (mel - 15.0)), linear)


def filterbank(rate: int, n_fft: int, mels: int, device: torch.device) -> torch.Tensor:
    """torchaudio's `melscale_fbanks(n_fft // 2 + 1, 0, rate / 2, mels, rate, norm="slaney", mel_scale="slaney")`."""
    freqs = torch.linspace(0, rate // 2, n_fft // 2 + 1, dtype=torch.float64, device=device)
    points = _hz(torch.linspace(float(_mel(torch.tensor(0.0, dtype=torch.float64))), float(_mel(torch.tensor(rate / 2.0, dtype=torch.float64))),
                                mels + 2, dtype=torch.float64, device=device))
    gaps = points[1:] - points[:-1]
    slopes = points.unsqueeze(0) - freqs.unsqueeze(1)
    bank = torch.clamp(torch.minimum(-slopes[:, :-2] / gaps[:-1], slopes[:, 2:] / gaps[1:]), min=0)
    return (bank * (2.0 / (points[2:mels + 2] - points[:mels])).unsqueeze(0)).float()


def log_mel(waveform: torch.Tensor, rate: int, target: int, mels: int, hop: int, *, channels: int,
            n_fft: int = 1024) -> torch.Tensor:
    """[1, channels, frames, mels]: a mono clip is both channels of a stereo VAE, as upstream's stereo read gives."""
    audio = resample(waveform, rate, target)
    if audio.shape[0] < channels:
        audio = audio.expand(channels, -1)
    audio = audio[:channels]
    window = torch.hann_window(n_fft, device=audio.device)
    spectrum = torch.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True,
                          pad_mode="reflect", return_complex=True).abs()
    mel = torch.matmul(filterbank(target, n_fft, mels, audio.device).T, spectrum)
    return torch.log(mel.clamp(min=1e-5)).transpose(1, 2).unsqueeze(0).contiguous()
