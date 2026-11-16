"""A soundtrack as LTX-2's audio VAE reads it: upstream's log-mel (ltx_core/model/audio_vae/ops.py `AudioProcessor`), with
the same torchaudio transforms, which the machine's PyTorch layer carries."""

from __future__ import annotations

import torch
import torchaudio


def log_mel(waveform: torch.Tensor, rate: int, target: int, mels: int, hop: int, *, channels: int,
            n_fft: int = 1024) -> torch.Tensor:
    """[1, channels, frames, mels] from [channels, samples]: resampled to `target`, magnitude Slaney mel, log of at
    least 1e-5. A mono clip is both channels of a stereo VAE."""
    audio = waveform.float()
    if rate != target:
        audio = torchaudio.functional.resample(audio, rate, target)
    audio = (audio.expand(channels, -1) if audio.shape[0] < channels else audio[:channels]).unsqueeze(0)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target, n_fft=n_fft, win_length=n_fft, hop_length=hop, f_min=0.0, f_max=target / 2.0, n_mels=mels,
        window_fn=torch.hann_window, center=True, pad_mode="reflect", power=1.0, mel_scale="slaney", norm="slaney",
    ).to(audio.device)
    mel = torch.log(torch.clamp(transform(audio), min=1e-5))
    return mel.permute(0, 1, 3, 2).contiguous()


def resample(waveform: torch.Tensor, rate: int, target: int) -> torch.Tensor:
    return waveform if rate == target else torchaudio.functional.resample(waveform.float(), rate, target)
