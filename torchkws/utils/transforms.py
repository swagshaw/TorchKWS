import librosa
import torch
from torch import Tensor


class ToMelSpectrogram:
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, x: Tensor) -> Tensor:
        x = x.numpy()
        mel_spectrogram = librosa.feature.melspectrogram(
            y=x, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return torch.FloatTensor(log_mel_spectrogram)


class ToTensor:
    def __call__(self, x: Tensor) -> Tensor:
        return x.unsqueeze(0)


class Normalize:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std