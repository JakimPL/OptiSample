from optisample.config.base import ConfigModel


class StftParams(ConfigModel):
    """Short-time Fourier transform sizing."""

    n_fft: int
    hop_length: int


class MelParams(ConfigModel):
    """Mel-spectrogram sizing (carries its own STFT sizing)."""

    n_fft: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: float | None

    def stft(self) -> StftParams:
        return StftParams(n_fft=self.n_fft, hop_length=self.hop_length)


class SpectralConfig(ConfigModel):
    """General analysis params for the standalone spectral primitives (centroid/rolloff/flux/...)."""

    stft: StftParams
    mel: MelParams
    rolloff_percent: float
