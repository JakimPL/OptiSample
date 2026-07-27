from optisample.config.base import ConfigModel
from optisample.config.dynamics import DynamicsConfig


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


class LoopConfig(ConfigModel):
    """Loop-point detection: search band, periodicity/sustain gates, placement, and seam crossfade."""

    min_hz: float
    max_hz: float
    min_correlation: float
    min_periods: int
    min_loop_s: float
    attack_skip_s: float
    tail_skip_s: float
    max_estimation_s: float
    sustain_decay_ratio: float
    crossfade_s: float


class QuantizeConfig(ConfigModel):
    """Requantization shaping: how far under full scale a sample is normalized before storing.

    Storing hot spends the whole depth on the recording; ``headroom_db`` is what the dither is left to
    move in above that peak (see :func:`~optisample.dsp.quantize.headroom_peak`).
    """

    headroom_db: float


class EncodeConfig(ConfigModel):
    """What :func:`optisample.dsp.surrogate.encode` needs beyond one swept ``EncodingParams`` point.

    ``peak_reference`` is the one amplitude every clip of an instrument is normalized against, set for
    a format keeping no per-sample multiplier so the balance between its samples is carried in the PCM.
    Left unset, each clip is normalized against its own peak and the balance is restored on playback.
    """

    loop: LoopConfig
    dynamics: DynamicsConfig
    headroom_db: float
    peak_reference: float | None = None
