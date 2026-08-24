from optisample.config.base import ConfigModel
from optisample.config.spectral import MelParams, StftParams


class MrstftConfig(ConfigModel):
    """Multi-resolution STFT distance: the set of ``(n_fft, hop)`` resolutions to average over."""

    resolutions: tuple[StftParams, ...]


class LogMelConfig(ConfigModel):
    """Log-mel L1 distance sizing."""

    mel: MelParams


class McdConfig(ConfigModel):
    """Mel-cepstral distortion: coefficient count and the mel sizing behind it."""

    n_mfcc: int
    mel: MelParams


class OnsetConfig(ConfigModel):
    """The attack term: how much of an onset is scored, and the mel sizing it is read under.

    ``pre_s`` is the stretch ahead of the onset the window opens at, which is where a stored copy's own
    noise floor shows; ``span_s`` how far past it the attack is read. Between them they say how much of a
    sound's beginning the objective hears apart from the note behind it, so a longer span reads more of
    the body and a shorter one concentrates on the transient. Every other threshold the reading uses is the
    convention :data:`~optisample.dsp.onset.ATTACK_READING` states.
    """

    mel: MelParams
    pre_s: float
    span_s: float


class SpectralWeights(ConfigModel):
    """Sub-weights for the four spectral-shape terms, applied to each term's relative delta."""

    centroid: float
    rolloff: float
    flatness: float
    flux_variance: float


class SpectralShapeConfig(ConfigModel):
    """Brightness/rolloff/flatness/flux-variance term: analysis sizing, sub-weights, rolloff fraction."""

    stft: StftParams
    weights: SpectralWeights
    rolloff_percent: float


class SegmentalSnrConfig(ConfigModel):
    """Segmental-SNR diagnostic: per-frame framing and the dB clip range."""

    frame_length: int
    hop_length: int
    clip_low_db: float
    clip_high_db: float


class PreprocessConfig(ConfigModel):
    """Shared comparison settings: loudness target, log dynamic-range floor, segmental-SNR framing."""

    target_lufs: float
    dynamic_range_db: float
    segmental: SegmentalSnrConfig


class MetricsConfig(ConfigModel):
    """The composite fidelity: term weights plus each component metric's configuration."""

    weights: dict[str, float]
    mrstft: MrstftConfig
    logmel: LogMelConfig
    mcd: McdConfig
    spectral_shape: SpectralShapeConfig
    onset: OnsetConfig
    preprocess: PreprocessConfig
