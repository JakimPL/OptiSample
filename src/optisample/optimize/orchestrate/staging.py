from optisample.config.dsp import EncodeConfig
from optisample.dsp.levels import peak_amplitude
from optisample.io.tracker.target import ExportTarget
from optisample.optimize.tasks import AudioMap

_SILENT_INSTRUMENT: float = 0.0  # an instrument whose every recording is silent has no peak to store against


def instrument_peak(audio: AudioMap) -> float:
    """The loudest sample of any recording the instrument keeps, which sets how hot they may be stored."""
    return max((peak_amplitude(signal) for signal in audio.values()), default=_SILENT_INSTRUMENT)


def staged_encode(encode: EncodeConfig, target: ExportTarget, audio: AudioMap) -> EncodeConfig:
    """The encode config one instrument is stored under, carrying what its normalization measures against.

    A format keeping a per-sample multiplier restores each level on playback, so every clip is stored as
    hot as its own depth allows and the balance is written beside the samples. A format without one
    carries that balance in the PCM: every clip is normalized against the loudest peak in the instrument,
    so the margins between the recordings survive into what is stored.

    The reference is fixed here, before anything is swept, because how hot a sample is stored decides how
    much quantization noise it earns -- which the objective does measure. Pinning it once means the
    bandwidth pre-pass, the sweep and the export all encode the way the written module will.
    """
    if target.stores_sample_gain:
        return encode

    return encode.model_copy(update={"peak_reference": instrument_peak(audio)})
