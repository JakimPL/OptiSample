import numpy as np

from optisample.dsp.resample import resample_to
from optisample.io.audio import read_wav
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec


def load_instrument_audio(
    instrument: InstrumentSpec,
) -> tuple[dict[tuple[int, int], Signal], int]:
    """Read one recording per ``(pitch, velocity)`` into a ``(pitch, velocity) -> signal`` map at a common rate.

    IT stores a single sample per key, so each ``(pitch, velocity)`` collapses to one representative: the
    first sample listed for that key wins (the samples arrive in ``render.index`` order, so this keeps the
    earliest recording). Each recording's ``lead_in_s`` pre-roll is dropped from the front so frame 0 lands
    on the note onset, then stereo recordings are downmixed to mono and resampled to the first rate seen.
    """
    audio: dict[tuple[int, int], Signal] = {}
    sample_rate = 0
    for sample in instrument.samples:
        key = (sample.pitch, sample.velocity)
        if key in audio:
            continue

        data, rate = read_wav(sample.file)
        lead_in_frames = round(sample.lead_in_s * rate)
        if lead_in_frames > 0:
            data = data[lead_in_frames:]
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sample_rate == 0:
            sample_rate = rate
        elif rate != sample_rate:
            data = resample_to(data, rate, sample_rate)
        audio[key] = np.asarray(data, dtype=np.float64)

    return audio, sample_rate
