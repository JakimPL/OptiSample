from __future__ import annotations

from typing import Final

import numpy as np

from optisample.config.dsp import LoopConfig
from optisample.config.reduce import DedupeConfig
from optisample.dsp.resample import resample_to
from optisample.io.audio import read_wav
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.dedupe import Selection, select_recordings
from optisample.optimize.reduce.keys import SampleKey
from optisample.progress import ProgressSink

_DECODE_LABEL: Final = "Decoding survivors"


def _read_onset_aligned(selection: Selection) -> tuple[Signal, int]:
    """Decode one survivor to mono with its pre-roll dropped, so frame 0 lands on the note onset."""
    data, rate = read_wav(selection.sample.file)
    lead_in_frames = round(selection.sample.lead_in_s * rate)
    if lead_in_frames > 0:
        data = data[lead_in_frames:]

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return np.asarray(data, dtype=np.float64), rate


def load_instrument_audio(
    instrument: InstrumentSpec,
    dedupe: DedupeConfig,
    loop: LoopConfig,
    progress: ProgressSink,
) -> tuple[dict[SampleKey, Signal], int]:
    """Read the surviving recording of each identity into a ``SampleKey -> signal`` map at a common rate.

    A tracker stores one sample per key, so :func:`~optisample.optimize.reduce.dedupe.select_recordings`
    reduces the recorded grid to one survivor per identity from the WAV headers first, and only those
    survivors are decoded here. Survivors arrive in key order; the first one's rate sets the common rate
    every later recording is resampled to.
    """
    audio: dict[SampleKey, Signal] = {}
    sample_rate = 0
    survivors = select_recordings(instrument, dedupe, loop, progress)
    for selection in progress.track(survivors, label=_DECODE_LABEL, total=len(survivors)):
        data, rate = _read_onset_aligned(selection)
        if sample_rate == 0:
            sample_rate = rate
        elif rate != sample_rate:
            data = resample_to(data, rate, sample_rate)
        audio[selection.key] = data

    return audio, sample_rate


def load_run_audio(instrument: InstrumentSpec, settings: OptimizeSettings) -> tuple[dict[SampleKey, Signal], int]:
    """The recordings one run works from, read off the settings that decide which of them survive.

    Both strategies' disk entry points start here, so an instrument loaded for grouping holds exactly the
    survivors it holds for the ungrouped solver.
    """
    return load_instrument_audio(instrument, settings.reduce.dedupe, settings.encode.loop, settings.progress)
