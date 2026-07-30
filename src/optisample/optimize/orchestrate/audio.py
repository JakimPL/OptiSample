from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.config.loop import GeometryConfig
from optisample.config.reduce import ReduceConfig, TrimConfig
from optisample.dsp.resample import resample_to
from optisample.io.audio import read_wav
from optisample.keys import SampleKey
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.dedupe import Selection, select_recordings
from optisample.optimize.reduce.trim import (
    RecordingScreen,
    carries_signal,
    screen_instrument,
    trim_recording,
)
from optisample.progress import ProgressSink

_DECODE_LABEL: Final = "Decoding survivors"
_UNSET_RATE: Final = 0  # the common rate stands open until the first survivor states its own


@dataclass(frozen=True)
class LoadedInstrument:
    """One instrument as a run receives it: the recordings it holds, and the material they can serve.

    ``instrument`` is narrowed to the notes a kept recording answers for, so the tasks a run builds cover
    exactly what the dataset holds. ``screen`` states what admitting the recordings cost, which is how a
    caller reports the recordings left out and the material that went with them.
    """

    instrument: InstrumentSpec
    audio: dict[SampleKey, Signal]
    sample_rate: int
    screen: RecordingScreen


@dataclass(frozen=True)
class _Decoded:
    """The survivors that carried signal, the identities that carried none, and their common rate."""

    audio: dict[SampleKey, Signal]
    silenced: tuple[SampleKey, ...]
    sample_rate: int


def _read_onset_aligned(selection: Selection) -> tuple[Signal, int]:
    """Decode one survivor to mono over the span its note sounds, so frame 0 lands on the note onset.

    The lead-in comes off the front and the release padding off the end, leaving the recording between
    the onset and the release end -- the span the material is scored against and the encoder is priced on.
    """
    data, rate = read_wav(selection.sample.file)
    onset_frame = round(selection.sample.lead_in_s * rate)
    release_frame = len(data) - round(selection.sample.trail_out_s * rate)
    data = data[onset_frame:release_frame]

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return np.asarray(data, dtype=np.float64), rate


def _decode_survivors(
    survivors: Sequence[Selection],
    trim: TrimConfig,
    progress: ProgressSink,
) -> _Decoded:
    """Decode every survivor at one common rate, keeping the trimmed span of each that carries signal.

    Survivors arrive in key order; the first one's rate sets the common rate every later recording is
    resampled to, which is also the rate the trim measures its length bound against. A recording whose
    peak stays under the silence floor is named apart instead, so the caller reports the slot it left.
    """
    audio: dict[SampleKey, Signal] = {}
    silenced: list[SampleKey] = []
    sample_rate = _UNSET_RATE
    for selection in progress.track(survivors, label=_DECODE_LABEL, total=len(survivors)):
        data, rate = _read_onset_aligned(selection)
        if sample_rate == _UNSET_RATE:
            sample_rate = rate
        elif rate != sample_rate:
            data = resample_to(data, rate, sample_rate)

        if carries_signal(data, trim.silence_floor):
            audio[selection.key] = trim_recording(data, sample_rate, trim)
        else:
            silenced.append(selection.key)

    return _Decoded(audio=audio, silenced=tuple(silenced), sample_rate=sample_rate)


def load_instrument_audio(
    instrument: InstrumentSpec,
    reduce: ReduceConfig,
    geometry: GeometryConfig,
    progress: ProgressSink,
) -> LoadedInstrument:
    """Read the surviving recording of each identity into a ``SampleKey -> signal`` map at a common rate.

    A tracker stores one sample per key, so :func:`~optisample.optimize.reduce.dedupe.select_recordings`
    reduces the recorded grid to one survivor per identity from the WAV headers first, and only those
    survivors are decoded here. Each is kept over the span that carries content, up to the trim's length
    bound; one that never rises above the silence floor is left out, and the material at any pitch that
    strips of every recording leaves with it.
    """
    decoded = _decode_survivors(select_recordings(instrument, reduce, geometry, progress), reduce.trim, progress)
    screened = screen_instrument(instrument, {key.pitch for key in decoded.audio}, decoded.silenced)
    return LoadedInstrument(
        instrument=screened.instrument,
        audio=decoded.audio,
        sample_rate=decoded.sample_rate,
        screen=screened.screen,
    )


def load_run_audio(instrument: InstrumentSpec, settings: OptimizeSettings) -> LoadedInstrument:
    """The recordings one run works from, read off the settings that decide which of them survive.

    Both strategies' disk entry points start here, so an instrument loaded for grouping holds exactly the
    survivors it holds for the ungrouped solver.
    """
    return load_instrument_audio(instrument, settings.reduce, settings.loop.geometry, settings.progress)
