from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from optisample.config.loop import LoopConfig
from optisample.config.reduce import ReduceConfig, TrimConfig
from optisample.dsp.resample import resample_to
from optisample.io.audio import mono, read_wav
from optisample.keys import SampleKey
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, SourceSample
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
class DecodedRecordings:
    """Every recording a decode was handed, in the order they were listed, brought to one common rate.

    ``signals`` holds one entry per recording: the trimmed span it sounds over, or nothing where its peak
    stayed under the silence floor -- so a caller reads which of its own recordings the screen left out
    straight off the position. ``sample_rate`` is the first recording's own rate, which every later one is
    resampled to and which the trim measures its length bound against.
    """

    signals: tuple[Signal | None, ...]
    sample_rate: int


@dataclass(frozen=True)
class _Decoded:
    """The survivors that carried signal, the identities that carried none, and their common rate."""

    audio: dict[SampleKey, Signal]
    silenced: tuple[SampleKey, ...]
    sample_rate: int


def _read_onset_aligned(sample: SourceSample) -> tuple[Signal, int]:
    """Decode one recording to mono over the span its note sounds, so frame 0 lands on the note onset.

    The lead-in comes off the front and the release padding off the end, leaving the recording between
    the onset and the release end -- the span the material is scored against and the encoder is priced on.
    """
    data, rate = read_wav(sample.file)
    onset_frame = round(sample.lead_in_s * rate)
    release_frame = len(data) - round(sample.trail_out_s * rate)
    return mono(data[onset_frame:release_frame]), rate


def _admitted(data: Signal, rate: int, *, sample_rate: int, trim: TrimConfig) -> Signal | None:
    """The span of one decoded recording worth keeping, at ``sample_rate``, where it carries signal.

    A recording whose peak stays under the silence floor answers with nothing, which is what names the
    slot it leaves open.
    """
    brought = data if rate == sample_rate else resample_to(data, rate, sample_rate)
    if carries_signal(brought, trim.silence_floor):
        return trim_recording(brought, sample_rate, trim)

    return None


def decode_recordings(
    samples: Sequence[SourceSample],
    trim: TrimConfig,
    progress: ProgressSink,
    *,
    label: str,
) -> DecodedRecordings:
    """Decode every recording of ``samples`` at one common rate, keeping the trimmed span each carries.

    The first listed recording's rate sets the common rate every later one is resampled to, so a set of
    recordings answers on one grid whatever each was captured at. This is the treatment a run gives the
    survivors it stores, and reading a whole stage's recordings through it puts them on those same terms.
    """
    signals: list[Signal | None] = []
    sample_rate = _UNSET_RATE
    for sample in progress.track(samples, label=label, total=len(samples)):
        data, rate = _read_onset_aligned(sample)
        if sample_rate == _UNSET_RATE:
            sample_rate = rate

        signals.append(_admitted(data, rate, sample_rate=sample_rate, trim=trim))

    return DecodedRecordings(signals=tuple(signals), sample_rate=sample_rate)


def _decode_survivors(
    survivors: Sequence[Selection],
    trim: TrimConfig,
    progress: ProgressSink,
) -> _Decoded:
    """Decode every survivor at one common rate, keeping the trimmed span of each that carries signal.

    Survivors arrive in key order, so the first one's rate is the common rate the whole map stands at. A
    recording whose peak stays under the silence floor is named apart instead, so the caller reports the
    slot it left.
    """
    decoded = decode_recordings(
        [selection.sample for selection in survivors],
        trim,
        progress,
        label=_DECODE_LABEL,
    )
    audio: dict[SampleKey, Signal] = {}
    silenced: list[SampleKey] = []
    for selection, signal in zip(survivors, decoded.signals):
        if signal is None:
            silenced.append(selection.key)
        else:
            audio[selection.key] = signal

    return _Decoded(audio=audio, silenced=tuple(silenced), sample_rate=decoded.sample_rate)


def load_instrument_audio(
    instrument: InstrumentSpec,
    reduce: ReduceConfig,
    loop: LoopConfig,
    progress: ProgressSink,
) -> LoadedInstrument:
    """Read the surviving recording of each identity into a ``SampleKey -> signal`` map at a common rate.

    A tracker stores one sample per key, so :func:`~optisample.optimize.reduce.dedupe.select_recordings`
    reduces the recorded grid to one survivor per identity from the WAV headers first, and only those
    survivors are decoded here. Each is kept over the span that carries content, up to the trim's length
    bound; one that never rises above the silence floor is left out, and the material at any pitch that
    strips of every recording leaves with it.
    """
    decoded = _decode_survivors(select_recordings(instrument, reduce, loop, progress), reduce.trim, progress)
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
    return load_instrument_audio(instrument, settings.reduce, settings.loop, settings.progress)
