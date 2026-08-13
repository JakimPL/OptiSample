from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from optisample.artifacts.instruments.normalize import normalized_recording, recording_instrument
from optisample.artifacts.instruments.sources import RecordingSource, dataset_recordings
from optisample.artifacts.paths import instrument_files_dir
from optisample.config.codec import EncodeConfig
from optisample.config.tracker import TrackerFormat
from optisample.io.audio import mono, read_wav
from optisample.io.dataset import SourceDataset
from optisample.io.tracker.envelope import EnvelopeGrid, envelope_grid
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.base import Signal
from optisample.progress import ProgressSink

INSTRUMENT_LABEL: Final = "Writing instruments"

_EVERY_FORMAT: Final = tuple(TrackerFormat)  # the formats a written recording is carried as, which is all of them
_NOTHING_STATED: Final = ()


@dataclass(frozen=True)
class Recording:
    """One recording an instrument is written from: what it is called, what it plays, and the audio itself.

    ``name`` is the stem every file about this recording shares, so an instrument stands beside the WAV it
    was written from and the two read as one recording in two forms.
    """

    name: str
    root_pitch: int
    sample_rate: int
    signal: Signal


@dataclass(frozen=True)
class InstrumentSettings:
    """What writing recordings as standalone instruments is carried out with.

    ``target`` supplies what every format's files are held to -- the compliance level and each format's
    own settings -- while the formats written are all of them, so a dataset carries its recordings as
    ``.iti`` and ``.xi`` alike. ``release_s`` is how long a released note takes to fall silent, which is
    the one part of the curve a recording never states, and ``configured_tempo_bpm`` is the clock the
    export plays at, which the recordings recorded away from a clock of their own are counted in ticks of.
    """

    encode: EncodeConfig
    target: ExportTarget
    release_s: float
    configured_tempo_bpm: float
    progress: ProgressSink

    def tempo_bpm(self, recorded: float | None) -> float:
        """The clock a set of recordings is written on: the one they were recorded at, where they state it."""
        if recorded is None:
            return self.configured_tempo_bpm

        return recorded


@dataclass(frozen=True)
class WrittenInstruments:
    """What writing one set of recordings as instruments produced, and what any format had no room for.

    ``understated`` names the recordings a format sounds under the level of the audio they were written
    from, which is what audio already filling its headroom asks for once its level is flattened into the
    envelope (:func:`~optisample.artifacts.instruments.normalize.stored_level`) -- a recording peaking
    within a decibel or two of full scale, and every waveform a plan already stored as hot as its depth
    allows. ``unreachable`` names the ones played at a pitch some format's keyboard leaves to the formats
    that reach it, so a set of files states for itself which recordings it carries in every format.
    """

    directories: tuple[Path, ...]
    files: int
    understated: tuple[str, ...]
    unreachable: tuple[str, ...]


@dataclass(frozen=True)
class _WrittenFormat:
    """One format a recording is carried as: what it is held to, the grid its curve is written on, and where.

    The grid is settled once for a whole set of recordings, since instruments written beside each other
    are played on one clock, and the directory follows from the extension the format writes.
    """

    target: ExportTarget
    grid: EnvelopeGrid
    directory: Path


@dataclass(frozen=True)
class _WrittenRecording:
    """What one recording landed as: the files written for it, and what a format had no room to state."""

    files: int
    understated: tuple[str, ...]
    unreachable: tuple[str, ...]


def _formats(samples_dir: Path, *, settings: InstrumentSettings, tempo_bpm: float) -> tuple[_WrittenFormat, ...]:
    """Every format a set of recordings is written as, each on the grid its own limits state.

    Both formats measure envelope time in ticks of the module's clock, so each is given the tempo the
    material was played at and the curves written for it hold at the rate that clock runs.
    """
    return tuple(
        _WrittenFormat(
            target=target,
            grid=envelope_grid(target, tempo=target.tempo(tempo_bpm), release_s=settings.release_s),
            directory=instrument_files_dir(samples_dir, target.instrument_extension),
        )
        for target in (replace(settings.target, format=written) for written in _EVERY_FORMAT)
    )


def _write_recording(
    recording: Recording,
    formats: Sequence[_WrittenFormat],
    encode: EncodeConfig,
) -> _WrittenRecording:
    """Write one recording as every format naming the key it was played at, answering what landed.

    The level is read off the recording once and every format is written from that one reading, since the
    split a curve is fitted to belongs to the recording while the corners it is fitted onto belong to the
    format.
    """
    reachable = [written for written in formats if written.target.names(recording.root_pitch)]
    normalized = normalized_recording(
        recording.signal,
        recording.sample_rate,
        name=recording.name,
        root_pitch=recording.root_pitch,
        encode=encode,
    )
    understated = False
    for written in reachable:
        instrument = recording_instrument(
            normalized,
            target=written.target,
            grid=written.grid,
            headroom_db=encode.headroom_db,
        )
        instrument_file = written.target.instrument_file(instrument.unit)
        instrument_file.save(written.directory / f"{recording.name}{written.target.instrument_extension}")
        understated = understated or not instrument.level.restored

    return _WrittenRecording(
        files=len(reachable),
        understated=(recording.name,) if understated else _NOTHING_STATED,
        unreachable=(recording.name,) if len(reachable) < len(formats) else _NOTHING_STATED,
    )


def _written(
    recordings: Iterable[Recording],
    formats: Sequence[_WrittenFormat],
    *,
    settings: InstrumentSettings,
    total: int,
) -> WrittenInstruments:
    """Write every recording as each format holds it, gathering what landed into one reading."""
    for written in formats:
        written.directory.mkdir(parents=True, exist_ok=True)

    landed = [
        _write_recording(recording, formats, settings.encode)
        for recording in settings.progress.track(recordings, label=INSTRUMENT_LABEL, total=total)
    ]
    return WrittenInstruments(
        directories=tuple(written.directory for written in formats),
        files=sum(recording.files for recording in landed),
        understated=tuple(name for recording in landed for name in recording.understated),
        unreachable=tuple(name for recording in landed for name in recording.unreachable),
    )


def write_instruments(
    recordings: Sequence[Recording],
    samples_dir: Path,
    *,
    settings: InstrumentSettings,
    recorded_tempo_bpm: float | None,
) -> WrittenInstruments:
    """Write every recording as a standalone instrument of each format, under ``samples_dir``.

    Each instrument holds the whole recording as one sample, played down by the volume envelope its own
    level was fitted to, so a player loading one sounds the recording the way it was captured -- at the
    amplitude it was captured at, and reaching five octaves either way of the key it was played on. The
    files land in a directory per format (:func:`~optisample.artifacts.paths.instrument_files_dir`), so
    the recordings and the instruments written from them read as one set.

    ``recorded_tempo_bpm`` is the clock the material was played at, which the envelopes are counted in
    ticks of; a set recorded away from a clock is written on the one the export plays at.
    """
    return _written(
        recordings,
        _formats(samples_dir, settings=settings, tempo_bpm=settings.tempo_bpm(recorded_tempo_bpm)),
        settings=settings,
        total=len(recordings),
    )


def _read_recordings(sources: Iterable[RecordingSource]) -> Iterator[Recording]:
    """Each source read off disk as the recording an instrument is written from, one file at a time.

    A recording captured on several channels is answered as the one waveform a tracker voice sounds
    (:func:`~optisample.io.audio.mono`), which is what the stages reading these datasets already do.
    """
    for source in sources:
        signal, sample_rate = read_wav(source.file)
        yield Recording(
            name=source.name,
            root_pitch=source.root_pitch,
            sample_rate=sample_rate,
            signal=mono(signal),
        )


def write_dataset_instruments(source: SourceDataset, *, settings: InstrumentSettings) -> WrittenInstruments:
    """Write every recording of a dataset as a standalone instrument of each format, beside its own WAVs.

    The dataset states which pitch each recording was played at and the clock the material was played on
    (:func:`~optisample.artifacts.instruments.sources.dataset_recordings`), so a tree written by any stage
    is filled in from what it already holds and a dataset carried off on its own keeps saying what its
    instruments were written from.
    """
    dataset = dataset_recordings(source)
    samples_dir = source.recordings_dir
    return _written(
        _read_recordings(dataset.sources),
        _formats(samples_dir, settings=settings, tempo_bpm=settings.tempo_bpm(dataset.tempo_bpm)),
        settings=settings,
        total=len(dataset.sources),
    )
