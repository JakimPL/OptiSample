from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisample.artifacts.documents.sample import ProvenanceRecord, sample_document
from optisample.artifacts.instruments.dump import InstrumentSettings, write_dataset_instruments, write_instruments
from optisample.artifacts.instruments.normalize import Recording
from optisample.artifacts.paths import SAMPLE_EXTENSION, instrument_files_dir
from optisample.artifacts.serialize import write_msgpack
from optisample.config.codec import EncodeConfig
from optisample.config.tracker import TrackerFormat
from optisample.dsp.envelope import decompose, level_reading
from optisample.dsp.level import Clock, level_readings, peak_amplitude, unit_level
from optisample.dsp.loop import Loop, LoopQuality
from optisample.dsp.surrogate import SettledLoop
from optisample.io.audio import mono, read_wav, write_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import NO_TEMPO, NoteRecord, dump_notes
from optisample.io.tracker.envelope import played_gain
from optisample.io.tracker.target import ExportTarget
from optisample.keys import SampleKey
from optisample.loop.settle import StoredLoop
from optisample.metrics.base import Signal
from optisample.music import midi_to_freq
from optisample.progress import NO_PROGRESS
from tests.artifacts.instruments.conftest import ROOT_PITCH, SR, TEMPO_BPM, Recorder
from trackmod.core.instruments.unit import InstrumentUnit
from trackmod.limits.compliance import Compliance
from trackmod.spec.levels import MAX_VOLUME
from trackmod.trackers.it.instrument_file import ITInstrumentFile
from trackmod.trackers.xm.instrument_file import XMInstrumentFile

NAME = "0000_p060_C4_v100"
CONFIGURED_TEMPO_BPM = 125.0
_READING_WINDOW_S = 0.05
_PLAYED_BACK_TOLERANCE_DB = -60.0  # how far under the recording its own reconstruction error stays
_ABOVE_EVERY_XM_KEY = 108  # a piano's top note, which FastTracker 2 leaves to the formats reaching it
_ONE_FORMAT = 1


@pytest.fixture
def settings(target: ExportTarget, encode_config: EncodeConfig, release_s: float) -> InstrumentSettings:
    """What these recordings are carried as instruments with, on the clock the export config plays at."""
    return InstrumentSettings(
        encode=encode_config,
        target=target,
        release_s=release_s,
        configured_tempo_bpm=CONFIGURED_TEMPO_BPM,
        progress=NO_PROGRESS,
    )


@pytest.fixture
def recording(recorded: Recorder, sample_rate: int) -> Recording:
    """One struck note as the recording an instrument is written from."""
    return Recording(name=NAME, root_pitch=ROOT_PITCH, sample_rate=sample_rate, signal=recorded(), loop=None)


def _read_back(path: Path, tracker_format: TrackerFormat) -> InstrumentUnit:
    """One written instrument as a player reads it: the unit its file holds, parsed off disk."""
    data = path.read_bytes()
    if tracker_format == TrackerFormat.IT:
        return ITInstrumentFile.parse(data, compliance=Compliance.CANONICAL).unit

    return XMInstrumentFile.parse(data, compliance=Compliance.CANONICAL).unit


def _delivered(unit: InstrumentUnit, *, tempo: int, sample_rate: int, level: int) -> Signal:
    """What a player puts out for a held note at this instrument's own key, over the whole waveform.

    The pitch a key sounds settles the rate the stored waveform plays at, so the recording's own rate is
    what the root key produces -- which FastTracker 2 states as a transposition rather than as a rate.
    """
    sample = unit.samples[0]
    gain = played_gain(
        unit.instrument.volume_envelope,
        tempo=tempo,
        frames=int(sample.pcm.size),
        sample_rate=sample_rate,
    )
    return np.asarray(sample.pcm * gain * (level / MAX_VOLUME), dtype=np.float64)


def _quietest_gap_db(played: Signal, recording: Recording) -> float:
    """How far the reconstruction stands from the recording, in decibels against the recording's own level."""
    scale = peak_amplitude(recording.signal) / peak_amplitude(played)
    truth = level_readings(recording.signal, recording.sample_rate, window_s=_READING_WINDOW_S)
    error = level_readings(played * scale - recording.signal, recording.sample_rate, window_s=_READING_WINDOW_S)
    return float(np.max(error.values - truth.values))


def _written_file(samples_dir: Path, tracker_format: TrackerFormat, name: str) -> Path:
    """Where one recording's instrument of ``tracker_format`` lands, the way a player browsing finds it."""
    extension = ".iti" if tracker_format == TrackerFormat.IT else ".xi"
    return instrument_files_dir(samples_dir, extension) / f"{name}{extension}"


def _ticks(samples_dir: Path, name: str) -> tuple[int, ...]:
    """The ticks one written curve turns on, which is the clock it was counted in."""
    unit = _read_back(_written_file(samples_dir, TrackerFormat.IT, name), TrackerFormat.IT)
    return tuple(point.tick for point in unit.instrument.volume_envelope.points)


# --- what a set of recordings lands as ---------------------------------------------------------------


def test_every_recording_is_carried_as_an_instrument_of_each_format(
    recording: Recording, settings: InstrumentSettings, tmp_path: Path
) -> None:
    """The recordings and the instruments written from them read as one set, filed by the extension each has."""
    written = write_instruments([recording], tmp_path, settings=settings, recorded_tempo_bpm=NO_TEMPO)

    assert written.files == len(TrackerFormat)
    assert [directory.name for directory in written.directories] == ["ITI", "XI"]
    for tracker_format in TrackerFormat:
        assert _written_file(tmp_path, tracker_format, NAME).is_file()


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_a_written_instrument_sounds_the_recording_it_was_written_from(
    tracker_format: TrackerFormat,
    recording: Recording,
    settings: InstrumentSettings,
    tmp_path: Path,
) -> None:
    """The whole point of the pair: read back off disk, waveform times envelope is the recording again."""
    write_instruments([recording], tmp_path, settings=settings, recorded_tempo_bpm=TEMPO_BPM)
    unit = _read_back(_written_file(tmp_path, tracker_format, NAME), tracker_format)
    sample = unit.samples[0]

    played = _delivered(
        unit,
        tempo=round(TEMPO_BPM),
        sample_rate=recording.sample_rate,
        level=sample.gain if tracker_format == TrackerFormat.IT else sample.volume,
    )

    assert _quietest_gap_db(played, recording) < _PLAYED_BACK_TOLERANCE_DB


def test_a_recording_played_above_a_formats_keyboard_reaches_the_formats_that_name_it(
    recorded: Recorder, sample_rate: int, settings: InstrumentSettings, tmp_path: Path
) -> None:
    """A piano's top note is one Impulse Tracker numbers, so the set states which recordings it carries whole."""
    recording = Recording(
        name=NAME,
        root_pitch=_ABOVE_EVERY_XM_KEY,
        sample_rate=sample_rate,
        signal=recorded(pitch=_ABOVE_EVERY_XM_KEY),
        loop=None,
    )

    written = write_instruments([recording], tmp_path, settings=settings, recorded_tempo_bpm=NO_TEMPO)

    assert written.files == _ONE_FORMAT
    assert written.unreachable == (NAME,)
    assert _written_file(tmp_path, TrackerFormat.IT, NAME).is_file()
    assert not _written_file(tmp_path, TrackerFormat.XM, NAME).exists()


# --- the clock the curve is counted in ---------------------------------------------------------------


def test_a_curve_is_counted_in_ticks_of_the_clock_the_material_was_played_at(
    recording: Recording, settings: InstrumentSettings, tmp_path: Path
) -> None:
    """A tick lasts ``2.5 / tempo`` seconds, so the same decline reaches further up a faster clock's ticks."""
    slow, fast = tmp_path / "slow", tmp_path / "fast"
    write_instruments([recording], slow, settings=settings, recorded_tempo_bpm=TEMPO_BPM)
    write_instruments([recording], fast, settings=settings, recorded_tempo_bpm=2.0 * TEMPO_BPM)

    assert _ticks(fast, NAME)[-1] > _ticks(slow, NAME)[-1]


def test_a_set_recorded_away_from_a_clock_is_written_on_the_one_the_export_plays_at(
    settings: InstrumentSettings,
) -> None:
    assert settings.tempo_bpm(NO_TEMPO) == CONFIGURED_TEMPO_BPM
    assert settings.tempo_bpm(TEMPO_BPM) == TEMPO_BPM


# --- a whole written dataset -------------------------------------------------------------------------


@pytest.fixture
def dataset(recorded: Recorder, tmp_path: Path) -> SourceDataset:
    """A written dataset: one recording per note, beside the manifest naming what each one plays."""
    samples_dir = tmp_path / "piano"
    samples_dir.mkdir()
    records = []
    for index, pitch in enumerate((ROOT_PITCH, ROOT_PITCH + 12)):
        write_wav(samples_dir / f"{index:04d}_p{pitch}_v100.wav", recorded(pitch=pitch), SR)
        records.append(NoteRecord(index=index, pitch=pitch, velocity=100, duration_s=1.5))

    notes_json = tmp_path / "piano.notes.json"
    dump_notes(records, notes_json, tempo_bpm=TEMPO_BPM)
    return SourceDataset(path=notes_json, samples_dir=samples_dir)


def test_a_written_dataset_is_carried_beside_the_recordings_it_holds(
    dataset: SourceDataset, settings: InstrumentSettings
) -> None:
    """A tree any stage wrote is filled in from what it already holds, which is what makes the stage playable."""
    written = write_dataset_instruments(dataset, settings=settings)

    assert written.files == 2 * len(TrackerFormat)
    assert written.unreachable == ()
    for index, pitch in enumerate((ROOT_PITCH, ROOT_PITCH + 12)):
        for tracker_format in TrackerFormat:
            assert _written_file(dataset.recordings_dir, tracker_format, f"{index:04d}_p{pitch}_v100").is_file()


@pytest.fixture
def settled_region() -> Loop:
    """The region a stage settled over each recording, which a held note wraps on."""
    return Loop(start=SR // 4, end=SR // 2)


@pytest.fixture
def calibrated(dataset: SourceDataset, encode_config: EncodeConfig, settled_region: Loop) -> SourceDataset:
    """``dataset`` with the ``.sample`` the loop stage leaves beside each WAV, each stating one offer."""
    offered = (
        StoredLoop(
            settled=SettledLoop(loop=settled_region, level=unit_level(Clock.RECORDED)),
            quality=LoopQuality(seam_step=0.4, level_drift_db=1.25, spectral_distance=3.5),
        ),
    )
    for index, wav in enumerate(sorted(dataset.recordings_dir.glob("*.wav"))):
        signal, sample_rate = read_wav(wav)
        pitch = ROOT_PITCH + 12 * index
        reading = level_reading(sample_rate, encode_config.envelope, midi_to_freq(pitch))
        write_msgpack(
            wav.with_suffix(SAMPLE_EXTENSION),
            sample_document(
                decompose(mono(signal), reading),
                offered,
                key=SampleKey(pitch, 100),
                sample_rate=sample_rate,
                provenance=ProvenanceRecord(stage="loop", instrument_id="piano", index=index, source=wav.name),
            ),
        )

    return dataset


def test_a_dataset_carrying_its_containers_wraps_each_instrument_where_the_stage_settled(
    calibrated: SourceDataset, settings: InstrumentSettings, settled_region: Loop
) -> None:
    """A stage's own decision reaches the file a player loads, so a note held past the recording sustains."""
    write_dataset_instruments(calibrated, settings=settings)

    for tracker_format in TrackerFormat:
        written = _written_file(calibrated.recordings_dir, tracker_format, f"0000_p{ROOT_PITCH}_v100")
        loop = _read_back(written, tracker_format).samples[0].loop

        assert loop is not None
        assert (loop.begin, loop.end) == (settled_region.start, settled_region.end)


def test_a_dataset_written_before_any_loop_was_settled_sounds_each_recording_end_to_end(
    dataset: SourceDataset, settings: InstrumentSettings
) -> None:
    """A tree carrying no container names no region, so its instruments hold their recordings as they stand."""
    write_dataset_instruments(dataset, settings=settings)
    written = _written_file(dataset.recordings_dir, TrackerFormat.IT, f"0000_p{ROOT_PITCH}_v100")

    assert _read_back(written, TrackerFormat.IT).samples[0].loop is None


def test_a_dataset_roots_each_instrument_at_the_pitch_its_own_recording_plays(
    dataset: SourceDataset, settings: InstrumentSettings
) -> None:
    """A key sounds the pitch it is named for, so the instrument written for a recording opens at its own key."""
    write_dataset_instruments(dataset, settings=settings)
    unit = _read_back(
        _written_file(dataset.recordings_dir, TrackerFormat.IT, f"0001_p{ROOT_PITCH + 12}_v100"), TrackerFormat.IT
    )

    assert unit.instrument.keymap[settings.target.key(ROOT_PITCH + 12).value] is not None


def test_a_directory_of_takes_is_written_on_the_clock_the_export_plays_at(
    recorded: Recorder, settings: InstrumentSettings, tmp_path: Path
) -> None:
    """A directory of takes says nothing of a song, so its curves are counted in the export's own ticks."""
    takes = tmp_path / "takes"
    takes.mkdir()
    write_wav(takes / f"p{ROOT_PITCH}_v100.wav", recorded(), SR)
    configured = tmp_path / "configured"
    write_instruments(
        [Recording(name=f"p{ROOT_PITCH}_v100", root_pitch=ROOT_PITCH, sample_rate=SR, signal=recorded(), loop=None)],
        configured,
        settings=settings,
        recorded_tempo_bpm=NO_TEMPO,
    )

    write_dataset_instruments(SourceDataset(path=takes, samples_dir=None), settings=settings)

    assert _ticks(takes, f"p{ROOT_PITCH}_v100") == _ticks(configured, f"p{ROOT_PITCH}_v100")
