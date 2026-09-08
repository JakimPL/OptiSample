from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts.dataset import recording_stem
from optisample.artifacts.documents.loops import read_loops, settled_loops
from optisample.artifacts.documents.sample import read_sample, sample_decomposition, sample_loops
from optisample.artifacts.instruments import InstrumentSettings
from optisample.artifacts.looped import dump_looped, loop_project
from optisample.artifacts.paths import calibrated_path
from optisample.config.tracker import TrackerFormat
from optisample.io.audio import read_wav
from optisample.io.note_extractor import IngestSettings, load_notes
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample
from optisample.optimize.orchestrate.audio import LoadedInstrument
from optisample.optimize.orchestrate.looping import LoopedInstrument, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.trim import NO_SCREEN
from optisample.optimize.tasks import AudioMap

SR = 22_050
_HELD_S = 3.0
_BRIEF_SHARE = 0.5  # the share of the shortest accepted loop a recording holds to be stored over its own span
_LOOPABLE = 60
_BRIEF = 72
_RECORDING_FILE = 1  # each folder opens with the recording, then one file per loop offered against it


def _tone(freq: float, duration_s: float) -> NDArray[np.float64]:
    times = np.arange(int(duration_s * SR), dtype=np.float64) / SR
    return np.asarray(0.7 * np.sin(2.0 * np.pi * freq * times) + 0.2 * np.sin(4.0 * np.pi * freq * times))


@pytest.fixture
def brief_s(loop_room_s: float) -> float:
    """A span the shortest accepted loop outruns, so the pitch holding it is stored over what it plays."""
    return _BRIEF_SHARE * loop_room_s


@pytest.fixture
def audio(brief_s: float) -> AudioMap:
    """One recording a loop fits inside, and one too short for the shortest accepted loop."""
    return {
        SampleKey(_LOOPABLE, 100): _tone(220.0, _HELD_S),
        SampleKey(_BRIEF, 100): _tone(440.0, brief_s),
    }


@pytest.fixture
def instrument(brief_s: float) -> InstrumentSpec:
    return InstrumentSpec(
        id="pad",
        budget_kb=96.0,
        samples=[SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in (_LOOPABLE, _BRIEF)],
        material=[
            NoteEvent(pitch=_LOOPABLE, velocity=100, duration_s=_HELD_S, count=2),
            NoteEvent(pitch=_BRIEF, velocity=100, duration_s=brief_s, count=1),
        ],
    )


@pytest.fixture
def looped(
    instrument: InstrumentSpec,
    audio: AudioMap,
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., object],
) -> LoopedInstrument:
    loaded = LoadedInstrument(instrument=instrument, audio=dict(audio), sample_rate=SR, screen=NO_SCREEN)
    return run_loops(loaded, optimize_settings(sweep=sweep()))


@pytest.fixture
def settings(optimize_settings: Callable[..., OptimizeSettings], sweep: Callable[..., object]) -> OptimizeSettings:
    return optimize_settings(sweep=sweep())


# --- the dataset the stage hands on ----------------------------------------------------------------


def test_the_stage_writes_a_dataset_a_later_ingest_reads_back(
    looped: LoopedInstrument,
    settings: OptimizeSettings,
    instrument_settings: InstrumentSettings,
    tmp_path: Path,
    ingest_settings: Callable[..., IngestSettings],
) -> None:
    """The frames a loop names index into these files, which is what makes the decision travel forward."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    assert written.paths.notes_json.is_file()
    reloaded = load_notes(written.paths.notes_json, written.paths.samples_dir, ingest_settings("pad"))
    assert [instrument.id for instrument in reloaded.instruments] == ["pad"]


def test_every_recording_is_written_as_the_stage_analyzed_it(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    assert written.recordings == len(looped.loaded.audio)
    for key in sorted(looped.loaded.audio):
        index = sorted(looped.loaded.audio).index(key)
        data, rate = read_wav(written.paths.samples_dir / f"{index:04d}_{key.label}.wav")
        assert rate == SR
        assert data.size == looped.loaded.audio[key].size


def test_every_recording_is_carried_as_a_calibrated_sample_beside_its_own_wav(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    """The pair a container holds puts the recording back together, which is what carries it into a stored copy."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    for index, key in enumerate(sorted(looped.loaded.audio)):
        document = read_sample(calibrated_path(written.paths.samples_dir, recording_stem(key, index)))
        assert (document.root_pitch, document.sample_rate, document.provenance.index) == (key.pitch, SR, index)
        assert document.provenance.instrument_id == "pad"
        assert sample_decomposition(document).recombined() == pytest.approx(looped.loaded.audio[key], abs=1e-6)


def test_the_loops_a_recording_offers_travel_in_the_container_holding_its_audio(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    """A stage reading a container back reaches the loops this run settled over the very audio beside them."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)
    key = SampleKey(_LOOPABLE, 100)
    index = sorted(looped.loaded.audio).index(key)

    assert (
        sample_loops(read_sample(calibrated_path(written.paths.samples_dir, recording_stem(key, index))))
        == looped.settled[key]
    )


def test_a_run_storing_no_loops_carries_every_recording_as_a_container_stating_none(
    instrument: InstrumentSpec,
    audio: AudioMap,
    settings: OptimizeSettings,
    instrument_settings: InstrumentSettings,
    tmp_path: Path,
) -> None:
    """The split is worth carrying wherever a run stores its samples, so the container states an empty offer."""
    loaded = LoadedInstrument(instrument=instrument, audio=dict(audio), sample_rate=SR, screen=NO_SCREEN)
    unlooped = run_loops(loaded, replace(settings, loops=False))

    written = dump_looped(unlooped, tmp_path, settings, instrument_settings)

    for index, key in enumerate(sorted(audio)):
        document = read_sample(calibrated_path(written.paths.samples_dir, recording_stem(key, index)))
        assert document.loops == []
        assert sample_decomposition(document).recombined() == pytest.approx(audio[key], abs=1e-6)


def test_the_material_reaches_the_dataset_note_for_note(
    looped: LoopedInstrument,
    settings: OptimizeSettings,
    instrument_settings: InstrumentSettings,
    tmp_path: Path,
    ingest_settings: Callable[..., IngestSettings],
) -> None:
    """The reduction reads its material from here, so an event standing for several notes is written once each."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    reloaded = load_notes(written.paths.notes_json, written.paths.samples_dir, ingest_settings("pad"))
    played = sum(event.count for event in reloaded.instruments[0].material)
    assert played == sum(event.count for event in looped.loaded.instrument.material)


def test_every_recording_is_carried_as_a_standalone_instrument_of_each_format(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    """The stage's own audio is playable in a tracker, which is what makes one stage audible against the next."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    assert written.instruments.files == len(looped.loaded.audio) * len(TrackerFormat)
    assert written.instruments.unreachable == ()
    for index, key in enumerate(sorted(looped.loaded.audio)):
        stem = recording_stem(key, index)
        assert (written.paths.samples_dir / "ITI" / f"{stem}.iti").is_file()
        assert (written.paths.samples_dir / "XI" / f"{stem}.xi").is_file()


# --- what the stage decided ------------------------------------------------------------------------


def test_the_document_states_the_loop_each_recording_was_settled_around(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    document = read_loops(written.paths.loops_json)

    assert document.instrument_id == "pad"
    assert document.sample_rate == SR
    assert [record.pitch for record in document.recordings] == [_LOOPABLE, _BRIEF]


def test_a_recording_too_short_to_loop_is_stated_as_offering_none(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    document = read_loops(written.paths.loops_json)
    brief = next(record for record in document.recordings if record.pitch == _BRIEF)

    assert brief.offered == []
    assert written.looped == 1


def test_the_document_lists_a_recordings_offers_from_the_cheapest_stored_span_upward(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    """An encoding names a loop by its place in this list, so the order it is written in is load-bearing."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    document = read_loops(written.paths.loops_json)
    loopable = next(record for record in document.recordings if record.pitch == _LOOPABLE)

    assert len(loopable.offered) > 1
    ends = [stored.end for stored in loopable.offered]
    assert ends == sorted(ends)


def test_the_loops_a_run_settled_read_back_as_the_encoder_receives_them(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    """A document read beside the dataset it was measured over names the same stretches of the same audio."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    assert settled_loops(read_loops(written.paths.loops_json)) == dict(looped.settled)


# --- what a listener judges it on ------------------------------------------------------------------


def test_every_loop_a_recording_offers_is_auditioned_against_the_recording_it_came_from(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    """Naming each audition by the offer it holds is what lets a listener hear the length axis a sweep prices."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    offered = looped.settlements[SampleKey(_LOOPABLE, 100)].offered
    folder = written.paths.auditions_dir / SampleKey(_LOOPABLE, 100).label
    assert (folder / "recording.wav").is_file()
    assert sorted(path.name for path in folder.glob("looped*.wav")) == sorted(
        f"looped{index}.wav" for index in range(len(offered))
    )
    assert written.auditions == _RECORDING_FILE + len(offered)


def test_a_recording_stored_over_the_span_it_plays_is_auditioned_against_nothing(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    """A recording with no loop has nothing to compare against itself, so the folders present are the loops."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    assert not (written.paths.auditions_dir / SampleKey(_BRIEF, 100).label).exists()


def test_an_audition_plays_the_loop_out_past_the_span_it_stores(
    looped: LoopedInstrument, settings: OptimizeSettings, instrument_settings: InstrumentSettings, tmp_path: Path
) -> None:
    """Wrapping the loop a few times is what turns a seam step or a level pulse into a rhythm a listener hears."""
    written = dump_looped(looped, tmp_path, settings, instrument_settings)

    folder = written.paths.auditions_dir / SampleKey(_LOOPABLE, 100).label
    for index, stored in enumerate(looped.settlements[SampleKey(_LOOPABLE, 100)].offered):
        held, _ = read_wav(folder / f"looped{index}.wav")
        assert held.size > stored.loop.end


# --- a whole project ------------------------------------------------------------------------------


def test_a_project_earns_one_looped_dataset_per_instrument(
    instrument: InstrumentSpec,
    audio: AudioMap,
    settings: OptimizeSettings,
    instrument_settings: InstrumentSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Instruments share the output root, each contributing the pair a later ingest resolves by default."""
    monkeypatch.setattr(
        "optisample.artifacts.looped.load_run_audio",
        lambda spec, _settings: LoadedInstrument(instrument=spec, audio=dict(audio), sample_rate=SR, screen=NO_SCREEN),
    )
    manifest = Manifest(project=ProjectSpec(name="pads"), instruments=[instrument])

    results = loop_project(manifest, tmp_path, settings, instrument_settings)

    assert [result.instrument_id for result in results] == ["pad"]
    assert (tmp_path / "pad.notes.json").is_file()
    assert (tmp_path / "pad").is_dir()
