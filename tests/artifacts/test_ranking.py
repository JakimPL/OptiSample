from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts.ranking import (
    ListeningSet,
    dump_ranking,
    pair_clips,
    ranking_project,
    read_label_sheet,
    read_ranking_set,
    write_label_sheet,
)
from optisample.calibrate.ranking import (
    PairQuota,
    RankingGrid,
    RankingSettings,
    Side,
    Verdict,
    read_labels,
    settled,
)
from optisample.config.optimize import SweepConfig
from optisample.io.audio import read_wav
from optisample.keys import SampleKey
from optisample.model import (
    InstrumentSpec,
    Manifest,
    NoteEvent,
    ProjectSpec,
    SourceSample,
)
from optisample.optimize.orchestrate.audio import LoadedInstrument
from optisample.optimize.orchestrate.looping import LoopedInstrument, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.trim import NO_SCREEN
from optisample.optimize.tasks import AudioMap, StoredRecordings

Recordings = Callable[..., StoredRecordings]

SR = 44_100
PITCHES = (60, 62)
_SEED = 137
_LADDER = (11_025, 16_000, 22_050)
_PAIR_FILES = 3  # the recording and the two sides, which is the whole of what a listener meets
_NOTE_S = 0.5
_REPEATED = 1


@pytest.fixture
def listening_audio(piano_note: Callable[..., NDArray[np.float64]]) -> AudioMap:
    return {SampleKey(pitch, 100): piano_note(pitch, 100, 0.6, seed=pitch * 137) for pitch in PITCHES}


@pytest.fixture
def listening_instrument() -> InstrumentSpec:
    return InstrumentSpec(
        id="piano",
        budget_kb=48.0,
        samples=[SourceSample(file=f"{pitch}.wav", pitch=pitch, velocity=100) for pitch in PITCHES],
        material=[NoteEvent(pitch=pitch, velocity=100, duration_s=_NOTE_S, count=1) for pitch in PITCHES],
    )


@pytest.fixture
def listening_settings(
    sweep: Callable[..., SweepConfig],
    optimize_settings: Callable[..., OptimizeSettings],
) -> OptimizeSettings:
    return optimize_settings(sweep=sweep(rates=_LADDER, depth=16), seed=_SEED)


@pytest.fixture
def ranking_settings() -> RankingSettings:
    return RankingSettings(
        grid=RankingGrid(depths=(16, 8), rate_steps=1),
        quota=PairQuota(loop=1, rate=1, depth=1, compress=0, trade=1),
        byte_tolerance=0.15,
        min_duration_s=_NOTE_S / 2,
        repeats=_REPEATED,
        seed=_SEED,
    )


@pytest.fixture
def looped(
    listening_instrument: InstrumentSpec,
    listening_audio: AudioMap,
    listening_settings: OptimizeSettings,
) -> LoopedInstrument:
    loaded = LoadedInstrument(
        instrument=listening_instrument,
        audio=dict(listening_audio),
        sample_rate=SR,
        screen=NO_SCREEN,
    )
    return run_loops(loaded, listening_settings)


@pytest.fixture
def written(
    looped: LoopedInstrument,
    listening_settings: OptimizeSettings,
    ranking_settings: RankingSettings,
    tmp_path: Path,
) -> ListeningSet:
    return dump_ranking(looped, tmp_path, listening_settings, ranking_settings)


def _manifest(written: ListeningSet) -> dict:
    return json.loads(written.paths.manifest_json.read_text(encoding="utf-8"))


def test_every_pair_lands_in_a_directory_of_its_own(written: ListeningSet) -> None:
    stated = [record["directory"] for record in _manifest(written)["pairs"]]

    assert all((written.paths.pairs_dir / directory).is_dir() for directory in stated)


def test_a_pair_holds_the_recording_and_the_two_sides(written: ListeningSet) -> None:
    directory = written.paths.pairs_dir / _manifest(written)["pairs"][0]["directory"]

    written_files = sorted(path.name for path in directory.glob("*.wav"))
    assert written_files == [f"{Side.A}.wav", f"{Side.B}.wav", "reference.wav"]
    assert len(written_files) == _PAIR_FILES


def test_the_two_sides_are_heard_over_the_same_stretch_as_the_recording(written: ListeningSet) -> None:
    directory = written.paths.pairs_dir / _manifest(written)["pairs"][0]["directory"]

    lengths = {read_wav(directory / f"{stem}.wav")[0].size for stem in ("reference", Side.A, Side.B)}
    assert len(lengths) == 1


def test_the_two_sides_are_different_audio(written: ListeningSet) -> None:
    directory = written.paths.pairs_dir / _manifest(written)["pairs"][0]["directory"]

    first, _ = read_wav(directory / f"{Side.A}.wav")
    second, _ = read_wav(directory / f"{Side.B}.wav")
    assert not np.array_equal(first, second)


def test_the_audio_is_written_at_the_rate_the_run_measures_at(written: ListeningSet) -> None:
    manifest = _manifest(written)
    directory = written.paths.pairs_dir / manifest["pairs"][0]["directory"]

    assert read_wav(directory / "reference.wav")[1] == manifest["sample_rate"]


def test_the_answer_sheet_holds_one_open_row_per_pair(written: ListeningSet) -> None:
    sheet = read_labels(written.paths.labels_csv.read_text(encoding="utf-8"))

    stated = [record["directory"] for record in _manifest(written)["pairs"]]
    assert [label.directory for label in sheet.labels] == stated
    assert sheet.outstanding == len(stated)


def test_the_set_explains_itself_beside_the_audio(written: ListeningSet) -> None:
    readme = written.paths.readme.read_text(encoding="utf-8")

    assert Verdict.TIE in readme and "reference.wav" in readme


def test_the_set_states_how_much_listening_it_asks_for(written: ListeningSet) -> None:
    assert written.pairs == len(_manifest(written)["pairs"])


def test_a_repeated_question_is_written_as_a_pair_of_its_own(written: ListeningSet) -> None:
    asked = [record["question_id"] for record in _manifest(written)["pairs"]]

    assert written.repeats == _REPEATED
    assert len(asked) - len(set(asked)) == _REPEATED


def test_the_set_states_what_its_pairs_were_chosen_from(written: ListeningSet) -> None:
    assert written.priced == _manifest(written)["priced_encodings"] > written.pairs


def test_the_written_manifest_reads_back_as_the_set_it_decodes(written: ListeningSet) -> None:
    document = read_ranking_set(written.paths)

    assert [record.directory for record in document.pairs] == [
        record["directory"] for record in _manifest(written)["pairs"]
    ]


def test_a_question_hands_over_the_recording_and_the_two_sides(written: ListeningSet) -> None:
    clips = pair_clips(written.paths, read_ranking_set(written.paths).pairs[0].directory)

    assert [path.is_file() for path in (clips.reference, clips.first, clips.second)] == [True] * 3


def test_an_answered_sheet_is_put_back_beside_the_set_it_answers(written: ListeningSet) -> None:
    sheet = read_label_sheet(written.paths)
    answered = settled(sheet, sheet.labels[0].directory, verdict=Verdict.A_CLEARLY, fault=None, note="")

    write_label_sheet(answered, written.paths)

    assert read_label_sheet(written.paths) == answered


def test_a_project_writes_one_set_per_instrument(
    listening_instrument: InstrumentSpec,
    listening_audio: AudioMap,
    recordings: Recordings,
    listening_settings: OptimizeSettings,
    ranking_settings: RankingSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = recordings(listening_audio, SR)
    monkeypatch.setattr(
        "optisample.artifacts.ranking.load_run_audio",
        lambda instrument, settings: LoadedInstrument(
            instrument=instrument, audio=dict(stored.audio), sample_rate=SR, screen=NO_SCREEN
        ),
    )
    manifest = Manifest(project=ProjectSpec(name="piano"), instruments=[listening_instrument])

    sets = ranking_project(manifest, tmp_path, listening_settings, ranking_settings)

    assert [one.instrument_id for one in sets] == ["piano"]
    assert sets[0].paths.manifest_json.is_file()
