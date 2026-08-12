from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisample.artifacts.paths import pipeline_paths
from optisample.cluster.representative import Group, grouping
from optisample.cluster.selection import Selection, selection, write_selection
from optisample.cluster.space import Coordinates
from optisample.cluster.stages import (
    Stage,
    StageCorpus,
    StageSettings,
    stage_dataset,
    stage_recordings,
)
from optisample.config import OptiConfig
from optisample.config.cluster import Representative
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NOTES_SUFFIX, NoteRecord, dump_notes
from optisample.music import midi_to_freq
from optisample.progress import NO_PROGRESS

SR = 8_000

_INSTRUMENT = "Piano"
_STRATEGY = "grouped"
_PITCHES = (48, 50, 52, 72, 74, 76)  # six takes, the lower three apart from the upper three
_VELOCITY = 100
_TAKE_S = 0.5
_HELD_S = 0.25  # how long one note sounds, which is the playing time it gives the recording answering it
_APART = 10.0  # the room between the two sets of coordinates, wide enough that a cut lands between them
_GROUPS = 2
_TO_THE_RELEASE = False


def _take(pitch: int) -> np.ndarray:
    """A recording of one note, loud enough to clear the silence floor a run screens recordings by."""
    times = np.arange(round(_TAKE_S * SR), dtype=np.float64) / SR
    return np.asarray(0.5 * np.sin(2.0 * np.pi * midi_to_freq(pitch) * times), dtype=np.float64)


def _notes() -> list[NoteRecord]:
    """One more note for each recording than the one before it, so every take earns a say of its own."""
    return [
        NoteRecord(index=index, pitch=pitch, velocity=_VELOCITY, duration_s=_HELD_S)
        for index, pitch in enumerate(_PITCHES)
        for _ in range(index + 1)
    ]


@pytest.fixture
def settings(config: OptiConfig) -> StageSettings:
    """What a reading of a stage is carried out with, on the terms the pipeline's own stages read by."""
    return StageSettings(
        instrument_id=_INSTRUMENT,
        strategy=_STRATEGY,
        dedupe=config.reduce.dedupe,
        trim=config.reduce.trim,
        keep_tail=_TO_THE_RELEASE,
        progress=NO_PROGRESS,
    )


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A run holding one dataset stage: six recordings, and the notes each of them answers for."""
    directory = pipeline_paths(tmp_path).subset_dir
    samples_dir = directory / _INSTRUMENT
    samples_dir.mkdir(parents=True)
    for index, pitch in enumerate(_PITCHES):
        write_wav(samples_dir / f"{index:04d}_p{pitch:03d}_v{_VELOCITY:03d}.wav", _take(pitch), SR)

    dump_notes(_notes(), directory / f"{_INSTRUMENT}{NOTES_SUFFIX}")
    return tmp_path


@pytest.fixture
def corpus(run: Path, settings: StageSettings) -> StageCorpus:
    """The stage's recordings as a space would place them, each carrying the file it is stored in."""
    return stage_recordings(run, Stage.SUBSET, settings)


@pytest.fixture
def coordinates(corpus: StageCorpus) -> Coordinates:
    """Two sets of points standing well apart, so which recordings a group gathers is settled up front."""
    placed = np.zeros((corpus.size, 2), dtype=np.float64)
    placed[len(_PITCHES) // 2 :, 0] = _APART
    placed[:, 1] = np.arange(corpus.size, dtype=np.float64) / corpus.size
    return placed


@pytest.fixture
def groups(corpus: StageCorpus, coordinates: Coordinates) -> tuple[Group, ...]:
    """The two groups those points fall into, read for the takes standing for each."""
    labels = np.asarray([0] * (len(_PITCHES) // 2) + [1] * (len(_PITCHES) // 2), dtype=np.intp)
    weights = np.asarray([recording.weight for recording in corpus.recordings], dtype=np.float64)
    return grouping(coordinates, labels, weights)


@pytest.fixture
def chosen(corpus: StageCorpus, groups: tuple[Group, ...]) -> Selection:
    """The take standing for each group, which is what a written selection holds."""
    return selection(corpus, groups, rule=Representative.MEDOID)


def test_a_selection_holds_one_take_for_every_group(chosen: Selection, groups: tuple[Group, ...]) -> None:
    assert chosen.stage == Stage.SUBSET
    assert chosen.instrument_id == _INSTRUMENT
    assert chosen.size == _GROUPS
    assert [pick.group for pick in chosen.picks] == [group.label for group in groups]


def test_every_chosen_take_is_a_member_of_the_group_it_stands_for(chosen: Selection, groups: tuple[Group, ...]) -> None:
    """A representative is a real recording out of its own group, which is what makes it playable."""
    for pick, group in zip(chosen.picks, groups, strict=True):
        assert pick.place in group.members.tolist()
        assert pick.place == group.representative(Representative.MEDOID)


def test_a_pick_carries_the_recording_the_point_stood_for(chosen: Selection, corpus: StageCorpus) -> None:
    for pick in chosen.picks:
        assert pick.recording == corpus.recordings[pick.place]
        assert pick.render_index == int(pick.recording.file.name.split("_", 1)[0])


def test_the_picks_stand_for_the_whole_corpus_they_were_cut_from(chosen: Selection, corpus: StageCorpus) -> None:
    """A selection reports what it covers, so a small set of takes is read beside the music it answers for."""
    assert chosen.covered == corpus.size
    assert chosen.playing_s == pytest.approx(sum(recording.weight for recording in corpus.recordings))
    assert chosen.render_indices == frozenset(pick.render_index for pick in chosen.picks)


def test_a_written_selection_holds_the_notes_its_takes_answer_for(chosen: Selection, run: Path, tmp_path: Path) -> None:
    written = write_selection(stage_dataset(run, Stage.SUBSET, _INSTRUMENT), chosen, tmp_path / "selection")
    answered = sum(pick.place + 1 for pick in chosen.picks)

    assert written.selection == chosen
    assert written.dataset.recordings == _GROUPS
    assert written.dataset.kept_notes == answered
    assert written.dataset.source_notes == len(_notes())


def test_a_written_selection_is_read_back_as_the_stage_it_was_chosen_from(
    chosen: Selection, run: Path, tmp_path: Path, settings: StageSettings
) -> None:
    """The output root is a dataset in its own right, so the pipeline picks a clustered choice up from it."""
    other = tmp_path / "next"
    write_selection(stage_dataset(run, Stage.SUBSET, _INSTRUMENT), chosen, pipeline_paths(other).looped_dir)

    carried = stage_recordings(other, Stage.LOOPED, settings)

    assert [recording.key for recording in carried.recordings] == [pick.recording.key for pick in chosen.picks]
    assert [recording.label for recording in carried.recordings] == [pick.recording.label for pick in chosen.picks]
    assert [recording.weight for recording in carried.recordings] == [
        pytest.approx((pick.place + 1) * _HELD_S) for pick in chosen.picks
    ]


def test_the_recordings_a_selection_leaves_behind_stay_where_they_were(
    chosen: Selection, run: Path, tmp_path: Path
) -> None:
    written = write_selection(stage_dataset(run, Stage.SUBSET, _INSTRUMENT), chosen, tmp_path / "selection")
    landed = {wav.name for wav in written.dataset.source.recordings_dir.glob("*.wav")}

    assert landed == {pick.recording.file.name for pick in chosen.picks}
    assert len(list((pipeline_paths(run).subset_dir / _INSTRUMENT).glob("*.wav"))) == len(_PITCHES)


def test_the_dataset_stages_are_the_ones_a_selection_is_written_out_of() -> None:
    assert [stage for stage in Stage if stage.is_dataset] == [Stage.SUBSET, Stage.LOOPED, Stage.REDUCED]


def test_the_allocated_stage_is_read_back_through_its_own_plan(run: Path) -> None:
    """What an allocation stored is named by its plan, so a slice of it is taken from the dataset behind it."""
    with pytest.raises(ValueError):
        stage_dataset(run, Stage.OPTIMIZED, _INSTRUMENT)
