from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisample.artifacts.paths import pipeline_paths
from optisample.cluster.representative import Group, MemberReadings, grouping
from optisample.cluster.selection import Selection, selection, write_selection
from optisample.cluster.space import Coordinates
from optisample.cluster.stages import (
    ReadingSettings,
    RecordingCorpus,
    RecordingSource,
    Stage,
    gathered_recordings,
    source_recordings,
    stage_dataset,
)
from optisample.config import OptiConfig
from optisample.config.dynamic_axis import AS_WRITTEN
from optisample.io.audio import write_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import NOTES_SUFFIX, IngestSettings, NoteRecord, dump_notes
from optisample.io.source import load_source
from optisample.model import ProjectSpec
from optisample.music import midi_to_freq
from optisample.progress import NO_PROGRESS

SR = 8_000

_INSTRUMENT = "Piano"
_OTHER_INSTRUMENT = "Rhodes"  # a second dataset of the same run, which a gathered space places beside the first
_STRATEGY = "grouped"
_PITCHES = (48, 50, 52, 72, 74, 76)  # six takes, the lower three apart from the upper three
_VELOCITY = 100
_TAKE_S = 0.5
_HELD_S = 0.25  # how long one note sounds, which is the playing time it gives the recording answering it
_APART = 10.0  # the room between the two sets of coordinates, wide enough that a cut lands between them
_GROUPS = 2
_TO_THE_RELEASE = False
_UNSPENT_KB = 1.0  # a reading of a written dataset allocates nothing and spends none of its budget
_STATED_BY_MANIFEST = 0.0  # the padding a written manifest records for itself


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


def _write_dataset(directory: Path, instrument_id: str) -> None:
    """One stage's dataset: six recordings, and the notes each of them answers for."""
    samples_dir = directory / instrument_id
    samples_dir.mkdir(parents=True, exist_ok=True)
    for index, pitch in enumerate(_PITCHES):
        write_wav(samples_dir / f"{index:04d}_p{pitch:03d}_v{_VELOCITY:03d}.wav", _take(pitch), SR)

    dump_notes(_notes(), directory / f"{instrument_id}{NOTES_SUFFIX}")


def _read_back(dataset: SourceDataset, instrument_id: str) -> list[tuple[int, int]]:
    """The keys a written dataset states, read through the ingest a later stage picks it up with."""
    manifest = load_source(
        dataset,
        IngestSettings(
            instrument_id=instrument_id,
            budget_kb=_UNSPENT_KB,
            project=ProjectSpec(name=instrument_id),
            pre_roll_s=_STATED_BY_MANIFEST,
            post_roll_s=_STATED_BY_MANIFEST,
            keep_tail=_TO_THE_RELEASE,
            dynamic_axis=AS_WRITTEN,
        ),
    )
    (instrument,) = manifest.instruments
    return [(note.pitch, note.velocity) for note in instrument.material]


@pytest.fixture
def settings(config: OptiConfig) -> ReadingSettings:
    """What a reading of a set is carried out with, on the terms the pipeline's own stages read by."""
    return ReadingSettings(
        strategy=_STRATEGY,
        dedupe=config.reduce.dedupe,
        trim=config.reduce.trim,
        keep_tail=_TO_THE_RELEASE,
        progress=NO_PROGRESS,
    )


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A run holding one dataset stage, carrying two instruments through it."""
    directory = pipeline_paths(tmp_path).subset_dir
    _write_dataset(directory, _INSTRUMENT)
    _write_dataset(directory, _OTHER_INSTRUMENT)
    return tmp_path


@pytest.fixture
def source(run: Path) -> RecordingSource:
    """The set the space below is built across."""
    return RecordingSource(root=run, instrument_id=_INSTRUMENT, stage=Stage.SUBSET)


@pytest.fixture
def corpus(source: RecordingSource, settings: ReadingSettings) -> RecordingCorpus:
    """The set's recordings as a space would place them, each carrying the file it is stored in."""
    return source_recordings(source, settings)


def _coordinates(corpus: RecordingCorpus) -> Coordinates:
    """Two sets of points standing well apart, so which recordings a group gathers is settled up front."""
    placed = np.zeros((corpus.size, 2), dtype=np.float64)
    placed[corpus.size // 2 :, 0] = _APART
    placed[:, 1] = np.arange(corpus.size, dtype=np.float64) / corpus.size
    return placed


def _groups(corpus: RecordingCorpus, config: OptiConfig) -> tuple[Group, ...]:
    """The two groups those points fall into, read for the takes standing for each."""
    coordinates = _coordinates(corpus)
    labels = np.asarray([0] * (corpus.size // 2) + [1] * (corpus.size // 2), dtype=np.intp)
    readings = MemberReadings(weights=corpus.weights, durations_s=corpus.durations_s)
    return grouping(coordinates, labels, readings, config=config.cluster.partition)


@pytest.fixture
def groups(corpus: RecordingCorpus, config: OptiConfig) -> tuple[Group, ...]:
    return _groups(corpus, config)


@pytest.fixture
def chosen(corpus: RecordingCorpus, groups: tuple[Group, ...]) -> Selection:
    """The take standing for each group, which is what a written selection holds."""
    return selection(corpus, groups)


def test_a_selection_holds_one_take_for_every_group(
    chosen: Selection, groups: tuple[Group, ...], source: RecordingSource
) -> None:
    assert chosen.sources == (source,)
    assert chosen.size == _GROUPS
    assert [pick.group for pick in chosen.picks] == [group.label for group in groups]


def test_every_chosen_take_is_a_member_of_the_group_it_stands_for(chosen: Selection, groups: tuple[Group, ...]) -> None:
    """A representative is a real recording out of its own group, which is what makes it playable."""
    for pick, group in zip(chosen.picks, groups, strict=True):
        assert pick.place in group.members.tolist()
        assert pick.place == group.representative


def test_a_pick_carries_the_recording_the_point_stood_for(
    chosen: Selection, corpus: RecordingCorpus, source: RecordingSource
) -> None:
    for pick in chosen.picks:
        assert pick.recording == corpus.recordings[pick.place]
        assert pick.source == source
        assert pick.render_index == int(pick.recording.file.name.split("_", 1)[0])


def test_the_picks_stand_for_the_whole_corpus_they_were_cut_from(
    chosen: Selection, corpus: RecordingCorpus, source: RecordingSource
) -> None:
    """A selection reports what it covers, so a small set of takes is read beside the music it answers for."""
    assert chosen.covered == corpus.size
    assert chosen.playing_s == pytest.approx(sum(recording.weight for recording in corpus.recordings))
    assert chosen.render_indices(source) == frozenset(pick.render_index for pick in chosen.picks)
    assert chosen.picks_from(source) == chosen.picks


def test_a_written_selection_holds_the_notes_its_takes_answer_for(chosen: Selection, tmp_path: Path) -> None:
    written = write_selection(chosen, tmp_path / "selection")
    answered = sum(pick.place + 1 for pick in chosen.picks)
    (entry,) = written.written

    assert written.selection == chosen
    assert written.recordings == _GROUPS
    assert written.kept_notes == answered
    assert entry.dataset.source_notes == len(_notes())


def test_a_written_set_lands_under_the_instrument_and_the_stage_it_came_from(
    chosen: Selection, tmp_path: Path, source: RecordingSource
) -> None:
    """Each set writes into a directory of its own, which is what keeps two of one instrument apart."""
    out_dir = tmp_path / "selection"

    (entry,) = write_selection(chosen, out_dir).written

    assert entry.source == source
    assert entry.dataset.source.path == out_dir / _INSTRUMENT / Stage.SUBSET.value / f"{_INSTRUMENT}{NOTES_SUFFIX}"


def test_a_written_selection_is_read_back_as_the_dataset_a_later_stage_picks_up(
    chosen: Selection, tmp_path: Path
) -> None:
    """What lands is a NoteExtractor dataset in its own right, so the pipeline reads a clustered choice."""
    (entry,) = write_selection(chosen, tmp_path / "selection").written

    carried = _read_back(entry.dataset.source, _INSTRUMENT)

    assert carried == [
        (pick.recording.key.pitch, pick.recording.key.velocity) for pick in chosen.picks for _ in range(pick.place + 1)
    ]


def test_the_recordings_a_selection_leaves_behind_stay_where_they_were(
    chosen: Selection, run: Path, tmp_path: Path
) -> None:
    (entry,) = write_selection(chosen, tmp_path / "selection").written
    landed = {wav.name for wav in entry.dataset.source.recordings_dir.glob("*.wav")}

    assert landed == {pick.recording.file.name for pick in chosen.picks}
    assert len(list((pipeline_paths(run).subset_dir / _INSTRUMENT).glob("*.wav"))) == len(_PITCHES)


def test_a_space_gathered_from_two_sets_writes_one_dataset_apiece(
    run: Path, settings: ReadingSettings, config: OptiConfig, tmp_path: Path
) -> None:
    """A choice made across two sets answers to each of them separately, every slice cut of its own dataset."""
    sources = (
        RecordingSource(root=run, instrument_id=_INSTRUMENT, stage=Stage.SUBSET),
        RecordingSource(root=run, instrument_id=_OTHER_INSTRUMENT, stage=Stage.SUBSET),
    )
    corpus = gathered_recordings(sources, settings)
    chosen = selection(corpus, _groups(corpus, config))
    out_dir = tmp_path / "selection"

    written = write_selection(chosen, out_dir)

    assert chosen.sources == sources
    assert [entry.source for entry in written.written] == list(sources)
    assert [entry.dataset.source.path for entry in written.written] == [
        out_dir / instrument / Stage.SUBSET.value / f"{instrument}{NOTES_SUFFIX}"
        for instrument in (_INSTRUMENT, _OTHER_INSTRUMENT)
    ]
    assert written.recordings == _GROUPS


def test_the_dataset_stages_are_the_ones_a_selection_is_written_out_of() -> None:
    assert [stage for stage in Stage if stage.is_dataset] == [Stage.SUBSET, Stage.LOOPED, Stage.REDUCED]


def test_the_allocated_stage_is_read_back_through_its_own_plan(run: Path) -> None:
    """What an allocation stored is named by its plan, so a slice of it is taken from the dataset behind it."""
    with pytest.raises(ValueError, match="stored samples"):
        stage_dataset(RecordingSource(root=run, instrument_id=_INSTRUMENT, stage=Stage.OPTIMIZED))
