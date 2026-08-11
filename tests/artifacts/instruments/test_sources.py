from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisample.artifacts.instruments.sources import dataset_recordings
from optisample.io.audio import write_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import NoteRecord, dump_notes
from tests.artifacts.instruments.conftest import SR, TEMPO_BPM

PITCHES = (48, 60, 72)
_HELD_S = 0.1


def _written_wav(samples_dir: Path, name: str) -> None:
    write_wav(samples_dir / name, np.zeros(round(_HELD_S * SR), dtype=np.float64), SR)


@pytest.fixture
def manifest_dataset(tmp_path: Path) -> SourceDataset:
    """A dataset in the shape every stage of this project writes: one manifest beside its own recordings."""
    samples_dir = tmp_path / "piano"
    samples_dir.mkdir()
    records = []
    for index, pitch in enumerate(PITCHES):
        _written_wav(samples_dir, f"{index:04d}_p{pitch}_v100.wav")
        records.append(NoteRecord(index=index, pitch=pitch, velocity=100, duration_s=_HELD_S))

    notes_json = tmp_path / "piano.notes.json"
    dump_notes(records, notes_json, tempo_bpm=TEMPO_BPM)
    return SourceDataset(path=notes_json, samples_dir=samples_dir)


@pytest.fixture
def take_directory(tmp_path: Path) -> SourceDataset:
    """A directory of takes, which names what it holds in the filenames alone."""
    samples_dir = tmp_path / "takes"
    samples_dir.mkdir()
    for pitch in PITCHES:
        _written_wav(samples_dir, f"p{pitch}_v100.wav")

    return SourceDataset(path=samples_dir, samples_dir=None)


def test_a_manifest_dataset_joins_each_recording_to_the_pitch_its_notes_play(manifest_dataset: SourceDataset) -> None:
    """A note reaching a recording is a note at that recording's own pitch, which is what roots its instrument."""
    dataset = dataset_recordings(manifest_dataset)

    assert [source.root_pitch for source in dataset.sources] == list(PITCHES)
    assert [source.name for source in dataset.sources] == [
        f"{index:04d}_p{pitch}_v100" for index, pitch in enumerate(PITCHES)
    ]


def test_a_manifest_dataset_states_the_clock_its_material_was_played_at(manifest_dataset: SourceDataset) -> None:
    assert dataset_recordings(manifest_dataset).tempo_bpm == TEMPO_BPM


def test_a_directory_of_takes_reads_the_pitch_each_filename_spells(take_directory: SourceDataset) -> None:
    dataset = dataset_recordings(take_directory)

    assert [source.root_pitch for source in dataset.sources] == list(PITCHES)
    assert [source.file.parent for source in dataset.sources] == [take_directory.path.resolve()] * len(PITCHES)


def test_a_directory_of_takes_leaves_the_clock_to_whoever_writes_its_instruments(
    take_directory: SourceDataset,
) -> None:
    """A set of recordings says nothing of a song, so the export supplies the clock its envelopes count in."""
    assert dataset_recordings(take_directory).tempo_bpm is None


def test_a_recording_the_manifest_names_no_note_at_is_refused(manifest_dataset: SourceDataset) -> None:
    """A dataset states one pitch per render index, so a file carrying an unknown one has no pitch to be rooted at."""
    _written_wav(manifest_dataset.recordings_dir, "0099_p90_v100.wav")

    with pytest.raises(ValueError, match="render index 99"):
        dataset_recordings(manifest_dataset)
