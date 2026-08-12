from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from optisample.config import load_config
from optisample.io.audio import write_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import IngestSettings, NoteRecord, dump_notes
from optisample.io.source import load_source, write_source_subset
from optisample.model import ProjectSpec

SR = 8_000
_SUBSONIC = load_config().subsonic
_INSTRUMENT = "Piano"
_FRAMES = 1_600
_TAKE_S = _FRAMES / SR
_PITCHES = (48, 55, 60, 67, 72)
_VELOCITY = 100
_SLICE = 0.4
_SLICED_TAKES = 2


@pytest.fixture
def recordings(tmp_path: Path) -> Path:
    """A directory of recordings named the way every dataset this project writes names them."""
    directory = tmp_path / _INSTRUMENT
    directory.mkdir()
    for index, pitch in enumerate(_PITCHES):
        write_wav(
            directory / f"{index:04d}_p{pitch:03d}_v{_VELOCITY:03d}.wav",
            np.random.default_rng(pitch).standard_normal(_FRAMES) * 0.2,
            SR,
        )

    return directory


@pytest.fixture
def manifest(recordings: Path) -> Path:
    """The same recordings with a manifest naming them, which is the other shape a source takes."""
    path = recordings.parent / f"{_INSTRUMENT}.notes.json"
    dump_notes(
        [
            NoteRecord(index=index, pitch=pitch, velocity=_VELOCITY, duration_s=_TAKE_S)
            for index, pitch in enumerate(_PITCHES)
        ],
        path,
    )
    return path


@pytest.fixture
def settings(ingest_settings: Callable[..., IngestSettings]) -> IngestSettings:
    return ingest_settings(_INSTRUMENT, project_name="song")


def test_a_manifest_and_the_directory_beside_it_reach_the_same_recordings(
    manifest: Path, recordings: Path, settings: IngestSettings
) -> None:
    """Both shapes answer with one single-instrument manifest, so every stage past the ingest reads one."""
    (joined,) = load_source(SourceDataset(path=manifest, samples_dir=None), settings).instruments
    (read,) = load_source(SourceDataset(path=recordings, samples_dir=None), settings).instruments

    assert [sample.file for sample in joined.samples] == [sample.file for sample in read.samples]
    assert [(sample.pitch, sample.velocity) for sample in joined.samples] == [
        (sample.pitch, sample.velocity) for sample in read.samples
    ]


def test_a_manifest_naming_shorter_notes_than_it_recorded_states_the_shorter_material(
    manifest: Path, recordings: Path, settings: IngestSettings
) -> None:
    """A performance says how long each note was held; a directory has the recorded span and says that."""
    held_s = _TAKE_S / 2
    dump_notes(
        [
            NoteRecord(index=index, pitch=pitch, velocity=_VELOCITY, duration_s=held_s)
            for index, pitch in enumerate(_PITCHES)
        ],
        manifest,
    )
    (joined,) = load_source(SourceDataset(path=manifest, samples_dir=None), settings).instruments
    (read,) = load_source(SourceDataset(path=recordings, samples_dir=None), settings).instruments

    assert {event.duration_s for event in joined.material} == {held_s}
    assert all(event.duration_s == pytest.approx(_TAKE_S) for event in read.material)


def test_a_slice_of_a_manifest_dataset_is_a_manifest_dataset(manifest: Path, tmp_path: Path) -> None:
    dataset = write_source_subset(
        SourceDataset(path=manifest, samples_dir=None),
        tmp_path / "out",
        instrument_id=_INSTRUMENT,
        fraction=_SLICE,
        subsonic=_SUBSONIC,
    )

    assert dataset.source.path == tmp_path / "out" / f"{_INSTRUMENT}.notes.json"
    assert dataset.source.is_directory is False


def test_a_slice_of_a_directory_is_a_directory(recordings: Path, tmp_path: Path) -> None:
    dataset = write_source_subset(
        SourceDataset(path=recordings, samples_dir=None),
        tmp_path / "out",
        instrument_id=_INSTRUMENT,
        fraction=_SLICE,
        subsonic=_SUBSONIC,
    )

    assert dataset.source.path == tmp_path / "out" / _INSTRUMENT
    assert dataset.source.is_directory is True


def test_either_shape_slices_to_the_same_share_of_its_source(manifest: Path, recordings: Path, tmp_path: Path) -> None:
    """One selection rule reads both shapes, so a fraction means the same thing whichever was handed in."""
    joined = write_source_subset(
        SourceDataset(path=manifest, samples_dir=None),
        tmp_path / "a",
        instrument_id=_INSTRUMENT,
        fraction=_SLICE,
        subsonic=_SUBSONIC,
    )
    read = write_source_subset(
        SourceDataset(path=recordings, samples_dir=None),
        tmp_path / "b",
        instrument_id=_INSTRUMENT,
        fraction=_SLICE,
        subsonic=_SUBSONIC,
    )

    assert joined.kept_notes == read.kept_notes == _SLICED_TAKES
    assert joined.pitches == read.pitches


def test_a_slice_is_read_back_by_the_loader_its_source_was(
    recordings: Path, tmp_path: Path, settings: IngestSettings
) -> None:
    """A stage hands its slice straight on, so the shape it wrote has to be one ``load_source`` reads."""
    dataset = write_source_subset(
        SourceDataset(path=recordings, samples_dir=None),
        tmp_path / "out",
        instrument_id=_INSTRUMENT,
        fraction=_SLICE,
        subsonic=_SUBSONIC,
    )
    (instrument,) = load_source(dataset.source, settings).instruments

    assert len(instrument.samples) == dataset.kept_notes
