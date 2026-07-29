from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from optisample.io.dataset import SourceDataset, instrument_name

_INSTRUMENT = "Piano"


@dataclass(frozen=True)
class _NameCase:
    """One ``instrument_name`` scenario: the filename handed in, and the instrument it names."""

    name: str
    filename: str
    expected: str


_NAME_CASES = (
    _NameCase("a manifest is named by its base", "Piano.notes.json", _INSTRUMENT),
    _NameCase("a dotted base survives the suffix", "Grand.Piano.notes.json", "Grand.Piano"),
    _NameCase("any other file falls back to its stem", "Piano.json", _INSTRUMENT),
)


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    """A manifest path beside the sibling directory an ingest resolves its recordings from."""
    (tmp_path / _INSTRUMENT).mkdir()
    path = tmp_path / f"{_INSTRUMENT}.notes.json"
    path.write_text("{}", encoding="utf-8")
    return path


@pytest.fixture
def recordings(tmp_path: Path) -> Path:
    """A directory of recordings, which is the other shape a source takes."""
    path = tmp_path / _INSTRUMENT
    path.mkdir()
    return path


@pytest.mark.parametrize("case", _NAME_CASES, ids=lambda case: case.name)
def test_a_manifest_path_names_the_instrument_behind_it(case: _NameCase, tmp_path: Path) -> None:
    assert instrument_name(tmp_path / case.filename) == case.expected


def test_a_directory_of_recordings_names_the_instrument_it_is(recordings: Path) -> None:
    """A directory carries no suffix to strip, so its own name is what the artifacts are filed under."""
    assert instrument_name(recordings) == _INSTRUMENT


def test_a_manifest_resolves_its_recordings_from_the_sibling_named_after_it(manifest: Path) -> None:
    source = SourceDataset(path=manifest, samples_dir=None)

    assert source.is_directory is False
    assert source.recordings_dir == manifest.parent / _INSTRUMENT


def test_naming_a_samples_directory_overrides_where_a_manifest_looks(manifest: Path, tmp_path: Path) -> None:
    """Datasets keeping their WAVs apart from the manifest are read by saying where they went."""
    elsewhere = tmp_path / "elsewhere"
    source = SourceDataset(path=manifest, samples_dir=elsewhere)

    assert source.recordings_dir == elsewhere


def test_a_directory_of_recordings_holds_its_own(recordings: Path, tmp_path: Path) -> None:
    """A directory is both the source and where its WAVs sit, so it answers with itself."""
    source = SourceDataset(path=recordings, samples_dir=tmp_path / "elsewhere")

    assert source.is_directory is True
    assert source.recordings_dir == recordings
