from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from optisample.config import load_config
from optisample.dsp.subsonic import remove_subsonic
from optisample.io.audio import read_wav, write_wav
from optisample.io.note_extractor import IngestSettings
from optisample.io.sample_dir import (
    UNKNOWN_VELOCITY,
    load_sample_dir,
    read_takes,
    write_sample_dir_subset,
)
from optisample.model import ProjectSpec

SR = 8_000
_SUBSONIC = load_config().subsonic
_INSTRUMENT = "Piano"
_FRAMES = 1_600
_TAKE_S = _FRAMES / SR
_PITCHES = (48, 60, 72)
_VELOCITIES = (20, 70, 120)
_LEAD_IN_S = 0.05
_TRAIL_OUT_S = 0.03


@dataclass(frozen=True)
class _SpellingCase:
    """One filename scenario: the name a recording carries, and the key and dynamic it spells."""

    name: str
    filename: str
    pitch: int
    velocity: int


_SPELLING_CASES = (
    _SpellingCase("a NoteExtractor name", "0000_p029_v018_cc0-0_cc1-0.wav", 29, 18),
    _SpellingCase("a reduced dataset name", "0000_p029_F1_v018.wav", 29, 18),
    _SpellingCase("a demo name", "0000_p60_v100.wav", 60, 100),
    _SpellingCase("a note name alone", "C4.wav", 60, UNKNOWN_VELOCITY),
    _SpellingCase("a flat spelling", "Bb2.wav", 46, UNKNOWN_VELOCITY),
    _SpellingCase("a note name inside a longer name", "Grand_F#3_take.wav", 54, UNKNOWN_VELOCITY),
)


def _write_take(path: Path) -> None:
    """One recording on disk, long enough for its header to state a duration worth measuring."""
    write_wav(path, np.random.default_rng(0).standard_normal(_FRAMES) * 0.2, SR)


@pytest.fixture
def named_grid(tmp_path: Path) -> Path:
    """A directory naming both axes, which is the shape every dataset this project writes carries."""
    directory = tmp_path / _INSTRUMENT
    directory.mkdir()
    for index, (pitch, velocity) in enumerate((p, v) for p in _PITCHES for v in _VELOCITIES):
        _write_take(directory / f"{index:04d}_p{pitch:03d}_v{velocity:03d}.wav")

    return directory


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A sample-library directory: one take a key, named by note name and silent on the dynamic."""
    directory = tmp_path / "library"
    directory.mkdir()
    for name in ("C3", "C4", "C5"):
        _write_take(directory / f"{name}.wav")

    return directory


@pytest.fixture
def settings(ingest_settings: Callable[..., IngestSettings]) -> IngestSettings:
    return ingest_settings(_INSTRUMENT, project_name="song")


@pytest.mark.parametrize("case", _SPELLING_CASES, ids=lambda case: case.name)
def test_a_filename_states_the_key_and_dynamic_of_the_take_it_holds(case: _SpellingCase, tmp_path: Path) -> None:
    _write_take(tmp_path / case.filename)
    (take,) = read_takes(tmp_path)

    assert (take.pitch, take.velocity) == (case.pitch, case.velocity)


def test_a_numbered_pitch_settles_the_two_spellings_a_reduced_name_carries(tmp_path: Path) -> None:
    """A reduced dataset names a key twice over, so reading the number first keeps them agreeing."""
    _write_take(tmp_path / "0000_p029_F1_v018.wav")
    (take,) = read_takes(tmp_path)

    assert take.pitch == 29


def test_controller_averages_carry_over_from_the_name(tmp_path: Path) -> None:
    _write_take(tmp_path / "0000_p060_v100_cc1-64.5_cc11-20.wav")
    (take,) = read_takes(tmp_path)

    assert take.cc_averages == {1: 64.5, 11: 20.0}


def test_takes_are_read_in_filename_order(named_grid: Path) -> None:
    """Filename order is what ranks duplicates, so a directory reduces the same way on every run."""
    takes = read_takes(named_grid)

    assert [take.file.name for take in takes] == sorted(take.file.name for take in takes)


def test_a_directory_naming_no_velocity_reads_every_take_at_one_dynamic(library: Path) -> None:
    """One take a key says nothing of how hard each was struck, so they share a single band."""
    takes = read_takes(library)

    assert {take.velocity for take in takes} == {UNKNOWN_VELOCITY}
    assert [take.pitch for take in takes] == [48, 60, 72]


def test_velocities_named_for_some_takes_alone_are_refused(library: Path) -> None:
    """Two readings of the same axis within one directory leave the dynamics undecided."""
    _write_take(library / "p60_v100.wav")

    with pytest.raises(ValueError, match="naming no velocity"):
        read_takes(library)


def test_a_recording_naming_no_pitch_is_refused(tmp_path: Path) -> None:
    _write_take(tmp_path / "take_one.wav")

    with pytest.raises(ValueError, match="names no pitch"):
        read_takes(tmp_path)


def test_a_directory_holding_no_recording_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no WAV recordings"):
        read_takes(tmp_path)


def test_every_take_becomes_one_recording_and_one_note(named_grid: Path, settings: IngestSettings) -> None:
    """A directory carries no song, so the recorded grid itself is what the allocation is optimized over."""
    (instrument,) = load_sample_dir(named_grid, settings).instruments

    assert len(instrument.samples) == len(_PITCHES) * len(_VELOCITIES)
    assert len(instrument.material) == len(instrument.samples)
    assert {(sample.pitch, sample.velocity) for sample in instrument.samples} == {
        (event.pitch, event.velocity) for event in instrument.material
    }


def test_a_note_is_held_for_as_long_as_its_recording_sounds(named_grid: Path, settings: IngestSettings) -> None:
    (instrument,) = load_sample_dir(named_grid, settings).instruments

    assert all(event.duration_s == pytest.approx(_TAKE_S) for event in instrument.material)


def test_the_padding_comes_off_both_ends_of_every_take(named_grid: Path, settings: IngestSettings) -> None:
    """A directory names no rolls of its own, so the flags are what say how its takes were padded."""
    padded = load_sample_dir(named_grid, replace(settings, pre_roll_s=_LEAD_IN_S, post_roll_s=_TRAIL_OUT_S))
    (instrument,) = padded.instruments

    assert all(sample.lead_in_s == _LEAD_IN_S for sample in instrument.samples)
    assert all(sample.trail_out_s == _TRAIL_OUT_S for sample in instrument.samples)
    assert all(event.duration_s == pytest.approx(_TAKE_S - _LEAD_IN_S - _TRAIL_OUT_S) for event in instrument.material)


def test_a_directory_run_asked_to_keep_the_tail_holds_each_take_to_its_end(
    named_grid: Path, settings: IngestSettings
) -> None:
    kept = load_sample_dir(named_grid, replace(settings, post_roll_s=_TRAIL_OUT_S, keep_tail=True))
    (instrument,) = kept.instruments

    assert all(sample.trail_out_s == 0.0 for sample in instrument.samples)
    assert all(event.duration_s == pytest.approx(_TAKE_S) for event in instrument.material)


def test_padding_claiming_a_whole_take_is_reported(named_grid: Path, settings: IngestSettings) -> None:
    """A note has to have something left to store, so padding that swallows one is a stated failure."""
    with pytest.raises(ValueError, match="all of it padding"):
        load_sample_dir(named_grid, replace(settings, pre_roll_s=_TAKE_S, post_roll_s=_TRAIL_OUT_S))


def test_a_slice_of_a_directory_is_a_directory(named_grid: Path, tmp_path: Path) -> None:
    """The slice reads back the way its source does, so the stage after it sees one shape."""
    dataset = write_sample_dir_subset(
        named_grid, tmp_path / "out", instrument_id=_INSTRUMENT, fraction=0.4, subsonic=_SUBSONIC
    )

    assert dataset.source.is_directory
    assert dataset.source.path == tmp_path / "out" / _INSTRUMENT
    assert len(list(dataset.source.path.glob("*.wav"))) == dataset.kept_notes


def test_a_slice_spans_the_ranges_its_source_covers(named_grid: Path, tmp_path: Path) -> None:
    dataset = write_sample_dir_subset(
        named_grid, tmp_path / "out", instrument_id=_INSTRUMENT, fraction=0.4, subsonic=_SUBSONIC
    )

    assert dataset.source_notes == len(_PITCHES) * len(_VELOCITIES)
    assert dataset.pitches == (_PITCHES[0], _PITCHES[-1])
    assert dataset.velocities == (_VELOCITIES[0], _VELOCITIES[-1])


def test_a_slice_of_a_directory_arrives_past_the_band_under_hearing(named_grid: Path, tmp_path: Path) -> None:
    """Either shape of source is taken in the same way, so a directory of takes is cleaned as a manifest is."""
    dataset = write_sample_dir_subset(
        named_grid, tmp_path / "out", instrument_id=_INSTRUMENT, fraction=0.4, subsonic=_SUBSONIC
    )
    originals = {wav.name: wav for wav in named_grid.glob("*.wav")}

    for wav in sorted(dataset.source.path.glob("*.wav")):
        kept, sample_rate = read_wav(wav)
        raw, _ = read_wav(originals[wav.name])
        np.testing.assert_allclose(kept, remove_subsonic(raw, sample_rate, _SUBSONIC), atol=1.0e-6)


def test_a_slice_keeps_the_takes_it_wrote_readable(named_grid: Path, tmp_path: Path, settings: IngestSettings) -> None:
    """Each take is written under its own name, so the slice spells the same keys the source did."""
    dataset = write_sample_dir_subset(
        named_grid, tmp_path / "out", instrument_id=_INSTRUMENT, fraction=0.4, subsonic=_SUBSONIC
    )
    (instrument,) = load_sample_dir(dataset.source.path, settings).instruments

    assert len(instrument.samples) == dataset.kept_notes
