from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from optisample.io.audio import write_wav
from optisample.io.note_extractor import IngestSettings, ManifestNote, NotesManifest, load_notes
from optisample.io.subset import even_ranks, select_positions, write_subset
from optisample.model import ProjectSpec

SR = 8_000
_PITCHES = tuple(range(60, 72))
_VELOCITIES = (20, 45, 70, 95, 120)
_TOTAL = len(_PITCHES) * len(_VELOCITIES)


@dataclass(frozen=True)
class _RankCase:
    """One ``even_ranks`` scenario: the axis length, how many picks, and the positions expected back."""

    name: str
    count: int
    picks: int
    expected: tuple[int, ...]


_RANK_CASES = (
    _RankCase("a single pick lands mid-axis", 5, 1, (2,)),
    _RankCase("two picks reach both ends", 5, 2, (0, 4)),
    _RankCase("three picks add the middle", 5, 3, (0, 2, 4)),
    _RankCase("picks beyond the axis keep every position", 3, 9, (0, 1, 2)),
    _RankCase("an empty axis yields nothing", 0, 4, ()),
    _RankCase("no picks yield nothing", 7, 0, ()),
)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A dataset covering every ``(pitch, velocity)`` pair once, so a subset's spread is readable."""
    samples = tmp_path / "Piano"
    samples.mkdir()
    generator = np.random.default_rng(0)
    notes = []
    for index, (pitch, velocity) in enumerate((pitch, vel) for pitch in _PITCHES for vel in _VELOCITIES):
        write_wav(samples / f"{index:04d}_p{pitch:03d}_v{velocity:03d}.wav", generator.standard_normal(SR) * 0.1, SR)
        notes.append(
            {
                "source_id": 1000 + index,
                "pitch": pitch,
                "velocity": velocity,
                "cc_averages": {"1": float(velocity)},
                "render": {"index": index, "start_seconds": float(index), "release_end_seconds": index + 0.75},
            }
        )

    notes_json = tmp_path / "Piano.notes.json"
    notes_json.write_text(json.dumps({"config": {"tracked_ccs": [1], "note_count": _TOTAL}, "notes": notes}))
    return notes_json


def _manifest(notes_json: Path) -> NotesManifest:
    return NotesManifest.model_validate(json.loads(notes_json.read_text(encoding="utf-8")))


@pytest.mark.parametrize("case", _RANK_CASES, ids=lambda case: case.name)
def test_even_ranks_spreads_its_picks_over_the_axis(case: _RankCase) -> None:
    assert even_ranks(case.count, case.picks) == case.expected


# --- what a subset spans ---------------------------------------------------------------------------


def test_a_subset_holds_the_share_of_notes_it_was_asked_for(source: Path) -> None:
    positions = select_positions(_manifest(source).notes, 0.2)
    assert len(positions) == round(0.2 * _TOTAL)
    assert list(positions) == sorted(set(positions))  # source order, each note taken once


def test_a_subset_reaches_every_pitch_before_any_pitch_repeats(source: Path) -> None:
    """Covering the keyboard first is what makes a small slice predict how the whole dataset behaves."""
    notes = _manifest(source).notes
    positions = select_positions(notes, len(_PITCHES) / _TOTAL)
    assert sorted(notes[position].pitch for position in positions) == sorted(_PITCHES)


def test_a_pitch_taking_two_notes_takes_its_quietest_and_its_loudest(source: Path) -> None:
    notes = _manifest(source).notes
    positions = select_positions(notes, 2 * len(_PITCHES) / _TOTAL)
    by_pitch: dict[int, list[int]] = {}
    for position in positions:
        by_pitch.setdefault(notes[position].pitch, []).append(notes[position].velocity)

    assert all(velocities == [_VELOCITIES[0], _VELOCITIES[-1]] for velocities in by_pitch.values())


def test_a_subset_smaller_than_the_keyboard_spreads_the_pitches_it_keeps(source: Path) -> None:
    notes = _manifest(source).notes
    kept = [notes[position].pitch for position in select_positions(notes, 3 / _TOTAL)]
    centre = (_PITCHES[0] + _PITCHES[-1]) / 2.0
    assert (kept[0], kept[-1]) == (_PITCHES[0], _PITCHES[-1])
    assert abs(kept[1] - centre) <= 1.0


def _lopsided() -> list[ManifestNote]:
    """Ten notes across two pitches, one of them played four times as often as the other."""
    return NotesManifest.model_validate(
        {
            "notes": [
                {
                    "pitch": pitch,
                    "velocity": velocity,
                    "render": {"index": 0, "start_seconds": 0.0, "release_end_seconds": 1.0},
                }
                for pitch, count in ((60, 8), (61, 2))
                for velocity in range(count)
            ]
        }
    ).notes


def test_every_pitch_takes_a_second_note_before_any_takes_a_third() -> None:
    """Notes go round the pitches in passes, so a subset spreads rather than deepening at one pitch."""
    notes = _lopsided()
    positions = select_positions(notes, 4 / len(notes))
    assert Counter(notes[position].pitch for position in positions) == {60: 2, 61: 2}


def test_a_pitch_that_has_given_all_it_holds_steps_aside_for_the_rest() -> None:
    notes = _lopsided()
    positions = select_positions(notes, 6 / len(notes))
    assert Counter(notes[position].pitch for position in positions) == {60: 4, 61: 2}


def test_a_whole_subset_keeps_every_note(source: Path) -> None:
    assert select_positions(_manifest(source).notes, 1.0) == tuple(range(_TOTAL))


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_a_fraction_outside_the_unit_interval_is_rejected(source: Path, fraction: float) -> None:
    with pytest.raises(ValueError):
        select_positions(_manifest(source).notes, fraction)


# --- the dataset it writes -------------------------------------------------------------------------


def test_the_written_subset_is_a_dataset_ingest_reads_back(source: Path, tmp_path: Path) -> None:
    dataset = write_subset(source, source.parent / "Piano", tmp_path / "out", instrument_id="Piano", fraction=0.2)

    assert (dataset.notes_json, dataset.samples_dir) == (
        tmp_path / "out" / "Piano.notes.json",
        tmp_path / "out" / "Piano",
    )
    assert dataset.recordings == dataset.kept_notes == round(0.2 * _TOTAL)
    assert dataset.source_notes == _TOTAL
    manifest = load_notes(
        dataset.notes_json,
        dataset.samples_dir,
        IngestSettings(instrument_id="Piano", budget_kb=64.0, project=ProjectSpec(name="song")),
    )
    assert len(manifest.instruments[0].samples) == dataset.kept_notes


def test_a_kept_note_is_written_exactly_as_the_source_states_it(source: Path, tmp_path: Path) -> None:
    """The subset measures the same material at the same lengths, so every field carries over verbatim."""
    dataset = write_subset(source, source.parent / "Piano", tmp_path / "out", instrument_id="Piano", fraction=0.2)
    written = json.loads(dataset.notes_json.read_text(encoding="utf-8"))
    original = {note["render"]["index"]: note for note in json.loads(source.read_text(encoding="utf-8"))["notes"]}

    assert all(note == original[note["render"]["index"]] for note in written["notes"])
    assert written["config"]["tracked_ccs"] == [1]  # the source's own config block carries over
    assert written["config"]["note_count"] == len(written["notes"])  # its declared count follows the subset


def test_the_subset_reports_the_ranges_it_spans(source: Path, tmp_path: Path) -> None:
    """Two notes a pitch is enough to reach both ends of the dynamics it was played across."""
    dataset = write_subset(source, source.parent / "Piano", tmp_path / "out", instrument_id="Piano", fraction=0.4)

    assert dataset.pitches == (_PITCHES[0], _PITCHES[-1])
    assert dataset.velocities == (_VELOCITIES[0], _VELOCITIES[-1])


def test_one_note_a_pitch_keeps_the_velocity_most_typical_of_it(source: Path, tmp_path: Path) -> None:
    dataset = write_subset(source, source.parent / "Piano", tmp_path / "out", instrument_id="Piano", fraction=0.2)

    assert dataset.velocities == (_VELOCITIES[2], _VELOCITIES[2])


def test_recordings_no_note_reaches_stay_behind(source: Path, tmp_path: Path) -> None:
    dataset = write_subset(source, source.parent / "Piano", tmp_path / "out", instrument_id="Piano", fraction=0.2)
    kept = {note["render"]["index"] for note in json.loads(dataset.notes_json.read_text(encoding="utf-8"))["notes"]}

    assert {int(wav.name.split("_", 1)[0]) for wav in dataset.samples_dir.glob("*.wav")} == kept
