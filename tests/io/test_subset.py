from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import load_config
from optisample.config.subset import IntakeConfig
from optisample.dsp.subsonic import remove_subsonic
from optisample.io.audio import read_wav, write_wav
from optisample.io.dataset import SourceDataset, SubsetDataset
from optisample.io.note_extractor import (
    IngestSettings,
    ManifestNote,
    NotesManifest,
    RenderWindow,
    load_notes,
)
from optisample.io.subset import (
    admit,
    even_ranks,
    notes_recorded_by,
    select_positions,
    sounding_positions,
    write_recording_subset,
    write_subset,
)
from optisample.model import ProjectSpec
from tests.conftest import TEST_CONFIG_DIR

SR = 8_000
_SUBSONIC = load_config(TEST_CONFIG_DIR).subsonic
_SILENT = 1.0e-300  # a floor keeping a band holding nothing off the logarithm
_PITCHES = tuple(range(60, 72))
_VELOCITIES = (20, 45, 70, 95, 120)
_TOTAL = len(_PITCHES) * len(_VELOCITIES)
_ADMIT_EVERY = 0.0  # a floor every note of the shared fixture clears, so a slice is read on its spread alone
_NOTE_S = 0.75  # how long each note of the shared fixture sounds, an exact binary fraction so spans read back whole
_BRIEF_S = 0.25  # how long the short notes of the ragged fixture sound, exact for the same reason
_BRIEF_EVERY = 3  # one note in this many of the ragged fixture is a short one


def _depth_db(signal: NDArray[np.float64], sample_rate: int, hz: float) -> float:
    """How much energy ``signal`` carries under ``hz``, in decibels, which is what a roll-off takes away.

    The reading is taken under a raised cosine, which holds a band this deep clear of what the ends of a
    finite stretch spread into it.
    """
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    return float(10.0 * np.log10(spectrum[freqs < hz].sum() + _SILENT))


def _dataset(source: Path) -> SourceDataset:
    """The source pair a slice is cut of: the manifest, beside the recordings it joins to."""
    return SourceDataset(path=source, samples_dir=source.parent / "Piano")


def _written(source: Path, out_dir: Path, *, fraction: float, min_duration_s: float) -> SubsetDataset:
    """The slice ``fraction`` of ``source`` writes into ``out_dir``, admitting notes at the floor named."""
    return write_subset(
        _dataset(source),
        out_dir,
        instrument_id="Piano",
        fraction=fraction,
        intake=IntakeConfig(min_duration_s=min_duration_s, subsonic=_SUBSONIC),
    )


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


def _write_source(tmp_path: Path, durations_s: Sequence[float]) -> Path:
    """A dataset covering every ``(pitch, velocity)`` pair once, each note sounding as long as it is given."""
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
                "render": {
                    "index": index,
                    "start_seconds": float(index),
                    "release_end_seconds": index + durations_s[index],
                },
            }
        )

    notes_json = tmp_path / "Piano.notes.json"
    document = {
        "config": {"tracked_ccs": [1], "note_count": _TOTAL},
        "settings": {"rolls": {"pre_roll_seconds": 0.0, "post_roll_seconds": 0.0}},
        "notes": notes,
    }
    notes_json.write_text(json.dumps(document))
    return notes_json


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A dataset covering every pair once, all notes alike in length, so a subset's spread is readable."""
    return _write_source(tmp_path, [_NOTE_S] * _TOTAL)


@pytest.fixture
def ragged(tmp_path: Path) -> Path:
    """The same grid with one note in three sounding for less than a slice admits."""
    return _write_source(
        tmp_path,
        [_BRIEF_S if index % _BRIEF_EVERY == 0 else _NOTE_S for index in range(_TOTAL)],
    )


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
    center = (_PITCHES[0] + _PITCHES[-1]) / 2.0
    assert (kept[0], kept[-1]) == (_PITCHES[0], _PITCHES[-1])
    assert abs(kept[1] - center) <= 1.0


def _lopsided() -> list[ManifestNote]:
    """Ten notes across two pitches, one of them played four times as often as the other."""
    return [
        ManifestNote(
            pitch=pitch,
            velocity=velocity,
            render=RenderWindow(index=0, start_seconds=0.0, release_end_seconds=1.0),
        )
        for pitch, count in ((60, 8), (61, 2))
        for velocity in range(count)
    ]


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


# --- the length a note sounds for to be drawn on ----------------------------------------------------


def test_only_the_notes_sounding_long_enough_are_drawn_on(ragged: Path) -> None:
    notes = _manifest(ragged).notes
    sounding = sounding_positions(notes, _NOTE_S)

    assert sounding == tuple(position for position in range(_TOTAL) if position % _BRIEF_EVERY)
    assert all(notes[position].duration_s >= _NOTE_S for position in sounding)


def test_a_note_sounding_exactly_the_floor_is_drawn_on(ragged: Path) -> None:
    """The floor is the shortest a note may sound, so the length it names is a length that qualifies."""
    assert len(sounding_positions(_manifest(ragged).notes, _BRIEF_S)) == _TOTAL


def test_the_share_is_counted_against_the_source_before_its_short_notes_leave(ragged: Path) -> None:
    """Raising the floor narrows what a slice draws on and leaves the size it comes out at alone."""
    notes = _manifest(ragged).notes
    admitted = admit(notes, fraction=0.5, min_duration_s=_NOTE_S)

    assert len(admitted.positions) == round(0.5 * _TOTAL)
    assert all(notes[position].duration_s >= _NOTE_S for position in admitted.positions)


def test_the_notes_held_out_for_sounding_briefly_are_counted(ragged: Path) -> None:
    admitted = admit(_manifest(ragged).notes, fraction=0.5, min_duration_s=_NOTE_S)

    assert admitted.brief == len(range(0, _TOTAL, _BRIEF_EVERY))


def test_a_share_wider_than_the_floor_leaves_keeps_everything_that_survives_it(ragged: Path) -> None:
    notes = _manifest(ragged).notes
    admitted = admit(notes, fraction=1.0, min_duration_s=_NOTE_S)

    assert admitted.positions == sounding_positions(notes, _NOTE_S)


def test_a_floor_no_note_reaches_leaves_nothing_to_draw_on(ragged: Path) -> None:
    with pytest.raises(ValueError, match="sounds for less than"):
        admit(_manifest(ragged).notes, fraction=0.5, min_duration_s=_NOTE_S + _BRIEF_S + 1.0)


# --- the dataset it writes -------------------------------------------------------------------------


def test_the_written_subset_is_a_dataset_ingest_reads_back(
    source: Path, tmp_path: Path, ingest_settings: Callable[..., IngestSettings]
) -> None:
    dataset = _written(source, tmp_path / "out", fraction=0.2, min_duration_s=_ADMIT_EVERY)

    assert (dataset.source.path, dataset.source.recordings_dir) == (
        tmp_path / "out" / "Piano.notes.json",
        tmp_path / "out" / "Piano",
    )
    assert dataset.recordings == dataset.kept_notes == round(0.2 * _TOTAL)
    assert dataset.source_notes == _TOTAL
    manifest = load_notes(
        dataset.source.path,
        dataset.source.recordings_dir,
        ingest_settings("Piano", project_name="song"),
    )
    assert len(manifest.instruments[0].samples) == dataset.kept_notes


def test_a_kept_note_is_written_exactly_as_the_source_states_it(source: Path, tmp_path: Path) -> None:
    """The subset measures the same material at the same lengths, so every field carries over verbatim."""
    dataset = _written(source, tmp_path / "out", fraction=0.2, min_duration_s=_ADMIT_EVERY)
    written = json.loads(dataset.source.path.read_text(encoding="utf-8"))
    original = {note["render"]["index"]: note for note in json.loads(source.read_text(encoding="utf-8"))["notes"]}

    assert all(note == original[note["render"]["index"]] for note in written["notes"])
    assert written["config"]["tracked_ccs"] == [1]  # the source's own config block carries over
    assert written["config"]["note_count"] == len(written["notes"])  # its declared count follows the subset
    assert written["settings"] == json.loads(source.read_text(encoding="utf-8"))["settings"]  # cut the same way


def test_the_subset_reports_the_ranges_it_spans(source: Path, tmp_path: Path) -> None:
    """Two notes a pitch is enough to reach both ends of the dynamics it was played across."""
    dataset = _written(source, tmp_path / "out", fraction=0.4, min_duration_s=_ADMIT_EVERY)

    assert dataset.pitches == (_PITCHES[0], _PITCHES[-1])
    assert dataset.velocities == (_VELOCITIES[0], _VELOCITIES[-1])


def test_one_note_a_pitch_keeps_the_velocity_most_typical_of_it(source: Path, tmp_path: Path) -> None:
    dataset = _written(source, tmp_path / "out", fraction=0.2, min_duration_s=_ADMIT_EVERY)

    assert dataset.velocities == (_VELOCITIES[2], _VELOCITIES[2])


def test_a_written_slice_states_how_many_notes_sounded_too_briefly(ragged: Path, tmp_path: Path) -> None:
    dataset = _written(ragged, tmp_path / "out", fraction=0.5, min_duration_s=_NOTE_S)

    assert dataset.brief_notes == len(range(0, _TOTAL, _BRIEF_EVERY))
    assert dataset.kept_notes == round(0.5 * _TOTAL)


def test_a_written_slice_of_a_source_it_all_admits_states_nothing_held_out(source: Path, tmp_path: Path) -> None:
    assert _written(source, tmp_path / "out", fraction=0.2, min_duration_s=_NOTE_S).brief_notes == 0


def test_recordings_no_note_reaches_stay_behind(source: Path, tmp_path: Path) -> None:
    dataset = _written(source, tmp_path / "out", fraction=0.2, min_duration_s=_ADMIT_EVERY)
    kept = {note["render"]["index"] for note in json.loads(dataset.source.path.read_text(encoding="utf-8"))["notes"]}

    assert {int(wav.name.split("_", 1)[0]) for wav in dataset.source.recordings_dir.glob("*.wav")} == kept


# --- the slice a chosen set of recordings accounts for ----------------------------------------------

_CHOSEN = (0, 7, 41)  # recordings a rule of the caller's own picked out of the source


def test_the_notes_a_recording_answers_for_are_the_ones_it_accounts_for(source: Path) -> None:
    notes = _manifest(source).notes
    positions = notes_recorded_by(notes, _CHOSEN)

    assert list(positions) == sorted(positions)
    assert {notes[position].render.index for position in positions} == set(_CHOSEN)


def test_a_recording_answering_several_notes_brings_every_one_of_them(source: Path, tmp_path: Path) -> None:
    """A stage storing one recording for a run of notes hands the slice all of the material it carries."""
    document = json.loads(source.read_text(encoding="utf-8"))
    for note in document["notes"][:4]:
        note["render"]["index"] = 0

    shared = tmp_path / "shared.notes.json"
    shared.write_text(json.dumps(document))

    assert notes_recorded_by(_manifest(shared).notes, {0}) == (0, 1, 2, 3)


def test_the_written_slice_holds_the_recordings_it_was_handed(source: Path, tmp_path: Path) -> None:
    dataset = write_recording_subset(_dataset(source), tmp_path / "out", instrument_id="Piano", recordings=_CHOSEN)
    written = json.loads(dataset.source.path.read_text(encoding="utf-8"))

    assert dataset.recordings == dataset.kept_notes == len(_CHOSEN)
    assert dataset.source_notes == _TOTAL
    assert {note["render"]["index"] for note in written["notes"]} == set(_CHOSEN)
    assert {int(wav.name.split("_", 1)[0]) for wav in dataset.source.recordings_dir.glob("*.wav")} == set(_CHOSEN)


def test_a_chosen_note_is_written_exactly_as_the_source_states_it(source: Path, tmp_path: Path) -> None:
    """The slice measures the same material at the same lengths, whichever rule chose its recordings."""
    dataset = write_recording_subset(_dataset(source), tmp_path / "out", instrument_id="Piano", recordings=_CHOSEN)
    written = json.loads(dataset.source.path.read_text(encoding="utf-8"))
    document = json.loads(source.read_text(encoding="utf-8"))
    original = {note["render"]["index"]: note for note in document["notes"]}

    assert all(note == original[note["render"]["index"]] for note in written["notes"])
    assert written["config"]["note_count"] == len(written["notes"])
    assert written["settings"] == document["settings"]


def test_the_written_slice_is_a_dataset_ingest_reads_back(
    source: Path, tmp_path: Path, ingest_settings: Callable[..., IngestSettings]
) -> None:
    dataset = write_recording_subset(_dataset(source), tmp_path / "out", instrument_id="Piano", recordings=_CHOSEN)
    manifest = load_notes(
        dataset.source.path, dataset.source.recordings_dir, ingest_settings("Piano", project_name="song")
    )

    assert len(manifest.instruments[0].samples) == len(_CHOSEN)
    assert {sample.file.name for sample in manifest.instruments[0].samples} == {
        wav.name for wav in dataset.source.recordings_dir.glob("*.wav")
    }


def test_a_written_subset_holds_its_recordings_past_the_band_under_hearing(source: Path, tmp_path: Path) -> None:
    """The way in is where the depth beneath hearing comes off, so what lands is the content a listener has."""
    dataset = _written(source, tmp_path / "out", fraction=0.2, min_duration_s=_ADMIT_EVERY)
    originals = {wav.name: wav for wav in (source.parent / "Piano").glob("*.wav")}

    for wav in sorted(dataset.source.recordings_dir.glob("*.wav")):
        kept, sample_rate = read_wav(wav)
        raw, _ = read_wav(originals[wav.name])
        taken = _depth_db(raw, sample_rate, _SUBSONIC.rejection_hz) - _depth_db(
            kept, sample_rate, _SUBSONIC.rejection_hz
        )
        assert taken >= _SUBSONIC.rejection_db


def test_a_written_subset_holds_exactly_what_the_roll_off_leaves(source: Path, tmp_path: Path) -> None:
    """One curve states the treatment, so what lands is the source read through it and nothing besides."""
    dataset = _written(source, tmp_path / "out", fraction=0.2, min_duration_s=_ADMIT_EVERY)
    originals = {wav.name: wav for wav in (source.parent / "Piano").glob("*.wav")}

    for wav in sorted(dataset.source.recordings_dir.glob("*.wav")):
        kept, sample_rate = read_wav(wav)
        raw, _ = read_wav(originals[wav.name])
        np.testing.assert_allclose(kept, remove_subsonic(raw, sample_rate, _SUBSONIC), atol=1.0e-6)


def test_a_chosen_recording_is_carried_over_as_it_stands(source: Path, tmp_path: Path) -> None:
    """A set chosen off a stage holds exactly the audio that stage did, the band having come off once."""
    dataset = write_recording_subset(_dataset(source), tmp_path / "out", instrument_id="Piano", recordings=_CHOSEN)
    originals = {wav.name: wav for wav in (source.parent / "Piano").glob("*.wav")}

    for wav in sorted(dataset.source.recordings_dir.glob("*.wav")):
        np.testing.assert_array_equal(read_wav(wav)[0], read_wav(originals[wav.name])[0])


def test_recordings_answering_for_no_note_leave_nothing_to_write(source: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_recording_subset(_dataset(source), tmp_path / "out", instrument_id="Piano", recordings=())
