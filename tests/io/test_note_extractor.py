from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from optisample.io.audio import write_wav
from optisample.io.note_extractor import (
    IngestSettings,
    NoteRecord,
    dump_notes,
    index_of_wav,
    load_notes,
    read_manifest,
)

SR = 8_000
RECORDED_S = 1.0

IngestFactory = Callable[..., IngestSettings]

_NOTES: Sequence[dict[str, Any]] = (
    {
        "pitch": 60,
        "velocity": 100,
        "cc_averages": {"0": 3.0, "1": 63.875},
        "render": {"index": 0, "start_seconds": 0.02, "release_end_seconds": 0.52},
    },
    {
        "pitch": 62,
        "velocity": 80,
        "cc_averages": {},
        "render": {"index": 1, "start_seconds": 5.0, "release_end_seconds": 6.5},
    },
)


def _write_wav(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    write_wav(path, rng.standard_normal(round(RECORDED_S * SR)).astype(np.float64) * 0.1, SR)


def _samples_dir(tmp_path: Path, indices: list[int]) -> Path:
    samples = tmp_path / "samples"
    samples.mkdir()
    for index in indices:
        _write_wav(samples / f"{index:04d}_p60_v100.wav", seed=index)
    return samples


def _stated(tmp_path: Path, document: dict[str, Any]) -> Path:
    notes_json = tmp_path / "inst.notes.json"
    notes_json.write_text(json.dumps(document), encoding="utf-8")
    return notes_json


def _dataset(tmp_path: Path, *, pre_roll_seconds: float, post_roll_seconds: float) -> tuple[Path, Path]:
    """A two-note manifest recording the given rolls, beside the recordings its notes join to."""
    document = {
        "settings": {"rolls": {"pre_roll_seconds": pre_roll_seconds, "post_roll_seconds": post_roll_seconds}},
        "notes": list(_NOTES),
    }
    return _stated(tmp_path, document), _samples_dir(tmp_path, [0, 1])


def test_index_of_wav_reads_leading_token() -> None:
    assert index_of_wav("0007_p060_v100_cc0-3.wav") == 7
    assert index_of_wav(Path("12_p60.wav")) == 12


def test_index_of_wav_rejects_missing_index() -> None:
    with pytest.raises(ValueError):
        index_of_wav("p60_v100.wav")


def test_load_notes_joins_by_index_and_derives_duration(tmp_path: Path, ingest_settings: IngestFactory) -> None:
    samples = _samples_dir(tmp_path, [0, 1])
    notes_json = tmp_path / "inst.notes.json"
    dump_notes(
        [
            NoteRecord(index=0, pitch=60, velocity=100, duration_s=0.5),
            NoteRecord(index=1, pitch=62, velocity=80, duration_s=1.25, cc_averages={1: 63.875}),
        ],
        notes_json,
        tracked_ccs=[1],
    )
    manifest = load_notes(notes_json, samples, ingest_settings("inst"))

    instrument = manifest.instruments[0]
    assert instrument.id == "inst"
    assert [sample.file.name for sample in instrument.samples] == ["0000_p60_v100.wav", "0001_p60_v100.wav"]
    assert all(sample.file.is_absolute() for sample in instrument.samples)
    assert [event.duration_s for event in instrument.material] == pytest.approx([0.5, 1.25])
    assert instrument.material[1].cc_averages == {1: 63.875}
    assert instrument.samples[1].cc_averages == {1: 63.875}


def test_a_recording_written_longer_than_its_note_keeps_every_frame(
    tmp_path: Path, ingest_settings: IngestFactory
) -> None:
    """A dataset this project writes declares no rolls, so the length it deliberately stored survives."""
    samples = _samples_dir(tmp_path, [0])
    notes_json = tmp_path / "inst.notes.json"
    dump_notes([NoteRecord(index=0, pitch=60, velocity=100, duration_s=0.5)], notes_json)

    instrument = load_notes(notes_json, samples, ingest_settings("inst")).instruments[0]

    assert instrument.samples[0].trail_out_s == pytest.approx(0.0)  # the recording holds twice the note
    assert (instrument.pre_roll_s, instrument.post_roll_s) == (0.0, 0.0)


def test_load_notes_coerces_cc_keys_and_clamps_lead_in(tmp_path: Path, ingest_settings: IngestFactory) -> None:
    notes_json, samples = _dataset(tmp_path, pre_roll_seconds=0.1, post_roll_seconds=0.2)

    instrument = load_notes(notes_json, samples, ingest_settings("inst")).instruments[0]

    assert instrument.samples[0].cc_averages == {0: 3.0, 1: 63.875}  # string CC keys coerce to int
    assert instrument.material[0].duration_s == pytest.approx(0.5)  # release_end - start
    assert instrument.material[1].duration_s == pytest.approx(1.5)
    assert instrument.samples[0].lead_in_s == pytest.approx(0.02)  # start < pre_roll, so clamped to start
    assert instrument.samples[1].lead_in_s == pytest.approx(0.1)  # pre_roll < start, so the full pre_roll


def test_the_rolls_a_manifest_records_are_what_a_run_reads_it_by(
    tmp_path: Path, ingest_settings: IngestFactory
) -> None:
    """The split run states the padding and the manifest carries it, so the ingest asks for none of it."""
    notes_json, samples = _dataset(tmp_path, pre_roll_seconds=0.1, post_roll_seconds=0.2)

    instrument = load_notes(notes_json, samples, ingest_settings("inst")).instruments[0]

    assert (instrument.pre_roll_s, instrument.post_roll_s) == pytest.approx((0.1, 0.2))
    assert instrument.samples[0].trail_out_s == pytest.approx(0.2)  # the whole declared post-roll is there


def test_a_recording_the_render_ran_out_for_drops_only_the_padding_it_holds(
    tmp_path: Path, ingest_settings: IngestFactory
) -> None:
    """The trail is measured from the file, so a cut clamped at the end of the render keeps its note."""
    notes_json, samples = _dataset(tmp_path, pre_roll_seconds=0.1, post_roll_seconds=0.2)

    instrument = load_notes(notes_json, samples, ingest_settings("inst")).instruments[0]

    assert instrument.material[1].duration_s > RECORDED_S  # the note outlasts what was recorded of it
    assert instrument.samples[1].trail_out_s == pytest.approx(0.0)


def test_a_run_asked_to_keep_the_tail_stores_the_padding_past_each_release(
    tmp_path: Path, ingest_settings: IngestFactory
) -> None:
    notes_json, samples = _dataset(tmp_path, pre_roll_seconds=0.1, post_roll_seconds=0.2)

    instrument = load_notes(notes_json, samples, ingest_settings("inst", keep_tail=True)).instruments[0]

    assert [sample.trail_out_s for sample in instrument.samples] == pytest.approx([0.0, 0.0])
    assert (instrument.pre_roll_s, instrument.post_roll_s) == pytest.approx((0.1, 0.2))


def test_a_manifest_stating_no_rolls_is_refused(tmp_path: Path, ingest_settings: IngestFactory) -> None:
    """A dataset says how it was cut, so one that says nothing is re-extracted rather than guessed at."""
    notes_json = _stated(tmp_path, {"config": {"tracked_ccs": [0, 1]}, "notes": list(_NOTES)})
    samples = _samples_dir(tmp_path, [0, 1])

    with pytest.raises(ValidationError, match="settings"):
        load_notes(notes_json, samples, ingest_settings("inst"))


def test_load_notes_raises_on_missing_wav(tmp_path: Path, ingest_settings: IngestFactory) -> None:
    samples = _samples_dir(tmp_path, [0])  # note index 1 has no WAV
    notes_json = tmp_path / "inst.notes.json"
    dump_notes(
        [
            NoteRecord(index=0, pitch=60, velocity=100, duration_s=0.5),
            NoteRecord(index=1, pitch=62, velocity=80, duration_s=0.5),
        ],
        notes_json,
    )
    with pytest.raises(ValueError):
        load_notes(notes_json, samples, ingest_settings("inst"))


def test_the_clock_a_render_ran_to_reaches_the_instrument_read_off_it(
    tmp_path: Path, ingest_settings: IngestFactory
) -> None:
    """A volume envelope is counted in ticks of the clock the material plays at, so the ingest carries it."""
    document = {
        "render": {"path": "song.mid", "tempo_bpm": 115.0, "time_signature": "4/4"},
        "settings": {"rolls": {"pre_roll_seconds": 0.0, "post_roll_seconds": 0.0}},
        "notes": list(_NOTES),
    }
    notes_json, samples = _stated(tmp_path, document), _samples_dir(tmp_path, [0, 1])

    instrument = load_notes(notes_json, samples, ingest_settings("inst")).instruments[0]

    assert instrument.tempo_bpm == pytest.approx(115.0)


def test_a_dataset_naming_no_clock_leaves_its_instrument_without_one(
    tmp_path: Path, ingest_settings: IngestFactory
) -> None:
    notes_json, samples = _dataset(tmp_path, pre_roll_seconds=0.0, post_roll_seconds=0.0)

    instrument = load_notes(notes_json, samples, ingest_settings("inst")).instruments[0]

    assert instrument.tempo_bpm is None


def test_a_written_dataset_states_the_clock_it_was_given(tmp_path: Path, ingest_settings: IngestFactory) -> None:
    """Every stage writes what it was handed, so the clock survives however many datasets it passes through."""
    samples = _samples_dir(tmp_path, [0])
    notes_json = tmp_path / "inst.notes.json"
    dump_notes(
        [NoteRecord(index=0, pitch=60, velocity=100, duration_s=0.5)],
        notes_json,
        tempo_bpm=115.0,
    )

    assert read_manifest(notes_json).tempo_bpm == pytest.approx(115.0)
    assert load_notes(notes_json, samples, ingest_settings("inst")).instruments[0].tempo_bpm == pytest.approx(115.0)


def test_a_dataset_written_without_a_clock_states_none(tmp_path: Path) -> None:
    notes_json = tmp_path / "inst.notes.json"
    dump_notes([NoteRecord(index=0, pitch=60, velocity=100, duration_s=0.5)], notes_json)

    assert read_manifest(notes_json).tempo_bpm is None
    assert "render" not in json.loads(notes_json.read_text(encoding="utf-8"))
