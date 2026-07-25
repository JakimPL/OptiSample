from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from optisample.io.audio import write_wav
from optisample.io.note_extractor import IngestSettings, NoteRecord, dump_notes, index_of_wav, load_notes
from optisample.model import ProjectSpec

SR = 8_000


def _write_wav(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    write_wav(path, rng.standard_normal(SR).astype(np.float64) * 0.1, SR)


def _samples_dir(tmp_path: Path, indices: list[int]) -> Path:
    samples = tmp_path / "samples"
    samples.mkdir()
    for index in indices:
        _write_wav(samples / f"{index:04d}_p60_v100.wav", seed=index)
    return samples


def _settings(**overrides: object) -> IngestSettings:
    data: dict[str, object] = {"instrument_id": "inst", "budget_kb": 48.0, "project": ProjectSpec(name="song")}
    data.update(overrides)
    return IngestSettings(**data)  # type: ignore[arg-type]


def test_index_of_wav_reads_leading_token() -> None:
    assert index_of_wav("0007_p060_v100_cc0-3.wav") == 7
    assert index_of_wav(Path("12_p60.wav")) == 12


def test_index_of_wav_rejects_missing_index() -> None:
    with pytest.raises(ValueError):
        index_of_wav("p60_v100.wav")


def test_load_notes_joins_by_index_and_derives_duration(tmp_path: Path) -> None:
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
    manifest = load_notes(notes_json, samples, _settings())

    instrument = manifest.instruments[0]
    assert instrument.id == "inst"
    assert [sample.file.name for sample in instrument.samples] == ["0000_p60_v100.wav", "0001_p60_v100.wav"]
    assert all(sample.file.is_absolute() for sample in instrument.samples)
    assert [event.duration_s for event in instrument.material] == pytest.approx([0.5, 1.25])
    assert instrument.material[1].cc_averages == {1: 63.875}
    assert instrument.samples[1].cc_averages == {1: 63.875}


def test_load_notes_coerces_cc_keys_and_clamps_lead_in(tmp_path: Path) -> None:
    samples = _samples_dir(tmp_path, [0, 1])
    notes_json = tmp_path / "inst.notes.json"
    notes_json.write_text(
        json.dumps(
            {
                "config": {"tracked_ccs": [0, 1]},
                "notes": [
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
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_notes(notes_json, samples, _settings(pre_roll_s=0.1, post_roll_s=0.2))

    instrument = manifest.instruments[0]
    assert instrument.samples[0].cc_averages == {0: 3.0, 1: 63.875}  # string CC keys coerce to int
    assert instrument.material[0].duration_s == pytest.approx(0.5)  # release_end - start
    assert instrument.material[1].duration_s == pytest.approx(1.5)
    assert instrument.samples[0].lead_in_s == pytest.approx(0.02)  # start < pre_roll, so clamped to start
    assert instrument.samples[1].lead_in_s == pytest.approx(0.1)  # pre_roll < start, so the full pre_roll
    assert instrument.pre_roll_s == pytest.approx(0.1)  # recorded for provenance
    assert instrument.post_roll_s == pytest.approx(0.2)


def test_load_notes_raises_on_missing_wav(tmp_path: Path) -> None:
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
        load_notes(notes_json, samples, _settings())
