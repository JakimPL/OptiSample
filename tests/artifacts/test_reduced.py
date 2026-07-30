from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts import DumpSettings, ReducedInstrument, dump_reduced
from optisample.artifacts.paths import reduced_paths
from optisample.artifacts.reduced import reduce_project, stored_frames
from optisample.config.reduce import DedupeKey
from optisample.io.audio import read_wav
from optisample.io.note_extractor import IngestSettings, load_notes
from optisample.keys import SampleKey
from optisample.model import (
    InstrumentSpec,
    Manifest,
    NoteEvent,
    ProjectSpec,
    SourceSample,
)
from optisample.optimize.orchestrate import prepare_run
from optisample.optimize.orchestrate.audio import LoadedInstrument
from optisample.optimize.orchestrate.looping import LoopedInstrument, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.summary import KeptRecording
from optisample.optimize.reduce.trim import NO_SCREEN
from optisample.optimize.tasks import AudioMap, StoredRecordings

Recordings = Callable[..., StoredRecordings]

SR = 44_100
PITCHES = (60, 62, 64)
VELOCITIES = (60, 100)
_TRIMMED_ONLY = 1  # the encodings a pitch offering no loop is swept at, which is the trimmed span alone


def _looped(instrument: InstrumentSpec, audio: AudioMap, settings: OptimizeSettings) -> LoopedInstrument:
    """The recordings a reduce run reads, with each one's loop settled the way the stage would."""
    return run_loops(_loaded(instrument, audio), settings)


def _loaded(instrument: InstrumentSpec, audio: AudioMap) -> LoadedInstrument:
    """Recordings handed straight in, standing in for a load that admitted every one of them."""
    return LoadedInstrument(instrument=instrument, audio=dict(audio), sample_rate=SR, screen=NO_SCREEN)


@pytest.fixture
def graded_audio(piano_note: Callable[..., NDArray[np.float64]]) -> AudioMap:
    """One recording per played ``(pitch, velocity)``, so dedup keeps every one of them."""
    return {
        SampleKey(pitch, velocity): piano_note(pitch, velocity, 0.6, seed=pitch * 137 + velocity)
        for pitch in PITCHES
        for velocity in VELOCITIES
    }


@pytest.fixture
def graded_instrument() -> InstrumentSpec:
    """An instrument whose material plays each recorded ``(pitch, velocity)`` twice."""
    return InstrumentSpec(
        id="piano",
        budget_kb=48.0,
        samples=[
            SourceSample(file=f"{pitch}_{velocity}.wav", pitch=pitch, velocity=velocity)
            for pitch in PITCHES
            for velocity in VELOCITIES
        ],
        material=[
            NoteEvent(pitch=pitch, velocity=velocity, duration_s=0.5, count=2)
            for pitch in PITCHES
            for velocity in VELOCITIES
        ],
    )


@pytest.fixture
def reduced(
    graded_instrument: InstrumentSpec,
    graded_audio: AudioMap,
    no_render_settings: DumpSettings,
    tmp_path: Path,
) -> ReducedInstrument:
    return dump_reduced(
        _looped(graded_instrument, graded_audio, no_render_settings.optimize), tmp_path, no_render_settings.optimize
    )


def _document(reduced: ReducedInstrument) -> dict[str, object]:
    return dict(json.loads(reduced.paths.reduction_json.read_text(encoding="utf-8")))


# --- the dataset that lands on disk -----------------------------------------------------------------


def test_a_reduced_run_writes_one_wav_per_surviving_recording(reduced: ReducedInstrument) -> None:
    written = sorted(path.name for path in reduced.paths.samples_dir.glob("*.wav"))
    assert len(written) == len(PITCHES) * len(VELOCITIES) == reduced.survivors
    assert written[0].startswith("0000_")


def test_a_survivor_is_named_by_its_join_index_and_its_key(reduced: ReducedInstrument) -> None:
    """The leading index is what a later ingest joins each note on; the rest says what the file holds."""
    for sample in _document(reduced)["samples"]:  # type: ignore[attr-defined]
        assert sample["file"] == f"{sample['index']:04d}_{sample['key']}.wav"
        assert (reduced.paths.samples_dir / sample["file"]).is_file()


def test_the_dataset_holds_one_note_per_played_note(
    reduced: ReducedInstrument, graded_instrument: InstrumentSpec
) -> None:
    """A material event standing for several played notes contributes one entry each, keeping its weight."""
    played = sum(event.count for event in graded_instrument.material)
    written = json.loads(reduced.paths.notes_json.read_text(encoding="utf-8"))["notes"]
    assert len(written) == played == reduced.notes


def test_every_note_points_at_a_written_survivor(reduced: ReducedInstrument) -> None:
    indices = {sample["index"] for sample in _document(reduced)["samples"]}  # type: ignore[attr-defined]
    written = json.loads(reduced.paths.notes_json.read_text(encoding="utf-8"))["notes"]
    assert {note["render"]["index"] for note in written} <= indices


def test_the_written_notes_start_at_the_onset(reduced: ReducedInstrument) -> None:
    """The survivors are already onset-aligned, so a reload trims no lead-in from them."""
    written = json.loads(reduced.paths.notes_json.read_text(encoding="utf-8"))["notes"]
    assert all(note["render"]["start_seconds"] == 0.0 for note in written)


# --- what the run reports ---------------------------------------------------------------------------


def test_the_document_names_the_key_the_survivors_were_kept_under(
    reduced: ReducedInstrument, no_render_settings: DumpSettings
) -> None:
    assert _document(reduced)["dedupe_key"] == no_render_settings.optimize.reduce.dedupe.key.value


def test_the_document_carries_the_full_reduction_summary(
    reduced: ReducedInstrument, graded_instrument: InstrumentSpec
) -> None:
    reduction = _document(reduced)["reduction"]
    assert reduction["kept_recordings"] == reduced.survivors  # type: ignore[index]
    assert reduction["played_notes"] == len(graded_instrument.material)  # type: ignore[index]
    assert len(reduction["grids"]) == len(PITCHES)  # type: ignore[index]


def test_the_run_reports_where_each_part_landed(reduced: ReducedInstrument, tmp_path: Path) -> None:
    assert reduced.paths == reduced_paths(tmp_path, "piano")
    assert reduced.instrument_id == "piano"
    assert reduced.elapsed_s >= 0.0


# --- auditions --------------------------------------------------------------------------------------


def test_each_played_pitch_gets_a_reference_beside_its_auditions(reduced: ReducedInstrument) -> None:
    folders = sorted(path.name for path in reduced.paths.auditions_dir.iterdir())
    assert folders == ["p060_C4", "p062_D4", "p064_E4"]
    assert all((reduced.paths.auditions_dir / folder / "reference.wav").is_file() for folder in folders)


def test_the_auditions_are_the_encodings_the_document_states(reduced: ReducedInstrument) -> None:
    """One rendering per swept encoding, so what the pre-pass settled is what can be listened to."""
    grids = _document(reduced)["reduction"]["grids"]  # type: ignore[index]
    swept = sum(grid["swept"] for grid in grids)
    rendered = list(reduced.paths.auditions_dir.rglob("*.wav"))
    assert len(rendered) == swept + len(grids) == reduced.auditions


def test_an_audition_is_named_by_the_encoding_it_holds(reduced: ReducedInstrument) -> None:
    """Every audition of a pitch is stored at the one format the reduction settled for it."""
    grid = _document(reduced)["reduction"]["grids"][0]  # type: ignore[index]
    folder = reduced.paths.auditions_dir / f"p{grid['pitch']:03d}_{grid['note']}"
    stored = grid["stored"]
    stem = f"r{stored['target_rate']}_d{stored['depth_bits']}" + ("_c" if stored["compress"] else "")

    auditions = sorted(path.name for path in folder.glob("*.wav") if path.stem != "reference")
    looped = [f"{stem}_loop.wav"] if grid["swept"] > _TRIMMED_ONLY else []

    assert auditions == sorted([f"{stem}.wav", *looped])


def test_an_audition_runs_as_long_as_the_note_it_stands_for(reduced: ReducedInstrument) -> None:
    reference, rate = read_wav(reduced.paths.auditions_dir / "p060_C4" / "reference.wav")
    for candidate_path in (reduced.paths.auditions_dir / "p060_C4").glob("r*.wav"):
        candidate, candidate_rate = read_wav(candidate_path)
        assert candidate_rate == rate
        assert candidate.size == reference.size


# --- the length a survivor is written at --------------------------------------------------------------


@pytest.mark.parametrize(("required_s", "frames"), [(0.5, 22_050), (0.5001, 22_055), (1.2, 52_920)])
def test_a_survivor_is_written_long_enough_for_what_its_pitch_asks(required_s: float, frames: int) -> None:
    """Rounding up keeps the written length at or above the requirement a reload measures it against."""
    recording = KeptRecording(SampleKey(60, 100), duration_s=2.0, required_duration_s=required_s)
    assert stored_frames(recording, SR) == frames
    assert stored_frames(recording, SR) / SR >= required_s


def test_a_survivor_longer_than_asked_is_trimmed_to_the_requirement(
    graded_instrument: InstrumentSpec,
    graded_audio: AudioMap,
    no_render_settings: DumpSettings,
    tmp_path: Path,
    recordings: Recordings,
) -> None:
    reduced = dump_reduced(
        _looped(graded_instrument, graded_audio, no_render_settings.optimize), tmp_path, no_render_settings.optimize
    )
    inputs = prepare_run(graded_instrument, recordings(graded_audio, SR), no_render_settings.optimize)
    required = {recording.key.label: recording.required_duration_s for recording in inputs.reduction.recordings}
    for sample in _document(reduced)["samples"]:  # type: ignore[attr-defined]
        held = len(graded_audio[SampleKey(*_key_parts(sample["key"]))]) / SR
        assert sample["duration_s"] == pytest.approx(min(held, stored_frames_seconds(required[sample["key"]])))


def stored_frames_seconds(required_s: float) -> float:
    """The written length ``required_s`` rounds up to, in seconds at the analysis rate."""
    return stored_frames(KeptRecording(SampleKey(0, 0), duration_s=0.0, required_duration_s=required_s), SR) / SR


def _key_parts(label: str) -> tuple[int, int]:
    """``p060_C4_v100`` -> ``(60, 100)``, the pitch and velocity the label spells out."""
    pitch, _, velocity = label.split("_")
    return int(pitch[1:]), int(velocity[1:])


# --- the round trip -----------------------------------------------------------------------------------


def test_a_reduced_dataset_reloads_into_the_same_survivors(
    graded_instrument: InstrumentSpec,
    graded_audio: AudioMap,
    no_render_settings: DumpSettings,
    tmp_path: Path,
    ingest_settings: Callable[..., IngestSettings],
    recordings: Recordings,
) -> None:
    """The point of the dataset: an allocation run reads it back and reduces to exactly what was written."""
    reduced = dump_reduced(
        _looped(graded_instrument, graded_audio, no_render_settings.optimize), tmp_path, no_render_settings.optimize
    )
    reloaded = load_notes(
        reduced.paths.notes_json,
        reduced.paths.samples_dir,
        ingest_settings("piano", budget_kb=48.0),
    )
    instrument = reloaded.instruments[0]
    assert len(instrument.material) == reduced.notes
    again = prepare_run(instrument, recordings(_decode(reduced.paths.samples_dir), SR), no_render_settings.optimize)
    written = _document(reduced)["reduction"]  # type: ignore[assignment]
    assert again.reduction.kept_recordings == written["kept_recordings"]  # type: ignore[index]
    assert [recording.key.label for recording in again.reduction.recordings] == [
        recording["key"] for recording in written["recordings"]  # type: ignore[index]
    ]


def _decode(samples_dir: Path) -> AudioMap:
    """The written survivors read back, keyed the way the label in each filename spells the identity."""
    audio: dict[SampleKey, NDArray[np.float64]] = {}
    for path in sorted(samples_dir.glob("*.wav")):
        pitch, velocity = _key_parts(path.stem.split("_", 1)[1])
        audio[SampleKey(pitch, velocity)] = read_wav(path)[0]
    return audio


def test_a_project_reduces_every_instrument_under_one_root(
    graded_instrument: InstrumentSpec,
    graded_audio: AudioMap,
    no_render_settings: DumpSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "optisample.artifacts.reduced.load_run_audio",
        lambda instrument, settings: _loaded(instrument, graded_audio),
    )
    manifest = Manifest(project=ProjectSpec(name="demo"), instruments=[graded_instrument])
    results = reduce_project(manifest, tmp_path, no_render_settings.optimize)
    assert [result.instrument_id for result in results] == ["piano"]
    assert (tmp_path / "piano.notes.json").is_file()
    assert (tmp_path / "piano").is_dir()


def test_the_dedupe_key_the_dataset_was_reduced_under_reaches_the_document(
    graded_instrument: InstrumentSpec,
    graded_audio: AudioMap,
    no_render_settings: DumpSettings,
    reduce: Callable[..., object],
    tmp_path: Path,
) -> None:
    """A coarser key keeps one recording per pitch, and the document says which key that was."""
    settings = OptimizeSettings(
        loop=no_render_settings.optimize.loop,
        sweep=no_render_settings.optimize.sweep,
        reduce=reduce(dedupe={"key": DedupeKey.PITCH}),  # type: ignore[arg-type]
        layers=no_render_settings.optimize.layers,
        encode=no_render_settings.optimize.encode,
        metrics=no_render_settings.optimize.metrics,
        velocity=no_render_settings.optimize.velocity,
        method=no_render_settings.optimize.method,
        energy_exponent=no_render_settings.optimize.energy_exponent,
        max_samples=no_render_settings.optimize.max_samples,
        target=no_render_settings.optimize.target,
    )
    loudest = {SampleKey(pitch, 100): graded_audio[SampleKey(pitch, 100)] for pitch in PITCHES}
    reduced = dump_reduced(_looped(graded_instrument, loudest, settings), tmp_path, settings)
    assert reduced.survivors == len(PITCHES)
    assert _document(reduced)["dedupe_key"] == DedupeKey.PITCH.value
