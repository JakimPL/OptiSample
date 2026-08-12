from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optisample.artifacts.documents.plan import (
    BudgetRecord,
    InstrumentRecord,
    KeyboardRecord,
    ModuleSizeRecord,
    PitchItemRecord,
    PlanDocument,
    ZoneItemRecord,
)
from optisample.artifacts.documents.reduction import ReductionDocument
from optisample.artifacts.documents.velocity import VelocityMapDocument
from optisample.artifacts.paths import pipeline_paths, plan_paths
from optisample.artifacts.serialize import write_json
from optisample.cluster.stages import (
    Stage,
    StageSettings,
    available_stages,
    stage_dir,
    stage_recordings,
)
from optisample.config import OptiConfig
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NOTES_SUFFIX, NoteRecord, dump_notes
from optisample.music import midi_to_freq, note_name
from optisample.progress import NO_PROGRESS

SR = 8_000

_INSTRUMENT = "Piano"
_STRATEGY = "grouped"
_PITCHES = (48, 60, 72)  # the notes a stage's dataset holds, one recording apiece
_VELOCITY = 100
_TAKE_S = 0.5
_HELD_S = 0.4  # how long each note sounds, which is the playing time one recording earns
_STORED_S = 0.3  # the span the allocation paid to keep of the one sample it stores
_STORED_RATE = 4_000  # a stored sample carries the rate the budget could afford, its own and no other's
_SILENT = 1.0e-6  # a peak this far down leaves a recording under any silence floor a run screens by
_TO_THE_RELEASE = False  # a reading that stops where the note does, which is the span a run stores
_BOUND_S = 0.25  # a length bound under every take, so what the trim keeps is what the reading answers
_DATASET_STAGES = (Stage.SUBSET, Stage.LOOPED, Stage.REDUCED)

_ENCODING = {
    "target_rate": _STORED_RATE,
    "depth_bits": 8,
    "compress": False,
    "trim_s": _STORED_S,
    "loop_index": None,
    "loop": None,
    "decay": None,
    "frames": round(_STORED_S * _STORED_RATE),
    "stored_bytes": round(_STORED_S * _STORED_RATE),
    "distortion": 2.5,
    "hull_size": 3,
}


def _take(pitch: int, duration_s: float = _TAKE_S, sample_rate: int = SR) -> np.ndarray:
    """A recording of one note: a tone at its own pitch, loud enough to clear any silence floor."""
    times = np.arange(round(duration_s * sample_rate), dtype=np.float64) / sample_rate
    return np.asarray(0.5 * np.sin(2.0 * np.pi * midi_to_freq(pitch) * times), dtype=np.float64)


def _write_dataset(directory: Path, notes: list[NoteRecord], takes: dict[int, np.ndarray]) -> None:
    """One stage's dataset: the recordings it wrote, beside the manifest naming the notes they answer."""
    samples_dir = directory / _INSTRUMENT
    samples_dir.mkdir(parents=True, exist_ok=True)
    for index, take in takes.items():
        write_wav(samples_dir / f"{index:04d}_p{index:03d}_v{_VELOCITY:03d}.wav", take, SR)

    dump_notes(notes, directory / f"{_INSTRUMENT}{NOTES_SUFFIX}")


def _with_post_roll(notes_json: Path, post_roll_s: float) -> None:
    """The padding a source render kept past each release, as that render's own manifest records it."""
    document = json.loads(notes_json.read_text(encoding="utf-8"))
    document["settings"]["rolls"]["post_roll_seconds"] = post_roll_s
    notes_json.write_text(json.dumps(document), encoding="utf-8")


def _plain_notes() -> list[NoteRecord]:
    """One note per recording, which is the grid a slice of a source holds."""
    return [
        NoteRecord(index=index, pitch=pitch, velocity=_VELOCITY, duration_s=_HELD_S)
        for index, pitch in enumerate(_PITCHES)
    ]


def _plan_document(strategy: str) -> PlanDocument:
    """One plan holding a single stored sample, which is what the allocated stage offers a reader."""
    shared = {
        "instrument_id": _INSTRUMENT,
        "objective": 12.0,
        "energy_exponent": 0.5,
        "budget": BudgetRecord(
            module_budget_bytes=8_192, sample_budget_bytes=8_000, used_bytes=1_200, module_bytes=1_400
        ),
        "module": ModuleSizeRecord(total_bytes=1_400, header_bytes=100, pcm_bytes=1_200, pattern_bytes=100),
        "keyboard": KeyboardRecord(numbered=120, played=3, answered=120),
        "reduction": ReductionDocument(
            listed_recordings=3, kept_recordings=3, played_notes=3, scored_classes=3, recordings=[], grids=[]
        ),
        "velocity_map": VelocityMapDocument(reference_volume=64, anchors=[], volumes=[64] * 128),
        "instruments": [
            InstrumentRecord(
                index=0,
                name=_INSTRUMENT,
                layer=0,
                band="v000-v127",
                lowest_velocity=0,
                highest_velocity=127,
                lowest_pitch=_PITCHES[0],
                highest_pitch=_PITCHES[-1],
                keys=len(_PITCHES),
                samples=1,
                stored_bytes=1_200,
                weight=_HELD_S,
                objective_share=12.0,
                envelope_drift_db=1.0,
            )
        ],
    }
    if strategy == "grouped":
        return PlanDocument(
            strategy="grouped",
            zones=[
                ZoneItemRecord(
                    layer=0,
                    keys=[_PITCHES[0], _PITCHES[-1]],
                    pitches=list(_PITCHES),
                    representative=_PITCHES[1],
                    representative_velocity=_VELOCITY,
                    weight=3.0,
                    **_ENCODING,
                )
            ],
            **shared,
        )

    return PlanDocument(
        strategy="ungrouped",
        method="exact",
        pitches=[
            PitchItemRecord(
                pitch=_PITCHES[1],
                note=note_name(_PITCHES[1]),
                weight=3.0,
                representative_velocity=_VELOCITY,
                **_ENCODING,
            )
        ],
        **shared,
    )


def _stored_label(strategy: str) -> str:
    """How each strategy names the one sample it stored, which is the WAV a reader resolves it through."""
    representative = _PITCHES[1]
    if strategy == "grouped":
        return f"zone00_rep{representative:03d}_{note_name(representative)}"

    return f"p{representative:03d}_{note_name(representative)}"


def _write_plan(root: Path, strategy: str) -> None:
    """One allocation as the dumper leaves it: its plan, beside the samples decoded back out of the module."""
    paths = plan_paths(pipeline_paths(root).optimized_dir / _INSTRUMENT, strategy)
    paths.samples_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths.plan_json, _plan_document(strategy))
    write_wav(
        paths.sample_wav(_stored_label(strategy)),
        _take(_PITCHES[1], _STORED_S, _STORED_RATE),
        _STORED_RATE,
    )


@pytest.fixture
def settings(config: OptiConfig) -> StageSettings:
    """What a reading of a stage is carried out with, on the terms the pipeline's own stages read by."""
    return StageSettings(
        instrument_id=_INSTRUMENT,
        strategy=_STRATEGY,
        dedupe=config.reduce.dedupe,
        trim=config.reduce.trim,
        keep_tail=_TO_THE_RELEASE,
        progress=NO_PROGRESS,
    )


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A whole chained run as it lands on disk: three datasets and one allocation, under one root."""
    paths = pipeline_paths(tmp_path)
    takes = {index: _take(pitch) for index, pitch in enumerate(_PITCHES)}
    for directory in (paths.subset_dir, paths.looped_dir, paths.reduced_dir):
        _write_dataset(directory, _plain_notes(), takes)

    _write_plan(tmp_path, _STRATEGY)
    return tmp_path


def test_the_stage_directories_are_the_ones_a_chained_run_files_its_steps_in(tmp_path: Path) -> None:
    paths = pipeline_paths(tmp_path)

    assert [stage_dir(paths, stage) for stage in Stage] == [
        paths.subset_dir,
        paths.looped_dir,
        paths.reduced_dir,
        paths.optimized_dir,
    ]


def test_a_whole_run_offers_every_stage_it_left_behind(run: Path, settings: StageSettings) -> None:
    assert available_stages(run, settings) == tuple(Stage)


def test_a_stage_a_run_never_reached_is_left_out(tmp_path: Path, settings: StageSettings) -> None:
    """A chain begun from an already-sliced dataset writes no subset, and a reader offers what is there."""
    paths = pipeline_paths(tmp_path)
    _write_dataset(paths.looped_dir, _plain_notes(), {index: _take(pitch) for index, pitch in enumerate(_PITCHES)})

    assert available_stages(tmp_path, settings) == (Stage.LOOPED,)


@pytest.mark.parametrize("stage", _DATASET_STAGES)
def test_a_dataset_stage_offers_one_recording_per_file_it_holds(
    stage: Stage, run: Path, settings: StageSettings
) -> None:
    corpus = stage_recordings(run, stage, settings)

    assert corpus.stage == stage
    assert corpus.instrument_id == _INSTRUMENT
    assert corpus.size == len(_PITCHES)
    assert [recording.key.pitch for recording in corpus.recordings] == list(_PITCHES)
    assert [recording.note for recording in corpus.recordings] == [note_name(pitch) for pitch in _PITCHES]
    assert [recording.root_hz for recording in corpus.recordings] == [midi_to_freq(pitch) for pitch in _PITCHES]


def test_a_recording_is_read_over_the_span_its_note_sounds(run: Path, settings: StageSettings) -> None:
    """The decode a run gives its survivors, so a point stands for the audio the pipeline worked from."""
    corpus = stage_recordings(run, Stage.SUBSET, settings)

    for recording in corpus.recordings:
        assert recording.sample_rate == SR
        assert recording.duration_s == pytest.approx(_TAKE_S)
        assert recording.label == recording.file.stem


def test_the_playing_time_a_recording_earns_is_what_the_material_gives_its_notes(
    run: Path, settings: StageSettings
) -> None:
    corpus = stage_recordings(run, Stage.SUBSET, settings)

    assert [recording.weight for recording in corpus.recordings] == [pytest.approx(_HELD_S)] * len(_PITCHES)


def test_a_recording_answering_several_notes_is_offered_once_carrying_all_of_their_time(
    tmp_path: Path, settings: StageSettings
) -> None:
    """A stage storing one recording for a run of notes names it once per note; the space places one point."""
    paths = pipeline_paths(tmp_path)
    served = _PITCHES[0]
    notes = [
        NoteRecord(index=0, pitch=served, velocity=_VELOCITY, duration_s=_HELD_S),
        NoteRecord(index=0, pitch=served, velocity=_VELOCITY, duration_s=_HELD_S),
    ]
    _write_dataset(paths.looped_dir, notes, {0: _take(served)})

    corpus = stage_recordings(tmp_path, Stage.LOOPED, settings)

    assert corpus.size == 1
    assert corpus.recordings[0].weight == pytest.approx(2.0 * _HELD_S)


def test_a_recording_carrying_no_signal_holds_no_point_to_place(tmp_path: Path, settings: StageSettings) -> None:
    paths = pipeline_paths(tmp_path)
    takes = {index: _take(pitch) for index, pitch in enumerate(_PITCHES)}
    takes[0] = np.full(round(_TAKE_S * SR), _SILENT)
    _write_dataset(paths.subset_dir, _plain_notes(), takes)

    corpus = stage_recordings(tmp_path, Stage.SUBSET, settings)

    assert [recording.key.pitch for recording in corpus.recordings] == list(_PITCHES[1:])


def test_a_recording_is_read_through_the_trim_a_run_stores_by(tmp_path: Path, settings: StageSettings) -> None:
    """The reading is the decode a run gives its survivors, so the length bound it stores under holds here."""
    paths = pipeline_paths(tmp_path)
    _write_dataset(paths.subset_dir, _plain_notes(), {index: _take(pitch) for index, pitch in enumerate(_PITCHES)})
    bounded = replace(settings, trim=settings.trim.model_copy(update={"max_length_s": _BOUND_S}))

    corpus = stage_recordings(tmp_path, Stage.SUBSET, bounded)

    for recording in corpus.recordings:
        assert recording.duration_s == pytest.approx(_BOUND_S)


def test_the_allocated_stage_offers_the_samples_the_module_actually_plays(run: Path, settings: StageSettings) -> None:
    """A stored sample stands at the rate the budget could afford, which is the audio a player hears."""
    corpus = stage_recordings(run, Stage.OPTIMIZED, settings)
    (stored,) = corpus.recordings

    assert corpus.stage == Stage.OPTIMIZED
    assert stored.key.pitch == _PITCHES[1]
    assert stored.key.velocity == _VELOCITY
    assert stored.sample_rate == _STORED_RATE
    assert stored.duration_s == pytest.approx(_STORED_S)
    assert stored.weight == pytest.approx(3.0)


def test_an_ungrouped_allocation_is_read_through_the_naming_that_wrote_it(
    tmp_path: Path, settings: StageSettings
) -> None:
    """Both strategies name a stored sample after the pitch it is rooted at, so either plan reads back."""
    _write_plan(tmp_path, "ungrouped")
    ungrouped = StageSettings(
        instrument_id=_INSTRUMENT,
        strategy="ungrouped",
        dedupe=settings.dedupe,
        trim=settings.trim,
        keep_tail=settings.keep_tail,
        progress=settings.progress,
    )

    corpus = stage_recordings(tmp_path, Stage.OPTIMIZED, ungrouped)
    (stored,) = corpus.recordings

    assert available_stages(tmp_path, ungrouped) == (Stage.OPTIMIZED,)
    assert stored.key.pitch == _PITCHES[1]
    assert stored.label == _stored_label("ungrouped")


def test_the_tail_past_a_release_is_read_where_a_reader_asks_for_it(tmp_path: Path, settings: StageSettings) -> None:
    """A run stores the span its material plays, and a decline is read as far as the reading was kept."""
    paths = pipeline_paths(tmp_path)
    _write_dataset(paths.subset_dir, _plain_notes(), {index: _take(pitch) for index, pitch in enumerate(_PITCHES)})
    _with_post_roll(paths.subset_dir / f"{_INSTRUMENT}{NOTES_SUFFIX}", _TAKE_S - _HELD_S)

    stopping = stage_recordings(tmp_path, Stage.SUBSET, settings).recordings
    tailing = stage_recordings(tmp_path, Stage.SUBSET, replace(settings, keep_tail=True)).recordings

    assert [recording.duration_s for recording in stopping] == [pytest.approx(_HELD_S)] * len(_PITCHES)
    assert [recording.duration_s for recording in tailing] == [pytest.approx(_TAKE_S)] * len(_PITCHES)
