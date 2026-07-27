"""The shared per-pitch tasks and the objective scorer both optimizers minimize (``optimize/tasks.py``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.config.reduce import ReduceConfig, Representatives
from optisample.dsp.surrogate import (
    EncodeContext,
    EncodingParams,
    StoredSample,
    encode,
    render,
)
from optisample.dsp.timebase import seconds_to_frames
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.orchestrate import prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import (
    AudioMap,
    EvalContext,
    PitchTask,
    build_tasks,
    render_event,
    score_event,
    score_events,
    score_reconstruction,
)
from optisample.optimize.velocity_map import VelocityVolumeMap

SR = 44_100
PITCHES = (60, 62)

ReduceFactory = Callable[..., ReduceConfig]


# --- task building --------------------------------------------------------------------------------


def _instrument(material: list[NoteEvent]) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in PITCHES]
    return InstrumentSpec(id="piano", budget_kb=64.0, samples=samples, material=material)


def test_build_tasks_orders_by_pitch_and_picks_the_loudest_representative(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    audio: AudioMap = {
        SampleKey(pitch, 100): piano_note(pitch, 100, dur=0.5, seed=pitch * 137 + 100) for pitch in PITCHES
    }
    material = [
        NoteEvent(pitch=62, velocity=80, duration_s=0.4, count=1),
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=2),
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),
    ]
    tasks = build_tasks(_instrument(material), audio, graded_velocity_map, reduce())
    assert [task.pitch for task in tasks] == [60, 62]  # ascending, the DP's segmentation order
    by_pitch = {task.pitch: task for task in tasks}
    assert by_pitch[60].representative_key == SampleKey(60, 100)  # nearest the loudest velocity played there
    assert by_pitch[60].weight == pytest.approx(2 * 0.5 + 1 * 0.5)  # summed usage weight of its events
    assert len(by_pitch[60].events) == 2  # two distinct dynamics at pitch 60


def test_each_note_class_carries_the_volume_it_renders_at(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1)]
    task = build_tasks(_instrument(material), audio, graded_velocity_map, reduce())[0]
    assert task.events[0].volume == graded_velocity_map.volume(40)  # scoring reads this, not the velocity


def test_notes_sharing_a_reference_and_a_volume_are_scored_once(
    piano_note: Callable[..., NDArray[np.float64]],
    flat_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    """Two dynamics the map cannot tell apart reconstruct identically, so they become one scored class."""
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=2),  # weight 1.0
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),  # weight 0.5, same reference and volume
    ]
    exact = reduce(events={"duration_bucket_ratio": 1.0})
    task = build_tasks(_instrument(material), audio, flat_velocity_map, exact)[0]
    assert len(task.events) == 1
    assert task.events[0].weight == pytest.approx(1.5)  # the two notes' playing time added up
    assert task.events[0].velocity == 100  # the class is labelled by the loudest note it covers


def test_duration_bucketing_scores_similar_lengths_as_one_class(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [
        NoteEvent(pitch=60, velocity=100, duration_s=0.52, count=1),
        NoteEvent(pitch=60, velocity=100, duration_s=0.60, count=1),  # within a 1.25 ratio of the above
    ]
    task = build_tasks(_instrument(material), audio, graded_velocity_map, reduce())[0]
    assert len(task.events) == 1
    assert task.events[0].duration_s >= 0.60  # scored at least as long as the longest note it covers
    assert task.events[0].weight == pytest.approx(0.52 + 0.60)  # weight stays the real playing time


@pytest.mark.parametrize(
    ("representatives", "expected"),
    [
        (Representatives.NEAREST_LOUDEST, (SampleKey(60, 100),)),
        (Representatives.ALL, (SampleKey(60, 100), SampleKey(60, 40))),
    ],
)
def test_candidates_offer_the_survivors_the_policy_allows(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    representatives: Representatives,
    expected: tuple[SampleKey, ...],
) -> None:
    audio: AudioMap = {
        SampleKey(60, velocity): piano_note(60, velocity, dur=0.5, seed=60 * 137 + velocity) for velocity in (40, 100)
    }
    material = [NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=1)]
    config = reduce(dedupe={"representatives": representatives})
    tasks = build_tasks(_instrument(material), audio, graded_velocity_map, config)
    assert tasks[0].candidates == expected  # the representative always leads


def test_a_note_class_names_the_recording_it_is_scored_against(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    """A class carries its reference's identity, so the classes at one pitch stay tellable apart."""
    audio: AudioMap = {
        SampleKey(60, velocity): piano_note(60, velocity, dur=0.5, seed=60 * 137 + velocity) for velocity in (40, 100)
    }
    material = [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=1),
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),
    ]
    task = build_tasks(_instrument(material), audio, graded_velocity_map, reduce())[0]
    assert {event.reference_key for event in task.events} == {SampleKey(60, 100), SampleKey(60, 40)}
    assert len({event.identity for event in task.events}) == len(task.events)
    for event in task.events:
        assert np.array_equal(event.reference, audio[event.reference_key])


def test_build_tasks_raises_when_a_material_pitch_has_no_recording(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [NoteEvent(pitch=99, velocity=100, duration_s=0.4, count=1)]  # pitch 99 not recorded
    with pytest.raises(ValueError, match="no recorded sample for pitch 99"):
        build_tasks(_instrument(material), audio, graded_velocity_map, reduce())


# --- scoring --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scoring:
    """A ready-to-score setup: one pitch task, the shared eval context, and its re-encoded sample."""

    task: PitchTask
    context: EvalContext
    stored: StoredSample


@pytest.fixture
def scoring(
    piano_note: Callable[..., NDArray[np.float64]],
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    make_encode_ctx: Callable[..., EncodeContext],
) -> _Scoring:
    audio: AudioMap = {
        SampleKey(pitch, 100): piano_note(pitch, 100, dur=0.6, seed=pitch * 137 + 100) for pitch in PITCHES
    }
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=2) for pitch in PITCHES]
    settings = optimize_settings(sweep=sweep(rates=(SR,), depths=(16,), dither=False))
    inputs = prepare_run(_instrument(material), audio, SR, settings)
    task = inputs.tasks[0]
    params = EncodingParams(target_rate=SR, depth_bits=16, dither=False)
    stored = encode(task.representative, SR, params, make_encode_ctx(task.pitch))
    return _Scoring(task=task, context=inputs.context, stored=stored)


def test_score_events_yields_one_weighted_score_per_event(scoring: _Scoring) -> None:
    scores = list(score_events(scoring.stored, scoring.task, scoring.context))
    assert len(scores) == len(scoring.task.events)
    for score in scores:
        assert score.weighted_fidelity == pytest.approx(score.event.weight * score.report.fidelity)
        assert score.report.fidelity >= 0.0


def test_scoring_one_class_reads_what_the_whole_pitch_scorer_reads_for_it(scoring: _Scoring) -> None:
    """The atom: a caller scoring some of a pitch's classes lands on the numbers the sum is built from."""
    for score in score_events(scoring.stored, scoring.task, scoring.context):
        alone = score_event(scoring.stored, score.event, pitch=scoring.task.pitch, context=scoring.context)
        assert alone.fidelity == score.report.fidelity


def test_score_reconstruction_is_the_weight_normalized_mean(scoring: _Scoring) -> None:
    scores = list(score_events(scoring.stored, scoring.task, scoring.context))
    reconstruction = score_reconstruction(scoring.stored, scoring.task, scoring.context)
    expected = sum(score.weighted_fidelity for score in scores) / scoring.task.weight
    assert reconstruction == pytest.approx(expected)
    assert reconstruction >= 0.0


def test_score_reconstruction_of_a_weightless_task_is_zero(scoring: _Scoring) -> None:
    # A task with no events has zero weight; the scorer must return 0 rather than divide by zero.
    empty = PitchTask(
        pitch=60,
        weight=0.0,
        representative_key=SampleKey(60, 100),
        representative=np.zeros(4),
        candidates=(SampleKey(60, 100),),
        events=(),
    )
    assert score_reconstruction(scoring.stored, empty, scoring.context) == 0.0


# --- the pieces every artifact renders one note through ---------------------------------------------


def test_a_class_is_compared_against_its_source_note_held_for_its_scored_length(scoring: _Scoring) -> None:
    event = scoring.task.events[0]
    reference = event.scored_reference(SR)
    assert reference.size == seconds_to_frames(event.duration_s, SR)
    assert np.array_equal(reference, event.reference[: reference.size])


def test_the_representative_is_the_most_played_class(
    piano_note: Callable[..., NDArray[np.float64]],
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
) -> None:
    """Ties go to the longest, so the pitch stands for the dynamic and length it spends the most time on."""
    audio: AudioMap = {
        SampleKey(60, velocity): piano_note(60, velocity, dur=0.6, seed=velocity) for velocity in (40, 100)
    }
    material = [
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=4),
    ]
    settings = optimize_settings(sweep=sweep(rates=(SR,), depths=(16,), dither=False))
    task = prepare_run(_instrument(material), audio, SR, settings).tasks[0]
    assert task.representative_event.velocity == 100
    assert task.representative_event.weight == max(event.weight for event in task.events)


def test_rendering_a_class_matches_what_the_objective_scored(scoring: _Scoring) -> None:
    """The reconstruction written to disk is the one the fidelity report was measured on."""
    event = scoring.task.events[0]
    rendered = render_event(scoring.stored, event, pitch=scoring.task.pitch, sample_rate=SR)
    assert rendered.size == seconds_to_frames(event.duration_s, SR)
    assert np.array_equal(
        rendered, render(scoring.stored, SR, pitch=scoring.task.pitch, volume=event.volume, duration_s=event.duration_s)
    )
