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
from optisample.metrics.composite import evaluate
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.orchestrate import prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import (
    AudioMap,
    EvalContext,
    PitchTask,
    TaskInputs,
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
TaskInputsFactory = Callable[..., TaskInputs]


# --- task building --------------------------------------------------------------------------------


def _instrument(material: list[NoteEvent]) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in PITCHES]
    return InstrumentSpec(id="piano", budget_kb=64.0, samples=samples, material=material)


def test_build_tasks_orders_by_pitch_and_picks_the_loudest_representative(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    task_inputs: TaskInputsFactory,
) -> None:
    audio: AudioMap = {
        SampleKey(pitch, 100): piano_note(pitch, 100, dur=0.5, seed=pitch * 137 + 100) for pitch in PITCHES
    }
    material = [
        NoteEvent(pitch=62, velocity=80, duration_s=0.4, count=1),
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=2),
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),
    ]
    tasks = build_tasks(_instrument(material), task_inputs(audio, graded_velocity_map))
    assert [task.pitch for task in tasks] == [60, 62]  # ascending, the DP's segmentation order
    by_pitch = {task.pitch: task for task in tasks}
    assert by_pitch[60].representative_key == SampleKey(60, 100)  # nearest the loudest velocity played there
    assert by_pitch[60].weight == pytest.approx(2 * 0.5 + 1 * 0.5)  # summed usage weight of its events
    assert len(by_pitch[60].events) == 2  # two distinct dynamics at pitch 60


def test_each_note_class_carries_the_volume_it_renders_at(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    task_inputs: TaskInputsFactory,
) -> None:
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1)]
    task = build_tasks(_instrument(material), task_inputs(audio, graded_velocity_map))[0]
    assert task.events[0].volume == graded_velocity_map.volume(40)  # scoring reads this, not the velocity


def test_notes_sharing_a_reference_and_a_volume_are_scored_once(
    piano_note: Callable[..., NDArray[np.float64]],
    flat_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    task_inputs: TaskInputsFactory,
) -> None:
    """Two dynamics the map cannot tell apart reconstruct identically, so they become one scored class."""
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=2),  # weight 1.0
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),  # weight 0.5, same reference and volume
    ]
    exact = reduce(events={"duration_bucket_ratio": 1.0})
    task = build_tasks(_instrument(material), task_inputs(audio, flat_velocity_map, reduce=exact))[0]
    assert len(task.events) == 1
    assert task.events[0].weight == pytest.approx(1.5)  # the two notes' playing time added up
    assert task.events[0].velocity == 100  # the class is labelled by the loudest note it covers


def test_duration_bucketing_scores_similar_lengths_as_one_class(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    task_inputs: TaskInputsFactory,
) -> None:
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [
        NoteEvent(pitch=60, velocity=100, duration_s=0.52, count=1),
        NoteEvent(pitch=60, velocity=100, duration_s=0.60, count=1),  # within a 1.25 ratio of the above
    ]
    task = build_tasks(_instrument(material), task_inputs(audio, graded_velocity_map))[0]
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
    task_inputs: TaskInputsFactory,
    representatives: Representatives,
    expected: tuple[SampleKey, ...],
) -> None:
    audio: AudioMap = {
        SampleKey(60, velocity): piano_note(60, velocity, dur=0.5, seed=60 * 137 + velocity) for velocity in (40, 100)
    }
    material = [NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=1)]
    config = reduce(dedupe={"representatives": representatives})
    tasks = build_tasks(_instrument(material), task_inputs(audio, graded_velocity_map, reduce=config))
    assert tasks[0].candidates == expected  # the representative always leads


def test_a_note_class_names_the_recording_it_is_scored_against(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    task_inputs: TaskInputsFactory,
) -> None:
    """A class carries its reference's identity, so the classes at one pitch stay tellable apart."""
    audio: AudioMap = {
        SampleKey(60, velocity): piano_note(60, velocity, dur=0.5, seed=60 * 137 + velocity) for velocity in (40, 100)
    }
    material = [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=1),
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),
    ]
    task = build_tasks(_instrument(material), task_inputs(audio, graded_velocity_map))[0]
    assert {event.reference_key for event in task.events} == {SampleKey(60, 100), SampleKey(60, 40)}
    assert len({event.identity for event in task.events}) == len(task.events)
    for event in task.events:
        assert np.array_equal(event.reference, audio[event.reference_key])


def test_a_quiet_class_carries_less_of_the_objective_than_a_loud_one(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    task_inputs: TaskInputsFactory,
) -> None:
    """The point of the energy weighting: the same playing time costs less where the note plays softly."""
    audio: AudioMap = {
        SampleKey(60, velocity): piano_note(60, velocity, dur=0.5, seed=60 * 137 + velocity) for velocity in (40, 100)
    }
    material = [NoteEvent(pitch=60, velocity=velocity, duration_s=0.5, count=1) for velocity in (40, 100)]
    task = build_tasks(_instrument(material), task_inputs(audio, graded_velocity_map))[0]
    by_velocity = {event.velocity: event for event in task.events}
    assert by_velocity[40].weight == pytest.approx(by_velocity[100].weight)  # same playing time
    assert by_velocity[40].objective_weight < by_velocity[100].objective_weight
    assert task.objective_weight == pytest.approx(sum(event.objective_weight for event in task.events))


def test_an_exponent_of_zero_prices_every_note_by_its_playing_time_alone(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    task_inputs: TaskInputsFactory,
) -> None:
    """The weighting switched off leaves the objective reading exactly the material's own weight."""
    audio: AudioMap = {
        SampleKey(60, velocity): piano_note(60, velocity, dur=0.5, seed=60 * 137 + velocity) for velocity in (40, 100)
    }
    material = [NoteEvent(pitch=60, velocity=velocity, duration_s=0.5, count=1) for velocity in (40, 100)]
    inputs = task_inputs(audio, graded_velocity_map, energy_exponent=0.0)
    task = build_tasks(_instrument(material), inputs)[0]
    assert task.objective_weight == pytest.approx(task.weight)


def test_a_class_is_weighed_over_the_span_it_is_scored_on(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    task_inputs: TaskInputsFactory,
) -> None:
    """A short note off a decaying recording keeps the energy of its own stretch, not the whole file's."""
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=2.0, seed=60 * 137 + 100)}
    exact = reduce(events={"duration_bucket_ratio": 1.0})
    lengths = [NoteEvent(pitch=60, velocity=100, duration_s=duration, count=1) for duration in (0.2, 2.0)]
    task = build_tasks(_instrument(lengths), task_inputs(audio, graded_velocity_map, reduce=exact))[0]
    by_length = {event.duration_s: event for event in task.events}
    assert by_length[0.2].energy_weight > by_length[2.0].energy_weight  # the attack outweighs the decay


def test_build_tasks_raises_when_a_material_pitch_has_no_recording(
    piano_note: Callable[..., NDArray[np.float64]],
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    task_inputs: TaskInputsFactory,
) -> None:
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [NoteEvent(pitch=99, velocity=100, duration_s=0.4, count=1)]  # pitch 99 not recorded
    with pytest.raises(ValueError, match="no recorded sample for pitch 99"):
        build_tasks(_instrument(material), task_inputs(audio, graded_velocity_map))


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
    settings = optimize_settings(sweep=sweep(rates=(SR,), depth=16, dither=False))
    inputs = prepare_run(_instrument(material), audio, SR, settings)
    task = inputs.tasks[0]
    params = EncodingParams(target_rate=SR, depth_bits=16, dither=False)
    stored = encode(task.representative, SR, params, make_encode_ctx(task.pitch))
    return _Scoring(task=task, context=inputs.context, stored=stored)


def test_score_events_yields_one_weighted_score_per_event(scoring: _Scoring) -> None:
    scores = list(score_events(scoring.stored, scoring.task, scoring.context))
    assert len(scores) == len(scoring.task.events)
    for score in scores:
        assert score.weighted_fidelity == pytest.approx(score.event.objective_weight * score.report.fidelity)
        assert score.report.fidelity >= 0.0


def test_scoring_one_class_reads_what_the_whole_pitch_scorer_reads_for_it(scoring: _Scoring) -> None:
    """The atom: a caller scoring some of a pitch's classes lands on the numbers the sum is built from."""
    for score in score_events(scoring.stored, scoring.task, scoring.context):
        alone = score_event(scoring.stored, score.event, pitch=scoring.task.pitch, context=scoring.context)
        assert alone.fidelity == score.report.fidelity


def test_score_reconstruction_is_the_weight_normalized_mean(scoring: _Scoring) -> None:
    scores = list(score_events(scoring.stored, scoring.task, scoring.context))
    reconstruction = score_reconstruction(scoring.stored, scoring.task, scoring.context)
    expected = sum(score.weighted_fidelity for score in scores) / scoring.task.objective_weight
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


def test_a_class_is_measured_over_its_source_note_held_for_its_scored_length(scoring: _Scoring) -> None:
    event = scoring.task.events[0]
    span = event.scored_span(SR)
    assert span.size == seconds_to_frames(event.duration_s, SR)
    assert np.array_equal(span, event.reference[: span.size])


def _stored_to_the_note(task: PitchTask, make_encode_ctx: Callable[..., EncodeContext]) -> StoredSample:
    """A pitch's recording stored for exactly the longest note it serves, so its ramp closes that note."""
    params = EncodingParams(target_rate=SR, depth_bits=16, dither=False, trim_s=task.max_duration_s)
    return encode(task.representative, SR, params, make_encode_ctx(task.pitch))


def test_a_note_held_to_the_stored_end_is_compared_to_a_ground_truth_closed_the_same_way(
    scoring: _Scoring, make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    stored = _stored_to_the_note(scoring.task, make_encode_ctx)
    event = scoring.task.representative_event
    span = event.scored_span(SR)

    closed = event.scored_reference(stored, SR, pitch=scoring.task.pitch)

    assert closed.size == span.size
    assert float(closed[-1]) == 0.0
    untouched = span.size - stored.release_frames
    assert np.array_equal(closed[:untouched], span[:untouched])


def test_the_ramp_a_sample_closes_on_is_left_out_of_what_the_score_charges(
    scoring: _Scoring, make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """The same ramp on both sides leaves the score reading the codec, where the bare recording reads the ramp."""
    stored = _stored_to_the_note(scoring.task, make_encode_ctx)
    event = scoring.task.representative_event
    candidate = render_event(stored, event, pitch=scoring.task.pitch, sample_rate=SR)

    closed = score_event(stored, event, pitch=scoring.task.pitch, context=scoring.context).fidelity
    against_the_bare_span = evaluate(event.scored_span(SR), candidate, SR, scoring.context.composite).fidelity

    assert closed < against_the_bare_span


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
    settings = optimize_settings(sweep=sweep(rates=(SR,), depth=16, dither=False))
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
