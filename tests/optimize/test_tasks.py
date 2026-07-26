"""The shared per-pitch tasks and the objective scorer both optimizers minimize (``optimize/tasks.py``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.config.reduce import Representatives
from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.orchestrate import prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import (
    AudioMap,
    EvalContext,
    MergedEvent,
    PitchTask,
    build_tasks,
    merge_events,
    nearest_key,
    score_events,
    score_reconstruction,
)

SR = 44_100
PITCHES = (60, 62)


# --- pure helpers ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("available", "target", "expected"),
    [
        ((50, 100), 70, 50),  # closer to the soft recording
        ((60, 80), 70, 80),  # equidistant → the louder one wins
        ((100,), 10, 100),  # only one recording to pick
    ],
)
def test_nearest_key_picks_the_closest_recorded_velocity(
    available: tuple[int, ...], target: int, expected: int
) -> None:
    keys = [SampleKey(60, velocity) for velocity in available]
    assert nearest_key(keys, target) == SampleKey(60, expected)


def test_nearest_key_breaks_a_velocity_tie_on_the_lowest_cc_bucket() -> None:
    dark = SampleKey(60, 100, ((1, 0),))
    bright = SampleKey(60, 100, ((1, 9),))
    assert nearest_key([bright, dark], 100) == dark


def test_merge_events_collapses_shared_velocity_and_duration() -> None:
    events = [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=2),  # weight 1.0
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=3),  # weight 1.5 → merges with the above
        NoteEvent(pitch=60, velocity=80, duration_s=1.0, count=1),  # weight 1.0, distinct key
    ]
    merged = merge_events(events)
    assert all(isinstance(item, MergedEvent) for item in merged)
    weights = {(item.velocity, item.duration_s): item.weight for item in merged}
    assert weights == {(100, 0.5): pytest.approx(2.5), (80, 1.0): pytest.approx(1.0)}


# --- task building --------------------------------------------------------------------------------


def _instrument(material: list[NoteEvent]) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in PITCHES]
    return InstrumentSpec(id="piano", budget_kb=64.0, samples=samples, material=material)


def test_build_tasks_orders_by_pitch_and_picks_the_loudest_representative(
    piano_note: Callable[..., NDArray[np.float64]],
) -> None:
    audio: AudioMap = {
        SampleKey(pitch, 100): piano_note(pitch, 100, dur=0.5, seed=pitch * 137 + 100) for pitch in PITCHES
    }
    material = [
        NoteEvent(pitch=62, velocity=80, duration_s=0.4, count=1),
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=2),
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),
    ]
    tasks = build_tasks(_instrument(material), audio, Representatives.NEAREST_LOUDEST)
    assert [task.pitch for task in tasks] == [60, 62]  # ascending, the DP's segmentation order
    by_pitch = {task.pitch: task for task in tasks}
    assert by_pitch[60].representative_key == SampleKey(60, 100)  # nearest the loudest velocity played there
    assert by_pitch[60].weight == pytest.approx(2 * 0.5 + 1 * 0.5)  # summed usage weight of its events
    assert len(by_pitch[60].events) == 2  # two distinct dynamics at pitch 60


@pytest.mark.parametrize(
    ("representatives", "expected"),
    [
        (Representatives.NEAREST_LOUDEST, (SampleKey(60, 100),)),
        (Representatives.ALL, (SampleKey(60, 100), SampleKey(60, 40))),
    ],
)
def test_candidates_offer_the_survivors_the_policy_allows(
    piano_note: Callable[..., NDArray[np.float64]],
    representatives: Representatives,
    expected: tuple[SampleKey, ...],
) -> None:
    audio: AudioMap = {
        SampleKey(60, velocity): piano_note(60, velocity, dur=0.5, seed=60 * 137 + velocity) for velocity in (40, 100)
    }
    material = [NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=1)]
    tasks = build_tasks(_instrument(material), audio, representatives)
    assert tasks[0].candidates == expected  # the representative always leads


def test_build_tasks_raises_when_a_material_pitch_has_no_recording(
    piano_note: Callable[..., NDArray[np.float64]],
) -> None:
    audio: AudioMap = {SampleKey(60, 100): piano_note(60, 100, dur=0.5, seed=60 * 137 + 100)}
    material = [NoteEvent(pitch=99, velocity=100, duration_s=0.4, count=1)]  # pitch 99 not recorded
    with pytest.raises(ValueError, match="no recorded sample for pitch 99"):
        build_tasks(_instrument(material), audio, Representatives.NEAREST_LOUDEST)


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
    _, context, tasks = prepare_run(_instrument(material), audio, SR, settings)
    task = tasks[0]
    params = EncodingParams(target_rate=SR, depth_bits=16, dither=False)
    stored = encode(task.representative, SR, params, make_encode_ctx(task.pitch))
    return _Scoring(task=task, context=context, stored=stored)


def test_score_events_yields_one_weighted_score_per_event(scoring: _Scoring) -> None:
    scores = list(score_events(scoring.stored, scoring.task, scoring.context))
    assert len(scores) == len(scoring.task.events)
    for score in scores:
        assert score.weighted_fidelity == pytest.approx(score.event.weight * score.report.fidelity)
        assert score.report.fidelity >= 0.0


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
