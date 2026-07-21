"""The shared per-pitch tasks and the objective scorer both optimizers minimize (``optimize/tasks.py``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import OptiConfig
from optisample.config.optimize import SweepConfig
from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.orchestrate import OptimizeSettings, prepare_run
from optisample.optimize.tasks import (
    AudioMap,
    EvalContext,
    MergedEvent,
    PitchTask,
    build_tasks,
    merge_events,
    nearest_velocity,
    score_events,
    score_reconstruction,
)
from optisample.synth import NoteSpec, render_sample

SR = 44_100
PITCHES = (60, 62)


def _note(config: OptiConfig, pitch: int, velocity: int, dur: float = 0.5) -> NDArray[np.float64]:
    spec = NoteSpec(pitch, velocity, 0.0, dur, SR)
    return render_sample("piano", spec, np.random.default_rng(pitch * 137 + velocity), config.synth)


# --- pure helpers ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("available", "target", "expected"),
    [
        ((50, 100), 70, 50),  # closer to the soft recording
        ((60, 80), 70, 80),  # equidistant → the louder one wins
        ((100,), 10, 100),  # only one recording to pick
    ],
)
def test_nearest_velocity(available: tuple[int, ...], target: int, expected: int) -> None:
    assert nearest_velocity(available, target) == expected


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


def test_build_tasks_orders_by_pitch_and_picks_the_loudest_representative(config: OptiConfig) -> None:
    audio: AudioMap = {(pitch, 100): _note(config, pitch, 100) for pitch in PITCHES}
    material = [
        NoteEvent(pitch=62, velocity=80, duration_s=0.4, count=1),
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=2),
        NoteEvent(pitch=60, velocity=40, duration_s=0.5, count=1),
    ]
    tasks = build_tasks(_instrument(material), audio)
    assert [task.pitch for task in tasks] == [60, 62]  # ascending, the DP's segmentation order
    by_pitch = {task.pitch: task for task in tasks}
    assert by_pitch[60].representative_velocity == 100  # nearest the loudest velocity played at pitch 60
    assert by_pitch[60].weight == pytest.approx(2 * 0.5 + 1 * 0.5)  # summed usage weight of its events
    assert len(by_pitch[60].events) == 2  # two distinct dynamics at pitch 60


def test_build_tasks_raises_when_a_material_pitch_has_no_recording(config: OptiConfig) -> None:
    audio: AudioMap = {(60, 100): _note(config, 60, 100)}
    material = [NoteEvent(pitch=99, velocity=100, duration_s=0.4, count=1)]  # pitch 99 not recorded
    with pytest.raises(ValueError, match="no recorded sample for pitch 99"):
        build_tasks(_instrument(material), audio)


# --- scoring --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scoring:
    """A ready-to-score setup: one pitch task, the shared eval context, and its re-encoded sample."""

    task: PitchTask
    ctx: EvalContext
    stored: StoredSample


@pytest.fixture
def scoring(
    config: OptiConfig,
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    make_encode_ctx: Callable[..., EncodeContext],
) -> _Scoring:
    audio: AudioMap = {(pitch, 100): _note(config, pitch, 100, dur=0.6) for pitch in PITCHES}
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=2) for pitch in PITCHES]
    settings = optimize_settings(sweep=sweep(rates=(SR,), depths=(16,), dither=False))
    _, ctx, tasks = prepare_run(_instrument(material), audio, SR, settings)
    task = tasks[0]
    params = EncodingParams(target_rate=SR, depth_bits=16, dither=False)
    stored = encode(task.representative, SR, params, make_encode_ctx(task.pitch))
    return _Scoring(task=task, ctx=ctx, stored=stored)


def test_score_events_yields_one_weighted_score_per_event(scoring: _Scoring) -> None:
    scores = list(score_events(scoring.stored, scoring.task, scoring.ctx))
    assert len(scores) == len(scoring.task.events)
    for score in scores:
        assert score.weighted_fidelity == pytest.approx(score.event.weight * score.report.fidelity)
        assert score.report.fidelity >= 0.0


def test_score_reconstruction_is_the_weight_normalized_mean(scoring: _Scoring) -> None:
    scores = list(score_events(scoring.stored, scoring.task, scoring.ctx))
    reconstruction = score_reconstruction(scoring.stored, scoring.task, scoring.ctx)
    expected = sum(score.weighted_fidelity for score in scores) / scoring.task.weight
    assert reconstruction == pytest.approx(expected)
    assert reconstruction >= 0.0


def test_score_reconstruction_of_a_weightless_task_is_zero(scoring: _Scoring) -> None:
    # A task with no events has zero weight; the scorer must return 0 rather than divide by zero.
    empty = PitchTask(pitch=60, weight=0.0, representative_velocity=100, representative=np.zeros(4), events=())
    assert score_reconstruction(scoring.stored, empty, scoring.ctx) == 0.0
