"""Per-layer pitch tasks built from band-restricted material (``layers/tasks.py``)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.reduce import ReduceConfig
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.music import MIDI_MAX_VELOCITY
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers, partitions, velocity_cells
from optisample.optimize.layers.tasks import band_instrument, band_tasks, layered_tasks
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import AudioMap, PitchTask, build_tasks
from optisample.optimize.velocity_map import VelocityVolumeMap

PITCHES = (60, 62)
RECORDED = (40, 100)

ReduceFactory = Callable[..., ReduceConfig]

WHOLE_AXIS = VelocityBand(0, MIDI_MAX_VELOCITY)
QUIET = VelocityBand(0, 63)
LOUD = VelocityBand(64, MIDI_MAX_VELOCITY)


@pytest.fixture
def audio(piano_note: Callable[..., NDArray[np.float64]]) -> AudioMap:
    """One recording per ``(pitch, velocity)`` the instrument was sampled at."""
    return {
        SampleKey(pitch, velocity): piano_note(pitch, velocity, dur=0.5, seed=pitch * 137 + velocity)
        for pitch in PITCHES
        for velocity in RECORDED
    }


def _instrument(material: list[NoteEvent]) -> InstrumentSpec:
    samples = [
        SourceSample(file=Path(f"{pitch}_{velocity}.wav"), pitch=pitch, velocity=velocity)
        for pitch in PITCHES
        for velocity in RECORDED
    ]
    return InstrumentSpec(id="piano", budget_kb=64.0, samples=samples, material=material)


def _both_dynamics() -> list[NoteEvent]:
    """Every key played once softly and once loudly, so both layers cover the whole keyboard."""
    return [
        NoteEvent(pitch=pitch, velocity=velocity, duration_s=0.5, count=1)
        for pitch in PITCHES
        for velocity in (30, 110)
    ]


def _shape(tasks: Sequence[PitchTask]) -> list[tuple[object, ...]]:
    """Everything a task states about what it stores and scores, in a form that compares key for key."""
    return [
        (
            task.pitch,
            task.weight,
            task.representative_key,
            task.candidates,
            tuple(event.identity for event in task.events),
            tuple(event.weight for event in task.events),
        )
        for task in tasks
    ]


def test_one_band_over_the_whole_axis_reproduces_the_unlayered_tasks(
    audio: AudioMap,
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    """The behaviour freeze: a single full-range layer is the plan the optimizer builds today."""
    instrument = _instrument(_both_dynamics())
    expected = build_tasks(instrument, audio, graded_velocity_map, reduce())
    layered = band_tasks(instrument, audio, graded_velocity_map, reduce(), WHOLE_AXIS)
    assert _shape(layered) == _shape(expected)
    for task, reference in zip(layered, expected):
        assert np.array_equal(task.representative, reference.representative)
        for event, source in zip(task.events, reference.events):
            assert np.array_equal(event.reference, source.reference)


def test_the_single_layer_split_is_the_whole_axis(
    audio: AudioMap,
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    instrument = _instrument(_both_dynamics())
    cells = velocity_cells(instrument.material, 6)
    (split,) = partitions(cells, 1)
    (only,) = layered_tasks(instrument, audio, graded_velocity_map, reduce(), split)
    assert _shape(only) == _shape(build_tasks(instrument, audio, graded_velocity_map, reduce()))


@pytest.mark.parametrize(("band", "expected"), [(QUIET, 40), (LOUD, 100)])
def test_a_layer_stores_the_recording_nearest_its_own_loudest_dynamic(
    audio: AudioMap,
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
    band: VelocityBand,
    expected: int,
) -> None:
    """The move layering adds: a quiet layer keeps a quiet recording instead of scaling the loud one down."""
    instrument = _instrument(_both_dynamics())
    tasks = band_tasks(instrument, audio, graded_velocity_map, reduce(), band)
    assert [task.representative_key for task in tasks] == [SampleKey(pitch, expected) for pitch in PITCHES]


def test_a_layer_holds_the_keys_its_own_dynamics_are_played_at(
    audio: AudioMap,
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    material = [
        NoteEvent(pitch=60, velocity=30, duration_s=0.5, count=1),
        NoteEvent(pitch=62, velocity=110, duration_s=0.5, count=1),
    ]
    instrument = _instrument(material)
    quiet = band_tasks(instrument, audio, graded_velocity_map, reduce(), QUIET)
    loud = band_tasks(instrument, audio, graded_velocity_map, reduce(), LOUD)
    assert [task.pitch for task in quiet] == [60]
    assert [task.pitch for task in loud] == [62]


def test_the_layers_share_out_every_note_the_material_plays(
    audio: AudioMap,
    graded_velocity_map: VelocityVolumeMap,
    reduce: ReduceFactory,
) -> None:
    instrument = _instrument(_both_dynamics())
    split = VelocityLayers((QUIET, LOUD))
    per_layer = layered_tasks(instrument, audio, graded_velocity_map, reduce(), split)
    assert len(per_layer) == split.count
    layered_weight = sum(task.weight for tasks in per_layer for task in tasks)
    whole = sum(task.weight for task in build_tasks(instrument, audio, graded_velocity_map, reduce()))
    assert layered_weight == pytest.approx(whole)


def test_a_layer_keeps_every_recording_the_instrument_has(audio: AudioMap) -> None:
    """A layer narrows what is played, so every survivor stays available as a reference and a candidate."""
    instrument = _instrument(_both_dynamics())
    quiet = band_instrument(instrument, QUIET)
    assert quiet.samples == instrument.samples
    assert [event.velocity for event in quiet.material] == [30, 30]
