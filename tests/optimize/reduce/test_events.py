from collections.abc import Callable

import pytest

from optisample.config.reduce import ReduceConfig
from optisample.keys import SampleKey
from optisample.model import NoteEvent
from optisample.optimize.reduce.events import bucket_duration_s, merge_events
from optisample.optimize.velocity_map import VelocityVolumeMap

_RATIO = 1.25  # the bundled duration_bucket_ratio, which the exactness tests override to 1.0
_EXACT_DURATIONS = {"duration_bucket_ratio": 1.0}

ReduceFactory = Callable[..., ReduceConfig]


def note(velocity: int, duration_s: float, count: int = 1) -> NoteEvent:
    return NoteEvent(pitch=60, velocity=velocity, duration_s=duration_s, count=count)


# --- the geometric duration grid --------------------------------------------------------------------


@pytest.mark.parametrize("duration_s", [0.01, 0.37, 0.5, 1.0, 2.0, 3.9, 12.5])
def test_a_bucketed_duration_covers_its_note_and_stays_inside_one_step(duration_s: float) -> None:
    """The defining property: long enough to hold the note, short enough to stay in the bucket above it."""
    edge = bucket_duration_s(duration_s, _RATIO)
    assert duration_s <= edge <= duration_s * _RATIO


def test_the_grid_is_anchored_at_one_second() -> None:
    assert bucket_duration_s(1.0, _RATIO) == 1.0
    assert bucket_duration_s(1.01, _RATIO) == pytest.approx(_RATIO)  # the next edge up


def test_durations_on_either_side_of_an_edge_land_on_different_buckets() -> None:
    assert bucket_duration_s(0.99, _RATIO) < bucket_duration_s(1.01, _RATIO)


@pytest.mark.parametrize("duration_s", [0.37, 1.0, 3.9])
def test_a_ratio_of_one_leaves_every_duration_where_it_is(duration_s: float) -> None:
    assert bucket_duration_s(duration_s, 1.0) == duration_s


# --- the exact merge ---------------------------------------------------------------------------------


def test_notes_agreeing_on_reference_volume_and_length_become_one_class(
    graded_velocity_map: VelocityVolumeMap, reduce: ReduceFactory
) -> None:
    events = [note(100, 0.5, count=2), note(100, 0.5, count=3), note(80, 1.0)]
    available = [SampleKey(60, 100)]

    merged = merge_events(events, available, graded_velocity_map, reduce(events=_EXACT_DURATIONS).events)

    assert {(item.velocity, item.duration_s): item.weight for item in merged} == {
        (100, 0.5): pytest.approx(2.5),
        (80, 1.0): pytest.approx(1.0),
    }


def test_dynamics_the_volume_map_cannot_tell_apart_merge(
    flat_velocity_map: VelocityVolumeMap, reduce: ReduceFactory
) -> None:
    """The reduction the old (velocity, duration) key missed: same reference, same volume, same length."""
    events = [note(100, 0.5), note(40, 0.5)]
    available = [SampleKey(60, 100)]

    merged = merge_events(events, available, flat_velocity_map, reduce(events=_EXACT_DURATIONS).events)

    assert len(merged) == 1
    assert merged[0].volume == 64
    assert merged[0].velocity == 100  # labelled by the loudest note it covers
    assert merged[0].weight == pytest.approx(1.0)


def test_dynamics_routed_to_different_recordings_stay_apart(
    flat_velocity_map: VelocityVolumeMap, reduce: ReduceFactory
) -> None:
    """One volume is not enough to merge: two notes scored against different recordings differ."""
    events = [note(100, 0.5), note(40, 0.5)]
    available = [SampleKey(60, 40), SampleKey(60, 100)]

    merged = merge_events(events, available, flat_velocity_map, reduce(events=_EXACT_DURATIONS).events)

    assert [item.reference_key for item in merged] == [SampleKey(60, 100), SampleKey(60, 40)]


def test_classes_come_back_in_the_order_the_material_first_plays_them(
    graded_velocity_map: VelocityVolumeMap, reduce: ReduceFactory
) -> None:
    events = [note(60, 1.0), note(100, 0.5), note(60, 1.0)]
    available = [SampleKey(60, 100)]

    merged = merge_events(events, available, graded_velocity_map, reduce(events=_EXACT_DURATIONS).events)

    assert [item.velocity for item in merged] == [60, 100]


# --- bucketing inside the merge ------------------------------------------------------------------------


def test_bucketing_merges_lengths_within_the_ratio_and_keeps_their_real_weight(
    graded_velocity_map: VelocityVolumeMap, reduce: ReduceFactory
) -> None:
    events = [note(100, 0.52), note(100, 0.60)]
    available = [SampleKey(60, 100)]

    merged = merge_events(events, available, graded_velocity_map, reduce().events)

    assert len(merged) == 1
    assert merged[0].duration_s == bucket_duration_s(0.60, _RATIO)  # the edge above both
    assert merged[0].weight == pytest.approx(0.52 + 0.60)  # playing time, not the bucket length


def test_bucketing_keeps_lengths_further_apart_than_the_ratio_separate(
    graded_velocity_map: VelocityVolumeMap, reduce: ReduceFactory
) -> None:
    events = [note(100, 0.5), note(100, 2.0)]
    available = [SampleKey(60, 100)]

    merged = merge_events(events, available, graded_velocity_map, reduce().events)

    assert len(merged) == 2
