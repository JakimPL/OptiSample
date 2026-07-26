from dataclasses import dataclass

import pytest

from optisample.config.reduce import DedupeConfig, DedupeKey, Representatives
from optisample.optimize.reduce.keys import (
    DedupeGroup,
    SampleKey,
    cc_buckets,
    dedupe_group,
    sample_key,
)


@dataclass(frozen=True)
class Recording:
    """A stand-in for the identity fields a source sample and a note event both carry."""

    pitch: int
    velocity: int
    cc_averages: dict[int, float]


def dedupe(key: DedupeKey, *, cc_quantum: float = 1.0) -> DedupeConfig:
    return DedupeConfig(
        key=key,
        cc_quantum=cc_quantum,
        transposition_headroom_semitones=12,
        representatives=Representatives.NEAREST_LOUDEST,
    )


@pytest.mark.parametrize(
    ("key", "expects_velocity", "expects_cc"),
    [
        (DedupeKey.PITCH, False, False),
        (DedupeKey.PITCH_VELOCITY, True, False),
        (DedupeKey.PITCH_VELOCITY_CC, True, True),
    ],
)
def test_key_states_which_axes_it_reads(key: DedupeKey, expects_velocity: bool, expects_cc: bool) -> None:
    assert key.includes_velocity is expects_velocity
    assert key.includes_cc is expects_cc


def test_cc_buckets_order_by_controller_and_floor_onto_the_quantum() -> None:
    assert cc_buckets({7: 64.9, 1: 12.0}, 10.0) == ((1, 1), (7, 6))


def test_cc_buckets_give_nearby_averages_one_identity() -> None:
    assert cc_buckets({1: 30.1}, 10.0) == cc_buckets({1: 39.9}, 10.0)
    assert cc_buckets({1: 30.1}, 10.0) != cc_buckets({1: 40.1}, 10.0)


def test_sample_key_reads_cc_only_when_the_key_distinguishes_variants() -> None:
    recording = Recording(pitch=60, velocity=100, cc_averages={1: 64.0})

    assert sample_key(recording, dedupe(DedupeKey.PITCH_VELOCITY)) == SampleKey(60, 100, ())
    assert sample_key(recording, dedupe(DedupeKey.PITCH_VELOCITY_CC)) == SampleKey(60, 100, ((1, 64),))


def test_sample_key_keeps_the_recorded_velocity_under_a_pitch_only_key() -> None:
    """A pitch-keyed survivor still reports the velocity it was recorded at, which the velocity map reads."""
    recording = Recording(pitch=60, velocity=37, cc_averages={})
    assert sample_key(recording, dedupe(DedupeKey.PITCH)).velocity == 37


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (DedupeKey.PITCH, DedupeGroup(60, None, ())),
        (DedupeKey.PITCH_VELOCITY, DedupeGroup(60, 100, ())),
        (DedupeKey.PITCH_VELOCITY_CC, DedupeGroup(60, 100, ((1, 64),))),
    ],
)
def test_dedupe_group_projects_onto_the_configured_axes(key: DedupeKey, expected: DedupeGroup) -> None:
    assert dedupe_group(SampleKey(60, 100, ((1, 64),)), key) == expected


def test_velocities_share_a_group_under_a_pitch_only_key() -> None:
    quiet = sample_key(Recording(60, 40, {}), dedupe(DedupeKey.PITCH))
    loud = sample_key(Recording(60, 110, {}), dedupe(DedupeKey.PITCH))

    assert dedupe_group(quiet, DedupeKey.PITCH) == dedupe_group(loud, DedupeKey.PITCH)
    assert dedupe_group(quiet, DedupeKey.PITCH_VELOCITY) != dedupe_group(loud, DedupeKey.PITCH_VELOCITY)


def test_sample_key_is_hashable_and_orders_by_pitch_then_velocity() -> None:
    keys = [SampleKey(67, 40), SampleKey(60, 110), SampleKey(60, 40)]
    assert sorted(keys) == [SampleKey(60, 40), SampleKey(60, 110), SampleKey(67, 40)]
    assert len({SampleKey(60, 40), SampleKey(60, 40)}) == 1


def test_label_names_the_note_and_marks_a_controller_variant() -> None:
    assert SampleKey(60, 100).label == "p060_C4_v100"
    assert SampleKey(60, 100, ((1, 6),)).label == "p060_C4_v100_cc1-6"
