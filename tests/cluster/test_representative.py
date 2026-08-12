from __future__ import annotations

import numpy as np
import pytest

from optisample.cluster.partition import Labels
from optisample.cluster.representative import (
    Durations,
    MemberReadings,
    Weights,
    grouping,
    uniform_weights,
)
from optisample.cluster.space import Coordinates, pairwise_distances
from optisample.config import OptiConfig
from optisample.config.cluster import PartitionConfig, Representative

_LEANED_ON = 40.0  # the say a recording carrying the music is given against a take the material barely asks for
_RINGS_ON = 4.0  # how long a take every rule is free to choose sounds for, well over any floor these ask of it
_CLIPPED_S = 0.05  # how long a take too short to shape sounds for
_FLOOR_S = 0.25  # the length a representative is held to here, which the clipped takes fall under
_NO_FLOOR_S = 0.0  # the floor leaving whichever take the rule names standing
_MIDDLE = 2  # where the medoid of the line stands, which is the take the shipped rule names
_NEXT_ALONG = 1  # the member standing one step from the middle, first of the two the walk reaches


@pytest.fixture
def coordinates() -> Coordinates:
    """A line of five recordings and a distant pair, which puts every representative somewhere plain."""
    return np.asarray([[0.0], [1.0], [2.0], [3.0], [10.0], [20.0], [20.5]], dtype=np.float64)


@pytest.fixture
def labels() -> Labels:
    """The line read as one group and the distant pair as another."""
    return np.asarray([0, 0, 0, 0, 0, 1, 1], dtype=np.intp)


@pytest.fixture
def weights(coordinates: Coordinates) -> Weights:
    """Every recording carrying the same say, which is what a corpus states before a playing time is read."""
    return uniform_weights(coordinates.shape[0])


@pytest.fixture
def durations(coordinates: Coordinates) -> Durations:
    """Every recording ringing well past any floor, so a rule's own choice is what stands."""
    return np.full(coordinates.shape[0], _RINGS_ON, dtype=np.float64)


@pytest.fixture
def readings(weights: Weights, durations: Durations) -> MemberReadings:
    """What the recordings carry beyond their coordinates, on the terms a corpus states before it is read."""
    return MemberReadings(weights=weights, durations_s=durations)


@pytest.fixture
def cutting(config: OptiConfig) -> PartitionConfig:
    """The shipped cut, held to a floor these takes all clear, so each test states the pressure it applies."""
    return config.cluster.partition.model_copy(update={"min_duration_s": _NO_FLOOR_S})


def _clipped(durations: Durations, *places: int) -> Durations:
    """The same recordings with the named ones cut down to a length too short to shape."""
    shortened = durations.copy()
    shortened[list(places)] = _CLIPPED_S
    return shortened


def _medoid_by_hand(coordinates: Coordinates, members: Labels, weights: Weights) -> int:
    """The member whose weighed distances to the rest of its group come to the least, found by trying each."""
    distances = pairwise_distances(coordinates)
    return min(members.tolist(), key=lambda member: float(distances[member, members] @ weights[members]))


def test_every_group_the_labels_name_is_reported_once(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    groups = grouping(coordinates, labels, readings, config=cutting)

    assert [group.label for group in groups] == [0, 1]
    assert [group.size for group in groups] == [5, 2]
    assert sorted(member for group in groups for member in group.members.tolist()) == list(range(labels.size))


def test_the_medoid_is_a_member_whose_distances_to_the_rest_come_to_the_least(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    """Read against every member tried in turn, which is the whole of what a medoid claims to be."""
    for group in grouping(coordinates, labels, readings, config=cutting):
        assert group.medoid in group.members.tolist()
        assert group.medoid == _medoid_by_hand(coordinates, group.members, readings.weights)


def test_the_centroid_is_the_mean_of_the_members(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    for group in grouping(coordinates, labels, readings, config=cutting):
        assert np.allclose(group.centroid, coordinates[group.members].mean(axis=0))


def test_the_weighted_medoid_follows_the_say_its_members_carry(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    """Leaning the say onto one end of the line walks the representative there, and the plain medoid holds."""
    leaning = readings.weights.copy()
    leaning[0] = _LEANED_ON
    plain = grouping(coordinates, labels, readings, config=cutting)[0]
    leaned = grouping(
        coordinates,
        labels,
        MemberReadings(weights=leaning, durations_s=readings.durations_s),
        config=cutting,
    )[0]

    assert leaned.medoid == plain.medoid
    assert leaned.weighted_medoid == 0


def test_uniform_weights_leave_the_weighted_medoid_where_the_medoid_stands(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    for group in grouping(coordinates, labels, readings, config=cutting):
        assert group.weighted_medoid == group.medoid


def test_the_nearest_centroid_is_the_member_closest_to_the_mean(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    """The mean is dragged out along the line by the distant take, and the member nearest it follows."""
    group = grouping(coordinates, labels, readings, config=cutting)[0]

    assert group.nearest_centroid == int(
        np.argmin(np.abs(coordinates[group.members, 0] - coordinates[group.members, 0].mean()))
    )
    assert group.nearest_centroid != group.medoid


def test_the_spread_states_how_far_the_group_reaches_from_its_medoid(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    group = grouping(coordinates, labels, readings, config=cutting)[0]
    reach = np.abs(coordinates[group.members, 0] - coordinates[group.medoid, 0])

    assert group.spread_max == pytest.approx(reach.max())
    assert group.spread_mean == pytest.approx(reach.sum() / (group.size - 1))
    assert group.farthest == int(group.members[np.argmax(reach)])


def test_a_group_of_one_stands_for_itself(cutting: PartitionConfig) -> None:
    coordinates = np.asarray([[0.0], [5.0], [5.5]], dtype=np.float64)
    readings = MemberReadings(weights=uniform_weights(3), durations_s=np.full(3, _RINGS_ON, dtype=np.float64))
    alone = grouping(coordinates, np.asarray([0, 1, 1], dtype=np.intp), readings, config=cutting)[0]

    assert alone.size == 1
    assert alone.medoid == alone.weighted_medoid == alone.nearest_centroid == alone.farthest == 0
    assert alone.representative == 0
    assert alone.spread_mean == 0.0
    assert alone.spread_max == 0.0


@pytest.mark.parametrize("rule", list(Representative))
def test_the_rule_names_which_recording_stands_for_the_group(
    rule: Representative,
    coordinates: Coordinates,
    labels: Labels,
    readings: MemberReadings,
    cutting: PartitionConfig,
) -> None:
    """A representative is settled once, at the cut, so every panel reading one reads the same take."""
    group = grouping(coordinates, labels, readings, config=cutting.model_copy(update={"representative": rule}))[0]
    named = {
        Representative.MEDOID: group.medoid,
        Representative.WEIGHTED_MEDOID: group.weighted_medoid,
        Representative.NEAREST_CENTROID: group.nearest_centroid,
    }

    assert group.representative == named[rule]
    assert group.representative in group.members.tolist()


def test_the_greedy_walk_reaches_past_every_take_that_falls_short(
    coordinates: Coordinates,
    labels: Labels,
    readings: MemberReadings,
    durations: Durations,
    cutting: PartitionConfig,
) -> None:
    """Each step out is the closest take yet untried, so the group keeps the most typical one that holds."""
    floored = cutting.model_copy(update={"min_duration_s": _FLOOR_S})
    clipped = MemberReadings(weights=readings.weights, durations_s=_clipped(durations, _MIDDLE))
    stepped = grouping(coordinates, labels, clipped, config=floored)[0]
    further = MemberReadings(weights=readings.weights, durations_s=_clipped(durations, _MIDDLE, _NEXT_ALONG))
    walked = grouping(coordinates, labels, further, config=floored)[0]

    assert stepped.medoid == _MIDDLE
    assert stepped.representative == _NEXT_ALONG
    assert walked.representative == _MIDDLE + 1


def test_a_group_whose_every_take_falls_short_stands_on_the_one_the_rule_named(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    """A representative stays a member of its own group, so the rule's choice holds where no take is longer."""
    clipped = MemberReadings(
        weights=readings.weights,
        durations_s=np.full(coordinates.shape[0], _CLIPPED_S, dtype=np.float64),
    )
    group = grouping(coordinates, labels, clipped, config=cutting.model_copy(update={"min_duration_s": _FLOOR_S}))[0]

    assert group.representative == group.medoid


def test_a_floor_of_nothing_leaves_whichever_take_the_rule_named(
    coordinates: Coordinates, labels: Labels, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    """Asking no length of a representative is what reads a corpus by the geometry alone."""
    clipped = MemberReadings(
        weights=readings.weights,
        durations_s=np.full(coordinates.shape[0], _CLIPPED_S, dtype=np.float64),
    )
    group = grouping(coordinates, labels, clipped, config=cutting)[0]

    assert group.representative == group.medoid


def test_labels_naming_another_corpus_are_rejected(
    coordinates: Coordinates, readings: MemberReadings, cutting: PartitionConfig
) -> None:
    with pytest.raises(ValueError, match="labels name"):
        grouping(coordinates, np.zeros(3, dtype=np.intp), readings, config=cutting)


def test_weights_stating_another_corpus_are_rejected(
    coordinates: Coordinates, labels: Labels, durations: Durations, cutting: PartitionConfig
) -> None:
    with pytest.raises(ValueError, match="weights state"):
        grouping(
            coordinates,
            labels,
            MemberReadings(weights=uniform_weights(3), durations_s=durations),
            config=cutting,
        )


def test_durations_stating_another_corpus_are_rejected(
    coordinates: Coordinates, labels: Labels, weights: Weights, cutting: PartitionConfig
) -> None:
    with pytest.raises(ValueError, match="durations state"):
        grouping(
            coordinates,
            labels,
            MemberReadings(weights=weights, durations_s=np.full(3, _RINGS_ON, dtype=np.float64)),
            config=cutting,
        )


def test_uniform_weights_give_every_recording_the_same_say() -> None:
    assert np.array_equal(uniform_weights(4), np.ones(4))
