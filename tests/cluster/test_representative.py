from __future__ import annotations

import numpy as np
import pytest

from optisample.cluster.partition import Labels
from optisample.cluster.representative import Weights, grouping, uniform_weights
from optisample.cluster.space import Coordinates, pairwise_distances
from optisample.config.cluster import Representative

_LEANED_ON = 40.0  # the say a recording carrying the music is given against a take the material barely asks for


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


def _medoid_by_hand(coordinates: Coordinates, members: Labels, weights: Weights) -> int:
    """The member whose weighed distances to the rest of its group come to the least, found by trying each."""
    distances = pairwise_distances(coordinates)
    return min(members.tolist(), key=lambda member: float(distances[member, members] @ weights[members]))


def test_every_group_the_labels_name_is_reported_once(
    coordinates: Coordinates, labels: Labels, weights: Weights
) -> None:
    groups = grouping(coordinates, labels, weights)

    assert [group.label for group in groups] == [0, 1]
    assert [group.size for group in groups] == [5, 2]
    assert sorted(member for group in groups for member in group.members.tolist()) == list(range(labels.size))


def test_the_medoid_is_a_member_whose_distances_to_the_rest_come_to_the_least(
    coordinates: Coordinates, labels: Labels, weights: Weights
) -> None:
    """Read against every member tried in turn, which is the whole of what a medoid claims to be."""
    for group in grouping(coordinates, labels, weights):
        assert group.medoid in group.members.tolist()
        assert group.medoid == _medoid_by_hand(coordinates, group.members, weights)


def test_the_centroid_is_the_mean_of_the_members(coordinates: Coordinates, labels: Labels, weights: Weights) -> None:
    groups = grouping(coordinates, labels, weights)

    for group in groups:
        assert np.allclose(group.centroid, coordinates[group.members].mean(axis=0))


def test_the_weighted_medoid_follows_the_say_its_members_carry(
    coordinates: Coordinates, labels: Labels, weights: Weights
) -> None:
    """Leaning the say onto one end of the line walks the representative there, and the plain medoid holds."""
    leaning = weights.copy()
    leaning[0] = _LEANED_ON
    plain = grouping(coordinates, labels, weights)[0]
    leaned = grouping(coordinates, labels, leaning)[0]

    assert leaned.medoid == plain.medoid
    assert leaned.weighted_medoid == 0


def test_uniform_weights_leave_the_weighted_medoid_where_the_medoid_stands(
    coordinates: Coordinates, labels: Labels, weights: Weights
) -> None:
    for group in grouping(coordinates, labels, weights):
        assert group.weighted_medoid == group.medoid


def test_the_nearest_centroid_is_the_member_closest_to_the_mean(
    coordinates: Coordinates, labels: Labels, weights: Weights
) -> None:
    """The mean is dragged out along the line by the distant take, and the member nearest it follows."""
    group = grouping(coordinates, labels, weights)[0]

    assert group.nearest_centroid == int(
        np.argmin(np.abs(coordinates[group.members, 0] - coordinates[group.members, 0].mean()))
    )
    assert group.nearest_centroid != group.medoid


def test_the_spread_states_how_far_the_group_reaches_from_its_medoid(
    coordinates: Coordinates, labels: Labels, weights: Weights
) -> None:
    group = grouping(coordinates, labels, weights)[0]
    reach = np.abs(coordinates[group.members, 0] - coordinates[group.medoid, 0])

    assert group.spread_max == pytest.approx(reach.max())
    assert group.spread_mean == pytest.approx(reach.sum() / (group.size - 1))
    assert group.farthest == int(group.members[np.argmax(reach)])


def test_a_group_of_one_stands_for_itself() -> None:
    coordinates = np.asarray([[0.0], [5.0], [5.5]], dtype=np.float64)
    groups = grouping(coordinates, np.asarray([0, 1, 1], dtype=np.intp), uniform_weights(3))
    alone = groups[0]

    assert alone.size == 1
    assert alone.medoid == alone.weighted_medoid == alone.nearest_centroid == alone.farthest == 0
    assert alone.spread_mean == 0.0
    assert alone.spread_max == 0.0


@pytest.mark.parametrize("rule", list(Representative))
def test_the_rule_names_which_recording_stands_for_the_group(
    rule: Representative, coordinates: Coordinates, labels: Labels, weights: Weights
) -> None:
    group = grouping(coordinates, labels, weights)[0]

    assert group.representative(rule) in group.members.tolist()


def test_labels_naming_another_corpus_are_rejected(coordinates: Coordinates, weights: Weights) -> None:
    with pytest.raises(ValueError, match="labels name"):
        grouping(coordinates, np.zeros(3, dtype=np.intp), weights)


def test_weights_stating_another_corpus_are_rejected(coordinates: Coordinates, labels: Labels) -> None:
    with pytest.raises(ValueError, match="weights state"):
        grouping(coordinates, labels, uniform_weights(3))


def test_uniform_weights_give_every_recording_the_same_say() -> None:
    assert np.array_equal(uniform_weights(4), np.ones(4))
