from __future__ import annotations

import numpy as np
import pytest

from optisample.cluster.partition import (
    Labels,
    centres,
    compact,
    cut,
    hierarchy,
    partition,
    silhouette,
    sweep,
    sweep_ceiling,
)
from optisample.cluster.space import Coordinates, pairwise_distances
from optisample.config.cluster import LinkageMethod, PartitionAlgorithm, PartitionConfig

_BLOB = 12  # recordings gathered around each corner of the material a cut is read on
_CORNERS = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))  # corners far enough apart that a cut has one right answer
_HUDDLE = 0.3  # how far a point strays from its corner, which keeps the corners plainly apart
_SEED = 19  # the entropy the scattered corners are drawn from


@pytest.fixture
def corners() -> Coordinates:
    """Three tight huddles of points, far enough apart that any rule cutting them agrees where they are."""
    generator = np.random.default_rng(_SEED)
    return np.asarray(
        np.concatenate([generator.normal(corner, _HUDDLE, size=(_BLOB, 2)) for corner in _CORNERS]),
        dtype=np.float64,
    )


@pytest.fixture
def truth() -> Labels:
    """Which corner every point of the huddles was drawn around."""
    return np.repeat(np.arange(len(_CORNERS)), _BLOB).astype(np.intp)


def _silhouette_by_hand(coordinates: Coordinates, labels: Labels) -> float:
    """The silhouette read one recording at a time, straight off the definition."""
    scores = []
    for index, label in enumerate(labels):
        reach = np.linalg.norm(coordinates - coordinates[index], axis=1)
        company = (labels == label) & (np.arange(labels.size) != index)
        if not company.any():
            scores.append(0.0)
            continue

        inside = float(reach[company].mean())
        outside = min(float(reach[labels == other].mean()) for other in set(labels.tolist()) - {label})
        scores.append((outside - inside) / max(inside, outside))

    return float(np.mean(scores))


def test_compacting_numbers_the_groups_that_drew_members_from_zero() -> None:
    assert np.array_equal(compact(np.asarray([4, 4, 9, 1], dtype=np.intp)), np.asarray([1, 1, 2, 0]))


@pytest.mark.parametrize("method", list(LinkageMethod))
def test_a_hierarchy_cut_at_three_finds_the_three_huddles(
    method: LinkageMethod, corners: Coordinates, truth: Labels
) -> None:
    labels = cut(hierarchy(corners, method), len(_CORNERS))

    assert {frozenset(np.flatnonzero(labels == group).tolist()) for group in np.unique(labels)} == {
        frozenset(np.flatnonzero(truth == group).tolist()) for group in np.unique(truth)
    }


def test_moving_centres_find_the_three_huddles(corners: Coordinates, truth: Labels) -> None:
    labels = centres(corners, len(_CORNERS))

    assert {frozenset(np.flatnonzero(labels == group).tolist()) for group in np.unique(labels)} == {
        frozenset(np.flatnonzero(truth == group).tolist()) for group in np.unique(truth)
    }


def test_one_space_read_twice_states_one_cut(corners: Coordinates) -> None:
    """The centres open from a settled seed, so a notebook re-reading a space sees the groups it saw before."""
    assert np.array_equal(centres(corners, len(_CORNERS)), centres(corners, len(_CORNERS)))


def test_the_silhouette_matches_a_reading_taken_recording_by_recording(corners: Coordinates, truth: Labels) -> None:
    assert silhouette(pairwise_distances(corners), truth) == pytest.approx(_silhouette_by_hand(corners, truth))


def test_a_recording_alone_in_its_group_scores_nothing() -> None:
    """A lone recording has no company of its own to compare against, and it carries a zero into the mean."""
    coordinates = np.asarray([[0.0], [0.1], [9.0]])
    labels = np.asarray([0, 0, 1], dtype=np.intp)
    paired = (9.0 - 0.1) / 9.0 + (8.9 - 0.1) / 8.9

    assert silhouette(pairwise_distances(coordinates), labels) == pytest.approx(paired / labels.size)


def test_a_cut_leaving_one_group_standing_states_no_separation(corners: Coordinates) -> None:
    assert silhouette(pairwise_distances(corners), np.zeros(corners.shape[0], dtype=np.intp)) == 0.0


def test_a_corpus_standing_at_one_point_draws_no_separation() -> None:
    coordinates = np.zeros((4, 2))
    labels = np.asarray([0, 0, 1, 1], dtype=np.intp)

    assert silhouette(pairwise_distances(coordinates), labels) == 0.0


@pytest.mark.parametrize("algorithm", list(PartitionAlgorithm))
def test_a_partition_names_the_group_every_recording_fell_into(
    algorithm: PartitionAlgorithm, corners: Coordinates, partition_config: PartitionConfig
) -> None:
    config = partition_config.model_copy(update={"algorithm": algorithm})

    cutting = partition(corners, groups=len(_CORNERS), config=config)

    assert cutting.labels.shape == (corners.shape[0],)
    assert cutting.groups == len(_CORNERS)
    assert cutting.silhouette > 0.5


def test_the_sweep_climbs_from_two_up_to_its_ceiling(corners: Coordinates, partition_config: PartitionConfig) -> None:
    config = partition_config.model_copy(update={"groups": 3, "max_groups": 7})

    climbed = sweep(corners, config)

    assert [cutting.groups for cutting in climbed] == list(range(2, 8))


def test_the_sweep_holds_inside_what_the_corpus_can_be_told_apart_into(
    partition_config: PartitionConfig,
) -> None:
    """A count leaving every recording alone reads nothing, so the climb stops one short of the corpus."""
    assert sweep_ceiling(5, partition_config) == 4
    assert len(sweep(np.arange(5.0).reshape(5, 1), partition_config)) == 3


@pytest.mark.parametrize("algorithm", list(PartitionAlgorithm))
def test_the_sweep_reads_the_count_the_material_was_built_with_as_its_best(
    algorithm: PartitionAlgorithm, corners: Coordinates, partition_config: PartitionConfig
) -> None:
    config = partition_config.model_copy(update={"algorithm": algorithm, "max_groups": 8})

    climbed = sweep(corners, config)

    assert max(climbed, key=lambda cutting: cutting.silhouette).groups == len(_CORNERS)
