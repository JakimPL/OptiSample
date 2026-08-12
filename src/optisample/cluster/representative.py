from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.cluster.partition import Labels
from optisample.cluster.space import Coordinates, pairwise_distances
from optisample.config.cluster import Representative

Members = NDArray[np.intp]
Weights = NDArray[np.float64]

_ALONE: Final = 1  # members a group holds when its recording keeps its own company


def uniform_weights(samples: int) -> Weights:
    """Every recording carrying the same say, which is what a corpus states before a playing time is read."""
    return np.ones(samples, dtype=np.float64)


@dataclass(frozen=True)
class Group:
    """One group of a cut sample space, stated by its middle and the recordings standing for it.

    ``centroid`` is the mean coordinate, a point in the space; the three that follow are real takes a
    listener can play, each an index into the space. ``medoid`` is the member standing closest to the rest
    of its group and is the dominant representative; ``weighted_medoid`` reads that same sum with each
    member carrying the say it was given, so the representative is the one the music leans on;
    ``nearest_centroid`` is the member sitting closest to the mean. ``spread_mean`` and ``spread_max`` say
    how far the group reaches from its medoid and ``farthest`` names the member out at that edge, which
    together say how much of a sound the representative genuinely covers.
    """

    label: int
    members: Members
    centroid: Coordinates
    medoid: int
    weighted_medoid: int
    nearest_centroid: int
    spread_mean: float
    spread_max: float
    farthest: int

    @property
    def size(self) -> int:
        """How many recordings the group holds."""
        return int(self.members.size)

    def representative(self, rule: Representative) -> int:
        """Which recording stands for the group under ``rule``, as an index into the space."""
        match rule:
            case Representative.MEDOID:
                return self.medoid

            case Representative.WEIGHTED_MEDOID:
                return self.weighted_medoid

            case Representative.NEAREST_CENTROID:
                return self.nearest_centroid


def grouping(coordinates: Coordinates, labels: Labels, weights: Weights) -> tuple[Group, ...]:
    """Every group ``labels`` cut out of ``coordinates``, read for its middle and its representatives.

    ``weights`` gives each recording the say it carries in the weighted medoid -- the playing time the
    material asks of its key is what the pipeline has to offer there, and :func:`uniform_weights` states a
    corpus that has yet to be read for one.

    Raises:
        ValueError: when ``labels`` or ``weights`` state a count of recordings other than the space holds.
    """
    samples = coordinates.shape[0]
    if labels.size != samples:
        raise ValueError(f"labels name {labels.size} recordings against the space's {samples}")

    if weights.size != samples:
        raise ValueError(f"weights state {weights.size} recordings against the space's {samples}")

    distances = pairwise_distances(coordinates)
    return tuple(
        _group(
            int(label),
            np.flatnonzero(labels == label),
            coordinates=coordinates,
            distances=distances,
            weights=weights,
        )
        for label in np.unique(labels)
    )


def _group(
    label: int,
    members: Members,
    *,
    coordinates: Coordinates,
    distances: Coordinates,
    weights: Weights,
) -> Group:
    """The group ``members`` make, with every representative read off the distances inside it."""
    inside = distances[np.ix_(members, members)]
    centroid = np.asarray(coordinates[members].mean(axis=0), dtype=np.float64)
    medoid = int(members[np.argmin(inside.sum(axis=1))])
    reach = distances[medoid, members]
    return Group(
        label=label,
        members=members,
        centroid=centroid,
        medoid=medoid,
        weighted_medoid=int(members[np.argmin(inside @ weights[members])]),
        nearest_centroid=int(members[np.argmin(np.linalg.norm(coordinates[members] - centroid, axis=1))]),
        spread_mean=float(reach.sum() / max(members.size - _ALONE, _ALONE)),
        spread_max=float(reach.max()),
        farthest=int(members[np.argmax(reach)]),
    )
