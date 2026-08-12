from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.cluster.partition import Labels
from optisample.cluster.space import Coordinates, pairwise_distances
from optisample.config.cluster import PartitionConfig, Representative

Members = NDArray[np.intp]
Weights = NDArray[np.float64]
Durations = NDArray[np.float64]

_ALONE: Final = 1  # members a group holds when its recording keeps its own company
_NEAREST_FIRST: Final = "stable"  # the sort keeping equal distances in the order the members stand


def uniform_weights(samples: int) -> Weights:
    """Every recording carrying the same say, which is what a corpus states before a playing time is read."""
    return np.ones(samples, dtype=np.float64)


@dataclass(frozen=True)
class MemberReadings:
    """What every recording of a space carries beyond its place in it, read once for the whole corpus.

    ``weights`` is the say each recording holds, which the weighted medoid reads its sum under -- the
    playing time the material asks of a key is what the pipeline has to offer there, and
    :func:`uniform_weights` states a corpus that has yet to be read for one. ``durations_s`` is how long
    each recording sounds for, which is what holds a representative to a length a listener can shape.
    """

    weights: Weights
    durations_s: Durations


@dataclass(frozen=True)
class Group:
    """One group of a cut sample space, stated by its middle and the recordings standing for it.

    ``centroid`` is the mean coordinate, a point in the space; the four that follow are real takes a
    listener can play, each an index into the space. ``medoid`` is the member standing closest to the rest
    of its group; ``weighted_medoid`` reads that same sum with each member carrying the say it was given, so
    it is the take the music leans on; ``nearest_centroid`` is the member sitting closest to the mean.
    ``representative`` is the one of the three the configured rule named, held to the length a take has to
    ring for to be worth shaping, which is what makes it the take the group is stood for by everywhere.
    ``spread_mean`` and ``spread_max`` say how far the group reaches from its medoid and ``farthest`` names
    the member out at that edge, which together say how much of a sound the representative genuinely covers.
    """

    label: int
    members: Members
    centroid: Coordinates
    medoid: int
    weighted_medoid: int
    nearest_centroid: int
    representative: int
    spread_mean: float
    spread_max: float
    farthest: int

    @property
    def size(self) -> int:
        """How many recordings the group holds."""
        return int(self.members.size)


@dataclass(frozen=True)
class _Placed:
    """One space as a group is read against it: where the recordings sit, how far apart, and what each carries."""

    coordinates: Coordinates
    distances: Coordinates
    readings: MemberReadings


def grouping(
    coordinates: Coordinates,
    labels: Labels,
    readings: MemberReadings,
    *,
    config: PartitionConfig,
) -> tuple[Group, ...]:
    """Every group ``labels`` cut out of ``coordinates``, read for its middle and the take standing for it.

    ``config`` settles the representative here rather than at every panel that reads one, so the take a
    table names, the point a picture rings and the recording a written selection carries are one take.

    Raises:
        ValueError: when ``labels`` or either of the ``readings`` states a count of recordings other than
            the space holds.
    """
    samples = coordinates.shape[0]
    if labels.size != samples:
        raise ValueError(f"labels name {labels.size} recordings against the space's {samples}")

    if readings.weights.size != samples:
        raise ValueError(f"weights state {readings.weights.size} recordings against the space's {samples}")

    if readings.durations_s.size != samples:
        raise ValueError(f"durations state {readings.durations_s.size} recordings against the space's {samples}")

    placed = _Placed(coordinates=coordinates, distances=pairwise_distances(coordinates), readings=readings)
    return tuple(
        _group(int(label), np.flatnonzero(labels == label), placed=placed, config=config) for label in np.unique(labels)
    )


def _group(label: int, members: Members, *, placed: _Placed, config: PartitionConfig) -> Group:
    """The group ``members`` make, with every candidate and the settled representative read off it."""
    inside = placed.distances[np.ix_(members, members)]
    centroid = np.asarray(placed.coordinates[members].mean(axis=0), dtype=np.float64)
    medoid = int(members[np.argmin(inside.sum(axis=1))])
    weighted_medoid = int(members[np.argmin(inside @ placed.readings.weights[members])])
    nearest_centroid = int(members[np.argmin(np.linalg.norm(placed.coordinates[members] - centroid, axis=1))])
    named = _candidate(
        config.representative,
        medoid=medoid,
        weighted_medoid=weighted_medoid,
        nearest_centroid=nearest_centroid,
    )
    reach = placed.distances[medoid, members]
    return Group(
        label=label,
        members=members,
        centroid=centroid,
        medoid=medoid,
        weighted_medoid=weighted_medoid,
        nearest_centroid=nearest_centroid,
        representative=_long_enough(named, members, placed=placed, floor_s=config.min_duration_s),
        spread_mean=float(reach.sum() / max(members.size - _ALONE, _ALONE)),
        spread_max=float(reach.max()),
        farthest=int(members[np.argmax(reach)]),
    )


def _candidate(rule: Representative, *, medoid: int, weighted_medoid: int, nearest_centroid: int) -> int:
    """Which member ``rule`` names, before the length a representative is held to is read."""
    match rule:
        case Representative.MEDOID:
            return medoid

        case Representative.WEIGHTED_MEDOID:
            return weighted_medoid

        case Representative.NEAREST_CENTROID:
            return nearest_centroid


def _long_enough(named: int, members: Members, *, placed: _Placed, floor_s: float) -> int:
    """The take standing for the group once the length one has to ring for is read, walking out from ``named``.

    A recording too short to loop or shape is a representative a later stage can do little with, so where
    the rule's own choice falls short the group is walked outward from it and the first member ringing for
    ``floor_s`` takes its place. Stepping out from the named take rather than searching the group afresh
    keeps the substitute as close to what the rule chose as the material allows, so the group is still stood
    for by a take from its own middle. A group whose every member falls short is left standing on the rule's
    choice, which keeps a representative a real member of the group it stands for.
    """
    durations_s = placed.readings.durations_s
    if durations_s[named] >= floor_s:
        return named

    outward = members[np.argsort(placed.distances[named, members], kind=_NEAREST_FIRST)]
    for member in outward.tolist():
        if durations_s[member] >= floor_s:
            return int(member)

    return named
