from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2

from optisample.cluster.space import Coordinates, pairwise_distances
from optisample.config.cluster import LinkageMethod, PartitionAlgorithm, PartitionConfig
from optisample.seed import DEFAULT_SEED

Labels = NDArray[np.intp]
Tree = NDArray[np.float64]

_MIN_GROUPS: Final = 2  # groups a separation is read across at the fewest
_ALONE: Final = 1  # members a group holds when its recording keeps its own company
_NO_SEPARATION: Final = 0.0  # what a cut with nothing to compare against states
_MAXCLUST: Final = "maxclust"  # scipy's rule for reading a hierarchy at the height leaving a count standing
_KMEANS_INIT: Final = "++"  # scipy's spread-out seeding, which opens the centres far apart across the material
_KMEANS_ROUNDS: Final = 50  # passes the centres are moved over, ample for a corpus of a few hundred takes


@dataclass(frozen=True)
class Partition:
    """One cut of a sample space: the group each recording fell into, and how far the groups stand apart.

    ``labels`` numbers the groups from zero and every number it holds names a group with members in it, so
    ``groups`` is the count that genuinely came out of the cut. ``silhouette`` runs from -1 to 1 and says
    how much closer a recording sits to its own group than to the nearest other one, averaged over the
    corpus.
    """

    labels: Labels
    groups: int
    silhouette: float


def compact(labels: Labels) -> Labels:
    """``labels`` renumbered from zero over the groups that drew members, holding their original order."""
    _, ordered = np.unique(labels, return_inverse=True)
    return np.asarray(ordered, dtype=np.intp)


def hierarchy(coordinates: Coordinates, method: LinkageMethod) -> Tree:
    """The tree ``coordinates`` join into under ``method``, built once and read at whichever height a cut asks.

    Holding the tree lets a sweep price every group count off one build, and it is what a dendrogram draws.
    """
    return np.asarray(linkage(coordinates, method=method.value), dtype=np.float64)


def cut(tree: Tree, groups: int) -> Labels:
    """The groups ``tree`` states when it is read at the height leaving ``groups`` of them standing."""
    return compact(np.asarray(fcluster(tree, t=groups, criterion=_MAXCLUST), dtype=np.intp))


def centres(coordinates: Coordinates, groups: int) -> Labels:
    """The groups ``coordinates`` fall into around ``groups`` centres moved until they settle.

    The centres open spread out across the material and the run dithers from :data:`DEFAULT_SEED`, so one
    space read twice states one cut. A centre that draws members away from every other leaves the cut
    holding the groups that kept theirs.
    """
    _, labels = kmeans2(
        coordinates,
        min(groups, coordinates.shape[0]),
        iter=_KMEANS_ROUNDS,
        minit=_KMEANS_INIT,
        rng=DEFAULT_SEED,
    )
    return compact(np.asarray(labels, dtype=np.intp))


def silhouette(distances: Coordinates, labels: Labels) -> float:
    """How far the groups ``labels`` names stand apart in ``distances``, from -1 to 1.

    Each recording is scored by how much closer it sits to its own group's members than to the nearest
    other group's, stated as a share of the wider of the two, and the mean of those scores is what the cut
    is worth. A recording alone in its group scores zero, having no company of its own to compare against,
    and a cut leaving one group standing states the same.
    """
    groups = int(labels.max()) + 1 if labels.size else 0
    if groups < _MIN_GROUPS:
        return _NO_SEPARATION

    counts = np.bincount(labels, minlength=groups).astype(np.float64)
    reach = np.stack([distances[:, labels == group].sum(axis=1) for group in range(groups)], axis=1)
    rows = np.arange(labels.size)
    company = counts[labels] - _ALONE
    inside = np.where(company > 0.0, reach[rows, labels] / np.where(company > 0.0, company, 1.0), _NO_SEPARATION)
    outside = reach / counts
    outside[rows, labels] = np.inf
    nearest = outside.min(axis=1)
    widest = np.maximum(inside, nearest)
    scored = (company > 0.0) & (widest > _NO_SEPARATION)
    return float(np.where(scored, (nearest - inside) / np.where(scored, widest, 1.0), _NO_SEPARATION).mean())


def sweep_ceiling(samples: int, config: PartitionConfig) -> int:
    """The most groups a sweep climbs to, held inside what ``samples`` recordings can be told apart into.

    Stopping one short of the corpus leaves at least one group holding a pair, which is what a silhouette
    reads a separation off.
    """
    return min(config.max_groups, samples - _ALONE)


def partition(coordinates: Coordinates, *, groups: int, config: PartitionConfig) -> Partition:
    """``coordinates`` cut into ``groups`` groups by the rule ``config`` names, scored for their separation."""
    return _scored(_labels(coordinates, groups=groups, config=config), pairwise_distances(coordinates))


def sweep(coordinates: Coordinates, config: PartitionConfig) -> tuple[Partition, ...]:
    """What every group count from two up to the sweep's ceiling is worth, one scored cut apiece.

    The distances and, for a hierarchy, the tree are built once and read at every count, so a reader picks
    the count off the curve for the price of one build.
    """
    distances = pairwise_distances(coordinates)
    counts = range(_MIN_GROUPS, sweep_ceiling(coordinates.shape[0], config) + 1)
    match config.algorithm:
        case PartitionAlgorithm.HIERARCHICAL:
            tree = hierarchy(coordinates, config.linkage)
            return tuple(_scored(cut(tree, groups), distances) for groups in counts)

        case PartitionAlgorithm.KMEANS:
            return tuple(_scored(centres(coordinates, groups), distances) for groups in counts)


def _labels(coordinates: Coordinates, *, groups: int, config: PartitionConfig) -> Labels:
    """The group each row of ``coordinates`` falls into under the rule ``config`` names."""
    match config.algorithm:
        case PartitionAlgorithm.HIERARCHICAL:
            return cut(hierarchy(coordinates, config.linkage), groups)

        case PartitionAlgorithm.KMEANS:
            return centres(coordinates, groups)


def _scored(labels: Labels, distances: Coordinates) -> Partition:
    """``labels`` gathered as a partition, with the separation they draw read off ``distances``."""
    return Partition(
        labels=labels,
        groups=int(labels.max()) + 1 if labels.size else 0,
        silhouette=silhouette(distances, labels),
    )
