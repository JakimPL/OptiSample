from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from optisample.cluster.descriptor import SampleDescriptor
from optisample.cluster.pooling import Reached
from optisample.config.cluster import SpaceConfig

Coordinates = NDArray[np.float64]

_MIN_PAIR: Final = 2  # recordings a pair is read across at the fewest
_NO_SPREAD: Final = 0.0  # the spread a reading holding one value across the corpus stands at
_PAIRED: Final = 2.0  # the two ordered pairs each unordered pair of recordings is counted as


@unique
class Block(StrEnum):
    """The four readings a sample space is assembled from, each scaled and weighed on its own.

    ``ONSET`` is the sound a recording opens on and ``SUSTAIN`` the sound it holds at each depth of its own
    decline, which are the two blocks carrying timbre. ``MOVEMENT`` says how far that sound travels while
    the note rings, and ``ENVELOPE`` carries the contour -- how long the note took to reach each depth and
    how straight its fall runs -- as a supplementary distinction.
    """

    ONSET = "onset"
    SUSTAIN = "sustain"
    MOVEMENT = "movement"
    ENVELOPE = "envelope"


@dataclass(frozen=True)
class BlockSpan:
    """Where one block's readings sit among the coordinates, and the say it was given there."""

    block: Block
    weight: float
    start: int
    stop: int

    @property
    def columns(self) -> int:
        """How many coordinates the block holds."""
        return self.stop - self.start


@dataclass(frozen=True)
class SampleSpace:
    """A set of recordings placed as points whose plain distance is the weighted distance across the blocks.

    ``coordinates`` holds one row per recording, ``spans`` says which columns each block laid down and the
    weight it carried there, and ``depths`` marks the fall depths the corpus shared, which is what the
    sustain and contour readings were taken over.
    """

    coordinates: Coordinates
    spans: Mapping[Block, BlockSpan]
    depths: Reached

    @property
    def samples(self) -> int:
        """How many recordings stand in the space."""
        return int(self.coordinates.shape[0])

    @property
    def dimensions(self) -> int:
        """How many coordinates a point is stated by."""
        return int(self.coordinates.shape[1])

    def block(self, block: Block) -> Coordinates:
        """The columns ``block`` holds, weighed as they stand in the space: ``(samples, columns)``."""
        span = self.spans[block]
        return np.asarray(self.coordinates[:, span.start : span.stop], dtype=np.float64)


def block_weight(block: Block, config: SpaceConfig) -> float:
    """The say ``config`` gives ``block`` in the space every group and representative is read in."""
    match block:
        case Block.ONSET:
            return config.onset_weight

        case Block.SUSTAIN:
            return config.sustain_weight

        case Block.MOVEMENT:
            return config.movement_weight

        case Block.ENVELOPE:
            return config.envelope_weight


def standardize(values: Coordinates) -> Coordinates:
    """``values`` with every column stated in spreads of its own, so a block's readings compare column by column.

    A block gathers readings in units of its own -- decibels of spectral shape beside log-seconds of
    timing -- and stating each column against its own spread has them weigh alike inside the block. A column
    holding one value across the corpus reads zero, which is the whole of the distinction it draws.
    """
    spread = values.std(axis=0)
    steady = spread > _NO_SPREAD
    centered = values - values.mean(axis=0)
    return np.asarray(np.where(steady, centered / np.where(steady, spread, 1.0), _NO_SPREAD), dtype=np.float64)


def mean_pairwise_square(values: Coordinates) -> float:
    """The mean squared distance between two rows of ``values``, read across every pair of them.

    The squared distances summed over all ordered pairs come to twice the count of rows times their total
    spread, so the mean follows from the columns' variances alone and the matrix of distances stays
    unbuilt. A corpus of one recording holds no pair and reads zero.
    """
    samples = values.shape[0]
    if samples < _MIN_PAIR:
        return _NO_SPREAD

    return float(_PAIRED * samples * values.var(axis=0).sum() / (samples - 1))


def scale_to_unit_spread(values: Coordinates) -> Coordinates:
    """``values`` scaled so two of its rows stand a squared distance of one apart on average.

    That leaves a block's say in the space settled by its weight alone, whatever its readings were stated
    in and however many columns it brought, so one weight means the same thing across datasets. A block
    whose rows already agree throughout stands where it is.
    """
    spread = mean_pairwise_square(values)
    if spread <= _NO_SPREAD:
        return values

    return np.asarray(values / np.sqrt(spread), dtype=np.float64)


def pairwise_distances(coordinates: Coordinates) -> Coordinates:
    """The distance between every two points of ``coordinates``, square and reading zero down its diagonal."""
    return np.asarray(squareform(pdist(coordinates)), dtype=np.float64)


def _reached_depths(descriptors: Sequence[SampleDescriptor], min_share: float) -> Reached:
    """The fall depths a ``min_share`` share of ``descriptors`` arrived at, which the corpus shares."""
    arrivals = np.asarray([descriptor.reached for descriptor in descriptors], dtype=np.bool_)
    return np.asarray(arrivals.mean(axis=0) >= min_share, dtype=np.bool_)


def _block_readings(descriptor: SampleDescriptor, block: Block, depths: Reached) -> Coordinates:
    """What ``descriptor`` states for ``block``, gathered as one row over the depths ``depths`` marks."""
    match block:
        case Block.ONSET:
            return descriptor.onset

        case Block.SUSTAIN:
            return np.asarray(descriptor.sustain[depths].ravel(), dtype=np.float64)

        case Block.MOVEMENT:
            return descriptor.movement.values

        case Block.ENVELOPE:
            return descriptor.envelope.at(depths)


def _weighed_block(
    descriptors: Sequence[SampleDescriptor],
    block: Block,
    *,
    depths: Reached,
    weight: float,
) -> Coordinates:
    """One block of ``descriptors``, stated in spreads of its own, scaled to unit pairwise spread and weighed.

    The square root of the weight is what the readings carry, so laying the blocks side by side has the
    squared distance between two points come to the weighted sum of the blocks' own squared distances.
    """
    readings = np.asarray(
        np.stack([_block_readings(descriptor, block, depths) for descriptor in descriptors]),
        dtype=np.float64,
    )
    return np.asarray(np.sqrt(weight) * scale_to_unit_spread(standardize(readings)), dtype=np.float64)


def sample_space(descriptors: Sequence[SampleDescriptor], config: SpaceConfig) -> SampleSpace:
    """``descriptors`` placed as points whose plain Euclidean distance is the weighted block distance.

    Each block is stated in spreads of its own columns, scaled to a mean pairwise squared distance of one,
    multiplied by the square root of its weight, and laid beside the rest. The squared distance between
    two points then comes to the weighted sum of the blocks' own squared distances exactly, so the
    principal components, the linkage and the medoid all read one geometry and agree on it.

    Raises:
        ValueError: when ``descriptors`` names no recordings, which is a space with nothing to place.
    """
    if not descriptors:
        raise ValueError("a sample space is built from at least one recording")

    depths = _reached_depths(descriptors, config.min_reached_share)
    spans: dict[Block, BlockSpan] = {}
    columns: list[Coordinates] = []
    start = 0
    for block in Block:
        weight = block_weight(block, config)
        weighed = _weighed_block(descriptors, block, depths=depths, weight=weight)
        spans[block] = BlockSpan(block=block, weight=weight, start=start, stop=start + weighed.shape[1])
        columns.append(weighed)
        start = spans[block].stop

    return SampleSpace(
        coordinates=np.asarray(np.concatenate(columns, axis=1), dtype=np.float64),
        spans=spans,
        depths=depths,
    )
