from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique

import numpy as np

from optisample.calibrate.ranking.renditions import ClipRenditions, Rendition
from optisample.dsp.surrogate import EncodingParams

_EVEN_ODDS = 0.5  # the draw that sends a pair's first member to one side, which is what blinds it


@unique
class Side(StrEnum):
    """Which of a blinded pair a rendition was written as, which is the whole of what a listener is told."""

    A = "a"
    B = "b"


@unique
class PairAxis(StrEnum):
    """What one pair asks a listener, read off the axes its two encodings differ along.

    The three single-axis questions ask whether the metric orders one degradation correctly:
    :attr:`LOOP` how much of the recording is stored, :attr:`RATE` the band it keeps, :attr:`DEPTH` the
    grid it is quantized onto. :attr:`TRADE` asks the question a byte budget asks -- two degradations of
    nearly the same size, one of which has to be bought.
    """

    LOOP = "loop"
    RATE = "rate"
    DEPTH = "depth"
    TRADE = "trade"


@dataclass(frozen=True)
class PairQuota:
    """How many pairs of each question a listening set spends a listener's time on."""

    loop: int
    rate: int
    depth: int
    trade: int

    def asked(self, axis: PairAxis) -> int:
        """How many pairs this quota buys of ``axis``."""
        match axis:
            case PairAxis.LOOP:
                return self.loop
            case PairAxis.RATE:
                return self.rate
            case PairAxis.DEPTH:
                return self.depth
            case PairAxis.TRADE:
                return self.trade

    @property
    def total(self) -> int:
        """Every pair the quota asks for, which is the listening the set costs."""
        return sum(self.asked(axis) for axis in PairAxis)


@dataclass(frozen=True)
class ListeningPair:
    """Two encodings of one note class, blinded, for a listener to place against the recording they came from.

    ``first`` is the member written as :attr:`Side.A` and ``second`` the one written as :attr:`Side.B`,
    settled by a seeded draw so the side a member takes carries nothing about what it is.
    """

    clip: ClipRenditions
    axis: PairAxis
    first: Rendition
    second: Rendition

    @property
    def margin(self) -> float:
        """How far apart the composite puts the two, which is the call a label either confirms or overturns."""
        return abs(self.first.distortion - self.second.distortion)

    @property
    def composite_side(self) -> Side:
        """The side the composite calls closer to the recording."""
        return Side.A if self.first.distortion <= self.second.distortion else Side.B


@dataclass(frozen=True)
class _Candidate:
    """One pair a set may ask about, before the quota has decided whether it is worth a listener's time."""

    clip: ClipRenditions
    axis: PairAxis
    left: Rendition
    right: Rendition

    @property
    def margin(self) -> float:
        return abs(self.left.distortion - self.right.distortion)


def _differing(left: EncodingParams, right: EncodingParams) -> tuple[PairAxis, ...]:
    """The axes two encodings of one clip differ along, in the order a pair is labelled by."""
    axes = []
    if left.loop_index != right.loop_index:
        axes.append(PairAxis.LOOP)

    if left.target_rate != right.target_rate:
        axes.append(PairAxis.RATE)

    if left.depth_bits != right.depth_bits:
        axes.append(PairAxis.DEPTH)

    return tuple(axes)


def _byte_matched(left: Rendition, right: Rendition, tolerance: float) -> bool:
    """Whether two encodings store within ``tolerance`` of one another, as a share of the larger."""
    larger, smaller = max(left.stored_bytes, right.stored_bytes), min(left.stored_bytes, right.stored_bytes)
    return larger - smaller <= tolerance * larger


def _question(left: Rendition, right: Rendition, *, tolerance: float) -> PairAxis | None:
    """What a listener would be asked by hearing these two, where the pair is worth asking about at all.

    A pair differing along one axis asks about that axis. A pair differing along several asks about the
    trade between them, which is a question only where the two cost nearly the same bytes: two encodings
    of different size are ordered by their price as much as by their sound, and a label on them says
    nothing a byte count did not already.
    """
    axes = _differing(left.params, right.params)
    if len(axes) == 1:
        return axes[0]

    if len(axes) > 1 and _byte_matched(left, right, tolerance):
        return PairAxis.TRADE

    return None


def _candidates(clips: Sequence[ClipRenditions], *, tolerance: float) -> Iterator[_Candidate]:
    """Every pair of encodings any clip offers that a listener could be asked about."""
    for clip in clips:
        for index, left in enumerate(clip.renditions):
            for right in clip.renditions[index + 1 :]:
                axis = _question(left, right, tolerance=tolerance)
                if axis is not None:
                    yield _Candidate(clip=clip, axis=axis, left=left, right=right)


def _spread(candidates: Sequence[_Candidate], count: int) -> tuple[_Candidate, ...]:
    """``count`` of ``candidates``, spread over the margins the composite reads and over the clips offering them.

    Ordering by margin and taking one from each equal share of that ordering fills the set from both ends
    of the metric's confidence: the near-ties, where any ranking is fragile and a label is worth most, and
    the clear calls, which are the anchor a metric failing them fails outright. Within a share the clip
    heard least so far is taken, so a register or a dynamic that happens to offer many pairs earns no more
    of the listening than the rest.
    """
    if count >= len(candidates):
        return tuple(candidates)

    ordered = sorted(candidates, key=lambda candidate: candidate.margin)
    heard: Counter[int] = Counter()
    taken: list[_Candidate] = []
    for share in range(count):
        opening, closing = share * len(ordered) // count, (share + 1) * len(ordered) // count
        chosen = min(ordered[opening:closing], key=lambda candidate: heard[candidate.clip.pitch])
        heard[chosen.clip.pitch] += 1
        taken.append(chosen)

    return tuple(taken)


def _blinded(candidate: _Candidate, draw: float) -> ListeningPair:
    """``candidate`` with its two members settled onto sides, ``draw`` deciding which takes which."""
    ordered = (candidate.left, candidate.right) if draw < _EVEN_ODDS else (candidate.right, candidate.left)
    return ListeningPair(clip=candidate.clip, axis=candidate.axis, first=ordered[0], second=ordered[1])


def listening_pairs(
    clips: Sequence[ClipRenditions],
    quota: PairQuota,
    *,
    byte_tolerance: float,
    seed: int,
) -> tuple[ListeningPair, ...]:
    """The pairs a listening set asks about, chosen to a quota, shuffled and blinded.

    Each question is filled from its own candidates (:func:`_spread`), so the axes the sweep holds fixed
    earn as much of the listening as the one it varies. The set is then shuffled, which leaves a listener
    reaching pair after pair with nothing in their order to say what any of them is about, and each pair
    is blinded, which leaves nothing in a side either.
    """
    offered = tuple(_candidates(clips, tolerance=byte_tolerance))
    chosen = [
        candidate
        for axis in PairAxis
        for candidate in _spread([offer for offer in offered if offer.axis is axis], quota.asked(axis))
    ]
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(chosen))
    draws = generator.random(len(chosen))
    return tuple(_blinded(chosen[int(place)], float(draw)) for place, draw in zip(order, draws))
