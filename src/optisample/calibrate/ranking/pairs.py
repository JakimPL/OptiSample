from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final

import numpy as np
from numpy.random import Generator

from optisample.calibrate.ranking.renditions import ClipRenditions, Rendition
from optisample.dsp.surrogate import EncodingParams

_EVEN_ODDS = 0.5  # the draw that sends a pair's first member to one side, which is what blinds it
_MIN_REPEAT_GAP: Final = 16  # questions a listener answers between meeting one and meeting it again


@unique
class Side(StrEnum):
    """Which of a blinded pair a rendition was written as, which is the whole of what a listener is told."""

    A = "a"
    B = "b"


@unique
class PairAxis(StrEnum):
    """What one pair asks a listener, read off the axes its two encodings differ along.

    The four single-axis questions ask whether the metric orders one degradation correctly: :attr:`LOOP`
    how much of the recording is stored, :attr:`RATE` the band it keeps, :attr:`DEPTH` the grid it is
    quantized onto, and :attr:`COMPRESS` how far its crest factor was narrowed to fit that grid.
    :attr:`TRADE` asks the question a byte budget asks -- two degradations of nearly the same size, one of
    which has to be bought.
    """

    LOOP = "loop"
    RATE = "rate"
    DEPTH = "depth"
    COMPRESS = "compress"
    TRADE = "trade"


@dataclass(frozen=True)
class PairQuota:
    """How many pairs of each question a listening set spends a listener's time on."""

    loop: int
    rate: int
    depth: int
    compress: int
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
            case PairAxis.COMPRESS:
                return self.compress
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
    settled by a seeded draw so the side a member takes carries nothing about what it is. ``question_id``
    names the comparison being put: a set asks a handful of them twice, blinded afresh and far apart, and
    the two occurrences share it, so how far a listener agrees with themselves is read off the same sheet
    as how far the metric agrees with them.
    """

    clip: ClipRenditions
    axis: PairAxis
    question_id: int
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

    @property
    def loudness_delta_lu(self) -> float:
        """How much louder :attr:`Side.A` plays than :attr:`Side.B`, which is the confound a preference risks.

        A listener reaches for the louder of two clips, so a preference that tracks this rather than the
        recording is a label about level. The sides are written at the level their encodings produce, which
        keeps the level the composite quotients out in front of the listener, and the gap is stated here so
        the reading can be checked against it.
        """
        return self.first.loudness_lufs - self.second.loudness_lufs


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
    """The axes two encodings of one clip differ along, in the order a pair is labeled by.

    The compressor counts among them because the pipeline ties it to depth: a pair reading as one
    question while it moves both the grid and the dynamics would carry a label that speaks for neither.
    """
    axes = []
    if left.loop_index != right.loop_index:
        axes.append(PairAxis.LOOP)

    if left.target_rate != right.target_rate:
        axes.append(PairAxis.RATE)

    if left.depth_bits != right.depth_bits:
        axes.append(PairAxis.DEPTH)

    if left.compress != right.compress:
        axes.append(PairAxis.COMPRESS)

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


def _blinded(candidate: _Candidate, question_id: int, draw: float) -> ListeningPair:
    """``candidate`` with its two members settled onto sides, ``draw`` deciding which takes which."""
    ordered = (candidate.left, candidate.right) if draw < _EVEN_ODDS else (candidate.right, candidate.left)
    return ListeningPair(
        clip=candidate.clip,
        axis=candidate.axis,
        question_id=question_id,
        first=ordered[0],
        second=ordered[1],
    )


def _repeat_gap(asked: int) -> int:
    """How many questions stand between one being met and being met again, held inside what a set has room for."""
    return min(_MIN_REPEAT_GAP, asked // 2)


def _with_repeats(order: Sequence[int], repeats: int, generator: Generator) -> tuple[int, ...]:
    """``order`` with up to ``repeats`` of its questions asked a second time, each well after the first.

    The sources are taken at even shares of the opening stretch, so the repeats read a listener across the
    whole session and a drift in how they answer late on shows up as a run of disagreements. Placing them
    from the last source backwards holds every gap at what it was drawn to be, since a repeat inserted
    earlier only ever pushes an already-placed one further from its source.
    """
    gap = _repeat_gap(len(order))
    asked = min(repeats, len(order) - gap)
    if asked <= 0:
        return tuple(order)

    sequence = list(order)
    sources = [share * (len(order) - gap) // asked for share in range(asked)]
    for source in sorted(sources, reverse=True):
        offset = int(generator.integers(gap, len(sequence) - source))
        sequence.insert(source + offset, order[source])

    return tuple(sequence)


def listening_pairs(
    clips: Sequence[ClipRenditions],
    quota: PairQuota,
    *,
    byte_tolerance: float,
    repeats: int,
    seed: int,
) -> tuple[ListeningPair, ...]:
    """The pairs a listening set asks about, chosen to a quota, repeated in part, shuffled and blinded.

    Each question is filled from its own candidates (:func:`_spread`), so the axes the sweep holds fixed
    earn as much of the listening as the one it varies. The set is then shuffled, which leaves a listener
    reaching pair after pair with nothing in their order to say what any of them is about, and each pair
    is blinded, which leaves nothing in a side either -- a repeated question draws its sides afresh, so
    the two occurrences read as two questions and answering them alike is a reading of the listener.
    """
    offered = tuple(_candidates(clips, tolerance=byte_tolerance))
    chosen = [
        candidate
        for axis in PairAxis
        for candidate in _spread([offer for offer in offered if offer.axis is axis], quota.asked(axis))
    ]
    generator = np.random.default_rng(seed)
    shuffled = [int(place) for place in generator.permutation(len(chosen))]
    order = _with_repeats(shuffled, repeats, generator)
    draws = generator.random(len(order))
    return tuple(_blinded(chosen[place], place, float(draw)) for place, draw in zip(order, draws))
