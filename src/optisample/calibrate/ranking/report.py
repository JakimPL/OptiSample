from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import median
from typing import Final

from scipy.stats import kendalltau

from optisample.calibrate.ranking.pairs import PairAxis, Side
from optisample.calibrate.ranking.verdicts import Verdict

_RANKABLE: Final = 2  # readings a rank correlation needs before it states anything
_PAIRED: Final = 2  # occurrences a repeated question is read across at a time
_AUDIBLE_GAP_LU: Final = 1.0  # the level difference a listener begins to hear on program material


def heard_strength(verdict: Verdict) -> int:
    """How far ahead of :attr:`Side.B` the listener placed :attr:`Side.A`, negative where they placed B ahead.

    Signing the scale is what puts a verdict and a metric's margin on one axis: both then state a
    direction and a size, and they agree exactly where they share a sign.
    """
    match verdict.closer:
        case Side.A:
            return verdict.strength
        case Side.B:
            return -verdict.strength
        case None:
            return 0


@dataclass(frozen=True)
class Judgment:
    """One answered question, read as the signed strength a metric is measured against.

    ``heard`` is signed toward :attr:`Side.A`, which is the way a listener met the pair. ``swapped``
    records that the blinding draw put the question's own first encoding on :attr:`Side.B`, which is what
    :attr:`aligned` reads the answer back through: the two occurrences of a repeated question are blinded
    afresh, so the encodings are all they share and only an answer signed toward those can be compared.
    ``identical`` marks the level answers the listener heard nothing at all between, which the margins
    read apart from the rest.
    """

    directory: str
    axis: PairAxis
    question_id: int
    heard: int
    identical: bool
    swapped: bool
    loudness_delta_lu: float

    @property
    def decided(self) -> bool:
        """Whether the listener placed one of the two sides closer to the recording."""
        return self.heard != 0

    @property
    def aligned(self) -> int:
        """The same answer signed toward the question's own first encoding rather than toward a side."""
        return -self.heard if self.swapped else self.heard


@dataclass(frozen=True)
class MetricReadings:
    """What one metric makes of every question in a set, signed the way a verdict is.

    Each entry states how far ahead of :attr:`Side.B` the metric puts :attr:`Side.A`, keyed by the
    directory the question's audio sits in, which is the name a label carries.
    """

    name: str
    read: dict[str, float]


@dataclass(frozen=True)
class Agreement:
    """How far one metric's reading of a set of questions stands from the listener's.

    ``tau`` reads direction and size together: a metric ordering every pair correctly still loses ground
    where it puts a clear call closer than a near-tie. ``matched`` counts direction alone over the calls
    the listener made, which is the plainer figure and the one a chance level of 0.5 is read against.
    """

    tau: float | None
    decided: int
    matched: int

    @property
    def confirmed(self) -> float | None:
        """The share of the listener's calls the metric makes the same way, where they made any."""
        return self.matched / self.decided if self.decided else None


@dataclass(frozen=True)
class MetricAgreement:
    """One metric ranked against the whole answer sheet, and against each question it holds.

    The three margins price the metric's confidence against the listener's: ``tied_margin`` is what it
    reads where the listener placed the sides level, ``identical_margin`` the part of that where they
    heard nothing at all between them, and ``decided_margin`` where they heard one ahead. A metric worth
    its ranking reads the last well clear of both.
    """

    name: str
    overall: Agreement
    by_axis: dict[PairAxis, Agreement]
    decided_margin: float | None
    tied_margin: float | None
    identical_margin: float | None

    @property
    def separation(self) -> float | None:
        """How much wider the metric reads a question the listener called than one they heard as level.

        Above 1.0 the metric is quiet where the ear was, which is what a ranking rests on beyond getting
        the order right; at 1.0 it reads the same distance whether or not there was one to hear.
        """
        if self.decided_margin is None or self.tied_margin is None or self.tied_margin == 0.0:
            return None

        return self.decided_margin / self.tied_margin

    @property
    def headroom(self) -> float | None:
        """How far the metric's calls stand above what it reads on a pair the listener met as one recording.

        :attr:`separation` is read against every question placed level, which holds pairs whose sides a
        listener told apart and ranked equal. This is read against the ones there was nothing between to
        hear, so it prices the metric against its own floor, which is the strictest reading a sheet offers.
        """
        if self.decided_margin is None or self.identical_margin is None or self.identical_margin == 0.0:
            return None

        return self.decided_margin / self.identical_margin


@dataclass(frozen=True)
class SelfAgreement:
    """The ceiling every metric is read against: how far the listener stands from themselves.

    A handful of questions are put twice, far apart and blinded afresh. How often the two occurrences
    name the same encoding is the most any metric could match, so a metric confirming as much of the
    sheet as the listener repeats has reached the evidence rather than fallen short of it.
    """

    repeated: int
    tau: float | None
    decided: int
    matched: int

    @property
    def confirmed(self) -> float | None:
        """The share of the repeated calls the listener made the same way twice, where they made any."""
        return self.matched / self.decided if self.decided else None


@dataclass(frozen=True)
class LevelConfound:
    """How far the labels track the level the two sides played at rather than the recording.

    Level is left as each encoding produces it, so a listener reaching for the louder of two clips is a
    risk the sheet carries and this is what reads it. ``louder`` counts the calls naming the louder side
    among those whose sides stood an audible gap apart; near 0.5 the preferences are about the recording,
    and near 1.0 they are about the volume.
    """

    tau: float | None
    gapped: int
    louder: int

    @property
    def followed(self) -> float | None:
        """The share of the calls with an audible level gap that named the louder side."""
        return self.louder / self.gapped if self.gapped else None


@dataclass(frozen=True)
class RankingReport:
    """Every metric read against one instrument's answer sheet, with the ceiling and the confound beside it.

    ``metrics`` holds the composite as the objective reads it, the composite as the listener heard it,
    each component of it alone, and the level diagnostics -- so the table states both how the metric in
    use fares and which of the terms available today would fare better.
    """

    instrument_id: str
    answered: int
    outstanding: int
    metrics: tuple[MetricAgreement, ...]
    ceiling: SelfAgreement
    level: LevelConfound


def _kendall_tau(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Kendall's tau-b between two readings of the same questions, absent where nothing can be ordered.

    Tau-b is the form that handles ties on both sides, and both sides hold them: a listener answers
    ``tie`` freely, and a metric can put two encodings level.
    """
    if len(first) < _RANKABLE:
        return None

    correlation, _ = kendalltau(first, second)
    return None if math.isnan(correlation) else float(correlation)


def _agreeing(stated: float, heard: float) -> bool:
    """Whether two signed readings of one question name the same side of it."""
    return (stated > 0 and heard > 0) or (stated < 0 and heard < 0)


def _agreement(judgments: Sequence[Judgment], read: Mapping[str, float]) -> Agreement:
    """One metric read against the questions in ``judgments``."""
    decided = [judgment for judgment in judgments if judgment.decided]
    return Agreement(
        tau=_kendall_tau(
            [judgment.heard for judgment in judgments],
            [read[judgment.directory] for judgment in judgments],
        ),
        decided=len(decided),
        matched=sum(1 for judgment in decided if _agreeing(read[judgment.directory], judgment.heard)),
    )


def _median_margin(judgments: Sequence[Judgment], read: Mapping[str, float]) -> float | None:
    """How far apart a metric puts the two sides of a typical question here, absent where there are none."""
    if not judgments:
        return None

    return median(abs(read[judgment.directory]) for judgment in judgments)


def metric_agreement(readings: MetricReadings, judgments: Sequence[Judgment]) -> MetricAgreement:
    """Rank one metric against the answered questions, whole and question by question.

    The per-axis reading is what turns a poor total into a change worth making: a metric agreeing on
    every axis but one has a term missing rather than a weighting wrong.
    """
    asked = {judgment.axis for judgment in judgments}
    return MetricAgreement(
        name=readings.name,
        overall=_agreement(judgments, readings.read),
        by_axis={
            axis: _agreement([judgment for judgment in judgments if judgment.axis is axis], readings.read)
            for axis in PairAxis
            if axis in asked
        },
        decided_margin=_median_margin([one for one in judgments if one.decided], readings.read),
        tied_margin=_median_margin([one for one in judgments if not one.decided], readings.read),
        identical_margin=_median_margin([one for one in judgments if one.identical], readings.read),
    )


def self_agreement(judgments: Sequence[Judgment]) -> SelfAgreement:
    """How far the listener stands from themselves on the questions the set put more than once."""
    grouped: dict[int, list[Judgment]] = defaultdict(list)
    for judgment in judgments:
        grouped[judgment.question_id].append(judgment)

    repeated = [group for group in grouped.values() if len(group) >= _PAIRED]
    occurrences = [met for group in repeated for met in combinations(group, _PAIRED)]
    decided = [(first, second) for first, second in occurrences if first.decided and second.decided]
    return SelfAgreement(
        repeated=len(repeated),
        tau=_kendall_tau(
            [first.aligned for first, _ in occurrences],
            [second.aligned for _, second in occurrences],
        ),
        decided=len(decided),
        matched=sum(1 for first, second in decided if _agreeing(first.aligned, second.aligned)),
    )


def level_confound(judgments: Sequence[Judgment]) -> LevelConfound:
    """Read the answered questions against the level their two sides played at rather than the recording."""
    gapped = [
        judgment for judgment in judgments if judgment.decided and abs(judgment.loudness_delta_lu) >= _AUDIBLE_GAP_LU
    ]
    return LevelConfound(
        tau=_kendall_tau(
            [judgment.heard for judgment in judgments],
            [judgment.loudness_delta_lu for judgment in judgments],
        ),
        gapped=len(gapped),
        louder=sum(1 for judgment in gapped if _agreeing(judgment.loudness_delta_lu, judgment.heard)),
    )


def ranking_report(
    judgments: Sequence[Judgment],
    readings: Sequence[MetricReadings],
    *,
    instrument_id: str,
    outstanding: int,
) -> RankingReport:
    """Rank every metric in ``readings`` against the answered questions, with the ceiling and the confound.

    ``outstanding`` is carried through from the sheet, so a report read part way through a session states
    how much of the listening it rests on rather than reading as a finished one.
    """
    return RankingReport(
        instrument_id=instrument_id,
        answered=len(judgments),
        outstanding=outstanding,
        metrics=tuple(metric_agreement(reading, judgments) for reading in readings),
        ceiling=self_agreement(judgments),
        level=level_confound(judgments),
    )
