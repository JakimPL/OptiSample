from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trackmod import BitDepth

from optisample.artifacts.serialize import Frozen
from optisample.calibrate.ranking import (
    Agreement,
    ListeningPair,
    MetricAgreement,
    PairAxis,
    RankingReport,
    Rendition,
    Side,
)


class RenditionRecord(Frozen):
    """One side of a blinded pair, decoded: the encoding it was stored under and what it earned.

    ``loop_index`` names the region of the recording the sample was stored around, and is absent where
    the encoding keeps the played span as it was recorded.
    """

    side: Side
    target_rate: int
    depth: BitDepth
    compress: bool
    loop_index: int | None
    stored_bytes: int
    distortion: float
    loudness_lufs: float


class ListeningPairRecord(Frozen):
    """One question a listening set asks, with the answer the composite already gives it.

    ``directory`` is where the audio sits, named so a listing reads in the order the pairs are met and
    carries nothing of what any of them is about. ``composite_side`` is the side the metric calls closer
    to the recording, which is the claim a label either confirms or overturns, and ``margin`` how far
    apart it puts the two -- a pair the metric all but ties is where a label is worth most.
    ``question_id`` is shared by the pairs putting one comparison twice, which is how a listener's
    agreement with themselves is read, and ``loudness_delta_lu`` states how much louder side A plays, so
    a preference can be checked against the level it was heard at. ``heard_gain_db`` is the one lift all
    three of the question's recordings were written with, so what each encoding delivers and what the
    listener met stay reconcilable; the gap between the sides is what it is either way.
    """

    directory: str
    axis: PairAxis
    question_id: int
    pitch: int
    key: str
    velocity: int
    duration_s: float
    first: RenditionRecord
    second: RenditionRecord
    composite_side: Side
    margin: float
    loudness_delta_lu: float
    heard_gain_db: float


class RankingSetDocument(Frozen):
    """What a listening set is, held apart from the audio a listener meets.

    Reading this decodes the set: every pair, both sides, and the ranking the composite gives them. The
    labels are collected beside it, so what a listener answers stays independent of what the metric
    already claims.
    """

    instrument_id: str
    sample_rate: int
    seed: int
    priced_encodings: int
    pairs: list[ListeningPairRecord]


def _rendition_record(rendition: Rendition, side: Side) -> RenditionRecord:
    """One member of a pair as the manifest states it."""
    return RenditionRecord(
        side=side,
        target_rate=rendition.params.target_rate,
        depth=rendition.params.depth,
        compress=rendition.params.compress,
        loop_index=rendition.params.loop_index,
        stored_bytes=rendition.stored_bytes,
        distortion=rendition.distortion,
        loudness_lufs=rendition.loudness_lufs,
    )


def pair_directory(pair: ListeningPair, index: int) -> str:
    """Where one pair's audio sits: its place in the set, then the recording both sides were encoded from.

    The question the pair asks is left to the manifest, so a listener browsing the folders meets each
    pair knowing which recording it came from and nothing about which degradation it holds.
    """
    return f"{index:03d}_{pair.clip.key.label}"


@dataclass(frozen=True)
class WrittenPair:
    """One question as it landed on disk: the pair it asks, and the lift its three recordings were written with."""

    pair: ListeningPair
    heard_gain_db: float


def pair_record(written: WrittenPair, index: int) -> ListeningPairRecord:
    """One pair as the manifest states it: both sides decoded, the composite's own call, and the lift it carries."""
    pair = written.pair
    return ListeningPairRecord(
        directory=pair_directory(pair, index),
        axis=pair.axis,
        question_id=pair.question_id,
        pitch=pair.clip.pitch,
        key=pair.clip.key.label,
        velocity=pair.clip.velocity,
        duration_s=pair.clip.event.duration_s,
        first=_rendition_record(pair.first, Side.A),
        second=_rendition_record(pair.second, Side.B),
        composite_side=pair.composite_side,
        margin=pair.margin,
        loudness_delta_lu=pair.loudness_delta_lu,
        heard_gain_db=written.heard_gain_db,
    )


def ranking_document(
    written: Sequence[WrittenPair],
    *,
    instrument_id: str,
    sample_rate: int,
    seed: int,
    priced_encodings: int,
) -> RankingSetDocument:
    """The manifest written beside a listening set's audio, in the order the questions were put on disk."""
    return RankingSetDocument(
        instrument_id=instrument_id,
        sample_rate=sample_rate,
        seed=seed,
        priced_encodings=priced_encodings,
        pairs=[pair_record(one, index) for index, one in enumerate(written)],
    )


class AgreementRecord(Frozen):
    """How far one metric stood from the listener over some set of questions.

    ``tau`` is Kendall's tau-b over the signed verdicts and the signed margins, so it reads direction and
    size at once, and is absent where fewer than two questions were answered or every reading is level.
    ``confirmed`` is ``matched / decided``, the share of the listener's calls the metric makes the same
    way, which is read against a chance level of 0.5.
    """

    tau: float | None
    decided: int
    matched: int
    confirmed: float | None


class MetricAgreementRecord(Frozen):
    """One metric ranked against a whole answer sheet, and against each question it holds.

    The three margins are the median distance the metric puts between two sides where the listener placed
    one ahead, where they placed the two level, and where they heard nothing at all between them.
    ``separation`` is the first over the second and ``headroom`` the first over the third, which prices
    the metric against its own floor. A metric worth its ranking reads well above 1.0 on both.
    """

    name: str
    overall: AgreementRecord
    by_axis: dict[PairAxis, AgreementRecord]
    decided_margin: float | None
    tied_margin: float | None
    identical_margin: float | None
    separation: float | None
    headroom: float | None


class SelfAgreementRecord(Frozen):
    """The ceiling every metric is read against: how far the listener stood from themselves.

    ``repeated`` counts the questions the set put more than once and the listener answered more than
    once. ``confirmed`` is how often the two occurrences named the same encoding, which is the most any
    metric could match.
    """

    repeated: int
    tau: float | None
    decided: int
    matched: int
    confirmed: float | None


class LevelConfoundRecord(Frozen):
    """How far the labels track the level the two sides played at rather than the recording.

    ``followed`` is the share of the calls whose sides stood an audible gap apart that named the louder
    one: near 0.5 the preferences are about the recording, near 1.0 they are about the volume.
    """

    tau: float | None
    gapped: int
    louder: int
    followed: float | None


class RankingReportDocument(Frozen):
    """What one instrument's answer sheet makes of every metric available to read it.

    ``metrics`` holds the composite as the objective reads it, the composite as the listener heard it,
    each of its components alone, and the level diagnostics -- so the table states both how the metric in
    use fares and which term available today fares better. ``ceiling`` and ``level`` are what any row of
    it is read against: the agreement the listener reaches with themselves, and how far the answers
    follow the volume.
    """

    instrument_id: str
    answered: int
    outstanding: int
    metrics: list[MetricAgreementRecord]
    ceiling: SelfAgreementRecord
    level: LevelConfoundRecord


def _agreement_record(agreement: Agreement) -> AgreementRecord:
    """One metric's standing over one set of questions as the report states it."""
    return AgreementRecord(
        tau=agreement.tau,
        decided=agreement.decided,
        matched=agreement.matched,
        confirmed=agreement.confirmed,
    )


def _metric_record(metric: MetricAgreement) -> MetricAgreementRecord:
    """One metric's whole standing as the report states it, question by question."""
    return MetricAgreementRecord(
        name=metric.name,
        overall=_agreement_record(metric.overall),
        by_axis={axis: _agreement_record(agreement) for axis, agreement in metric.by_axis.items()},
        decided_margin=metric.decided_margin,
        tied_margin=metric.tied_margin,
        identical_margin=metric.identical_margin,
        separation=metric.separation,
        headroom=metric.headroom,
    )


def report_document(report: RankingReport) -> RankingReportDocument:
    """The report written beside the answer sheet it was read from."""
    return RankingReportDocument(
        instrument_id=report.instrument_id,
        answered=report.answered,
        outstanding=report.outstanding,
        metrics=[_metric_record(metric) for metric in report.metrics],
        ceiling=SelfAgreementRecord(
            repeated=report.ceiling.repeated,
            tau=report.ceiling.tau,
            decided=report.ceiling.decided,
            matched=report.ceiling.matched,
            confirmed=report.ceiling.confirmed,
        ),
        level=LevelConfoundRecord(
            tau=report.level.tau,
            gapped=report.level.gapped,
            louder=report.level.louder,
            followed=report.level.followed,
        ),
    )
