from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from optisample.calibrate.ranking import (
    Judgement,
    MetricReadings,
    PairAxis,
    Side,
    Verdict,
    heard_strength,
    level_confound,
    metric_agreement,
    ranking_report,
    self_agreement,
)

_NO_GAP = 0.0
_LOUD_GAP = 6.0
_QUIET_GAP = 0.2
_PERFECT = 1.0
_INVERTED = -1.0


def _judgement(
    directory: str,
    heard: int,
    *,
    axis: PairAxis = PairAxis.DEPTH,
    question_id: int = 0,
    identical: bool = False,
    swapped: bool = False,
    loudness_delta_lu: float = _NO_GAP,
) -> Judgement:
    """One answered question, named so a reading can be attached to it."""
    return Judgement(
        directory=directory,
        axis=axis,
        question_id=question_id,
        heard=heard,
        identical=identical,
        swapped=swapped,
        loudness_delta_lu=loudness_delta_lu,
    )


def _readings(*read: tuple[str, float]) -> MetricReadings:
    """One metric's reading of every question it was given."""
    return MetricReadings(name="composite", read=dict(read))


@dataclass(frozen=True)
class _Scale:
    """One point of the verdict scale beside the signed strength it is read as."""

    verdict: Verdict
    heard: int
    closer: Side | None


_SCALE = (
    _Scale(Verdict.A_CLEARLY, 2, Side.A),
    _Scale(Verdict.A_SLIGHTLY, 1, Side.A),
    _Scale(Verdict.TIE, 0, None),
    _Scale(Verdict.IDENTICAL, 0, None),
    _Scale(Verdict.B_SLIGHTLY, -1, Side.B),
    _Scale(Verdict.B_CLEARLY, -2, Side.B),
)


@pytest.mark.parametrize("point", _SCALE, ids=lambda point: point.verdict.value)
def test_the_scale_is_read_as_a_strength_signed_towards_the_side_it_names(point: _Scale) -> None:
    assert heard_strength(point.verdict) == point.heard
    assert point.verdict.closer is point.closer


def test_a_metric_calling_every_pair_the_listeners_way_confirms_all_of_them() -> None:
    judgements = [_judgement("a", 2), _judgement("b", -1), _judgement("c", 1)]

    read = metric_agreement(_readings(("a", 0.8), ("b", -0.2), ("c", 0.3)), judgements)

    assert read.overall.tau == pytest.approx(_PERFECT)
    assert read.overall.confirmed == pytest.approx(_PERFECT)


def test_a_metric_calling_every_pair_the_other_way_confirms_none_of_them() -> None:
    judgements = [_judgement("a", 2), _judgement("b", -1), _judgement("c", 1)]

    read = metric_agreement(_readings(("a", -0.8), ("b", 0.2), ("c", -0.3)), judgements)

    assert read.overall.tau == pytest.approx(_INVERTED)
    assert read.overall.matched == 0
    assert read.overall.decided == len(judgements)


def test_a_tie_is_ranked_without_being_counted_among_the_calls_a_listener_made() -> None:
    judgements = [_judgement("a", 2), _judgement("b", 0), _judgement("c", -2)]

    read = metric_agreement(_readings(("a", 0.9), ("b", 0.1), ("c", -0.9)), judgements)

    assert read.overall.decided == len(judgements) - 1
    assert read.overall.matched == read.overall.decided


def test_a_metric_is_read_against_each_question_apart_from_the_whole_sheet() -> None:
    judgements = [
        _judgement("a", 2, axis=PairAxis.DEPTH),
        _judgement("b", 2, axis=PairAxis.DEPTH),
        _judgement("c", 2, axis=PairAxis.LOOP),
    ]

    read = metric_agreement(_readings(("a", 0.5), ("b", 0.5), ("c", -0.5)), judgements)

    assert read.by_axis[PairAxis.DEPTH].matched == 2
    assert read.by_axis[PairAxis.LOOP].matched == 0


def test_a_question_the_sheet_never_asked_is_left_out_of_the_reading() -> None:
    read = metric_agreement(_readings(("a", 0.5)), [_judgement("a", 2, axis=PairAxis.RATE)])

    assert set(read.by_axis) == {PairAxis.RATE}


def test_the_margins_separate_what_a_listener_called_from_what_they_heard_as_level() -> None:
    judgements = [_judgement("a", 2), _judgement("b", -2), _judgement("c", 0), _judgement("d", 0)]

    read = metric_agreement(_readings(("a", 0.8), ("b", -1.2), ("c", 0.1), ("d", 0.3)), judgements)

    assert read.decided_margin == pytest.approx(1.0)
    assert read.tied_margin == pytest.approx(0.2)
    assert read.separation == pytest.approx(5.0)


def test_a_sheet_holding_calls_alone_leaves_the_separation_unstated() -> None:
    read = metric_agreement(_readings(("a", 0.8), ("b", -1.2)), [_judgement("a", 2), _judgement("b", -2)])

    assert read.tied_margin is None
    assert read.separation is None


def test_the_margins_read_a_pair_heard_as_one_recording_apart_from_the_rest() -> None:
    judgements = [_judgement("a", 2), _judgement("b", 0), _judgement("c", 0, identical=True)]

    read = metric_agreement(_readings(("a", 1.0), ("b", 0.4), ("c", 0.1)), judgements)

    assert read.tied_margin == pytest.approx(0.25)
    assert read.identical_margin == pytest.approx(0.1)
    assert read.headroom == pytest.approx(10.0)


def test_a_sheet_where_every_pair_was_told_apart_leaves_the_headroom_unstated() -> None:
    read = metric_agreement(_readings(("a", 0.8), ("b", 0.2)), [_judgement("a", 2), _judgement("b", 0)])

    assert read.identical_margin is None
    assert read.headroom is None


def test_a_metric_reading_nothing_between_an_identical_pair_states_no_headroom() -> None:
    judgements = [_judgement("a", 2), _judgement("b", 0, identical=True)]

    read = metric_agreement(_readings(("a", 0.8), ("b", 0.0)), judgements)

    assert read.identical_margin == pytest.approx(0.0)
    assert read.headroom is None


def test_a_metric_reading_every_pair_level_states_no_separation() -> None:
    judgements = [_judgement("a", 2), _judgement("b", 0)]

    read = metric_agreement(_readings(("a", 0.0), ("b", 0.0)), judgements)

    assert read.separation is None
    assert read.overall.matched == 0


def test_one_answered_question_is_too_few_to_correlate() -> None:
    read = metric_agreement(_readings(("a", 0.5)), [_judgement("a", 2)])

    assert read.overall.tau is None
    assert read.overall.confirmed == pytest.approx(_PERFECT)


def test_an_unanswered_sheet_ranks_nothing() -> None:
    read = metric_agreement(_readings(), [])

    assert read.overall.tau is None
    assert read.overall.confirmed is None
    assert read.decided_margin is None


def test_a_listener_answering_a_repeated_question_alike_reaches_the_ceiling() -> None:
    judgements = [
        _judgement("a", 2, question_id=7),
        _judgement("b", 2, question_id=7),
        _judgement("c", -1, question_id=9),
        _judgement("d", -1, question_id=9),
    ]

    ceiling = self_agreement(judgements)

    assert ceiling.repeated == 2
    assert ceiling.confirmed == pytest.approx(_PERFECT)
    assert ceiling.tau == pytest.approx(_PERFECT)


def test_a_repeated_question_is_read_through_the_blinding_each_occurrence_was_given() -> None:
    judgements = [
        _judgement("a", 2, question_id=7, swapped=False),
        _judgement("b", -2, question_id=7, swapped=True),
    ]

    ceiling = self_agreement(judgements)

    assert ceiling.matched == 1
    assert ceiling.decided == 1


def test_a_listener_reversing_themselves_confirms_none_of_the_repeats() -> None:
    judgements = [_judgement("a", 2, question_id=7), _judgement("b", -2, question_id=7)]

    ceiling = self_agreement(judgements)

    assert ceiling.matched == 0
    assert ceiling.repeated == 1


def test_a_repeat_the_listener_tied_once_is_left_out_of_the_calls_it_is_read_over() -> None:
    judgements = [_judgement("a", 2, question_id=7), _judgement("b", 0, question_id=7)]

    ceiling = self_agreement(judgements)

    assert ceiling.repeated == 1
    assert ceiling.decided == 0
    assert ceiling.confirmed is None


def test_a_set_putting_every_question_once_states_no_ceiling() -> None:
    ceiling = self_agreement([_judgement("a", 2, question_id=1), _judgement("b", 2, question_id=2)])

    assert ceiling.repeated == 0
    assert ceiling.tau is None


def test_a_listener_reaching_for_the_louder_side_reads_as_a_level_confound() -> None:
    judgements = [
        _judgement("a", 2, loudness_delta_lu=_LOUD_GAP),
        _judgement("b", -2, loudness_delta_lu=-_LOUD_GAP),
    ]

    level = level_confound(judgements)

    assert level.gapped == len(judgements)
    assert level.followed == pytest.approx(_PERFECT)
    assert level.tau == pytest.approx(_PERFECT)


def test_a_gap_too_small_to_hear_is_left_out_of_the_confound() -> None:
    judgements = [_judgement("a", 2, loudness_delta_lu=_QUIET_GAP), _judgement("b", -2, loudness_delta_lu=_NO_GAP)]

    level = level_confound(judgements)

    assert level.gapped == 0
    assert level.followed is None


def test_a_report_ranks_every_metric_it_is_handed_over_the_same_answers() -> None:
    judgements: Sequence[Judgement] = [_judgement("a", 2), _judgement("b", -2)]
    readings = [
        MetricReadings(name="composite", read={"a": 0.5, "b": -0.5}),
        MetricReadings(name="mcd", read={"a": -0.5, "b": -0.5}),
    ]

    report = ranking_report(judgements, readings, instrument_id="piano", outstanding=3)

    assert [metric.name for metric in report.metrics] == ["composite", "mcd"]
    assert report.metrics[0].overall.matched == 2
    assert report.metrics[1].overall.matched == 1
    assert (report.instrument_id, report.answered, report.outstanding) == ("piano", 2, 3)
