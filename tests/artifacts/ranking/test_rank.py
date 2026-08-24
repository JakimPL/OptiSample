from __future__ import annotations

from collections.abc import Iterable

import pytest

from optisample.artifacts.documents.ranking import RankingReportDocument
from optisample.artifacts.paths import RankingPaths
from optisample.artifacts.ranking import (
    ListeningSet,
    rank_listening_set,
    read_label_sheet,
    read_ranking_set,
    write_label_sheet,
)
from optisample.artifacts.serialize import read_json
from optisample.calibrate.ranking import RankingReport, Side, Verdict, settled
from optisample.config import OptiConfig
from optisample.metrics.composite import build_composite
from optisample.progress import NO_PROGRESS

_AUDIBLE_GAP_LU = 1.0  # what the report counts as a level gap, restated so a test names its own threshold
_CALLED = {Side.A: Verdict.A_CLEARLY, Side.B: Verdict.B_CLEARLY}
_EXPECTED_METRICS = (
    "objective",
    "composite",
    "mrstft",
    "logmel_l1",
    "spectral_shape",
    "mcd",
    "onset",
    "segmental_snr_db",
    "snr_db",
    "si_sdr_db",
)


def _answer(paths: RankingPaths, verdicts: Iterable[tuple[str, Verdict]]) -> None:
    """Fill in the answer sheet beside a written set, as a listener working through it would."""
    sheet = read_label_sheet(paths)
    for directory, verdict in verdicts:
        sheet = settled(sheet, directory, verdict=verdict, fault=None, note="")

    write_label_sheet(sheet, paths)


def _agreeing_with_the_metric(written: ListeningSet) -> list[tuple[str, Verdict]]:
    """The answers a listener would give who heard exactly what the composite reads."""
    return [(record.directory, _CALLED[record.composite_side]) for record in read_ranking_set(written.paths).pairs]


def _ranked(written: ListeningSet, config: OptiConfig) -> RankingReport:
    """Rank the metrics against whatever the sheet beside ``written`` now holds."""
    return rank_listening_set(written.paths.pairs_dir, build_composite(config.analysis.metrics), NO_PROGRESS)


def test_a_report_ranks_the_composite_its_components_and_the_level_diagnostics(
    written: ListeningSet,
    config: OptiConfig,
) -> None:
    _answer(written.paths, _agreeing_with_the_metric(written))

    report = _ranked(written, config)

    assert tuple(metric.name for metric in report.metrics) == _EXPECTED_METRICS


def test_a_listener_agreeing_with_the_sweep_confirms_every_call_it_made(
    written: ListeningSet,
    config: OptiConfig,
) -> None:
    _answer(written.paths, _agreeing_with_the_metric(written))

    objective = _ranked(written, config).metrics[0]

    assert objective.name == "objective"
    assert objective.overall.matched == objective.overall.decided > 0


def test_only_the_questions_a_listener_settled_are_ranked(written: ListeningSet, config: OptiConfig) -> None:
    answered = _agreeing_with_the_metric(written)[:1]
    _answer(written.paths, answered)

    report = _ranked(written, config)

    assert report.answered == len(answered)
    assert report.outstanding == written.pairs - len(answered)


def test_an_untouched_sheet_ranks_nothing_and_says_how_much_listening_is_left(
    written: ListeningSet,
    config: OptiConfig,
) -> None:
    report = _ranked(written, config)

    assert (report.answered, report.metrics) == (0, ())
    assert report.outstanding == written.pairs


def test_a_question_put_twice_and_answered_alike_reaches_the_ceiling(
    written: ListeningSet,
    config: OptiConfig,
) -> None:
    _answer(written.paths, _agreeing_with_the_metric(written))

    ceiling = _ranked(written, config).ceiling

    assert ceiling.repeated == written.repeats > 0
    assert ceiling.matched == ceiling.decided


def test_the_level_gap_the_manifest_states_is_what_the_confound_is_read_over(
    written: ListeningSet,
    config: OptiConfig,
) -> None:
    _answer(written.paths, _agreeing_with_the_metric(written))

    report = _ranked(written, config)

    gapped = [
        record for record in read_ranking_set(written.paths).pairs if abs(record.loudness_delta_lu) >= _AUDIBLE_GAP_LU
    ]
    assert report.level.gapped == len(gapped)


def test_the_report_is_written_beside_the_answer_sheet_it_was_read_from(
    written: ListeningSet,
    config: OptiConfig,
) -> None:
    _answer(written.paths, _agreeing_with_the_metric(written))

    report = _ranked(written, config)

    stated = read_json(written.paths.report_json, RankingReportDocument)
    assert stated.instrument_id == report.instrument_id
    assert [metric.name for metric in stated.metrics] == [metric.name for metric in report.metrics]


def test_a_metric_reading_the_two_sides_apart_is_stated_as_a_ratio_over_the_ones_it_ties(
    written: ListeningSet,
    config: OptiConfig,
) -> None:
    pairs = read_ranking_set(written.paths).pairs
    _answer(written.paths, [(record.directory, Verdict.TIE) for record in pairs[:1]])
    _answer(written.paths, [(record.directory, _CALLED[record.composite_side]) for record in pairs[1:]])

    composite = next(metric for metric in _ranked(written, config).metrics if metric.name == "composite")

    assert composite.decided_margin is not None and composite.tied_margin is not None
    assert composite.separation == pytest.approx(composite.decided_margin / composite.tied_margin)
