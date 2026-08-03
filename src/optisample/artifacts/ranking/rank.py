from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from optisample.artifacts.documents.ranking import (
    ListeningPairRecord,
    RenditionRecord,
    report_document,
)
from optisample.artifacts.paths import RankingPaths, listening_set_paths
from optisample.artifacts.ranking.sets import (
    PairClips,
    pair_clips,
    read_label_sheet,
    read_ranking_set,
)
from optisample.artifacts.serialize import write_json
from optisample.calibrate.ranking import (
    Judgement,
    MetricReadings,
    RankingReport,
    Verdict,
    heard_strength,
    ranking_report,
)
from optisample.io.audio import read_wav
from optisample.metrics.composite import CompositeFidelity, QualityReport, evaluate
from optisample.progress import ProgressSink

_OBJECTIVE: Final = "objective"  # the composite as the sweep read it, against the reference each side closes
_COMPOSITE: Final = "composite"  # the composite as the listener heard it, against the recording itself
_QUALITIES: Final = ("segmental_snr_db", "snr_db", "si_sdr_db")  # read high-is-closer, so flipped to distances
_UNLOOPED_ORDER: Final = -1  # where an encoding storing the played span sorts among the loops
_SCORE_LABEL: Final = "Scoring answered pairs"


def _encoding_order(record: RenditionRecord) -> tuple[int, int, int, int]:
    """One encoding as a sortable key, which is what gives a question an order its blinding cannot move.

    A repeated question is blinded afresh, so the sides carry nothing the two occurrences share and only
    the encodings do. Ordering them settles which of the two a verdict is read as naming.
    """
    return (
        record.target_rate,
        record.depth_bits,
        int(record.compress),
        _UNLOOPED_ORDER if record.loop_index is None else record.loop_index,
    )


def _judgement(record: ListeningPairRecord, verdict: Verdict) -> Judgement:
    """One answered question as the report reads it: the verdict signed, and the blinding it was given under."""
    return Judgement(
        directory=record.directory,
        axis=record.axis,
        question_id=record.question_id,
        heard=heard_strength(verdict),
        swapped=_encoding_order(record.first) > _encoding_order(record.second),
        loudness_delta_lu=record.loudness_delta_lu,
    )


def _distances(report: QualityReport) -> dict[str, float]:
    """Every distance one side earns against the recording, the qualities flipped so all of them read low-is-closer.

    Holding one polarity across the table is what lets the composite, its components and the SNR
    diagnostics be ranked against each other by the same comparison.
    """
    return {
        _COMPOSITE: report.fidelity,
        **report.breakdown,
        **{name: -report.diagnostics[name] for name in _QUALITIES},
    }


def _read_apart(clips: PairClips, composite: CompositeFidelity) -> dict[str, float]:
    """How far ahead of side B every metric puts side A, read off the three recordings a listener met.

    The sides are scored against the recording as it was played, which is the comparison the listener
    made. The sweep's own reading closes that recording the way each side closes it, and stands in the
    table beside this one so the two can be told apart.
    """
    recording, sample_rate = read_wav(clips.reference)
    first = _distances(evaluate(recording, read_wav(clips.first)[0], sample_rate, composite))
    second = _distances(evaluate(recording, read_wav(clips.second)[0], sample_rate, composite))
    return {name: second[name] - distance for name, distance in first.items()}


def _objective_apart(record: ListeningPairRecord) -> float:
    """How far ahead of side B the sweep put side A, which is the ranking the set was chosen by."""
    return record.second.distortion - record.first.distortion


def _metric_readings(apart: Mapping[str, Mapping[str, float]]) -> tuple[MetricReadings, ...]:
    """The per-question readings regrouped under the metric that made them, which is what a report ranks."""
    named = dict.fromkeys(name for readings in apart.values() for name in readings)
    return tuple(
        MetricReadings(name=name, read={directory: readings[name] for directory, readings in apart.items()})
        for name in named
    )


def rank_metrics(paths: RankingPaths, composite: CompositeFidelity, progress: ProgressSink) -> RankingReport:
    """Rank every metric available today against one written listening set's answers.

    Only the answered questions are scored, so a sheet filled part way through a session reports on what
    it holds at the cost of what it holds. The audio is read back from the set rather than rebuilt, which
    is what keeps the ranking about the very recordings the verdicts were given on.
    """
    document = read_ranking_set(paths)
    sheet = read_label_sheet(paths)
    verdicts = {label.directory: label.verdict for label in sheet.answered if label.verdict is not None}
    answered = [record for record in document.pairs if record.directory in verdicts]
    apart = {
        record.directory: {
            _OBJECTIVE: _objective_apart(record),
            **_read_apart(pair_clips(paths, record.directory), composite),
        }
        for record in progress.track(answered, label=_SCORE_LABEL, total=len(answered))
    }
    return ranking_report(
        [_judgement(record, verdicts[record.directory]) for record in answered],
        _metric_readings(apart),
        instrument_id=document.instrument_id,
        outstanding=sheet.outstanding,
    )


def write_ranking_report(report: RankingReport, paths: RankingPaths) -> None:
    """Put ``report`` beside the answer sheet it was read from."""
    write_json(paths.report_json, report_document(report))


def rank_listening_set(directory: Path, composite: CompositeFidelity, progress: ProgressSink) -> RankingReport:
    """Rank the metrics against the listening set written in ``directory``, and write the report beside it."""
    paths = listening_set_paths(directory)
    report = rank_metrics(paths, composite, progress)
    write_ranking_report(report, paths)
    return report
