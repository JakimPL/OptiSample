from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum, unique
from io import StringIO
from typing import Final

from optisample.calibrate.ranking.pairs import Side

LABEL_COLUMNS: Final = ("directory", "verdict", "fault", "note")
UNANSWERED: Final = ""  # what a row holds while the question is still open

_CLEARLY: Final = 2
_SLIGHTLY: Final = 1
_EQUALLY: Final = 0


@unique
class Verdict(StrEnum):
    """Which side of a blinded pair a listener placed closer to the recording, and how clearly.

    The grades earn their place twice over. They let a near-miss cost a metric less than a blown call,
    which is what makes agreement measure something a listener would recognise; and they let the size of
    a perceived gap be compared with the size of the margin the composite reads, which is what separates
    two metrics that order every pair alike and disagree on how far apart the sides stand.

    Two of them stand level and say different things. :attr:`TIE` places both sides the same distance
    from the recording, each with its own fault; :attr:`IDENTICAL` says there was nothing between them
    to hear. The second holds a metric to a floor the first leaves open, since a pair a listener met as
    one recording is one the metric is due to read as one.
    """

    A_CLEARLY = "aa"
    A_SLIGHTLY = "a"
    TIE = "tie"
    IDENTICAL = "same"
    B_SLIGHTLY = "b"
    B_CLEARLY = "bb"

    @property
    def closer(self) -> Side | None:
        """The side placed closer to the recording, where the listener placed one."""
        match self:
            case Verdict.A_CLEARLY | Verdict.A_SLIGHTLY:
                return Side.A
            case Verdict.B_CLEARLY | Verdict.B_SLIGHTLY:
                return Side.B
            case Verdict.TIE | Verdict.IDENTICAL:
                return None

    @property
    def strength(self) -> int:
        """How far apart the two stood: clearly, slightly, or level with one another."""
        match self:
            case Verdict.A_CLEARLY | Verdict.B_CLEARLY:
                return _CLEARLY
            case Verdict.A_SLIGHTLY | Verdict.B_SLIGHTLY:
                return _SLIGHTLY
            case Verdict.TIE | Verdict.IDENTICAL:
                return _EQUALLY


@unique
class Fault(StrEnum):
    """What the side a listener rejected does wrong, named from the faults these encodings produce.

    Each token points at a term the composite either carries or lacks, so grouping the labels by fault
    says which change the evidence calls for: :attr:`HISS` is the quantization noise a level-invariant
    metric dilutes, :attr:`DULL` the band a spectral distance already reads, :attr:`FLUTTER` and
    :attr:`BEATING` the two forms a loop returns the material's own motion in, :attr:`CLICK` the wrap,
    and :attr:`STOP` a stored span ending before the note does.
    """

    HISS = "hiss"
    DULL = "dull"
    FLUTTER = "flutter"
    BEATING = "beating"
    CLICK = "click"
    STOP = "stop"
    OTHER = "other"


_VERDICT_TOKENS: Final = frozenset(verdict.value for verdict in Verdict)
_FAULT_TOKENS: Final = frozenset(fault.value for fault in Fault)


@dataclass(frozen=True)
class PairLabel:
    """One row of the answer sheet: what a listener made of one question.

    ``fault`` and ``note`` are the listener's own, left open on any row: a verdict alone is enough for
    the ranking, and what is written beside it is what turns a disagreement into a change worth making.
    """

    directory: str
    verdict: Verdict | None
    fault: Fault | None
    note: str

    @property
    def answered(self) -> bool:
        """Whether the listener has settled this question."""
        return self.verdict is not None


@dataclass(frozen=True)
class LabelSheet:
    """The answer sheet as it stands, whether part-filled or finished.

    Holding the open rows beside the settled ones is what lets a session be put down and picked up: the
    sheet reads back as it was written, and a listener meets the questions still waiting for them.
    """

    labels: tuple[PairLabel, ...]

    @property
    def answered(self) -> tuple[PairLabel, ...]:
        """The rows a listener has settled, in the order they were met."""
        return tuple(label for label in self.labels if label.answered)

    @property
    def outstanding(self) -> int:
        """How many questions are still open."""
        return len(self.labels) - len(self.answered)


def _verdict(token: str, row: int) -> Verdict | None:
    """The verdict ``token`` states, read leniently over case and surrounding space.

    Raises:
        ValueError: where the token is filled in and stands outside the scale.
    """
    stated = token.strip().lower()
    if stated == UNANSWERED:
        return None

    if stated not in _VERDICT_TOKENS:
        raise ValueError(f"row {row}: verdict {token!r} stands outside the scale {sorted(_VERDICT_TOKENS)}")

    return Verdict(stated)


def _fault(token: str, row: int) -> Fault | None:
    """The fault ``token`` names, read leniently over case and surrounding space.

    Raises:
        ValueError: where the token is filled in and names a fault outside the vocabulary.
    """
    stated = token.strip().lower()
    if stated == UNANSWERED:
        return None

    if stated not in _FAULT_TOKENS:
        raise ValueError(f"row {row}: fault {token!r} stands outside the vocabulary {sorted(_FAULT_TOKENS)}")

    return Fault(stated)


def _label(row: Sequence[str], number: int) -> PairLabel:
    """One sheet row as a label.

    Raises:
        ValueError: where the row holds a field count other than the sheet's columns.
    """
    if len(row) != len(LABEL_COLUMNS):
        raise ValueError(f"row {number}: holds {len(row)} fields where the sheet states {len(LABEL_COLUMNS)}")

    directory, verdict, fault, note = row
    return PairLabel(
        directory=directory.strip(),
        verdict=_verdict(verdict, number),
        fault=_fault(fault, number),
        note=note.strip(),
    )


def blank_sheet(directories: Iterable[str]) -> LabelSheet:
    """An open answer sheet, one row per question, in the order the questions are met."""
    return LabelSheet(
        tuple(PairLabel(directory=directory, verdict=None, fault=None, note="") for directory in directories)
    )


def read_labels(text: str) -> LabelSheet:
    """The answer sheet ``text`` holds, with every filled-in token checked against the vocabulary.

    Reading strictly is what keeps a hand-filled sheet worth the listening that went into it: a token
    outside the scale is a question whose answer is unknown, and it is worth more as a failure than as a
    silent tie.

    Raises:
        ValueError: where the sheet opens with columns other than :data:`LABEL_COLUMNS`.
    """
    rows = list(csv.reader(StringIO(text)))
    if not rows or tuple(rows[0]) != LABEL_COLUMNS:
        raise ValueError(f"the answer sheet must open with the columns {LABEL_COLUMNS}")

    return LabelSheet(tuple(_label(row, number) for number, row in enumerate(rows[1:], start=2)))


def settled(
    sheet: LabelSheet, directory: str, *, verdict: Verdict | None, fault: Fault | None, note: str
) -> LabelSheet:
    """``sheet`` with one question answered, the rest of it standing as it was.

    Raises:
        KeyError: where the sheet holds no row for ``directory``.
    """
    if directory not in {label.directory for label in sheet.labels}:
        raise KeyError(f"the sheet holds no question at {directory!r}")

    return LabelSheet(
        tuple(
            replace(label, verdict=verdict, fault=fault, note=note) if label.directory == directory else label
            for label in sheet.labels
        )
    )


def next_open(sheet: LabelSheet, after: int) -> int | None:
    """Where the next question waiting for an answer sits, looking on from ``after`` and around the end.

    Wrapping is what lets a session be worked through in one pass and then swept for whatever was left
    behind, which is how a listener returning to the hard questions reaches them.
    """
    total = len(sheet.labels)
    for step in range(1, total + 1):
        place = (after + step) % total
        if not sheet.labels[place].answered:
            return place

    return None


def labels_text(sheet: LabelSheet) -> str:
    """``sheet`` as the CSV a listener fills in, which reads back through :func:`read_labels` as it stands."""
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(LABEL_COLUMNS)
    for label in sheet.labels:
        writer.writerow(
            (
                label.directory,
                label.verdict.value if label.verdict is not None else UNANSWERED,
                label.fault.value if label.fault is not None else UNANSWERED,
                label.note,
            )
        )

    return buffer.getvalue()
