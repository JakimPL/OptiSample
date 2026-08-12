from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.artifacts.paths import RankingPaths, ranking_paths
from optisample.artifacts.ranking import (
    PairClips,
    pair_clips,
    read_label_sheet,
    read_ranking_set,
    write_label_sheet,
)
from optisample.calibrate.ranking import (
    Fault,
    LabelSheet,
    PairLabel,
    Verdict,
    next_open,
    settled,
)

OPEN_CHOICE: Final = "(open)"  # what a control shows while the listener has settled nothing

_FIRST: Final = 0


@dataclass(frozen=True)
class LabellingSession:
    """One written listening set opened for answering: the questions in order, and the sheet so far.

    The manifest is read for the order alone, so what a panel can put on screen is the audio and the
    listener's own answers -- the ranking the composite gives sits in ``pairs.json`` and stays there until
    the sheet is finished.
    """

    paths: RankingPaths
    directories: tuple[str, ...]
    sheet: LabelSheet

    @property
    def answered(self) -> int:
        """How many questions the listener has settled."""
        return len(self.sheet.labels) - self.sheet.outstanding

    @property
    def total(self) -> int:
        """How many questions the set puts."""
        return len(self.directories)

    def label(self, place: int) -> PairLabel:
        """The row standing at ``place`` in the order the questions are met."""
        return self.sheet.labels[place]

    def clips(self, place: int) -> PairClips:
        """The three recordings the question at ``place`` puts in front of a listener."""
        return pair_clips(self.paths, self.directories[place])


def open_session(root: Path, instrument_id: str) -> LabellingSession:
    """The listening set written under ``root`` for ``instrument_id``, with its answer sheet as it stands."""
    paths = ranking_paths(root, instrument_id)
    document = read_ranking_set(paths)
    return LabellingSession(
        paths=paths,
        directories=tuple(record.directory for record in document.pairs),
        sheet=read_label_sheet(paths),
    )


def answer(
    session: LabellingSession,
    place: int,
    *,
    verdict: Verdict | None,
    fault: Fault | None,
    note: str,
) -> LabellingSession:
    """``session`` with the question at ``place`` settled and the sheet put back on disk.

    Writing on every answer is what makes the session survive a closed browser: the sheet on disk is
    always the whole of what has been heard.
    """
    updated = settled(session.sheet, session.directories[place], verdict=verdict, fault=fault, note=note)
    write_label_sheet(updated, session.paths)
    return LabellingSession(paths=session.paths, directories=session.directories, sheet=updated)


def following(session: LabellingSession, place: int) -> int:
    """Where to go after answering ``place``: the next question still open, or ``place`` once none are."""
    waiting = next_open(session.sheet, place)
    return place if waiting is None else waiting


def verdict_choices() -> dict[str, Verdict | None]:
    """The scale as a control offers it, from A being clearly closer through the level pair to B being so."""
    return {
        "A clearly closer": Verdict.A_CLEARLY,
        "A slightly closer": Verdict.A_SLIGHTLY,
        "neither is closer": Verdict.TIE,
        "no difference to hear": Verdict.IDENTICAL,
        "B slightly closer": Verdict.B_SLIGHTLY,
        "B clearly closer": Verdict.B_CLEARLY,
        OPEN_CHOICE: None,
    }


def fault_choices() -> dict[str, Fault | None]:
    """The fault vocabulary as a control offers it, with an open entry for a question that names none."""
    return {OPEN_CHOICE: None} | {fault.value: fault for fault in Fault}


def choice_label(choices: dict[str, Verdict | None] | dict[str, Fault | None], held: object) -> str:
    """Which entry of ``choices`` stands for ``held``, so a control opens on what the sheet already says."""
    for shown, value in choices.items():
        if value == held:
            return shown

    return OPEN_CHOICE


def session_summary(session: LabellingSession) -> str:
    """How far through the set the listener is, as a line a panel prints above the controls."""
    return f"{session.answered} of {session.total} answered, {session.sheet.outstanding} to go"


def first_open(session: LabellingSession) -> int:
    """Where a session opens: the first question still waiting, or the start once every one is settled."""
    if session.sheet.labels and not session.sheet.labels[_FIRST].answered:
        return _FIRST

    waiting = next_open(session.sheet, _FIRST)
    return _FIRST if waiting is None else waiting
