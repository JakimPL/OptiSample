from __future__ import annotations

from dataclasses import dataclass

import pytest

from optisample.calibrate.ranking import (
    LABEL_COLUMNS,
    Fault,
    LabelSheet,
    PairLabel,
    Side,
    Verdict,
    blank_sheet,
    labels_text,
    next_open,
    read_labels,
    settled,
)

_DIRECTORIES = ("000_C4_v100", "001_D4_v100", "002_E4_v100")
_HEADER = ",".join(LABEL_COLUMNS)


@dataclass(frozen=True)
class _Reading:
    """One verdict as a listener writes it, beside what it says about the pair."""

    token: str
    closer: Side | None
    strength: int


_READINGS = (
    _Reading("aa", Side.A, 2),
    _Reading("a", Side.A, 1),
    _Reading("tie", None, 0),
    _Reading("same", None, 0),
    _Reading("b", Side.B, 1),
    _Reading("bb", Side.B, 2),
)


@pytest.fixture
def filled() -> LabelSheet:
    """A sheet with one settled row, one settled row carrying a fault, and one still open."""
    return LabelSheet(
        (
            PairLabel(directory=_DIRECTORIES[0], verdict=Verdict.A_CLEARLY, fault=Fault.HISS, note="grainy tail"),
            PairLabel(directory=_DIRECTORIES[1], verdict=Verdict.TIE, fault=None, note=""),
            PairLabel(directory=_DIRECTORIES[2], verdict=None, fault=None, note=""),
        )
    )


@pytest.mark.parametrize("reading", _READINGS, ids=lambda reading: reading.token)
def test_a_verdict_names_the_side_a_listener_placed_closer(reading: _Reading) -> None:
    assert Verdict(reading.token).closer is reading.closer


@pytest.mark.parametrize("reading", _READINGS, ids=lambda reading: reading.token)
def test_a_verdict_states_how_far_apart_the_two_stood(reading: _Reading) -> None:
    assert Verdict(reading.token).strength == reading.strength


def test_an_open_sheet_holds_one_row_per_question() -> None:
    sheet = blank_sheet(_DIRECTORIES)

    assert [label.directory for label in sheet.labels] == list(_DIRECTORIES)
    assert sheet.outstanding == len(_DIRECTORIES)


def test_a_sheet_reads_back_as_it_was_written(filled: LabelSheet) -> None:
    assert read_labels(labels_text(filled)) == filled


def test_an_open_sheet_reads_back_as_it_was_written() -> None:
    sheet = blank_sheet(_DIRECTORIES)

    assert read_labels(labels_text(sheet)) == sheet


def test_a_sheet_states_which_questions_are_settled(filled: LabelSheet) -> None:
    assert [label.directory for label in filled.answered] == list(_DIRECTORIES[:2])
    assert filled.outstanding == 1


def test_a_verdict_is_read_through_the_case_and_space_around_it() -> None:
    sheet = read_labels(f"{_HEADER}\n{_DIRECTORIES[0]}, BB , Hiss ,\n")

    assert sheet.labels[0].verdict is Verdict.B_CLEARLY
    assert sheet.labels[0].fault is Fault.HISS


def test_a_verdict_outside_the_scale_stops_the_reading() -> None:
    with pytest.raises(ValueError, match="stands outside the scale"):
        read_labels(f"{_HEADER}\n{_DIRECTORIES[0]},maybe,,\n")


def test_a_fault_outside_the_vocabulary_stops_the_reading() -> None:
    with pytest.raises(ValueError, match="stands outside the vocabulary"):
        read_labels(f"{_HEADER}\n{_DIRECTORIES[0]},aa,warbly,\n")


def test_a_sheet_opening_with_other_columns_stops_the_reading() -> None:
    with pytest.raises(ValueError, match="must open with the columns"):
        read_labels("directory,closer,note\n")


def test_an_empty_sheet_stops_the_reading() -> None:
    with pytest.raises(ValueError, match="must open with the columns"):
        read_labels("")


def test_a_row_holding_the_wrong_field_count_stops_the_reading() -> None:
    with pytest.raises(ValueError, match="row 2"):
        read_labels(f"{_HEADER}\n{_DIRECTORIES[0]},aa\n")


def test_answering_a_question_leaves_the_rest_of_the_sheet_as_it_was(filled: LabelSheet) -> None:
    answered = settled(filled, _DIRECTORIES[2], verdict=Verdict.B_SLIGHTLY, fault=Fault.DULL, note="thin")

    assert answered.labels[2].verdict is Verdict.B_SLIGHTLY
    assert answered.labels[:2] == filled.labels[:2]


def test_answering_a_question_the_sheet_lacks_stops_the_writing(filled: LabelSheet) -> None:
    with pytest.raises(KeyError, match="holds no question"):
        settled(filled, "099_G9_v001", verdict=Verdict.TIE, fault=None, note="")


def test_the_next_open_question_is_the_one_after_the_place_reached(filled: LabelSheet) -> None:
    assert next_open(filled, 0) == 2


def test_the_search_for_an_open_question_carries_round_the_end(filled: LabelSheet) -> None:
    assert next_open(filled, 2) == 2


def test_a_finished_sheet_leaves_no_question_open() -> None:
    finished = LabelSheet(
        tuple(PairLabel(directory=directory, verdict=Verdict.TIE, fault=None, note="") for directory in _DIRECTORIES)
    )

    assert next_open(finished, 0) is None


def test_a_note_survives_the_commas_a_listener_writes_in_it() -> None:
    written = labels_text(
        LabelSheet(
            (PairLabel(directory=_DIRECTORIES[0], verdict=Verdict.B_SLIGHTLY, fault=None, note="dull, and thin"),)
        )
    )

    assert read_labels(written).labels[0].note == "dull, and thin"
