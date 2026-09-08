from __future__ import annotations

from pathlib import Path

import pytest

from notebooks.utils import labeling
from notebooks.utils.labeling import LabelingSession
from optisample.artifacts.documents.ranking import (
    ListeningPairRecord,
    RankingSetDocument,
    RenditionRecord,
)
from optisample.artifacts.paths import ranking_paths
from optisample.artifacts.serialize import write_json, write_text
from optisample.calibrate.ranking import (
    Fault,
    PairAxis,
    Side,
    Verdict,
    blank_sheet,
    labels_text,
)

_INSTRUMENT = "piano"
_SAMPLE_RATE = 44_100
_SEED = 137
_PRICED = 6
_DIRECTORIES = ("000_C4_v100", "001_D4_v100", "002_E4_v100")
_STEMS = ("reference", f"{Side.A}", f"{Side.B}")


def _rendition(side: Side) -> RenditionRecord:
    return RenditionRecord(
        side=side,
        target_rate=16_000,
        depth_bits=16,
        compress=False,
        loop_index=None,
        stored_bytes=8_000,
        distortion=1.0,
        loudness_lufs=-23.0,
    )


def _record(directory: str, index: int) -> ListeningPairRecord:
    return ListeningPairRecord(
        directory=directory,
        axis=PairAxis.DEPTH,
        question_id=index,
        pitch=60 + index,
        key=directory.split("_")[1],
        velocity=100,
        duration_s=1.5,
        first=_rendition(Side.A),
        second=_rendition(Side.B),
        composite_side=Side.A,
        margin=0.5,
        loudness_delta_lu=0.0,
        heard_gain_db=12.0,
    )


@pytest.fixture
def written(tmp_path: Path) -> Path:
    """A listening set on disk: its manifest, its open answer sheet, and a silent clip per side."""
    paths = ranking_paths(tmp_path, _INSTRUMENT)
    paths.pairs_dir.mkdir(parents=True)
    for directory in _DIRECTORIES:
        held = paths.pairs_dir / directory
        held.mkdir()
        for stem in _STEMS:
            (held / f"{stem}.wav").write_bytes(b"")

    write_json(
        paths.manifest_json,
        RankingSetDocument(
            instrument_id=_INSTRUMENT,
            sample_rate=_SAMPLE_RATE,
            seed=_SEED,
            priced_encodings=_PRICED,
            pairs=[_record(directory, index) for index, directory in enumerate(_DIRECTORIES)],
        ),
    )
    write_text(paths.labels_csv, labels_text(blank_sheet(_DIRECTORIES)))
    return tmp_path


@pytest.fixture
def session(written: Path) -> LabelingSession:
    return labeling.open_session(written, _INSTRUMENT)


def test_a_session_meets_the_questions_in_the_order_the_manifest_states(session: LabelingSession) -> None:
    assert session.directories == _DIRECTORIES


def test_a_fresh_session_opens_on_the_first_question(session: LabelingSession) -> None:
    assert labeling.first_open(session) == 0


def test_a_session_hands_over_the_three_recordings_a_question_puts(session: LabelingSession) -> None:
    clips = session.clips(1)

    assert [path.parent.name for path in (clips.reference, clips.first, clips.second)] == [_DIRECTORIES[1]] * 3
    assert all(path.is_file() for path in (clips.reference, clips.first, clips.second))


def test_answering_a_question_writes_it_through_to_the_sheet(session: LabelingSession) -> None:
    answered = labeling.answer(session, 0, verdict=Verdict.B_CLEARLY, fault=Fault.HISS, note="grainy")

    reopened = labeling.open_session(session.paths.pairs_dir.parent, _INSTRUMENT)
    assert reopened.label(0).verdict is Verdict.B_CLEARLY
    assert reopened.label(0).fault is Fault.HISS
    assert answered.answered == 1


def test_answering_moves_the_session_on_to_the_next_open_question(session: LabelingSession) -> None:
    answered = labeling.answer(session, 0, verdict=Verdict.TIE, fault=None, note="")

    assert labeling.following(answered, 0) == 1


def test_a_session_with_every_question_settled_stays_where_it_is(session: LabelingSession) -> None:
    for place in range(session.total):
        session = labeling.answer(session, place, verdict=Verdict.A_SLIGHTLY, fault=None, note="")

    assert labeling.following(session, 1) == 1
    assert labeling.first_open(session) == 0


def test_a_session_reopened_part_way_through_lands_on_what_is_left(session: LabelingSession) -> None:
    answered = labeling.answer(session, 0, verdict=Verdict.A_CLEARLY, fault=None, note="")

    assert labeling.first_open(answered) == 1


def test_a_session_states_how_far_through_the_set_it_is(session: LabelingSession) -> None:
    answered = labeling.answer(session, 2, verdict=Verdict.TIE, fault=None, note="")

    assert labeling.session_summary(answered) == "1 of 3 answered, 2 to go"


@pytest.mark.parametrize("verdict", list(Verdict))
def test_every_verdict_is_offered_as_a_choice(verdict: Verdict) -> None:
    choices = labeling.verdict_choices()

    assert labeling.choice_label(choices, verdict) in choices
    assert choices[labeling.choice_label(choices, verdict)] is verdict


@pytest.mark.parametrize("fault", list(Fault))
def test_every_fault_is_offered_as_a_choice(fault: Fault) -> None:
    choices = labeling.fault_choices()

    assert choices[labeling.choice_label(choices, fault)] is fault


def test_an_unanswered_question_opens_its_controls_on_the_open_entry(session: LabelingSession) -> None:
    label = session.label(0)

    assert labeling.choice_label(labeling.verdict_choices(), label.verdict) == labeling.OPEN_CHOICE
    assert labeling.choice_label(labeling.fault_choices(), label.fault) == labeling.OPEN_CHOICE
