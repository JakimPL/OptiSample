from __future__ import annotations

import io

import pytest

from optisample.progress import (
    NO_PROGRESS,
    SilentProgress,
    TqdmProgress,
    bars_are_watchable,
    counting,
    progress_sink,
)

_ITEMS = ("a", "b", "c")
_LABEL = "Stage"
_BOUND = 4  # units a counted stage may spend, which is what its bar is sized by


class _FakeStream(io.StringIO):
    """A writable stream that answers the terminal probe however a test needs it to."""

    def __init__(self, *, terminal: bool) -> None:
        super().__init__()
        self.terminal = terminal

    def isatty(self) -> bool:
        return self.terminal


@pytest.fixture
def stream() -> _FakeStream:
    return _FakeStream(terminal=True)


# --- what a sink yields ---------------------------------------------------------------------------


def test_a_silent_sink_yields_every_item_untouched() -> None:
    assert list(SilentProgress().track(_ITEMS, label=_LABEL, total=len(_ITEMS))) == list(_ITEMS)


def test_a_drawn_sink_yields_every_item_untouched(stream: _FakeStream) -> None:
    """A bar reports the work; the items reaching the caller are the ones it was handed."""
    assert list(TqdmProgress(stream).track(_ITEMS, label=_LABEL, total=len(_ITEMS))) == list(_ITEMS)


def test_an_empty_stage_yields_nothing_and_still_completes(stream: _FakeStream) -> None:
    assert list(TqdmProgress(stream).track((), label=_LABEL, total=0)) == []


# --- what a sink reports --------------------------------------------------------------------------


def test_a_drawn_sink_writes_its_label_and_total_to_the_stream(stream: _FakeStream) -> None:
    list(TqdmProgress(stream).track(_ITEMS, label=_LABEL, total=len(_ITEMS)))
    written = stream.getvalue()
    assert _LABEL in written
    assert f"{len(_ITEMS)}" in written


def test_a_disabled_run_leaves_the_stream_clear(stream: _FakeStream) -> None:
    """``--no-progress`` and a redirected stderr both land here, so the stream must stay untouched."""
    list(progress_sink(stream, enabled=False).track(_ITEMS, label=_LABEL, total=len(_ITEMS)))
    assert stream.getvalue() == ""


def test_a_stage_reports_against_the_total_it_states_rather_than_the_items_it_walks(stream: _FakeStream) -> None:
    """A caller counting encodes may iterate something else, so the stated total is what the bar fills."""
    list(TqdmProgress(stream).track(_ITEMS, label=_LABEL, total=10))
    assert "/10" in stream.getvalue()


# --- counting a stage off -------------------------------------------------------------------------


def test_a_counted_stage_draws_against_the_bound_it_states(stream: _FakeStream) -> None:
    """A search sizes its bar by the most it may spend, so the bound is what the bar reads against."""
    with counting(TqdmProgress(stream), label=_LABEL, total=_BOUND) as step:
        step()

    assert _LABEL in stream.getvalue()
    assert f"/{_BOUND}" in stream.getvalue()


def test_a_counted_stage_reaches_its_bound_however_much_of_it_was_spent(stream: _FakeStream) -> None:
    """A stage ending under its bound ends its line, so the bar after it starts on a fresh one."""
    with counting(TqdmProgress(stream), label=_LABEL, total=_BOUND) as step:
        step()

    assert f"{_BOUND}/{_BOUND}" in stream.getvalue()


def test_a_counted_stage_spending_past_its_bound_keeps_drawing(stream: _FakeStream) -> None:
    """The bound sizes the bar rather than gating the work, so a stage running over still finishes."""
    with counting(TqdmProgress(stream), label=_LABEL, total=_BOUND) as step:
        for _ in range(_BOUND * 2):
            step()


def test_a_counted_stage_stays_silent_where_the_run_does() -> None:
    with counting(NO_PROGRESS, label=_LABEL, total=_BOUND) as step:
        step()


# --- choosing a sink ------------------------------------------------------------------------------


@pytest.mark.parametrize("terminal", [True, False])
def test_a_bar_is_watchable_only_at_a_terminal(terminal: bool) -> None:
    assert bars_are_watchable(_FakeStream(terminal=terminal)) is terminal


def test_an_enabled_run_draws_and_a_disabled_one_stays_silent(stream: _FakeStream) -> None:
    assert isinstance(progress_sink(stream, enabled=True), TqdmProgress)
    assert isinstance(progress_sink(stream, enabled=False), SilentProgress)


def test_the_shared_silent_sink_is_one_every_caller_may_hold() -> None:
    """It carries no state, so defaulting to a single instance keeps every quiet caller equivalent."""
    assert isinstance(NO_PROGRESS, SilentProgress)
    assert list(NO_PROGRESS.track(_ITEMS, label=_LABEL, total=len(_ITEMS))) == list(_ITEMS)
