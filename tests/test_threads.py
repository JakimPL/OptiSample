from os import environ

import pytest

from optisample.threads import _THREAD_LIMITS, bind_compute_threads

_ONE_THREAD = "1"
_TWO_THREADS = "2"


@pytest.fixture
def unstated(monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment saying nothing about compute threads, restored to whatever this one states."""
    for name in _THREAD_LIMITS:
        monkeypatch.delenv(name, raising=False)


def test_a_run_computes_on_one_thread_per_process(unstated: None) -> None:
    """A worker sizing a thread pool of its own would multiply the cores a stage was asked for."""
    bind_compute_threads()
    assert [environ[name] for name in _THREAD_LIMITS] == [_ONE_THREAD] * len(_THREAD_LIMITS)


def test_a_thread_count_the_caller_states_stands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", _TWO_THREADS)
    bind_compute_threads()
    assert environ["OMP_NUM_THREADS"] == _TWO_THREADS
