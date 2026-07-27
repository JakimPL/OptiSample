from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from os import getpid, process_cpu_count

from optisample.parallel import ALL_CORES, IN_PROCESS, map_workers, worker_count
from optisample.progress import NO_PROGRESS

_ITEMS = (1, 2, 3, 4)
_LABEL = "Stage"
_SHARED = 2  # enough workers to prove the work left this process, few enough to leave the machine usable
_NO_ITEMS = 0


def _doubled(value: int) -> tuple[int, int]:
    """Twice ``value``, beside the process that computed it, so a test reads where the work ran.

    Written at module level because a worker receives its function by importing the module holding it.
    """
    return getpid(), value * 2


@dataclass
class _RecordingSink:
    """A sink keeping what each stage reported, so a test reads the bar a caller would have watched."""

    labels: list[str] = field(default_factory=list)
    totals: list[int] = field(default_factory=list)
    yielded: int = 0

    def track[ItemT](self, items: Iterable[ItemT], *, label: str, total: int) -> Iterator[ItemT]:
        self.labels.append(label)
        self.totals.append(total)
        for item in items:
            self.yielded += 1
            yield item


def _cores() -> int:
    return process_cpu_count() or IN_PROCESS


# --- how many processes a stage asks for ------------------------------------------------------------


def test_asking_for_every_core_starts_one_worker_per_core() -> None:
    assert worker_count(ALL_CORES, _cores() + 1) == _cores()


def test_a_stage_shorter_than_the_machine_starts_one_worker_per_item() -> None:
    """Startup is paid per worker, so a five-pitch stage pays for five however wide the machine is."""
    assert worker_count(ALL_CORES, len(_ITEMS)) == min(_cores(), len(_ITEMS))


def test_an_explicit_count_is_taken_as_asked_when_the_work_fills_it() -> None:
    assert worker_count(_SHARED, len(_ITEMS)) == _SHARED


def test_a_stage_holding_nothing_stays_in_the_calling_process() -> None:
    assert worker_count(ALL_CORES, _NO_ITEMS) == IN_PROCESS


# --- what a mapped stage answers --------------------------------------------------------------------


def test_a_run_in_the_calling_process_answers_every_item_there() -> None:
    results = map_workers(_doubled, _ITEMS, workers=IN_PROCESS, label=_LABEL, progress=NO_PROGRESS)
    assert [value for _, value in results] == [2, 4, 6, 8]
    assert {pid for pid, _ in results} == {getpid()}


def test_a_shared_run_answers_the_same_items_from_other_processes() -> None:
    """The point of the fan-out: identical answers in the order given, computed somewhere else."""
    results = map_workers(_doubled, _ITEMS, workers=_SHARED, label=_LABEL, progress=NO_PROGRESS)
    assert [value for _, value in results] == [2, 4, 6, 8]
    assert getpid() not in {pid for pid, _ in results}


def test_a_stage_holding_no_items_answers_with_none() -> None:
    assert map_workers(_doubled, (), workers=ALL_CORES, label=_LABEL, progress=NO_PROGRESS) == []


# --- what a mapped stage reports --------------------------------------------------------------------


def test_a_stage_reports_its_label_and_the_items_it_holds() -> None:
    sink = _RecordingSink()
    map_workers(_doubled, _ITEMS, workers=IN_PROCESS, label=_LABEL, progress=sink)
    assert (sink.labels, sink.totals) == ([_LABEL], [len(_ITEMS)])
    assert sink.yielded == len(_ITEMS)


def test_a_shared_stage_reports_the_same_work_as_one_kept_in_process() -> None:
    """A bar reads the same whichever way the stage was scheduled, so an ETA means one thing."""
    sink = _RecordingSink()
    map_workers(_doubled, _ITEMS, workers=_SHARED, label=_LABEL, progress=sink)
    assert (sink.labels, sink.totals) == ([_LABEL], [len(_ITEMS)])
    assert sink.yielded == len(_ITEMS)
