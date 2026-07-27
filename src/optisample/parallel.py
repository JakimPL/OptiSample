from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_all_start_methods, get_context
from multiprocessing.context import BaseContext
from os import process_cpu_count
from typing import Final

from optisample.progress import ProgressSink

IN_PROCESS: Final = 1  # one worker is the calling process itself, which carries the work as it stands
ALL_CORES: Final = 0  # the configured count asking for a worker on every core the process may run on

_ONE_CORE: Final = 1  # what a platform leaving its core count unstated is taken to offer
_FROM_A_SERVER: Final = "forkserver"


def _pool_context() -> BaseContext:
    """How worker processes are started: from a single-threaded server where the platform offers one.

    A process that already holds threads can leave a forked child with a lock no thread of its own will
    release, so the workers are forked from a server started before any of that. Platforms without a
    forkserver start them the way that platform starts processes.
    """
    if _FROM_A_SERVER in get_all_start_methods():
        return get_context(_FROM_A_SERVER)

    return get_context()  # pragma: no cover - every platform the suite runs on offers a forkserver


def worker_count(workers: int, items: int) -> int:
    """How many processes share ``items`` pieces of work, given the ``workers`` a caller asked for.

    ``ALL_CORES`` asks for one worker per core this process may run on. The answer is capped at the
    number of items, so a short stage pays startup proportional to the work it has: five pitches start
    five workers on a machine that would otherwise start two dozen.
    """
    requested = (process_cpu_count() or _ONE_CORE) if workers == ALL_CORES else workers
    return max(IN_PROCESS, min(requested, items))


def _map_over_processes[ItemT, ResultT](
    function: Callable[[ItemT], ResultT],
    items: Sequence[ItemT],
    *,
    workers: int,
    label: str,
    progress: ProgressSink,
) -> list[ResultT]:
    """Run ``function`` over ``items`` in a pool of ``workers`` processes, back in the order given.

    Each result is reported as it lands and placed by the index it was submitted under, so the bar
    tracks work finishing while the caller still reads the answers in its own order.
    """
    with ProcessPoolExecutor(max_workers=workers, mp_context=_pool_context()) as pool:
        pending = {pool.submit(function, item): index for index, item in enumerate(items)}
        landed: dict[int, ResultT] = {}
        for future in progress.track(as_completed(pending), label=label, total=len(pending)):
            landed[pending[future]] = future.result()

    return [landed[index] for index in range(len(items))]


def map_workers[ItemT, ResultT](
    function: Callable[[ItemT], ResultT],
    items: Sequence[ItemT],
    *,
    workers: int,
    label: str,
    progress: ProgressSink,
) -> list[ResultT]:
    """Apply ``function`` to every item, sharing them across processes when more than one is asked for.

    Results come back in ``items`` order however the work ran, so a stage reads the same whichever way
    it was scheduled, and progress is reported under ``label`` as each result lands.

    A caller reaching for this states two things about its work: every item is computed from itself
    alone, in whatever order a pool happens to reach it, and both ``function`` and the items pickle --
    which is what a worker process is handed. A run-wide input belongs on ``function``, bound with
    :func:`functools.partial` so it travels once per call rather than once per item.
    """
    count = worker_count(workers, len(items))
    if count == IN_PROCESS:
        return list(progress.track((function(item) for item in items), label=label, total=len(items)))

    return _map_over_processes(function, items, workers=count, label=label, progress=progress)
