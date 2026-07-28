from collections.abc import Iterable, Iterator
from typing import Final, Protocol, TextIO

from tqdm import tqdm

BAR_FORMAT: Final = "{desc:.<34} {percentage:3.0f}% {n_fmt:>6}/{total_fmt:<6} [{elapsed}<{remaining}]"


class ProgressSink(Protocol):
    """Where a long stage says how far through it is, so a run states its shape while it is still running.

    Stated as a protocol so the pipeline reports the same way whether a terminal is watching or a test is,
    and so every stage names its own total: the count is known before the loop starts in each case, which
    is what turns a bar into a time estimate rather than a spinner.
    """

    def track[ItemT](self, items: Iterable[ItemT], *, label: str, total: int) -> Iterator[ItemT]:
        """Yield every item of ``items``, reporting progress towards ``total`` under ``label``."""


class SilentProgress:
    """A sink that passes its items straight through, leaving the caller's output to the caller.

    This is what the library reports through by default and what the test suite sees, so importing
    OptiSample as a package keeps stdout and stderr clear.
    """

    def track[ItemT](self, items: Iterable[ItemT], *, label: str, total: int) -> Iterator[ItemT]:
        """Yield every item of ``items``, keeping the run silent."""
        # pylint: disable=unused-argument
        yield from items


class TqdmProgress:
    """A sink drawing one bar per stage on stderr, so stdout carries the run's results alone.

    ``total`` comes from the caller rather than from the iterable's length, which lets a stage report
    against the work it counted (encodings, notes) even while iterating something else.
    """

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def track[ItemT](self, items: Iterable[ItemT], *, label: str, total: int) -> Iterator[ItemT]:
        """Yield every item of ``items``, drawing its progress towards ``total`` as a labelled bar."""
        yield from tqdm(items, desc=label, total=total, file=self.stream, bar_format=BAR_FORMAT, leave=True)


NO_PROGRESS: Final = SilentProgress()  # the sink a caller that reports nothing shares; it holds no state


def bars_are_watchable(stream: TextIO) -> bool:
    """Whether ``stream`` is a terminal, which is where a redrawing bar reads as progress.

    Redirected output keeps every redraw as literal text, so a run whose stderr is a file or a pipe
    reports through :class:`SilentProgress` and leaves that file holding the errors alone.
    """
    return stream.isatty()


def progress_sink(stream: TextIO, *, enabled: bool) -> ProgressSink:
    """The sink a run reports through: a bar drawn on ``stream`` when asked for, and silence otherwise."""
    return TqdmProgress(stream) if enabled else SilentProgress()
