from os import environ
from typing import Final

_ONE_THREAD: Final = "1"
_THREAD_LIMITS: Final = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def bind_compute_threads() -> None:
    """Hold this process and every worker it starts to one compute thread each.

    The numerical libraries a run computes with size a thread pool of their own from the machine's core
    count as they load, and that count multiplies against the processes a stage shares its work across:
    six workers measured at thirty-one cores of a twenty-four core machine. Stating the limit first holds
    a run to the workers it asked for.

    It also holds every part of a run to the same arithmetic. A sum reduced over one thread reads the
    same however many cores were free, so a score is the score whether its stage was shared out or
    carried in the calling process — which is the property the fan-out stages are gated on.

    Callers state this before the libraries load, since each reads its count once, as it is imported.
    A limit the environment already carries is the caller's own, and stands.
    """
    for name in _THREAD_LIMITS:
        environ.setdefault(name, _ONE_THREAD)
