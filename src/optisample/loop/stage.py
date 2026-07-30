from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final

from optisample.config.loop import LoopConfig
from optisample.keys import SampleKey
from optisample.loop.settle import Settlement, settle_loop
from optisample.metrics.base import Signal
from optisample.music import midi_to_freq
from optisample.parallel import map_workers
from optisample.progress import ProgressSink

SETTLE_LABEL: Final = "Settling loops"


@dataclass(frozen=True)
class LoopRequest:
    """One recording as the stage receives it: which sound it is, its audio, and the span asked of it.

    ``search_s`` is the longest stretch the material plays at this recording's pitch, which bounds where a
    loop is worth placing: a loop ending past it stores more than keeping the played span would.
    """

    key: SampleKey
    signal: Signal
    search_s: float


def settle_one(request: LoopRequest, config: LoopConfig, sample_rate: int) -> Settlement:
    """Settle the loop for one recording, reading the request and the config alone.

    Depending on nothing else is what lets the stage share its recordings out over processes and reach the
    same settlement in whichever order they come back. The key names the pitch the recording sounds, so the
    period searched and the level read are both taken over the stretch this note's own material repeats in.
    """
    return settle_loop(
        request.signal,
        sample_rate,
        config,
        root_hz=midi_to_freq(request.key.pitch),
        search_s=request.search_s,
    )


def settle_loops(
    requests: Sequence[LoopRequest],
    config: LoopConfig,
    sample_rate: int,
    *,
    workers: int,
    progress: ProgressSink,
) -> dict[SampleKey, Settlement]:
    """Settle every recording's loop, sharing the recordings across ``workers`` processes.

    Answers keyed by identity, so a later stage reaches a recording's loop by the same key the audio is
    held under and the order the settlements came back in carries no meaning.
    """
    settled = map_workers(
        partial(settle_one, config=config, sample_rate=sample_rate),
        list(requests),
        workers=workers,
        label=SETTLE_LABEL,
        progress=progress,
    )
    return {request.key: settlement for request, settlement in zip(requests, settled)}
