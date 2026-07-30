from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.loop import LoopConfig
from optisample.keys import SampleKey
from optisample.loop.stage import LoopRequest, settle_loops, settle_one
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.orchestrate.audio import LoadedInstrument
from optisample.optimize.orchestrate.looping import run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.trim import NO_SCREEN
from optisample.parallel import IN_PROCESS
from optisample.progress import NO_PROGRESS

SR = 22_050
_HELD_S = 3.0
_SHARED = 2  # workers, enough to run the stage apart without asking the machine for every core
_PITCHED: Final = ((57, 220.0), (69, 440.0))


def _tone(freq: float) -> NDArray[np.float64]:
    times = np.arange(int(_HELD_S * SR), dtype=np.float64) / SR
    return np.asarray(0.7 * np.sin(2.0 * np.pi * freq * times) + 0.2 * np.sin(4.0 * np.pi * freq * times))


def _requests() -> tuple[LoopRequest, ...]:
    """Two recordings of different pitches, each asked for the whole of what it holds."""
    return tuple(
        LoopRequest(key=SampleKey(pitch, 100), signal=_tone(freq), search_s=_HELD_S) for pitch, freq in _PITCHED
    )


def _loaded() -> LoadedInstrument:
    """A run holding both recordings, its material playing each pitch for as long as the recording lasts."""
    return LoadedInstrument(
        instrument=InstrumentSpec(
            id="pad",
            budget_kb=96.0,
            samples=[SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch, _ in _PITCHED],
            material=[NoteEvent(pitch=pitch, velocity=100, duration_s=_HELD_S, count=1) for pitch, _ in _PITCHED],
        ),
        audio={request.key: request.signal for request in _requests()},
        sample_rate=SR,
        screen=NO_SCREEN,
    )


def test_every_recording_earns_a_settlement_under_its_own_identity(loop_config: LoopConfig) -> None:
    requests = _requests()

    settled = settle_loops(requests, loop_config, SR, workers=IN_PROCESS, progress=NO_PROGRESS)

    assert set(settled) == {request.key for request in requests}
    assert all(settlement.search_s == _HELD_S for settlement in settled.values())


def test_a_recording_is_settled_from_itself_and_the_config_alone(loop_config: LoopConfig) -> None:
    """That is the property the stage shares its recordings out on, so one request answers for itself."""
    request = _requests()[0]

    assert (
        settle_one(request, loop_config, SR)
        == settle_loops((request,), loop_config, SR, workers=IN_PROCESS, progress=NO_PROGRESS)[request.key]
    )


def test_a_stage_shared_across_processes_states_the_same_settlements(loop_config: LoopConfig) -> None:
    requests = _requests()

    shared = settle_loops(requests, loop_config, SR, workers=_SHARED, progress=NO_PROGRESS)

    assert shared == settle_loops(requests, loop_config, SR, workers=IN_PROCESS, progress=NO_PROGRESS)


def test_settling_nothing_answers_with_nothing(loop_config: LoopConfig) -> None:
    assert settle_loops((), loop_config, SR, workers=IN_PROCESS, progress=NO_PROGRESS) == {}


# --- a run that stores no loops -------------------------------------------------------------------


def test_a_run_storing_no_loops_settles_nothing_and_leaves_every_span_played(
    loop_config: LoopConfig, optimize_settings: Callable[..., OptimizeSettings], sweep: Callable[..., object]
) -> None:
    """``--no-loop`` leaves the stage out, so the sweep has nothing to price a loop against."""
    settings = optimize_settings(sweep=sweep(), loop=loop_config)

    without = run_loops(_loaded(), dataclasses.replace(settings, loops=False))

    assert without.settlements == {}
    assert without.looped_recordings == 0
    assert without.recordings.settled == {}


def test_a_run_storing_loops_carries_them_on_the_recordings_it_hands_forward(
    loop_config: LoopConfig, optimize_settings: Callable[..., OptimizeSettings], sweep: Callable[..., object]
) -> None:
    """What the stage decided reaches every later stage through the recordings bundle, keyed as the audio is."""
    loaded = _loaded()

    with_loops = run_loops(loaded, optimize_settings(sweep=sweep(), loop=loop_config))

    assert set(with_loops.recordings.settled) == set(loaded.audio)
    assert with_loops.recordings.sample_rate == SR
    assert with_loops.looped_recordings == len(loaded.audio)
