from collections.abc import Callable, Sequence

import numpy as np
import pytest
from trackmod import Instrument, Sample, Song, TrackerModule
from trackmod.core.instruments.keymap import KeyAssignment, routed_keymap
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.spec.pitch import RATE_NOTE

from optisample.config.render import PlaybackConfig
from optisample.io.tracker.target import ExportTarget
from optisample.io.tracker.voices import instrument_voices

PROBE_ROWS = 40
PROBE_KEY = Note(RATE_NOTE)
_RELEASE_ROW = 12
_CHANNELS = 1
_CHANNEL = 0

ProbeModule = Callable[..., TrackerModule]


@pytest.fixture
def tone() -> Callable[..., np.ndarray]:
    """Factory: a short tone spanning a handful of cycles, so a render has something audible in it."""

    def _tone(frames: int = 4096, cycles: float = 40.0) -> np.ndarray:
        time = np.arange(frames) / frames
        return np.asarray(0.8 * np.sin(2 * np.pi * cycles * time), dtype=np.float64)

    return _tone


@pytest.fixture
def probe_module(playback_config: PlaybackConfig, target: ExportTarget) -> ProbeModule:
    """Factory: a module playing ``samples[0]`` at the reference key, then releasing it.

    Every sample gets a key of its own, counting up from the reference, so a format that stores only
    the samples its keymap reaches still writes them all.
    """

    def _module(samples: Sequence[Sample], bind_to: ExportTarget | None = None) -> TrackerModule:
        bound = bind_to if bind_to is not None else target
        builder = PatternBuilder(rows=PROBE_ROWS, channels=_CHANNELS)
        builder.place(0, _CHANNEL, Cell(note=PROBE_KEY, instrument=0, volume=64))
        builder.place(_RELEASE_ROW, _CHANNEL, bound.release_cell())
        keymap = routed_keymap(
            {
                Note(PROBE_KEY.value + index): KeyAssignment(sample=index, note=Note(PROBE_KEY.value + index))
                for index in range(len(samples))
            }
        )
        song = Song(
            name="probe",
            channels=_CHANNELS,
            patterns=(builder.build(),),
            order=OrderList.sequential(1),
            voices=instrument_voices((Instrument(name="probe", keymap=keymap),), samples),
            playback=Playback(speed=playback_config.speed, tempo=playback_config.tempo),
        )
        return bound.bind(song)

    return _module
