"""Shared fixtures for the file/format boundary tests: small modules to write, read and render."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from optisample.config.render import PlaybackConfig
from optisample.io.tracker.target import ExportTarget
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, routed_keymap
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.module.protocol import TrackerModule
from trackmod.spec.pitch import RATE_NOTE

PROBE_ROWS = 40
PROBE_KEY = Note(RATE_NOTE)
_RELEASE_ROW = 12
_CHANNELS = 1
_CHANNEL = 0


@pytest.fixture
def tone() -> Callable[..., np.ndarray]:
    """Factory: a short tone spanning a handful of cycles, so a render has something audible in it."""

    def _tone(frames: int = 4096, cycles: float = 40.0) -> np.ndarray:
        time = np.arange(frames) / frames
        return np.asarray(0.8 * np.sin(2 * np.pi * cycles * time), dtype=np.float64)

    return _tone


@pytest.fixture
def probe_module(playback_config: PlaybackConfig, target: ExportTarget) -> Callable[[Sequence[Sample]], TrackerModule]:
    """Factory: a one-note module playing ``samples[0]`` at the reference key, then releasing it."""

    def _module(samples: Sequence[Sample]) -> TrackerModule:
        builder = PatternBuilder(rows=PROBE_ROWS, channels=_CHANNELS)
        builder.place(0, _CHANNEL, Cell(note=PROBE_KEY, instrument=0, volume=64))
        builder.place(_RELEASE_ROW, _CHANNEL, Cell(note=NoteCommand.CUT))
        keymap = routed_keymap({PROBE_KEY: KeyAssignment(sample=0, note=PROBE_KEY)})
        song = Song(
            name="probe",
            channels=_CHANNELS,
            patterns=(builder.build(),),
            order=OrderList.sequential(1),
            instruments=(Instrument(name="probe", keymap=keymap),),
            samples=tuple(samples),
            playback=Playback(speed=playback_config.speed, tempo=playback_config.tempo),
        )
        return target.bind(song)

    return _module
