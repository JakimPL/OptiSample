from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from optisample.config.tracker import ITTrackerConfig, TrackerConfig, TrackerFormat
from optisample.io.tracker.target import ExportTarget, export_target
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, routed_keymap
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.limits.compliance import Compliance
from trackmod.spec.pitch import RATE_NOTE

_KEY = Note(RATE_NOTE)
_CHANNELS = 1
_GLOBAL_VOLUME = 100
_MIX_VOLUME = 40


def song(rows: int) -> Song:
    """A one-note, one-sample song of ``rows`` rows -- the smallest thing a target can bind."""
    builder = PatternBuilder(rows=rows, channels=_CHANNELS)
    builder.place(0, 0, Cell(note=_KEY, instrument=0, volume=64))
    return Song(
        name="probe",
        channels=_CHANNELS,
        patterns=(builder.build(),),
        order=OrderList.sequential(1),
        instruments=(Instrument(name="probe", keymap=routed_keymap({_KEY: KeyAssignment(sample=0, note=_KEY)})),),
        samples=(Sample(name="s", pcm=np.zeros(64, dtype=np.float64), rate=22_050, depth=BitDepth.SIXTEEN),),
        playback=Playback(speed=6, tempo=125),
    )


@pytest.fixture
def tracker_config() -> TrackerConfig:
    return TrackerConfig(
        format=TrackerFormat.IT,
        compliance=Compliance.CANONICAL,
        it=ITTrackerConfig(global_volume=_GLOBAL_VOLUME, mix_volume=_MIX_VOLUME),
    )


def test_the_target_carries_the_format_settings_the_config_states(tracker_config: TrackerConfig) -> None:
    built = export_target(tracker_config)
    assert built.format is tracker_config.format
    assert built.compliance is tracker_config.compliance
    assert (built.it.global_volume, built.it.mix_volume) == (_GLOBAL_VOLUME, _MIX_VOLUME)


def test_binding_a_song_gives_a_module_that_writes_itself(target: ExportTarget) -> None:
    module = target.bind(song(target.min_rows))
    assert module.song.name == "probe"
    assert module.storage is target.storage
    assert module.extension.startswith(".")
    assert module.size().total == len(module.to_bytes())


def test_the_storage_table_prices_a_stored_sample(target: ExportTarget) -> None:
    frames = 1000
    cost = target.storage.sample_bytes(frames=frames, depth=BitDepth.SIXTEEN)
    assert cost == target.storage.sample + frames * BitDepth.SIXTEEN.bytes_per_frame
    assert target.storage.frames_budget(cost, depth=BitDepth.SIXTEEN) == frames  # the inverse of the same table


def test_the_release_cell_silences_a_channel(target: ExportTarget) -> None:
    cell = target.release_cell()
    assert not cell.is_empty
    assert cell.note is NoteCommand.CUT


def test_the_row_bounds_come_from_the_compliance_level(target: ExportTarget) -> None:
    extended = dataclasses.replace(target, compliance=Compliance.EXTENDED)
    assert target.compliance is Compliance.CANONICAL
    assert target.min_rows > extended.min_rows  # the original tracker refuses patterns its format could hold
    assert target.max_rows == extended.max_rows


def test_a_pattern_below_the_canonical_floor_is_reported(target: ExportTarget) -> None:
    module = target.bind(song(target.min_rows - 1))
    assert module.violations()  # the floor is what the exporter pads material up to
    assert target.bind(song(target.min_rows)).violations() == ()
