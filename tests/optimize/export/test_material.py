from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.dsp.timebase import row_seconds
from optisample.io.render import openmpt123_available, render_module
from optisample.io.tracker.target import ExportTarget
from optisample.model import NoteEvent
from optisample.optimize.export.material import Voicing, event_rows, material_patterns
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.cell import Cell
from trackmod.module.protocol import TrackerModule

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

_CHANNEL = 0

# 2.5 and 3.5 rows: durations that land mid-row, so rounding up to whole rows is unambiguous.
_TWO_AND_A_HALF_ROWS = 2.5
_THREE_AND_A_HALF_ROWS = 3.5


def song_cells(module: TrackerModule) -> list[Cell]:
    """Every occupied cell of every pattern, in play order."""
    return [
        cell
        for pattern in module.song.patterns
        for row in range(pattern.rows)
        if not (cell := pattern.cell(row, _CHANNEL)).is_empty
    ]


def test_pattern_applies_velocity_volume_map_and_releases_each_note(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    plan, module = build()
    cells = song_cells(module)
    notes = [cell for cell in cells if cell.note != NoteCommand.CUT]
    releases = [cell for cell in cells if cell.note == NoteCommand.CUT]
    assert [cell.note for cell in notes] == [
        Note.from_midi(60),
        Note.from_midi(60),
        Note.from_midi(67),
    ]  # material order
    assert notes[0].volume == plan.velocity_map.volume(100)  # loudest velocity -> full volume
    assert notes[1].volume == plan.velocity_map.volume(50)  # quieter event uses the mapped volume
    assert notes[0].volume > notes[1].volume
    assert len(releases) == 3  # every event is released


def test_every_note_names_the_only_instrument_the_module_carries(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    _, module = build()
    notes = [cell for cell in song_cells(module) if cell.note != NoteCommand.CUT]
    assert all(cell.instrument == 0 for cell in notes)
    assert len(module.song.instruments) == 1


def test_long_material_spills_into_multiple_ordered_patterns(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]], target: ExportTarget
) -> None:
    _, module = build([NoteEvent(pitch=60, velocity=100, duration_s=0.5) for _ in range(60)])
    song = module.song
    assert len(song.patterns) >= 2  # 60 events of 5 rows apiece overflow one pattern
    assert song.order.entries == tuple(range(len(song.patterns)))
    assert all(target.min_rows <= pattern.rows <= target.max_rows for pattern in song.patterns)
    assert module.violations() == ()  # every pattern is one the target's tracker accepts


def test_a_short_song_is_padded_up_to_the_compliance_floor(
    plain_voicing: Voicing, playback_config: PlaybackConfig, target: ExportTarget
) -> None:
    seconds_per_row = row_seconds(playback_config.speed, playback_config.tempo)
    material = [NoteEvent(pitch=60, velocity=100, duration_s=seconds_per_row)]
    patterns, order = material_patterns(material, plain_voicing, playback_config, target)
    assert order.entries == (0,)
    assert patterns[0].rows == target.min_rows  # one short note still fills a pattern the tracker accepts


def test_material_with_no_events_still_yields_one_playable_pattern(
    plain_voicing: Voicing, playback_config: PlaybackConfig, target: ExportTarget
) -> None:
    patterns, order = material_patterns([], plain_voicing, playback_config, target)
    assert len(patterns) == 1
    assert patterns[0].rows == target.min_rows
    assert order.entries == (0,)


def test_notes_are_laid_end_to_end_each_followed_by_its_release(
    plain_voicing: Voicing, playback_config: PlaybackConfig, target: ExportTarget
) -> None:
    seconds_per_row = row_seconds(playback_config.speed, playback_config.tempo)
    material = [
        NoteEvent(pitch=60, velocity=100, duration_s=_THREE_AND_A_HALF_ROWS * seconds_per_row),
        NoteEvent(pitch=67, velocity=100, duration_s=_TWO_AND_A_HALF_ROWS * seconds_per_row),
    ]
    pattern = material_patterns(material, plain_voicing, playback_config, target)[0][0]
    assert pattern.cell(0, _CHANNEL).note == Note.from_midi(60)
    assert pattern.cell(4, _CHANNEL).note == NoteCommand.CUT  # 3.5 rows rounds up to 4 held rows
    assert pattern.cell(5, _CHANNEL).note == Note.from_midi(67)  # the next note starts right after the release
    assert pattern.cell(8, _CHANNEL).note == NoteCommand.CUT


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(0.0, 1), (0.01, 1), (_TWO_AND_A_HALF_ROWS, 3), (10_000.0, 198)],
    ids=["silent", "shorter-than-a-row", "rounds-up", "capped-to-fit-a-pattern"],
)
def test_event_rows_spans_at_least_one_row_and_leaves_room_for_the_release(
    rows: float, expected: int, playback_config: PlaybackConfig, target: ExportTarget
) -> None:
    seconds_per_row = row_seconds(playback_config.speed, playback_config.tempo)
    assert event_rows(rows * seconds_per_row, seconds_per_row, target.max_rows) == expected


@requires_openmpt
def test_grouped_module_renders_through_openmpt(
    grouped_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]], render_config: RenderConfig
) -> None:
    _, module = grouped_build()  # both keys share one repitched sample
    audio, rate = render_module(module, render_config)
    assert rate == render_config.sample_rate
    assert audio.ndim == 1 and audio.size > 0
    assert float(np.max(np.abs(audio))) > 0.0  # the repitched zone actually sounds in the real engine
