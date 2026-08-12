from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from optisample.config.tracker import (
    ITTrackerConfig,
    TrackerConfig,
    TrackerFormat,
    XMTrackerConfig,
)
from optisample.io.tracker.target import ExportTarget, balanced_gains, export_target, sample_label
from trackmod.core.effects.effect import Effect
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, routed_keymap
from trackmod.core.instruments.transfer import extract
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.limits.capability import Capability
from trackmod.limits.compliance import Compliance
from trackmod.spec.levels import MAX_VOLUME
from trackmod.spec.pitch import RATE_NOTE

_KEY = Note(RATE_NOTE)
PROBE_ROWS = 32
_CHANNELS = 1
_GLOBAL_VOLUME = 100
_MIX_VOLUME = 40
_TRACKER_NAME = "probe"

_FORMATS = tuple(TrackerFormat)

# every format numbers the keyboard from C-0, and each names a different stretch of it
_KEY_RANGES = ((TrackerFormat.IT, 12, 131), (TrackerFormat.XM, 12, 107))

# what each format calls a standalone instrument on disk
_INSTRUMENT_EXTENSIONS = ((TrackerFormat.IT, ".iti"), (TrackerFormat.XM, ".xi"))


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
        xm=XMTrackerConfig(tracker=_TRACKER_NAME),
    )


def test_the_target_carries_the_format_settings_the_config_states(tracker_config: TrackerConfig) -> None:
    built = export_target(tracker_config)
    assert built.format is tracker_config.format
    assert built.compliance is tracker_config.compliance
    assert (built.it.global_volume, built.it.mix_volume) == (_GLOBAL_VOLUME, _MIX_VOLUME)
    assert built.xm.tracker == _TRACKER_NAME


@pytest.mark.parametrize("tracker_format", _FORMATS)
def test_binding_a_song_gives_a_module_that_writes_itself(
    tracker_format: TrackerFormat, retarget: Callable[[TrackerFormat], ExportTarget]
) -> None:
    target = retarget(tracker_format)
    module = target.bind(song(target.min_rows))
    assert module.song.name == "probe"
    assert module.storage is target.storage
    assert module.extension == f".{tracker_format}"
    assert module.size().total == len(module.to_bytes())


@pytest.mark.parametrize(("tracker_format", "extension"), _INSTRUMENT_EXTENSIONS)
def test_an_instrument_is_bound_as_the_file_its_own_format_writes(
    tracker_format: TrackerFormat,
    extension: str,
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    """One voice out of a song stands on its own, written the way the run's format writes instruments."""
    target = retarget(tracker_format)
    written = target.instrument_file(extract(song(target.min_rows), 0))
    assert written.extension == extension
    assert written.violations() == ()
    assert written.size().total == len(written.to_bytes())
    assert written.limits == target.limits  # the file answers to the bounds the module was graded against


@pytest.mark.parametrize("tracker_format", _FORMATS)
def test_an_instrument_file_carries_the_samples_its_keymap_reaches(
    tracker_format: TrackerFormat, retarget: Callable[[TrackerFormat], ExportTarget]
) -> None:
    """The file is portable because the unit numbers its own samples, so nothing points back at the song."""
    target = retarget(tracker_format)
    unit = extract(song(target.min_rows), 0)
    written = target.instrument_file(unit)
    assert written.unit.samples == unit.samples
    assert written.unit.instrument.keymap == unit.instrument.keymap


@pytest.mark.parametrize("tracker_format", _FORMATS)
def test_the_storage_table_prices_a_stored_sample(
    tracker_format: TrackerFormat, retarget: Callable[[TrackerFormat], ExportTarget]
) -> None:
    frames = 1000
    storage = retarget(tracker_format).storage
    cost = storage.sample_bytes(frames=frames, depth=BitDepth.SIXTEEN)
    assert cost == storage.sample + frames * BitDepth.SIXTEEN.bytes_per_frame
    assert storage.frames_budget(cost, depth=BitDepth.SIXTEEN) == frames  # the inverse of the same table


def test_only_the_format_with_a_per_sample_multiplier_reports_one(
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    """Which format restores a stored level on playback, and which has to carry it in the PCM instead."""
    assert retarget(TrackerFormat.IT).stores_sample_gain
    assert not retarget(TrackerFormat.XM).stores_sample_gain


@pytest.mark.parametrize("tracker_format", _FORMATS)
def test_the_release_cell_silences_a_channel(
    tracker_format: TrackerFormat, retarget: Callable[[TrackerFormat], ExportTarget]
) -> None:
    target = retarget(tracker_format)
    cell = target.release_cell()
    assert not cell.is_empty
    assert target.bind(song(target.min_rows)).violations() == ()  # a cell the format can write


def test_each_format_spells_the_release_the_way_it_can(retarget: Callable[[TrackerFormat], ExportTarget]) -> None:
    assert retarget(TrackerFormat.IT).release_cell().note is NoteCommand.CUT
    xm_cell = retarget(TrackerFormat.XM).release_cell()
    assert xm_cell.note is None  # this format numbers its note column for keys alone
    assert isinstance(xm_cell.effect, Effect)


@pytest.mark.parametrize(("tracker_format", "lowest", "highest"), _KEY_RANGES)
def test_each_format_reaches_its_own_stretch_of_the_keyboard(
    tracker_format: TrackerFormat,
    lowest: int,
    highest: int,
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    target = retarget(tracker_format)
    assert (target.min_pitch, target.max_pitch) == (lowest, highest)
    assert target.key(target.max_pitch).midi == target.max_pitch
    with pytest.raises(ValueError, match="key range"):
        target.key(target.max_pitch + 1)


@pytest.mark.parametrize("tracker_format", _FORMATS)
def test_a_recorded_tempo_reaches_the_clock_a_format_counts_its_ticks_in(
    tracker_format: TrackerFormat,
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    """A format states its clock as a whole number, so the nearest one to the material's own is written."""
    target = retarget(tracker_format)
    bound = target.limits.bound(Capability.TEMPO)

    assert target.tempo(115.4) == 115
    assert target.tempo(float(bound.minimum)) == bound.minimum
    assert target.tempo(float(bound.maximum)) == bound.maximum


@pytest.mark.parametrize("tracker_format", _FORMATS)
def test_a_tempo_a_format_has_no_room_for_is_refused(
    tracker_format: TrackerFormat,
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    target = retarget(tracker_format)
    bound = target.limits.bound(Capability.TEMPO)

    with pytest.raises(ValueError, match="is outside the"):
        target.tempo(bound.maximum + 1.0)

    with pytest.raises(ValueError, match="is outside the"):
        target.tempo(bound.minimum - 1.0)


def test_a_split_transposition_is_refused_by_the_format_that_cannot_hold_it(
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    """One stored sample carries one transposition in FastTracker 2, and two keys must agree on it.

    The exporter routes every key of a unit at the same interval from the unit's own recorded pitch, so
    this is the boundary telling a future caller which keymaps stay writable.
    """
    split = routed_keymap(
        {
            _KEY: KeyAssignment(sample=0, note=_KEY),
            Note(_KEY.value + 1): KeyAssignment(sample=0, note=_KEY),  # sounds the same note off another key
        }
    )
    built = song(PROBE_ROWS)
    disagreeing = built.model_copy(update={"instruments": (Instrument(name="probe", keymap=split),)})

    assert retarget(TrackerFormat.IT).bind(disagreeing).to_bytes()  # a per-key keymap holds it
    with pytest.raises(ValueError, match="transposes sample 0 differently"):
        retarget(TrackerFormat.XM).bind(disagreeing).to_bytes()


def test_the_row_bounds_come_from_the_compliance_level(target: ExportTarget) -> None:
    extended = dataclasses.replace(target, compliance=Compliance.EXTENDED)
    assert target.compliance is Compliance.CANONICAL
    assert target.min_rows > extended.min_rows  # the original tracker refuses patterns its format could hold
    assert target.max_rows == extended.max_rows


def test_a_pattern_below_the_canonical_floor_is_reported(target: ExportTarget) -> None:
    module = target.bind(song(target.min_rows - 1))
    assert module.violations()  # the floor is what the exporter pads material up to
    assert target.bind(song(target.min_rows)).violations() == ()


# --- how a set of samples is balanced and named -------------------------------------------------------------


def test_the_sample_asking_for_most_takes_the_top_step(target: ExportTarget) -> None:
    """Every sample is stored hot, so the balance between them is restored on the step beside each."""
    assert balanced_gains([1.0, 0.5, 0.25], target) == (MAX_VOLUME, MAX_VOLUME // 2, MAX_VOLUME // 4)


def test_a_sample_far_under_the_loudest_is_heard_rather_than_silenced(target: ExportTarget) -> None:
    """The softest step that still sounds is the floor, so a quiet sample keeps a voice."""
    assert balanced_gains([1.0, 1e-6], target)[1] == 1


def test_a_format_pinning_its_gain_reports_the_top_step_throughout(
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    """FastTracker 2 keeps no per-sample multiplier, so its balance rides in the PCM instead."""
    assert balanced_gains([1.0, 0.25], retarget(TrackerFormat.XM)) == (MAX_VOLUME, MAX_VOLUME)


def test_a_set_holding_nothing_is_balanced_against_nothing(target: ExportTarget) -> None:
    """An empty set states no steps, which is what a caller with no samples writes."""
    assert balanced_gains([], target) == ()


def test_a_sample_is_named_for_the_recording_it_holds() -> None:
    """One key may be stored once per velocity band, so the name states the dynamic as well as the note."""
    assert sample_label("Piano", pitch=60, velocity=100) == "Piano C4 v100"


def test_a_long_instrument_id_is_cut_to_the_field_that_holds_it() -> None:
    """The whole name fits the narrowest field a target format keeps for it."""
    name = sample_label("AnUnreasonablyLongInstrumentName", pitch=60, velocity=100)

    assert name.endswith(" C4 v100")
    assert len(name) <= 22
