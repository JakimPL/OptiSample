from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.artifacts.instruments.normalize import (
    STORED_DEPTH,
    NormalizedRecording,
    Recording,
    level_curve,
    normalized_recording,
    recording_instrument,
    stored_level,
)
from optisample.config.codec import EncodeConfig
from optisample.config.tracker import TrackerFormat
from optisample.dsp.level import gain_to_db, level_readings, peak_amplitude
from optisample.dsp.loop import Loop
from optisample.dsp.quantize import headroom_peak
from optisample.io.tracker.envelope import EnvelopeGrid, envelope_grid, played_gain
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.base import Signal
from tests.artifacts.instruments.conftest import ROOT_PITCH, Recorder
from trackmod.core.instruments.unit import InstrumentUnit
from trackmod.core.notes.pitch import Note
from trackmod.spec.levels import MAX_VOLUME

NAME = "0000_p060_C4_v100"
_READING_WINDOW_S = 0.05
_PLAYED_BACK_TOLERANCE_DB = -60.0  # how far under the recording its own reconstruction error stays
_LATTICE_TOLERANCE_DB = -0.2  # how far under its headroom a waveform lands where the lattice runs coarsest

Grids = Callable[[ExportTarget], EnvelopeGrid]


def _recording(
    signal: Signal, sample_rate: int, *, root_pitch: int = ROOT_PITCH, loop: Loop | None = None
) -> Recording:
    """``signal`` as the recording an instrument is written from, at the pitch it was played at."""
    return Recording(name=NAME, root_pitch=root_pitch, sample_rate=sample_rate, signal=signal, loop=loop)


@pytest.fixture
def normalized(recorded: Recorder, sample_rate: int, encode_config: EncodeConfig) -> NormalizedRecording:
    """A struck note read for playing back, at the pitch it was recorded at."""
    return normalized_recording(_recording(recorded(), sample_rate), encode_config)


def _unit(
    recording: NormalizedRecording,
    target: ExportTarget,
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> InstrumentUnit:
    return recording_instrument(
        recording,
        target=target,
        grid=grid_for(target),
        headroom_db=encode_config.headroom_db,
    ).unit


def _played_back(unit: InstrumentUnit, recording: NormalizedRecording, grid: EnvelopeGrid) -> Signal:
    """What a tracker puts out for the written pair, sounded at the key the recording was played at."""
    envelope = unit.instrument.volume_envelope
    assert envelope is not None
    gain = played_gain(
        envelope,
        tempo=grid.tempo,
        frames=int(recording.signal.size),
        sample_rate=recording.sample_rate,
    )
    return np.asarray(unit.samples[0].pcm * gain, dtype=np.float64)


def _stated_level(unit: InstrumentUnit, target: ExportTarget) -> float:
    """The level the written file applies to every note it starts, as a fraction of what it states at full.

    A tracker multiplies the sample's own gain, the volume a bare note plays at and the instrument's
    global volume, so what a key delivers is all three over the waveform under them.
    """
    sample = unit.samples[0]
    instrument_maximum = target.sample_level_bounds[1].maximum
    return (
        (sample.volume / MAX_VOLUME) * (sample.gain / MAX_VOLUME) * (unit.instrument.global_volume / instrument_maximum)
    )


def _quietest_gap_db(played: Signal, recording: NormalizedRecording) -> float:
    """How far the reconstruction stands from the recording, in decibels against the recording's own level.

    The pair reproduces the recording up to the one constant the normalization spent, so the reading is
    taken after that constant is divided back out.
    """
    scale = peak_amplitude(recording.signal) / peak_amplitude(played)
    truth = level_readings(recording.signal, recording.sample_rate, window_s=_READING_WINDOW_S)
    error = level_readings(played * scale - recording.signal, recording.sample_rate, window_s=_READING_WINDOW_S)
    return float(np.max(error.values - truth.values))


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_the_written_pair_plays_the_recording_back(
    tracker_format: TrackerFormat,
    normalized: NormalizedRecording,
    retarget: Callable[[TrackerFormat], ExportTarget],
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    target = retarget(tracker_format)
    unit = _unit(normalized, target, grid_for, encode_config)
    played = _played_back(unit, normalized, grid_for(target))
    assert _quietest_gap_db(played, normalized) < _PLAYED_BACK_TOLERANCE_DB


def test_the_region_a_recording_wraps_on_indexes_the_waveform_written_from_it(
    recorded: Recorder,
    sample_rate: int,
    encode_config: EncodeConfig,
    target: ExportTarget,
    grid_for: Grids,
) -> None:
    """The curve scales each frame where it stands, so a stage's frames name the same material after it."""
    region = Loop(start=sample_rate // 4, end=sample_rate // 2)
    recording = normalized_recording(_recording(recorded(), sample_rate, loop=region), encode_config)

    sample = _unit(recording, target, grid_for, encode_config).samples[0]

    assert sample.pcm.size == recording.signal.size
    assert sample.loop is not None
    assert (sample.loop.begin, sample.loop.end) == (region.start, region.end)


@pytest.mark.parametrize("decay_db", [12.0, 36.0, 90.0])
def test_a_fall_past_what_the_envelope_grid_states_is_carried_by_the_waveform(
    decay_db: float,
    recorded: Recorder,
    sample_rate: int,
    encode_config: EncodeConfig,
    target: ExportTarget,
    grid_for: Grids,
) -> None:
    recording = normalized_recording(_recording(recorded(decay_db=decay_db), sample_rate), encode_config)
    unit = _unit(recording, target, grid_for, encode_config)
    played = _played_back(unit, recording, grid_for(target))
    assert _quietest_gap_db(played, recording) < _PLAYED_BACK_TOLERANCE_DB


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_the_curve_turns_through_the_corners_the_format_numbers(
    tracker_format: TrackerFormat,
    normalized: NormalizedRecording,
    retarget: Callable[[TrackerFormat], ExportTarget],
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    target = retarget(tracker_format)
    envelope = _unit(normalized, target, grid_for, encode_config).instrument.volume_envelope
    assert envelope is not None
    assert envelope.length <= target.envelope_point_bound.maximum
    assert len(level_curve(normalized, target).nodes) == envelope.length - 1


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_the_waveform_holds_the_whole_recording_inside_the_headroom_it_is_stored_under(
    tracker_format: TrackerFormat,
    normalized: NormalizedRecording,
    retarget: Callable[[TrackerFormat], ExportTarget],
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    sample = _unit(normalized, retarget(tracker_format), grid_for, encode_config).samples[0]
    assert sample.frames == normalized.signal.size
    assert sample.loop is None
    assert sample.depth == STORED_DEPTH
    assert peak_amplitude(sample.pcm) <= headroom_peak(encode_config.headroom_db)


@pytest.mark.parametrize("peak", [0.9, 0.3, 0.05, 0.015])
def test_a_format_stating_its_level_across_two_grids_stores_every_recording_at_its_headroom(
    peak: float,
    recorded: Recorder,
    sample_rate: int,
    encode_config: EncodeConfig,
    retarget: Callable[[TrackerFormat], ExportTarget],
    grid_for: Grids,
) -> None:
    """A lattice fine enough to meet the level asked leaves the waveform nothing to give up reaching it."""
    target = retarget(TrackerFormat.IT)
    recording = normalized_recording(_recording(recorded(peak=peak), sample_rate), encode_config)
    sample = _unit(recording, target, grid_for, encode_config).samples[0]

    stored = peak_amplitude(sample.pcm) / headroom_peak(encode_config.headroom_db)
    assert gain_to_db(stored) > _LATTICE_TOLERANCE_DB


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_the_pair_delivers_the_recording_at_the_level_it_was_captured_at(
    tracker_format: TrackerFormat,
    normalized: NormalizedRecording,
    retarget: Callable[[TrackerFormat], ExportTarget],
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    """The absolute level is stated in the file, so two instruments stand as far apart as their recordings."""
    target = retarget(tracker_format)
    unit = _unit(normalized, target, grid_for, encode_config)
    delivered = _played_back(unit, normalized, grid_for(target)) * _stated_level(unit, target)

    assert peak_amplitude(delivered) == pytest.approx(peak_amplitude(normalized.signal), rel=1e-6)


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_the_level_is_stated_on_the_step_the_format_keeps_and_the_other_stands_at_full(
    tracker_format: TrackerFormat,
    normalized: NormalizedRecording,
    retarget: Callable[[TrackerFormat], ExportTarget],
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    target = retarget(tracker_format)
    sample = _unit(normalized, target, grid_for, encode_config).samples[0]
    stating, full = (sample.gain, sample.volume) if target.stores_sample_gain else (sample.volume, sample.gain)

    assert full == MAX_VOLUME
    assert target.sample_level_bounds[0].contains(stating)


def test_the_waveform_is_level_flat_where_the_envelope_reaches(
    normalized: NormalizedRecording,
    target: ExportTarget,
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    """The stored timbre holds one loudness throughout, which is what spends the whole grid on the waveform."""
    sample = _unit(normalized, target, grid_for, encode_config).samples[0]
    readings = level_readings(sample.pcm, normalized.sample_rate, window_s=_READING_WINDOW_S)
    recorded_swing = np.ptp(
        level_readings(normalized.signal, normalized.sample_rate, window_s=_READING_WINDOW_S).values
    )
    assert np.ptp(readings.values) < recorded_swing / 2


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_every_key_the_recording_reaches_sounds_its_own_pitch(
    tracker_format: TrackerFormat,
    normalized: NormalizedRecording,
    retarget: Callable[[TrackerFormat], ExportTarget],
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    target = retarget(tracker_format)
    keymap = _unit(normalized, target, grid_for, encode_config).instrument.keymap
    root = target.key(ROOT_PITCH)
    root_assignment = keymap[root.value]
    assert root_assignment is not None
    answered = [(key, keymap[key]) for key in range(len(keymap)) if keymap[key] is not None]
    assert len(answered) > 1
    for key, assignment in answered:
        assert assignment is not None
        assert assignment.sample == 0
        assert assignment.note.value - key == root_assignment.note.value - root.value


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_the_instrument_writes_as_a_standalone_file(
    tracker_format: TrackerFormat,
    normalized: NormalizedRecording,
    retarget: Callable[[TrackerFormat], ExportTarget],
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    target = retarget(tracker_format)
    written = target.instrument_file(_unit(normalized, target, grid_for, encode_config))
    assert written.violations() == ()
    assert len(written.to_bytes()) == written.size().total


@pytest.mark.parametrize("decay_db", [12.0, 30.0])
def test_the_level_read_off_a_recording_states_the_decline_it_makes(
    decay_db: float,
    recorded: Recorder,
    sample_rate: int,
    encode_config: EncodeConfig,
) -> None:
    recording = normalized_recording(_recording(recorded(decay_db=decay_db), sample_rate), encode_config)
    assert np.ptp(recording.levels.values) == pytest.approx(decay_db, abs=1.0)


def test_the_level_is_read_at_the_pitch_the_recording_is_declared_at(
    recorded: Recorder,
    sample_rate: int,
    encode_config: EncodeConfig,
) -> None:
    """The weighting spans two periods of the declared pitch, so the same audio reads differently under two."""
    signal = recorded(pitch=29)
    deep = normalized_recording(_recording(signal, sample_rate, root_pitch=29), encode_config)
    high = normalized_recording(_recording(signal, sample_rate, root_pitch=84), encode_config)
    assert not np.allclose(deep.levels.values, high.levels.values)


def test_a_silent_recording_states_a_curve_that_writes(
    sample_rate: int,
    encode_config: EncodeConfig,
    target: ExportTarget,
    grid_for: Grids,
) -> None:
    """A take that never sounded still writes, so a dataset holding one is carried through whole."""
    recording = normalized_recording(_recording(np.zeros(sample_rate, dtype=np.float64), sample_rate), encode_config)
    written = target.instrument_file(_unit(recording, target, grid_for, encode_config))
    assert written.violations() == ()


def test_the_envelope_states_the_level_against_its_own_loudest_moment(
    normalized: NormalizedRecording,
    target: ExportTarget,
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    """An instrument written on its own carries the whole of its level, so its peak takes the top step."""
    envelope = _unit(normalized, target, grid_for, encode_config).instrument.volume_envelope
    assert envelope is not None
    assert max(point.value for point in envelope.points) == MAX_VOLUME


def test_a_curve_written_for_a_faster_clock_reaches_further_along_its_ticks(
    normalized: NormalizedRecording,
    target: ExportTarget,
    encode_config: EncodeConfig,
    release_s: float,
) -> None:
    """Ticks last ``2.5 / tempo`` seconds, so one curve counts out more of them the faster the clock runs."""

    def _written(tempo: int) -> tuple[int, ...]:
        envelope = recording_instrument(
            normalized,
            target=target,
            grid=envelope_grid(target, tempo=tempo, release_s=release_s),
            headroom_db=encode_config.headroom_db,
        ).unit.instrument.volume_envelope
        assert envelope is not None
        return tuple(point.tick for point in envelope.points)

    assert _written(128)[-1] > _written(64)[-1]


def test_a_key_a_recording_reaches_sounds_the_interval_it_stands_at(
    normalized: NormalizedRecording,
    target: ExportTarget,
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    keymap = _unit(normalized, target, grid_for, encode_config).instrument.keymap
    root = target.key(ROOT_PITCH)
    above = Note(root.value + 12)
    assignment, root_assignment = keymap[above.value], keymap[root.value]
    assert assignment is not None and root_assignment is not None
    assert assignment.note.value - root_assignment.note.value == 12


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_the_fields_take_as_much_of_the_level_as_they_state_and_the_waveform_the_rest(
    tracker_format: TrackerFormat,
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    """The fields take the lowest level at or above what is asked, so the waveform is only ever scaled down."""
    steps = retarget(tracker_format).sample_level_bounds
    for room in (1.0, 1.6, 4.0, 49.6, 300.0):
        level = stored_level(room, steps=steps)
        assert all(bound.contains(step) for step, bound in zip(level.steps, steps))
        assert 0.0 < level.share <= 1.0
        assert level.restored


def test_a_waveform_already_filling_its_headroom_is_delivered_under_the_level_it_was_captured_at(
    target: ExportTarget,
) -> None:
    """A recording the leveling leaves no room to scale up is stored as hot as it allows and reports the gap."""
    level = stored_level(0.25, steps=target.sample_level_bounds)

    assert level.steps == tuple(bound.maximum for bound in target.sample_level_bounds)
    assert level.share == pytest.approx(1.0)
    assert not level.restored
    assert level.gap_db < 0.0


def test_a_format_pinning_its_gain_states_the_level_in_the_volume_a_bare_note_plays_at(
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    it_target, xm_target = retarget(TrackerFormat.IT), retarget(TrackerFormat.XM)
    it_levels = it_target.sample_levels((20, it_target.sample_level_bounds[1].maximum))
    xm_levels = xm_target.sample_levels((20, xm_target.sample_level_bounds[1].maximum))

    assert (it_levels.volume, it_levels.gain) == (MAX_VOLUME, 20)
    assert (xm_levels.volume, xm_levels.gain) == (20, MAX_VOLUME)


def test_a_recording_captured_at_full_scale_reports_the_level_it_falls_short_of(
    recorded: Recorder,
    sample_rate: int,
    encode_config: EncodeConfig,
    target: ExportTarget,
    grid_for: Grids,
) -> None:
    """A leveled waveform asking for more than full scale is stored as hot as it can be and says so."""
    written = recording_instrument(
        normalized_recording(_recording(recorded(peak=1.0), sample_rate), encode_config),
        target=target,
        grid=grid_for(target),
        headroom_db=encode_config.headroom_db,
    )

    assert not written.level.restored
    assert written.level.gap_db < 0.0
    assert peak_amplitude(written.unit.samples[0].pcm) <= headroom_peak(encode_config.headroom_db)


def test_a_recording_with_room_to_spare_is_restored_exactly(
    normalized: NormalizedRecording,
    target: ExportTarget,
    grid_for: Grids,
    encode_config: EncodeConfig,
) -> None:
    written = recording_instrument(
        normalized,
        target=target,
        grid=grid_for(target),
        headroom_db=encode_config.headroom_db,
    )

    assert written.level.restored
