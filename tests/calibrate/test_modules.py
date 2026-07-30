from __future__ import annotations

from collections.abc import Callable

from optisample.calibrate import NoteProbe, single_note_module
from optisample.config.render import PlaybackConfig
from optisample.dsp.surrogate import StoredSample
from optisample.io.tracker.target import ExportTarget
from trackmod.core.notes.pitch import Note
from trackmod.core.timing.clock import row_seconds
from trackmod.spec.pitch import RATE_NOTE

_PROBE_PITCH = 60
_ROOT_PITCH = 48
_ROW_MARGIN = 2


def test_single_note_module_wires_one_sample_to_the_probed_key(
    stored: Callable[..., StoredSample], playback_config: PlaybackConfig, target: ExportTarget
) -> None:
    sample = stored(root_pitch=_ROOT_PITCH, rate=22_050)
    probe = NoteProbe(pitch=_PROBE_PITCH, volume=50, duration_s=0.5)
    song = single_note_module(sample, probe, playback_config, target).song
    key = Note.from_midi(_PROBE_PITCH)
    assert len(song.samples) == 1
    assert song.samples[0].rate == 22_050  # the sample states the rate it was actually stored at
    assignment = song.instruments[0].assignment(key)
    assert assignment is not None
    assert assignment.sample == 0
    # a whole octave above the recording's own key, so the key sounds an octave above the reference
    assert assignment.note == Note(RATE_NOTE + _PROBE_PITCH - _ROOT_PITCH)
    assert song.patterns[0].cell(0, 0).note == key
    assert song.patterns[0].cell(0, 0).volume == 50


def test_single_note_module_rows_cover_the_duration(
    stored: Callable[..., StoredSample], playback_config: PlaybackConfig, target: ExportTarget
) -> None:
    sample = stored()
    seconds_per_row = row_seconds(playback_config.speed, playback_config.tempo)
    long_s = (target.min_rows + 20) * seconds_per_row
    short = single_note_module(sample, NoteProbe(pitch=_PROBE_PITCH, duration_s=0.5), playback_config, target)
    longer = single_note_module(sample, NoteProbe(pitch=_PROBE_PITCH, duration_s=long_s), playback_config, target)
    assert longer.song.patterns[0].rows > short.song.patterns[0].rows
    assert longer.song.patterns[0].rows == int(long_s / seconds_per_row) + _ROW_MARGIN


def test_a_probe_shorter_than_the_compliance_floor_is_padded_up_to_it(
    stored: Callable[..., StoredSample], playback_config: PlaybackConfig, target: ExportTarget
) -> None:
    module = single_note_module(stored(), NoteProbe(pitch=_PROBE_PITCH, duration_s=0.01), playback_config, target)
    assert module.song.patterns[0].rows == target.min_rows
    assert module.violations() == ()  # a pattern the target's tracker refuses would be reported here
