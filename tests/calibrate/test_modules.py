from __future__ import annotations

from collections.abc import Callable

from optisample.calibrate import NoteProbe, single_note_module
from optisample.dsp.surrogate import StoredSample
from optisample.io.it_writer import ITPlayback
from optisample.optimize.export import c5speed_for_pitch


def test_single_note_module_wires_one_sample_to_the_probed_key(
    stored: Callable[..., StoredSample], playback: ITPlayback
) -> None:
    sample = stored(root_pitch=48, rate=22_050)
    module = single_note_module(sample, NoteProbe(pitch=60, volume=50, duration_s=0.5), playback)
    assert len(module.samples) == 1
    assert module.samples[0].c5speed == c5speed_for_pitch(22_050, 48)  # root key plays natural
    assert module.instruments[0].note_map[60] == (60, 1)  # probed key -> (identity note, sample 1)
    row, channel, cell = module.patterns[0].cells[0]
    assert (row, channel) == (0, 0)
    assert (cell.note, cell.instrument, cell.volume) == (60, 1, 50)


def test_single_note_module_rows_cover_the_duration(stored: Callable[..., StoredSample], playback: ITPlayback) -> None:
    sample = stored()
    short = single_note_module(sample, NoteProbe(pitch=60, duration_s=0.5), playback)
    longer = single_note_module(sample, NoteProbe(pitch=60, duration_s=2.0), playback)
    assert longer.patterns[0].rows > short.patterns[0].rows
    assert short.patterns[0].rows == int(0.5 / (6 * 2.5 / 125)) + 2  # speed 6, tempo 125 -> 0.12 s/row
