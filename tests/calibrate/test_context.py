from __future__ import annotations

from optisample.calibrate import NoteProbe
from optisample.dsp.surrogate import MAX_VOLUME


def test_note_probe_defaults_to_full_volume() -> None:
    assert NoteProbe(pitch=60).volume == MAX_VOLUME
