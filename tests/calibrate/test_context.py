from __future__ import annotations

from trackmod.spec.levels import MAX_VOLUME

from optisample.calibrate import NoteProbe


def test_note_probe_defaults_to_full_volume() -> None:
    assert NoteProbe(pitch=60).volume == MAX_VOLUME
