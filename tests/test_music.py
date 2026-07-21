from __future__ import annotations

import pytest

from optisample.music import midi_to_freq, note_name, semitone_ratio


def test_semitone_ratio_octaves() -> None:
    assert semitone_ratio(0.0) == pytest.approx(1.0)
    assert semitone_ratio(12.0) == pytest.approx(2.0)
    assert semitone_ratio(-12.0) == pytest.approx(0.5)


def test_midi_to_freq_a4() -> None:
    assert midi_to_freq(69) == pytest.approx(440.0)
    assert midi_to_freq(81) == pytest.approx(880.0)


@pytest.mark.parametrize(
    ("pitch", "name"),
    [(0, "C-1"), (12, "C0"), (60, "C4"), (61, "C#4"), (69, "A4"), (71, "B4")],
)
def test_note_name(pitch: int, name: str) -> None:
    assert note_name(pitch) == name
