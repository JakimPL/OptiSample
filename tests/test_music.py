from __future__ import annotations

import pytest

from optisample.music import midi_to_freq, note_name, pitch_label, semitone_ratio


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


@pytest.mark.parametrize(("pitch", "label"), [(0, "p000_C-1"), (60, "p060_C4"), (108, "p108_C8")])
def test_a_pitch_label_carries_its_number_and_its_note(pitch: int, label: str) -> None:
    assert pitch_label(pitch) == label


def test_pitch_labels_sort_into_pitch_order() -> None:
    """The zero-padded number leads, so a directory of per-pitch folders reads low to high."""
    pitches = [72, 9, 60, 100]
    assert [pitch_label(pitch) for pitch in sorted(pitches)] == sorted(pitch_label(pitch) for pitch in pitches)
