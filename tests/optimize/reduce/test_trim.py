from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from optisample.config.reduce import ReduceConfig
from optisample.keys import SampleKey
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.reduce.trim import (
    NO_SCREEN,
    carries_signal,
    content_frames,
    screen_instrument,
    trim_recording,
)

SR = 1000  # a round rate, so a frame count reads straight off the seconds a bound states

ReduceFactory = Callable[..., ReduceConfig]


def ramp(*, loud_frames: int, quiet_frames: int, quiet_level: float = 1.0e-9) -> Signal:
    """A stretch at full scale followed by one below any floor a test sets."""
    return np.concatenate(
        [
            np.ones(loud_frames, dtype=np.float64),
            np.full(quiet_frames, quiet_level, dtype=np.float64),
        ]
    )


# --- where a recording stops carrying content ---------------------------------------------------


def test_content_ends_at_the_last_frame_reaching_the_floor() -> None:
    assert content_frames(ramp(loud_frames=300, quiet_frames=700), 1.0e-4) == 300


def test_a_frame_exactly_on_the_floor_still_counts_as_content() -> None:
    signal = np.array([1.0, 1.0e-4, 1.0e-9], dtype=np.float64)
    assert content_frames(signal, 1.0e-4) == 2


def test_a_recording_under_the_floor_throughout_carries_no_content() -> None:
    assert content_frames(np.full(500, 1.0e-9), 1.0e-4) == 0


def test_content_is_measured_on_the_magnitude_so_a_negative_peak_counts() -> None:
    assert content_frames(np.array([-1.0, 0.0, 0.0]), 0.5) == 1


def test_a_recording_that_never_falls_quiet_keeps_every_frame() -> None:
    assert content_frames(np.ones(400), 1.0e-4) == 400


# --- what a screen admits ------------------------------------------------------------------------


def test_a_recording_reaching_the_floor_somewhere_is_admitted() -> None:
    assert carries_signal(ramp(loud_frames=1, quiet_frames=999), 1.0e-3)


def test_a_recording_staying_under_the_floor_is_turned_away() -> None:
    assert not carries_signal(np.full(1000, 1.0e-4), 1.0e-3)


# --- the two bounds on a kept span ---------------------------------------------------------------


def test_the_tail_is_cut_where_the_decay_falls_under_the_floor(reduce: ReduceFactory) -> None:
    trim = reduce(trim={"max_length_s": 10.0}).trim
    kept = trim_recording(ramp(loud_frames=200, quiet_frames=5000), SR, trim)
    assert kept.size == 200


def test_a_recording_running_past_the_length_bound_is_cut_to_it(reduce: ReduceFactory) -> None:
    trim = reduce(trim={"max_length_s": 0.4}).trim
    kept = trim_recording(np.ones(5000, dtype=np.float64), SR, trim)
    assert kept.size == 400


def test_the_shorter_of_the_two_bounds_decides(reduce: ReduceFactory) -> None:
    """Both bounds shorten from the end, so the kept span runs to whichever comes first."""
    trim = reduce(trim={"max_length_s": 1.0}).trim
    content_first = trim_recording(ramp(loud_frames=300, quiet_frames=5000), SR, trim)
    length_first = trim_recording(np.ones(5000, dtype=np.float64), SR, trim)
    assert (content_first.size, length_first.size) == (300, 1000)


def test_a_recording_inside_both_bounds_is_kept_whole(reduce: ReduceFactory) -> None:
    trim = reduce(trim={"max_length_s": 10.0}).trim
    signal = np.ones(700, dtype=np.float64)
    np.testing.assert_array_equal(trim_recording(signal, SR, trim), signal)


# --- the material a screened-out recording takes with it ------------------------------------------


def _instrument(pitches: tuple[int, ...], *, count: int = 1) -> InstrumentSpec:
    return InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in pitches],
        material=[NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=count) for pitch in pitches],
    )


def test_material_a_kept_recording_answers_for_stays() -> None:
    screened = screen_instrument(_instrument((60, 67)), {60, 67}, ())
    assert [event.pitch for event in screened.instrument.material] == [60, 67]
    assert screened.screen == NO_SCREEN


def test_a_pitch_left_with_no_recording_loses_its_notes() -> None:
    screened = screen_instrument(_instrument((60, 67)), {60}, (SampleKey(67, 100),))
    assert [event.pitch for event in screened.instrument.material] == [60]
    assert screened.screen.unplayable == (67,)
    assert screened.screen.silenced == (SampleKey(67, 100),)


def test_the_screen_counts_the_notes_played_rather_than_the_events() -> None:
    """An event standing for several played notes takes all of them with it."""
    screened = screen_instrument(_instrument((60, 67), count=4), {60}, (SampleKey(67, 100),))
    assert screened.screen.dropped_notes == 4


def test_an_instrument_whose_every_recording_was_silent_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no recording carrying signal"):
        screen_instrument(_instrument((60, 67)), set(), (SampleKey(60, 100), SampleKey(67, 100)))
