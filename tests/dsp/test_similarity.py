from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.loop import FeatureConfig
from optisample.dsp.level import gain_to_db
from optisample.dsp.similarity import change_rate, frame_series, settling_frame, window_frames

SR = 8_000
FREQ = 200.0
_QUIET = 0.25  # the gain a copy of a recording is taken at, which its shape stays clear of
_TRAVEL_HZ = 2_000.0  # the pitch a sweep climbs to, which is a shape moving as fast as one can


def _sine(frames: int, freq: float = FREQ, amp: float = 0.8) -> NDArray[np.float64]:
    times = np.arange(frames, dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * times)


def _sweep(frames: int) -> NDArray[np.float64]:
    """A tone climbing from the played pitch to ``_TRAVEL_HZ``, so its shape travels the whole way."""
    times = np.arange(frames, dtype=np.float64) / SR
    rising = FREQ + (_TRAVEL_HZ - FREQ) * times / max(times[-1], 1e-9)
    return 0.8 * np.sin(2.0 * np.pi * np.cumsum(rising) / SR)


def test_a_frame_spans_whole_periods_of_the_pitch_it_is_read_at(features_config: FeatureConfig) -> None:
    """Reading each note over its own periods is what resolves a deep note's partials and a high one's alike."""
    deep = window_frames(SR, features_config, 100.0)
    high = window_frames(SR, features_config, 4_000.0)

    assert deep == round(features_config.window_periods * SR / 100.0)
    assert high == round(features_config.min_window_s * SR)  # the floor carries a spectrum for the shortest
    assert deep % 2 == 0 and high % 2 == 0


def test_the_shape_a_frame_states_is_the_sound_past_the_level_it_was_played_at(
    features_config: FeatureConfig,
) -> None:
    """Scaling a recording moves the level curve by that gain and leaves the shape exactly as it stands."""
    signal = _sine(SR)
    loud = frame_series(signal, SR, features_config, FREQ)
    quiet = frame_series(_QUIET * signal, SR, features_config, FREQ)

    assert np.allclose(loud.shape, quiet.shape)
    assert np.allclose(loud.level_db - quiet.level_db, -gain_to_db(_QUIET))


def test_every_frame_states_its_shape_around_the_level_it_carries(features_config: FeatureConfig) -> None:
    series = frame_series(_sine(SR), SR, features_config, FREQ)

    assert np.allclose(series.shape.mean(axis=1), 0.0, atol=1e-9)
    assert series.frames == series.level_db.size
    assert series.shape.shape == (series.frames, features_config.bands)
    assert series.frame_start(3) == 3 * series.hop_length
    assert series.hop_s == pytest.approx(series.hop_length / SR)


def test_a_recording_frame_is_read_as_the_series_frame_covering_it_or_the_one_past_it(
    features_config: FeatureConfig,
) -> None:
    """A stretch stated in recording frames stays inside itself: its near end rounds up, its far end down."""
    series = frame_series(_sine(SR), SR, features_config, FREQ)
    inside = series.frame_start(3) + series.hop_length // 2

    assert series.frame_index(inside) == 3
    assert series.frame_after(inside) == 4
    assert series.frame_index(series.frame_start(3)) == series.frame_after(series.frame_start(3)) == 3
    assert series.frame_after(2 * SR) == series.frames  # a bound past the recording lands on its last frame


def test_material_holding_one_sound_moves_slower_than_material_that_travels(
    features_config: FeatureConfig,
) -> None:
    held = change_rate(frame_series(_sine(2 * SR), SR, features_config, FREQ), features_config)
    traveling = change_rate(frame_series(_sweep(2 * SR), SR, features_config, FREQ), features_config)

    assert float(np.max(held)) < features_config.settle_db_per_s < float(np.median(traveling))


def test_a_recording_shorter_than_the_span_states_no_rate_and_settles_at_once(
    features_config: FeatureConfig,
) -> None:
    """A reading needs a span of material either side of it, which a blip carries none of."""
    series = frame_series(_sine(SR // 100), SR, features_config, FREQ)

    assert change_rate(series, features_config).size == 0
    assert settling_frame(series, features_config) == 0


def test_a_note_holding_one_sound_settles_at_its_first_frame(features_config: FeatureConfig) -> None:
    assert settling_frame(frame_series(_sine(2 * SR), SR, features_config, FREQ), features_config) == 0


def test_a_note_whose_sound_travels_settles_where_it_stops_traveling(features_config: FeatureConfig) -> None:
    """The onset each recording states is its own, so a sound still moving is read as still in its onset."""
    traveling = _sweep(SR)
    signal = np.concatenate([traveling, _sine(3 * SR, freq=_TRAVEL_HZ)])

    settled = settling_frame(frame_series(signal, SR, features_config, FREQ), features_config)

    assert traveling.size // 2 < settled < traveling.size + SR // 2


def test_one_reading_dipping_under_the_threshold_leaves_the_material_unsettled(
    features_config: FeatureConfig,
) -> None:
    """Noise holds a shape that jitters where it stands, so a reading falls under the rate on its own noise."""
    noise = np.random.default_rng(0).standard_normal(2 * SR)
    series = frame_series(noise, SR, features_config, FREQ)

    dips = np.nonzero(change_rate(series, features_config) <= features_config.settle_db_per_s)[0]

    assert dips.size > 0
    assert settling_frame(series, features_config) > series.frame_start(int(dips[0]))
