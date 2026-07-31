from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.loop import FeatureConfig, FrontierConfig
from optisample.dsp.loopability import FrontierBounds, LoopReach, loop_frontier
from optisample.dsp.similarity import frame_series

SR = 8_000
FREQ = 200.0
_SHORTEST = 800  # 0.1 s, the shortest round these tests ask for
_REACH = 2 * SR  # the stretch past the opening rounds are looked for over

Frontier = Callable[..., tuple[LoopReach, ...]]


def _sine(frames: int, freq: float = FREQ) -> NDArray[np.float64]:
    times = np.arange(frames, dtype=np.float64) / SR
    return np.asarray(0.7 * np.sin(2.0 * np.pi * freq * times) + 0.2 * np.sin(4.0 * np.pi * freq * times))


def _decaying(frames: int) -> NDArray[np.float64]:
    """A struck note: one pitch throughout, its partials thinning as it rings."""
    fading = np.exp(-np.arange(frames, dtype=np.float64) / (0.5 * SR))
    return np.asarray(_sine(frames) * fading + 0.3 * _sine(frames, freq=4 * FREQ) * fading**3)


@pytest.fixture
def frontier(features_config: FeatureConfig, loop_frontier_config: FrontierConfig) -> Frontier:
    """Factory: the rounds a signal offers, with the frontier knobs a test varies."""

    def _read(signal: NDArray[np.float64], *, shortest: int = _SHORTEST, **overrides: object) -> tuple[LoopReach, ...]:
        series = frame_series(signal, SR, features_config, FREQ)
        bounds = FrontierBounds(opens=0, reach=min(signal.size, _REACH), ends=signal.size, shortest=shortest)
        config = loop_frontier_config.model_copy(update=overrides)
        return loop_frontier(series, bounds, features_config, config)

    return _read


def test_a_window_too_short_to_hold_one_round_offers_nothing(frontier: Frontier) -> None:
    assert frontier(_sine(SR // 4), shortest=SR) == ()


def test_every_round_offered_reaches_at_least_the_shortest_asked_for(frontier: Frontier) -> None:
    reaches = frontier(_decaying(3 * SR))

    assert reaches
    assert all(reach.length >= _SHORTEST - reach.length % _SHORTEST for reach in reaches)
    assert all(reach.stored_frames == reach.start + reach.length for reach in reaches)


def test_what_a_round_gives_up_is_the_wrap_it_makes_and_the_movement_it_stands_in_for(
    frontier: Frontier,
) -> None:
    """Both readings are the same distance in the same units, so one number states the whole of the cost."""
    reaches = frontier(_decaying(3 * SR))

    assert all(reach.distance_db == pytest.approx(reach.wrap_db + reach.stand_in_db) for reach in reaches)
    assert all(reach.wrap_db >= 0.0 and reach.stand_in_db >= 0.0 for reach in reaches)


def test_the_offers_run_from_the_cheapest_stored_span_upward_and_each_buys_less(frontier: Frontier) -> None:
    """The frontier is the hull of a cost-per-byte curve, so it is what a Lagrangian sweep walks."""
    reaches = frontier(_decaying(3 * SR))
    assert len(reaches) > 1

    stored = [reach.stored_frames for reach in reaches]
    distances = [reach.distance_db for reach in reaches]
    assert stored == sorted(stored) and len(set(stored)) == len(stored)
    assert distances == sorted(distances, reverse=True)

    slopes = [(high - low) / (far - near) for low, high, near, far in zip(distances, distances[1:], stored, stored[1:])]
    assert slopes == sorted(slopes)


def test_a_round_replaced_by_material_that_sounds_the_same_gives_up_less(frontier: Frontier) -> None:
    """A round plays in place of what comes next, so what it gives up is how far that material has moved."""
    held = frontier(_sine(3 * SR))
    moving = frontier(_decaying(3 * SR))

    assert held and moving
    assert max(reach.stand_in_db for reach in held) < max(reach.stand_in_db for reach in moving)


def test_what_a_round_gives_up_falls_as_it_reaches_further_into_the_material(frontier: Frontier) -> None:
    """A note settles as it rings, so a round reaching past the movement replaces material that has stopped."""
    reaches = frontier(_decaying(3 * SR))

    assert len(reaches) > 2
    assert reaches[0].stand_in_db > reaches[-1].stand_in_db


def test_material_no_round_wraps_onto_is_turned_away_by_the_bound_on_the_wrap(frontier: Frontier) -> None:
    """Noise holds a shape that never comes back, so the wrap it makes is what says it cannot be looped."""
    noise = np.random.default_rng(0).standard_normal(3 * SR)
    wandering = frontier(noise, max_wrap_distance_db=1e9)
    repeating = frontier(_sine(3 * SR), max_wrap_distance_db=1e9)

    assert wandering and repeating
    between = (min(reach.wrap_db for reach in wandering) + max(reach.wrap_db for reach in repeating)) / 2
    assert frontier(noise, max_wrap_distance_db=between) == ()
    assert frontier(_sine(3 * SR), max_wrap_distance_db=between) != ()


def test_the_sweep_is_asked_to_price_no_more_offers_than_it_was_told_to(frontier: Frontier) -> None:
    """Each offer costs the sweep an encoding, so how many reach it is stated rather than left to the hull."""
    signal = _decaying(3 * SR)
    capped = frontier(signal, max_offers=2)
    whole = frontier(signal, max_offers=64)

    assert len(capped) <= 2 < len(whole)
    assert capped[0] == whole[0]  # the cheapest offer is kept, which is what a budget under pressure reaches for
    assert capped[-1] == whole[-1]  # and the dearest, which is what one with room reaches for


def test_rounds_are_looked_for_no_further_than_the_reach_the_search_is_given(
    features_config: FeatureConfig, loop_frontier_config: FrontierConfig
) -> None:
    """The reach is what holds a note that rings for a minute to the same work as one that rings for a second."""
    signal = _decaying(4 * SR)
    series = frame_series(signal, SR, features_config, FREQ)
    bounds = FrontierBounds(opens=0, reach=SR, ends=signal.size, shortest=_SHORTEST)

    reaches = loop_frontier(series, bounds, features_config, loop_frontier_config)

    assert reaches
    assert all(reach.stored_frames <= SR + series.hop_length for reach in reaches)
