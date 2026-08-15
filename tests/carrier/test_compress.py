from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from optisample.carrier.compress import compressed_source, held_sources, source_reading
from optisample.carrier.instrument import written_shape
from optisample.carrier.source import CarrierSource
from optisample.carrier.store import CarrierSettings
from optisample.config import OptiConfig
from optisample.dsp.level import gain_to_db
from optisample.dsp.loop import Loop
from tests.carrier.conftest import ROOT_PITCH, SR, Sourcer

Settings = Callable[..., CarrierSettings]

_LOOP = Loop(start=SR // 4, end=SR // 2)
_SPIKE_DB = 18.0  # a burst far enough over the body that a written curve has no room to follow it


def _held(sources: Sequence[CarrierSource], settings: CarrierSettings) -> tuple[CarrierSource, ...]:
    """The set restated at the level the shape fitted from it left them."""
    shape = written_shape(sources, target=settings.target, grid=settings.grid)
    return held_sources(sources, shape.envelope, settings=settings)


def _level_range_db(source: CarrierSource) -> float:
    """How far a source's own level travels between its loudest and its quietest moment."""
    level = source.decomposition.level
    return gain_to_db(float(np.max(level)) / float(np.min(level)))


def test_the_split_a_held_source_carries_still_multiplies_back(source: Sourcer, config: OptiConfig) -> None:
    """The pair is restated, not broken: level times carrier is the held-back recording exactly."""
    one = source(spike_db=_SPIKE_DB)
    reading = source_reading(one, config.encode.envelope)
    gain = np.linspace(1.0, 0.25, one.frames, dtype=np.float64)
    held = compressed_source(one, gain, reading)

    assert held.decomposition.recombined() == pytest.approx(one.recording * gain)


def test_a_held_source_keeps_everything_but_its_level(source: Sourcer, config: OptiConfig) -> None:
    """A restated source stands for the same take at the same key, so what follows reads it unchanged."""
    one = source(spike_db=_SPIKE_DB, loop=_LOOP)
    reading = source_reading(one, config.encode.envelope)
    held = compressed_source(one, np.full(one.frames, 0.5), reading)

    assert (held.key, held.root_pitch, held.sample_rate) == (one.key, one.root_pitch, one.sample_rate)
    assert (held.loops, held.loop_index, held.weight) == (one.loops, one.loop_index, one.weight)
    assert held.frames == one.frames


def test_a_take_standing_apart_from_the_shared_curve_is_held_back_to_it(
    source: Sourcer, carrier_settings: Settings
) -> None:
    """One take carrying a burst its neighbours do not is what a single shared curve has no room to state."""
    apart, *rest = [source(pitch=ROOT_PITCH, spike_db=_SPIKE_DB)] + [source(pitch=ROOT_PITCH + step) for step in (2, 4)]
    restated, *others = _held([apart, *rest], carrier_settings())

    assert _level_range_db(restated) < _level_range_db(apart)
    for one, other in zip(others, rest):
        assert one.recording == pytest.approx(other.recording)


def test_a_set_the_shape_already_states_comes_back_as_it_stands(source: Sourcer, carrier_settings: Settings) -> None:
    """The threshold opens over the body, so a set the curve follows closely reaches the encoder untouched.

    Every take declines the same way here, so the one curve fitted from them states the whole of what they
    hold and the pass finds nothing left over -- a burst they all carry included.
    """
    sources = [source(pitch=ROOT_PITCH + step, spike_db=_SPIKE_DB) for step in (0, 2, 4)]
    held = _held(sources, carrier_settings())

    for one, other in zip(held, sources):
        assert one.recording == pytest.approx(other.recording)


def test_a_set_holding_nothing_is_held_back_by_nothing(carrier_settings: Settings) -> None:
    """An empty set states no shape to answer to, which leaves the pass with nothing to restate."""
    assert held_sources([], None, settings=carrier_settings()) == ()
