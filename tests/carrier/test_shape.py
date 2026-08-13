from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from optisample.carrier.instrument import written_shape
from optisample.carrier.shape import NO_SHAPE, carrier_shape, shared_rate
from optisample.io.tracker.envelope import NO_ENVELOPE, envelope_grid, shape_nodes
from optisample.io.tracker.target import ExportTarget
from tests.carrier.conftest import ROOT_PITCH, Sourcer

_ALIKE_DB = 3.0  # how far apart a set declining together is allowed to read, which a shared curve follows


@pytest.fixture
def nodes(target: ExportTarget) -> int:
    """Corners the format has room for in a fitted shape."""
    return shape_nodes(target.envelope_point_bound)


def test_a_set_read_at_one_rate_states_that_rate(source: Sourcer) -> None:
    """Every source of a set is read on one clock, which is the clock their shared shape is fitted on."""
    sources = [source(pitch=ROOT_PITCH), source(pitch=ROOT_PITCH + 7)]

    assert shared_rate(sources) == sources[0].sample_rate


def test_a_set_holding_nothing_has_no_rate_to_state(source: Sourcer) -> None:
    """A shape is shared by material, so an empty set is refused rather than fitted."""
    with pytest.raises(ValueError, match="at least one source"):
        shared_rate([])


def test_sources_read_at_rates_that_differ_are_refused(source: Sourcer) -> None:
    """Two rates would place one moment at two positions, so the set is refused before it is fitted."""
    sources = [source(), replace(source(), sample_rate=44_100)]

    with pytest.raises(ValueError, match="one rate"):
        shared_rate(sources)


def test_the_shape_follows_the_decline_its_sources_make(source: Sourcer, nodes: int) -> None:
    """A struck note falls, so the curve a set of them shares falls too."""
    shape = carrier_shape([source(decay_db=30.0), source(decay_db=30.0, pitch=ROOT_PITCH + 5)], nodes=nodes)

    assert shape is not NO_SHAPE
    assert shape.curve.values[-1] < shape.curve.values[0]


def test_a_set_declining_alike_shares_one_curve_closely(source: Sourcer, nodes: int) -> None:
    """Recordings falling at one rate cost little to write under a single envelope."""
    alike = [source(pitch=ROOT_PITCH + step, decay_db=30.0) for step in (0, 2, 4)]
    shape = carrier_shape(alike, nodes=nodes)

    assert shape is not NO_SHAPE
    assert shape.dispersion_db < _ALIKE_DB


def test_a_set_declining_apart_costs_more_to_share(source: Sourcer, nodes: int) -> None:
    """The whole reading: an envelope fits a set worse the more its members' own declines disagree."""
    alike = carrier_shape([source(decay_db=30.0), source(pitch=ROOT_PITCH + 2, decay_db=30.0)], nodes=nodes)
    apart = carrier_shape([source(decay_db=6.0), source(pitch=ROOT_PITCH + 2, decay_db=60.0)], nodes=nodes)

    assert alike is not NO_SHAPE
    assert apart is not NO_SHAPE
    assert apart.dispersion_db > alike.dispersion_db


def test_each_source_answers_to_a_level_of_its_own(source: Sourcer, nodes: int) -> None:
    """One group per source is what leaves the curve stating the decline and the steps stating the loudness."""
    quiet, loud = source(peak=0.05), source(pitch=ROOT_PITCH + 3, peak=0.5)
    shape = carrier_shape([quiet, loud], nodes=nodes)

    assert shape is not NO_SHAPE
    offsets = shape.offsets_db[0]
    assert len(offsets) == 2
    assert offsets[0] < offsets[1]


def test_a_set_too_short_to_read_leaves_no_shape(source: Sourcer, nodes: int) -> None:
    """A stretch shorter than one whole window states nothing a curve could follow."""
    brief = replace(source(), decomposition=replace(source().decomposition, level=np.ones(1), carrier=np.ones(1)))

    assert carrier_shape([brief], nodes=nodes) is NO_SHAPE


def test_a_source_worth_nothing_is_refused(source: Sourcer, nodes: int) -> None:
    """A member carrying no playing time would weigh nothing in the shape it is meant to shift."""
    with pytest.raises(ValueError, match="worth more than nothing"):
        carrier_shape([source(weight=0.0)], nodes=nodes)


def test_a_set_too_short_to_read_is_written_without_a_curve(source: Sourcer, target: ExportTarget) -> None:
    """No shape means no envelope, which sounds every waveform at the level its own material carries."""
    brief = replace(source(), decomposition=replace(source().decomposition, level=np.ones(1), carrier=np.ones(1)))
    grid = envelope_grid(target, tempo=125, release_s=0.25)
    written = written_shape([brief], target=target, grid=grid)

    assert written.envelope is NO_ENVELOPE
    assert written.dispersion_db == 0.0
