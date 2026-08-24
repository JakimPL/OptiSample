from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.metrics import MetricsConfig
from optisample.metrics import MetricContext, build_metric
from optisample.metrics.base import Metric

SR = 16_000
_FREQ = 400.0
_SECOND = SR
_ATTACK_S = 0.002  # a rise short enough that a note lasting a second holds almost none of it
_SMEARED_S = 0.03  # the same rise softened, which is what a curve written on a coarse grid gives back
_TAIL_FROM_S = 0.3  # where a change is put to stand well clear of the attack the term reads


def _tone(frames: int) -> NDArray[np.float64]:
    return np.sin(2.0 * np.pi * _FREQ * np.arange(frames, dtype=np.float64) / SR)


def _struck(attack_s: float) -> NDArray[np.float64]:
    """A tone rising over ``attack_s`` and holding, which is the shape an attack term is asked about."""
    rise = max(1, round(attack_s * SR))
    envelope = np.concatenate([np.linspace(0.0, 1.0, rise), np.ones(_SECOND - rise)])
    return _tone(_SECOND) * envelope


@pytest.fixture(name="onset")
def _onset(metrics_config: MetricsConfig) -> Metric:
    return build_metric("onset", metrics_config)


@pytest.fixture(name="context")
def _context() -> MetricContext:
    return MetricContext(sample_rate=SR)


def test_a_take_measures_no_distance_from_itself(onset: Metric, context: MetricContext) -> None:
    """The term is a distance, so the one thing it has to answer for is a copy of the reference."""
    struck = _struck(_ATTACK_S)

    assert onset.distance(struck, struck, context) == pytest.approx(0.0, abs=1e-9)


def test_a_softened_attack_states_a_distance_the_whole_span_averages_away(
    onset: Metric,
    metrics_config: MetricsConfig,
    context: MetricContext,
) -> None:
    """The whole of why the term exists: a mean over a note's frames has almost no say about its first few.

    The same softening is read twice -- over the attack alone and over the note it opens -- and the attack
    reads it several times over, since what a second of held tone does to a mean over frames is dilute
    whatever the opening milliseconds hold.
    """
    reference = _struck(_ATTACK_S)
    softened = _struck(_SMEARED_S)
    spanning = build_metric("logmel_l1", metrics_config)

    assert onset.distance(reference, softened, context) > spanning.distance(reference, softened, context)


def test_a_change_past_the_attack_leaves_the_term_where_it_was(onset: Metric, context: MetricContext) -> None:
    """The term reads the attack alone, so what a note does once it is under way belongs to the other terms."""
    reference = _struck(_ATTACK_S)
    altered = np.array(reference)
    altered[round(_TAIL_FROM_S * SR) :] *= 0.25

    assert onset.distance(reference, altered, context) == pytest.approx(0.0, abs=1e-9)
