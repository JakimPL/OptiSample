"""Metric protocol and a small plug-in registry.

A metric is any object exposing ``name`` and ``distance(reference, candidate, context) -> float``
where the distance is a non-negative dissimilarity (``0.0`` = identical, larger = worse). The
registry maps a metric name to a *factory* ``MetricsConfig -> Metric`` (not to a pre-built
instance): every metric's parameters come from config, so instances can only be built once a
configuration is known. :func:`build_composite` assembles the weighted set from these factories.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from optisample.config.metrics import MetricsConfig

Signal = NDArray[np.float64]


@dataclass(frozen=True)
class MetricContext:
    """Side information a metric may need beyond the two signals."""

    sample_rate: int
    normalized: bool = False


@runtime_checkable
class Metric(Protocol):
    """A comparison metric returning a non-negative distance (0 = identical)."""

    @property
    def name(self) -> str:
        """Registry key / display name (read-only; frozen-dataclass fields satisfy this)."""

    def distance(self, reference: Signal, candidate: Signal, context: MetricContext) -> float: ...


MetricFactory = Callable[[MetricsConfig], Metric]

_FACTORIES: dict[str, MetricFactory] = {}


def register_metric(name: str, factory: MetricFactory, *, overwrite: bool = False) -> None:
    """Register a ``MetricsConfig -> Metric`` factory under ``name`` (called at metric-module import)."""
    if not overwrite and name in _FACTORIES:
        raise ValueError(f"metric {name!r} is already registered")
    _FACTORIES[name] = factory


def build_metric(name: str, config: MetricsConfig) -> Metric:
    """Instantiate the metric registered under ``name`` from ``config``."""
    if name not in _FACTORIES:
        raise KeyError(f"unknown metric {name!r}; registered: {sorted(_FACTORIES)}")
    return _FACTORIES[name](config)


def available_metrics() -> list[str]:
    """Names of all registered metric factories, sorted."""
    return sorted(_FACTORIES)


def unregister_metric(name: str) -> None:
    """Remove a metric factory from the registry (mainly for tests)."""
    _FACTORIES.pop(name, None)
