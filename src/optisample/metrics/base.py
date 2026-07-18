"""Metric protocol and a small plug-in registry.

A metric is any object exposing ``name`` and ``distance(reference, candidate, ctx) -> float``
where the distance is a non-negative dissimilarity (``0.0`` = identical, larger = worse). The
registry lets lightweight and (later) research-grade metrics be swapped in/out by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

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

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float: ...


_REGISTRY: dict[str, Metric] = {}


def register(metric: Metric, *, overwrite: bool = False) -> Metric:
    """Register ``metric`` under its ``name``; returns it so it can be used as a decorator target."""
    if not overwrite and metric.name in _REGISTRY:
        raise ValueError(f"metric {metric.name!r} is already registered")
    _REGISTRY[metric.name] = metric
    return metric


def get(name: str) -> Metric:
    """Look up a registered metric by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown metric {name!r}; registered: {sorted(_REGISTRY)}") from exc


def available() -> list[str]:
    """Names of all registered metrics, sorted."""
    return sorted(_REGISTRY)


def unregister(name: str) -> None:
    """Remove a metric from the registry (mainly for tests)."""
    _REGISTRY.pop(name, None)
