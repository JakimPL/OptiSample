"""Per-sample rate-distortion operating points and their lower convex hull.

For one source recording we sweep encoding configurations (stored sample rate x bit depth) and, for
each, measure ``(stored_bytes, distortion)``. Distortion is the composite fidelity between the source
and the sample *encoded then rendered back at its own pitch and duration*, so it isolates the
**encoding** loss (resampling + requantization); repitching and grouping loss are handled in later
phases.

The lower convex hull of those points is the sample's rate-distortion frontier -- the only
configurations a Lagrangian budget sweep (P3) can ever select, one per slope ``lambda``. Points that
lie on or above the chord between two neighbours are dominated and dropped, so the hull is the exact
menu of "bytes bought, distortion saved" trades the allocator reasons about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.dsp.surrogate import EncodeContext, EncodingParams, Signal, encode, render
from optisample.metrics.composite import CompositeFidelity, evaluate

_HULL_EPS = 1e-12


@dataclass(frozen=True, eq=False)
class SourceClip:
    """A recording to encode, its rate/natural pitch, and the longest duration the material needs."""

    signal: Signal
    sample_rate: int
    root_pitch: int
    duration_s: float | None = None


@dataclass(frozen=True)
class OperatingPoint:
    """One encoding's cost/quality: stored bytes vs. composite distortion (lower is better)."""

    params: EncodingParams
    stored_bytes: int
    distortion: float
    frames: int

    @property
    def kib(self) -> float:
        return self.stored_bytes / 1024.0


def default_rates(sample_rate: int, divisors: Sequence[int], min_rate: int) -> list[int]:
    """Candidate stored rates: ``sample_rate`` divided by ``divisors``, floored at ``min_rate``, deduped."""
    rates = {min(sample_rate, max(min_rate, int(round(sample_rate / divisor)))) for divisor in divisors}
    return sorted(rates, reverse=True)


def sweep_rates(sweep: SweepConfig, sample_rate: int) -> list[int]:
    """Stored rates to try: the explicit ``sweep.rates`` override, else derived from ``sample_rate``."""
    if sweep.rates is not None:
        return list(sweep.rates)
    return default_rates(sample_rate, sweep.rate_divisors, sweep.min_rate)


def _reference(clip: SourceClip) -> Signal:
    if clip.duration_s is None:
        return clip.signal
    return np.asarray(clip.signal[: max(0, int(round(clip.duration_s * clip.sample_rate)))], dtype=np.float64)


def evaluate_encoding(
    clip: SourceClip,
    params: EncodingParams,
    *,
    composite: CompositeFidelity,
    encode_config: EncodeConfig,
    rng: np.random.Generator | None = None,
) -> OperatingPoint:
    """Encode ``clip`` with ``params``, render it back at its own pitch, and score the encoding loss."""
    encode_ctx = EncodeContext(root_pitch=clip.root_pitch, config=encode_config, rng=rng)
    stored = encode(clip.signal, clip.sample_rate, params, encode_ctx)
    candidate = render(stored, clip.sample_rate, pitch=clip.root_pitch, duration_s=clip.duration_s)
    report = evaluate(_reference(clip), candidate, clip.sample_rate, composite)
    return OperatingPoint(
        params=params, stored_bytes=stored.stored_bytes, distortion=report.fidelity, frames=stored.frames
    )


def sample_operating_points(
    clip: SourceClip,
    sweep: SweepConfig,
    *,
    composite: CompositeFidelity,
    encode_config: EncodeConfig,
    rng: np.random.Generator | None = None,
) -> list[OperatingPoint]:
    """Evaluate every ``(rate, depth)`` in ``sweep`` for ``clip`` (trimmed to its material duration)."""
    rates = sweep_rates(sweep, clip.sample_rate)
    points: list[OperatingPoint] = []
    for loop in sweep.loops:
        for depth in sweep.depths:
            for rate in rates:
                params = EncodingParams(
                    target_rate=rate,
                    depth_bits=depth,
                    trim_s=clip.duration_s,
                    dither=sweep.dither,
                    noise_shaping=sweep.noise_shaping,
                    loop=loop,
                )
                points.append(
                    evaluate_encoding(clip, params, composite=composite, encode_config=encode_config, rng=rng)
                )
    return points


class RDPoint(Protocol):
    """A byte-cost/distortion point -- what the rate-distortion hull needs (per-sample or per-zone)."""

    @property
    def stored_bytes(self) -> int: ...

    @property
    def distortion(self) -> float: ...


_RDPointT = TypeVar("_RDPointT", bound=RDPoint)


def _slope(low: RDPoint, high: RDPoint) -> float:
    """Distortion change per byte between two points (negative: more bytes buy less distortion)."""
    return (high.distortion - low.distortion) / (high.stored_bytes - low.stored_bytes)


def _pareto_frontier(points: Sequence[_RDPointT]) -> list[_RDPointT]:
    """Pass 1 -- Pareto filter: sorted by bytes ascending, keep points strictly better than all cheaper ones.

    A point is dominated (and dropped) if some cheaper point already reaches its distortion or lower;
    the ``stored_bytes`` guard also collapses ties in bytes to the lowest-distortion representative.
    """
    ordered = sorted(points, key=lambda point: (point.stored_bytes, point.distortion))
    frontier: list[_RDPointT] = []
    best = np.inf
    for point in ordered:
        if point.distortion < best - _HULL_EPS and (not frontier or point.stored_bytes > frontier[-1].stored_bytes):
            frontier.append(point)
            best = point.distortion
    return frontier


def _hull_pop(frontier: Sequence[_RDPointT]) -> list[_RDPointT]:
    """Pass 2 -- convex-hull pop: drop Pareto points that sit above the chord of their neighbours.

    Walking the byte-ordered frontier, a point whose incoming slope is no steeper than the previous
    edge's marks a concave kink; pop the middle point until every successive edge gets strictly
    steeper, leaving only the vertices of the lower convex hull (a Lagrangian sweep's candidates).
    """
    hull: list[_RDPointT] = []
    for point in frontier:
        while len(hull) >= 2 and _slope(hull[-2], hull[-1]) >= _slope(hull[-1], point) - _HULL_EPS:
            hull.pop()
        hull.append(point)
    return hull


def lower_convex_hull(points: Sequence[_RDPointT]) -> list[_RDPointT]:
    """Rate-distortion frontier: Pareto-optimal points on the lower convex hull, ordered by bytes.

    Generic over the point type so it serves both per-sample operating points and the per-zone
    ``(representative, encoding)`` options of :mod:`optisample.optimize.grouping`. Built in two passes:
    :func:`_pareto_frontier` drops dominated points, then :func:`_hull_pop` drops the concave ones.
    """
    return _hull_pop(_pareto_frontier(points))


def rd_frontier(
    clip: SourceClip,
    sweep: SweepConfig,
    *,
    composite: CompositeFidelity,
    encode_config: EncodeConfig,
    rng: np.random.Generator | None = None,
) -> list[OperatingPoint]:
    """Convenience: sweep ``clip`` over ``sweep`` and return only the rate-distortion hull."""
    points = sample_operating_points(clip, sweep, composite=composite, encode_config=encode_config, rng=rng)
    return lower_convex_hull(points)
