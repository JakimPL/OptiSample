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

import numpy as np

from optisample.dsp.surrogate import EncodingParams, Signal, encode, render
from optisample.metrics.composite import CompositeFidelity, default_composite, evaluate

DEFAULT_RATE_DIVISORS = (1, 2, 3, 4, 6, 8)
DEFAULT_DEPTHS = (16, 8)
DEFAULT_MIN_RATE = 4_000
_HULL_EPS = 1e-12


@dataclass(frozen=True, eq=False)
class SourceClip:
    """A recording to encode, its rate/natural pitch, and the longest duration the material needs."""

    signal: Signal
    sample_rate: int
    root_pitch: int
    duration_s: float | None = None


@dataclass(frozen=True)
class SweepGrid:
    """The encoding axes to sweep for one clip (rates default to fractions of the source rate)."""

    rates: tuple[int, ...] | None = None
    depths: tuple[int, ...] = DEFAULT_DEPTHS
    dither: bool = True
    noise_shaping: bool = False


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


def default_rates(
    sample_rate: int, divisors: Sequence[int] = DEFAULT_RATE_DIVISORS, min_rate: int = DEFAULT_MIN_RATE
) -> list[int]:
    """Candidate stored rates: ``sample_rate`` divided by ``divisors``, floored at ``min_rate``, deduped."""
    rates = {min(sample_rate, max(min_rate, int(round(sample_rate / divisor)))) for divisor in divisors}
    return sorted(rates, reverse=True)


def _reference(clip: SourceClip) -> Signal:
    if clip.duration_s is None:
        return clip.signal
    return np.asarray(clip.signal[: max(0, int(round(clip.duration_s * clip.sample_rate)))], dtype=np.float64)


def evaluate_encoding(
    clip: SourceClip,
    params: EncodingParams,
    *,
    composite: CompositeFidelity | None = None,
    rng: np.random.Generator | None = None,
) -> OperatingPoint:
    """Encode ``clip`` with ``params``, render it back at its own pitch, and score the encoding loss."""
    composite = composite if composite is not None else default_composite()
    stored = encode(clip.signal, clip.sample_rate, params, root_pitch=clip.root_pitch, rng=rng)
    candidate = render(stored, clip.sample_rate, pitch=clip.root_pitch, duration_s=clip.duration_s)
    report = evaluate(_reference(clip), candidate, clip.sample_rate, composite)
    return OperatingPoint(
        params=params, stored_bytes=stored.stored_bytes, distortion=report.fidelity, frames=stored.frames
    )


def sample_operating_points(
    clip: SourceClip,
    grid: SweepGrid = SweepGrid(),
    *,
    composite: CompositeFidelity | None = None,
    rng: np.random.Generator | None = None,
) -> list[OperatingPoint]:
    """Evaluate every ``(rate, depth)`` in ``grid`` for ``clip`` (trimmed to its material duration)."""
    composite = composite if composite is not None else default_composite()
    rates = grid.rates if grid.rates is not None else default_rates(clip.sample_rate)
    points: list[OperatingPoint] = []
    for depth in grid.depths:
        for rate in rates:
            params = EncodingParams(
                target_rate=rate,
                depth_bits=depth,
                trim_s=clip.duration_s,
                dither=grid.dither,
                noise_shaping=grid.noise_shaping,
            )
            points.append(evaluate_encoding(clip, params, composite=composite, rng=rng))
    return points


def _slope(low: OperatingPoint, high: OperatingPoint) -> float:
    """Distortion change per byte between two points (negative: more bytes buy less distortion)."""
    return (high.distortion - low.distortion) / (high.stored_bytes - low.stored_bytes)


def lower_convex_hull(points: Sequence[OperatingPoint]) -> list[OperatingPoint]:
    """Rate-distortion frontier: Pareto-optimal points on the lower convex hull, ordered by bytes."""
    ordered = sorted(points, key=lambda point: (point.stored_bytes, point.distortion))
    frontier: list[OperatingPoint] = []
    best = np.inf
    for point in ordered:
        if point.distortion < best - _HULL_EPS and (not frontier or point.stored_bytes > frontier[-1].stored_bytes):
            frontier.append(point)
            best = point.distortion
    hull: list[OperatingPoint] = []
    for point in frontier:
        while len(hull) >= 2 and _slope(hull[-2], hull[-1]) >= _slope(hull[-1], point) - _HULL_EPS:
            hull.pop()
        hull.append(point)
    return hull


def rd_frontier(
    clip: SourceClip,
    grid: SweepGrid = SweepGrid(),
    *,
    composite: CompositeFidelity | None = None,
    rng: np.random.Generator | None = None,
) -> list[OperatingPoint]:
    """Convenience: sweep ``clip`` over ``grid`` and return only the rate-distortion hull."""
    return lower_convex_hull(sample_operating_points(clip, grid, composite=composite, rng=rng))
