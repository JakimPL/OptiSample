from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from optisample.config.loop import FeatureConfig, FrontierConfig
from optisample.dsp.similarity import FrameSeries, shape_travel, span_frames
from optisample.frontier import lower_convex_hull

Signal = NDArray[np.float64]

_MIN_ROUND: Final = 1  # series frames a round spans, which is the shortest wrap the series can state


@dataclass(frozen=True)
class LoopReach:
    """One round a recording may be looped over, and the two things storing it gives up.

    ``start`` and ``length`` are frames of the recording the series was read from. ``wrap_db`` is how far
    the sound moves across the wrap -- the shape the round opens with against the shape that would have
    followed its close -- which is the step heard once per round. ``stand_in_db`` is how far the round's own
    sound stands from the sound the rest of the note holds, which is the movement a stored round plays in
    place of. The first rises as a round runs longer and the second falls as it reaches further into the
    material, so together they put a genuine trade on the bytes a round costs.
    """

    start: int
    length: int
    wrap_db: float
    stand_in_db: float

    @property
    def stored_frames(self) -> int:
        """What storing this round keeps: the attack plus one round, which is what prices the frontier."""
        return self.start + self.length

    @property
    def distance_db(self) -> float:
        """What a listener is left with, in decibels: the step across the wrap and the movement given up.

        Both readings are the same ``logmel_l1`` distance in the same units, so their sum is one number
        stating the whole of what storing this round costs.
        """
        return self.wrap_db + self.stand_in_db


@dataclass(frozen=True)
class FrontierBounds:
    """The stretch of a recording rounds are looked for in, in frames of that recording.

    ``opens`` is where the material has settled and a round may first begin, ``reach`` the frame a stored
    round runs no further than, ``ends`` the frame the material a round stands in for runs to, and
    ``shortest`` the round every offer clears.
    """

    opens: int
    reach: int
    ends: int
    shortest: int


@dataclass(frozen=True)
class _ReachPoint:
    """A reach priced for the hull, whose stored bytes are affine in the frames the reach keeps."""

    reach: LoopReach

    @property
    def stored_bytes(self) -> int:
        return self.reach.stored_frames

    @property
    def distortion(self) -> float:
        return self.reach.distance_db


@dataclass(frozen=True)
class _Window:
    """The frames of a timbre series rounds are looked for over, and the material they stand in for."""

    series: FrameSeries
    bounds: FrontierBounds
    span: int

    @property
    def opens(self) -> int:
        """The series frame the window starts at, which every position inside it counts from.

        Held at or past where the material settled, so no round is looked for in the onset the recording
        was read as still making.
        """
        return self.series.frame_after(self.bounds.opens)

    @property
    def size(self) -> int:
        """How many positions a round's start or its end may take inside the window."""
        return self.series.frame_index(self.bounds.reach) - self.opens + 1

    @property
    def rest(self) -> int:
        """Frames of material a round standing at the window's opening plays in place of."""
        return self.series.frame_index(self.bounds.ends) - self.opens

    @property
    def shortest(self) -> int:
        """The shortest round worth offering, in series frames."""
        return max(_MIN_ROUND, ceil(self.bounds.shortest / self.series.hop_length))

    @property
    def shape(self) -> Signal:
        """The frames every wrap the window holds is read over, ``(frames, bands)``.

        The window reaches ``span`` frames past the last position a round may end at, which is the material
        a wrap there is read against.
        """
        return self.series.shape[self.opens : self.opens + self.size + self.span]

    @property
    def totals(self) -> Signal:
        """Cumulative shape from the window's opening to the end of the material, ``(rest + 1, bands)``.

        One pass of prefix sums, which answers the mean shape over any stretch of the material in constant
        time -- what lets every round be measured against everything past it in one sweep of the window.
        """
        frames = self.series.shape[self.opens : self.opens + self.rest]
        return np.concatenate([np.zeros((1, frames.shape[1]), dtype=np.float64), np.cumsum(frames, axis=0)])

    def reach(self, start: int, end: int, wrap_db: float, stand_in_db: float) -> LoopReach:
        """The reach a round running from window frame ``start`` to ``end`` names in the recording."""
        return LoopReach(
            start=self.series.frame_start(self.opens + start),
            length=(end - start) * self.series.hop_length,
            wrap_db=wrap_db,
            stand_in_db=stand_in_db,
        )


def _wrap_distances(window: _Window) -> Signal:
    """How far the sound moves across every wrap the window holds, ``(starts, ends)`` in decibels.

    Entry ``[start, end]`` reads the material a round opening at ``start`` begins with against the material
    that would have followed a close at ``end``, over ``span`` frames of each. Reading a run of frames
    rather than one gives the stochastic part of a recording -- breath, bow noise, the room tail -- the
    treatment the change rate got: two runs of noise read as the one sound they are, while two runs of
    genuinely different timbre stay apart. A wrap near the end of the material is read over the frames left
    there, which is all the evidence it carries; a wrap the window holds no frame for stays at infinity,
    which leaves it off the frontier.
    """
    shape = window.shape
    size = window.size
    distances = np.full((size, size), np.inf, dtype=np.float64)
    for lag in range(1, size):
        travel = shape_travel(shape, lag)
        if travel.size == 0:
            continue

        held = sliding_window_view(travel, min(window.span, travel.size)).mean(axis=1)
        readable = min(held.size, size - lag)
        starts = np.arange(readable)
        distances[starts, starts + lag] = held[:readable]

    return distances


def _stand_in(totals: Signal, starts: NDArray[np.int_], end: int) -> Signal:
    """How far a round from each of ``starts`` to ``end`` stands from the material it replaces, in decibels.

    What a stored round plays in place of is the round's worth of recording that would have come next, so
    that is what it is read against: the mean shape over the round beside the mean shape over the stretch
    of the same length past its close, or over what is left where the recording ends sooner. Reading one
    round ahead rather than the whole remainder is what keeps the comparison at the resolution a loop
    actually plays at -- a long round taken from an evolving passage averages to a sound no moment of the
    recording holds, and reads as the mismatch it is.

    The reading falls as a round reaches further into the material, which is what puts a rate axis on the
    frontier and a reason on the bytes a later round costs. A round reaching the end of the material
    replaces nothing and reads ``0.0``.
    """
    rest = totals.shape[0] - 1
    if end >= rest:
        return np.zeros(starts.size, dtype=np.float64)

    ahead = np.minimum(2 * end - starts, rest)
    held = (totals[end] - totals[starts]) / (end - starts)[:, None]
    following = (totals[ahead] - totals[end]) / (ahead - end)[:, None]
    return np.asarray(np.mean(np.abs(held - following), axis=1), dtype=np.float64)


def _reach_at(window: _Window, wrap: Signal, totals: Signal, end: int, config: FrontierConfig) -> LoopReach | None:
    """The best round closing at window frame ``end``: the placement giving up least of the recording.

    Storing a round keeps everything up to its close, so every start closing here costs the same bytes and
    what separates them is only what they sound like -- which makes this one point of a rate-distortion
    curve, read straight off a column of the self-similarity the window carries.

    Answers ``None`` where no round reaching this far clears ``max_wrap_distance_db``, which is material
    this length has no purchase on.
    """
    starts = np.arange(end - window.shortest + 1)
    if starts.size == 0:
        return None

    across = wrap[starts, end]
    cost = np.where(across <= config.max_wrap_distance_db, across + _stand_in(totals, starts, end), np.inf)
    chosen = int(np.argmin(cost))
    if not np.isfinite(cost[chosen]):
        return None

    return window.reach(chosen, end, float(across[chosen]), float(cost[chosen] - across[chosen]))


def _spread(reaches: Sequence[LoopReach], count: int) -> tuple[LoopReach, ...]:
    """At most ``count`` of ``reaches``, spread evenly along the stored span they cover.

    Whatever prices the frontier reads it along its byte axis, so offers spread evenly along that axis
    leave a choice at every budget while holding the sweep to a fixed number of encodings per recording.
    The cheapest and the dearest are always among them, which are the two ends a budget under pressure and
    a budget with room reach for.
    """
    if len(reaches) <= count:
        return tuple(reaches)

    stored = np.array([reach.stored_frames for reach in reaches], dtype=np.float64)
    wanted = np.linspace(stored[0], stored[-1], count)
    chosen = sorted({int(np.argmin(np.abs(stored - target))) for target in wanted})
    return tuple(reaches[index] for index in chosen)


def holds_a_round(series: FrameSeries, bounds: FrontierBounds, features: FeatureConfig) -> bool:
    """Whether the window ``bounds`` names has room for one round of the shortest length it accepts.

    This is the room :func:`loop_frontier` needs before any wrap is worth reading, so a caller finding no
    round on offer separates a recording that held too little material from one whose every wrap landed on
    a sound it had moved away from.
    """
    window = _Window(series=series, bounds=bounds, span=span_frames(series, features))
    return window.size > window.shortest


def loop_frontier(
    series: FrameSeries,
    bounds: FrontierBounds,
    features: FeatureConfig,
    config: FrontierConfig,
) -> tuple[LoopReach, ...]:
    """Every round worth offering for ``series``, cheapest stored span first.

    Storing a round keeps ``[0, start + length)``, so the bytes are affine in where the round closes and
    every close is one point of a cost-per-byte curve: the placement giving up least of the recording
    (:func:`_reach_at`), read off the self-similarity of the recording's own timbre frames. What a
    placement gives up is the step across its wrap plus the movement it plays in place of, the first rising
    as a round runs longer and the second falling as it reaches further -- which is what makes the curve a
    rate-distortion frontier rather than a ranking. Its lower convex hull keeps the closes that buy their
    bytes and ``max_offers`` holds how many of them the sweep is asked to price.

    Both readings are absolute distances rather than rates, because what a listener hears once per round is
    the mismatch itself: dividing it by the round's own length would read a long round's wrap as small
    merely for being spread out.

    Answers an empty tuple for material no round has purchase on -- a window too short to hold one, or one
    whose every wrap lands on a sound the recording has moved away from.
    """
    window = _Window(series=series, bounds=bounds, span=span_frames(series, features))
    if window.size <= window.shortest:
        return ()

    wrap = _wrap_distances(window)
    totals = window.totals
    reaches: list[LoopReach] = []
    for end in range(window.shortest, window.size):
        placed = _reach_at(window, wrap, totals, end, config)
        if placed is not None:
            reaches.append(placed)

    hull = lower_convex_hull([_ReachPoint(reach) for reach in reaches])
    return _spread([point.reach for point in hull], config.max_offers)
