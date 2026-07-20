"""Per-pitch rate-distortion cost model for the ungrouped optimizer.

For each pitch the material uses, sweep the encoding grid (rate x depth x loop), scoring the stored
sample's reconstruction distortion against every event played at that pitch, and reduce the swept
points to one knapsack item plus its lower convex hull -- the (bytes, distortion) trade-offs the
allocator chooses from.
"""

from __future__ import annotations

from collections.abc import Sequence

from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode
from optisample.optimize.knapsack import KnapsackItem
from optisample.optimize.operating_points import OperatingPoint, lower_convex_hull, sweep_rates
from optisample.optimize.tasks import EvalContext, PitchTask, score_reconstruction


def _evaluate_config(task: PitchTask, ctx: EvalContext, params: EncodingParams) -> OperatingPoint:
    """Encode the pitch's own representative, then score reconstruction against every event at it."""
    encode_ctx = EncodeContext(root_pitch=task.pitch, config=ctx.encode, rng=ctx.rng)
    stored = encode(task.representative, ctx.sample_rate, params, encode_ctx)
    distortion = score_reconstruction(stored, task, ctx)
    return OperatingPoint(params=params, stored_bytes=stored.stored_bytes, distortion=distortion, frames=stored.frames)


def _pitch_points(task: PitchTask, ctx: EvalContext) -> list[OperatingPoint]:
    """Sweep the rate x depth grid for one pitch, trimming storage to its longest note."""
    rates = sweep_rates(ctx.sweep, ctx.sample_rate)
    trim_s = task.max_duration_s
    points: list[OperatingPoint] = []
    for loop in ctx.sweep.loops:
        for depth in ctx.sweep.depths:
            for rate in rates:
                params = EncodingParams(
                    target_rate=rate,
                    depth_bits=depth,
                    trim_s=trim_s,
                    dither=ctx.sweep.dither,
                    noise_shaping=ctx.sweep.noise_shaping,
                    loop=loop,
                )
                points.append(_evaluate_config(task, ctx, params))
    return points


def build_items(
    tasks: Sequence[PitchTask], ctx: EvalContext
) -> tuple[tuple[KnapsackItem, ...], dict[int, tuple[OperatingPoint, ...]]]:
    """Turn each pitch task into a knapsack item plus its lower-convex-hull configs."""
    items: list[KnapsackItem] = []
    hulls: dict[int, tuple[OperatingPoint, ...]] = {}
    for task in tasks:
        points = tuple(_pitch_points(task, ctx))
        hulls[task.pitch] = tuple(lower_convex_hull(points))
        items.append(KnapsackItem(key=str(task.pitch), weight=task.weight, points=points))
    return tuple(items), hulls
