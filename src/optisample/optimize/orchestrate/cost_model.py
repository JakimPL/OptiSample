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
from optisample.optimize.operating_points import OperatingPoint, lower_convex_hull, sweep_param_grid
from optisample.optimize.tasks import EvalContext, PitchTask, score_reconstruction


def _evaluate_config(task: PitchTask, context: EvalContext, params: EncodingParams) -> OperatingPoint:
    """Encode the pitch's own representative, then score reconstruction against every event at it."""
    encode_context = EncodeContext(root_pitch=task.pitch, config=context.encode, rng=context.rng)
    stored = encode(task.representative, context.sample_rate, params, encode_context)
    distortion = score_reconstruction(stored, task, context)
    return OperatingPoint(params=params, stored_bytes=stored.stored_bytes, distortion=distortion, frames=stored.frames)


def _pitch_points(task: PitchTask, context: EvalContext) -> list[OperatingPoint]:
    """Sweep the rate x depth grid for one pitch, trimming storage to its longest note."""
    return [
        _evaluate_config(task, context, params)
        for params in sweep_param_grid(context.sweep, context.sample_rate, trim_s=task.max_duration_s)
    ]


def build_items(
    tasks: Sequence[PitchTask], context: EvalContext
) -> tuple[tuple[KnapsackItem, ...], dict[int, tuple[OperatingPoint, ...]]]:
    """Turn each pitch task into a knapsack item plus its lower-convex-hull configs."""
    items: list[KnapsackItem] = []
    hulls: dict[int, tuple[OperatingPoint, ...]] = {}
    for task in tasks:
        points = tuple(_pitch_points(task, context))
        hulls[task.pitch] = tuple(lower_convex_hull(points))
        items.append(KnapsackItem(key=str(task.pitch), weight=task.weight, points=points))
    return tuple(items), hulls
