from collections.abc import Mapping, Sequence
from typing import Final

from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode
from optisample.optimize.knapsack import KnapsackItem
from optisample.optimize.operating_points import OperatingPoint, lower_convex_hull
from optisample.optimize.tasks import EvalContext, PitchTask, score_reconstruction
from optisample.progress import ProgressSink

Shortlists = Mapping[int, tuple[EncodingParams, ...]]  # the encodings to sweep, by the pitch storing them
_SWEEP_LABEL: Final = "Sweeping encodings"


def _evaluate_config(
    task: PitchTask,
    context: EvalContext,
    params: EncodingParams,
) -> OperatingPoint:
    """Encode the pitch's own representative, then score reconstruction against every event at it."""
    encode_context = EncodeContext(root_pitch=task.pitch, config=context.encode, rng=context.rng)
    stored = encode(task.representative, context.sample_rate, params, encode_context)
    distortion = score_reconstruction(stored, task, context)
    return OperatingPoint(
        params=params,
        stored_bytes=context.storage.sample_bytes(frames=stored.frames, depth=stored.depth),
        distortion=distortion,
        frames=stored.frames,
    )


def _sweep_plan(tasks: Sequence[PitchTask], shortlists: Shortlists) -> list[tuple[PitchTask, EncodingParams]]:
    """Every encode the sweep will run, pitch-major then in shortlist order.

    Listing the work before starting it gives the run a total to report against, and enumerating it in
    the order the sweep consumes it keeps each encode drawing the same dither as it did unlisted.
    ``shortlists`` is what the bandwidth pre-pass
    (:func:`~optisample.optimize.reduce.summary.summarize_reduction`) left of the full grid per pitch, so
    the sweep spends its encodes on the encodings the budget puts within reach.
    """
    return [(task, params) for task in tasks for params in shortlists[task.pitch]]


def build_items(
    tasks: Sequence[PitchTask],
    shortlists: Shortlists,
    context: EvalContext,
    progress: ProgressSink,
) -> tuple[tuple[KnapsackItem, ...], dict[int, tuple[OperatingPoint, ...]]]:
    """Turn each pitch task into a knapsack item plus its lower-convex-hull configs."""
    swept = _sweep_plan(tasks, shortlists)
    points_by_pitch: dict[int, list[OperatingPoint]] = {task.pitch: [] for task in tasks}
    for task, params in progress.track(swept, label=_SWEEP_LABEL, total=len(swept)):
        points_by_pitch[task.pitch].append(_evaluate_config(task, context, params))

    items: list[KnapsackItem] = []
    hulls: dict[int, tuple[OperatingPoint, ...]] = {}
    for task in tasks:
        points = tuple(points_by_pitch[task.pitch])
        hulls[task.pitch] = tuple(lower_convex_hull(points))
        items.append(
            KnapsackItem(
                key=str(task.pitch),
                weight=task.objective_weight,
                points=points,
            ),
        )

    return tuple(items), hulls
