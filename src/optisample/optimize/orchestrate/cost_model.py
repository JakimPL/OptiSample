from collections.abc import Mapping, Sequence

from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode
from optisample.optimize.knapsack import KnapsackItem
from optisample.optimize.operating_points import OperatingPoint, lower_convex_hull
from optisample.optimize.tasks import EvalContext, PitchTask, score_reconstruction

Shortlists = Mapping[int, tuple[EncodingParams, ...]]  # the encodings to sweep, by the pitch storing them


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


def _pitch_points(
    task: PitchTask,
    shortlist: Sequence[EncodingParams],
    context: EvalContext,
) -> list[OperatingPoint]:
    """Sweep one pitch's shortlisted encodings, each trimming storage to the longest note played here.

    ``shortlist`` is what the bandwidth pre-pass
    (:func:`~optisample.optimize.reduce.summary.summarize_reduction`) left of the full grid for this
    pitch, so the sweep spends its encodes on the encodings the budget puts within reach.
    """
    return [_evaluate_config(task, context, params) for params in shortlist]


def build_items(
    tasks: Sequence[PitchTask],
    shortlists: Shortlists,
    context: EvalContext,
) -> tuple[tuple[KnapsackItem, ...], dict[int, tuple[OperatingPoint, ...]]]:
    """Turn each pitch task into a knapsack item plus its lower-convex-hull configs."""
    items: list[KnapsackItem] = []
    hulls: dict[int, tuple[OperatingPoint, ...]] = {}
    for task in tasks:
        points = tuple(_pitch_points(task, shortlists[task.pitch], context))
        hulls[task.pitch] = tuple(lower_convex_hull(points))
        items.append(
            KnapsackItem(
                key=str(task.pitch),
                weight=task.weight,
                points=points,
            ),
        )

    return tuple(items), hulls
