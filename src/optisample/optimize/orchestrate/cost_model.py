from collections.abc import Sequence
from typing import Final

from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode
from optisample.optimize.knapsack import KnapsackItem
from optisample.optimize.operating_points import OperatingPoint, lower_convex_hull
from optisample.optimize.reduce.bandwidth import ClipDemand, candidate_params
from optisample.optimize.tasks import EvalContext, PitchTask, score_reconstruction

_OWN_KEY: Final = 0  # every key here sounds its own recording, so nothing is transposed
_ONE_KEY: Final = 1  # and each stored sample answers for that one key alone


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
    context: EvalContext,
) -> list[OperatingPoint]:
    """Sweep the shortlisted encodings for one pitch, trimming storage to its longest note.

    The shortlist comes from the bandwidth pre-pass, which prices and scores the whole rate x depth grid
    from the recording alone and hands back the part of it the budget puts within reach. Every key here
    plays its own recording, so the pre-pass bounds the stored band by the recording's own content.
    """
    demand = ClipDemand(trim_s=task.max_duration_s, delta_semitones=_OWN_KEY, key_count=_ONE_KEY)
    return [
        _evaluate_config(task, context, params)
        for params in candidate_params(
            task.representative,
            demand,
            context,
        )
    ]


def build_items(
    tasks: Sequence[PitchTask],
    context: EvalContext,
) -> tuple[tuple[KnapsackItem, ...], dict[int, tuple[OperatingPoint, ...]]]:
    """Turn each pitch task into a knapsack item plus its lower-convex-hull configs."""
    items: list[KnapsackItem] = []
    hulls: dict[int, tuple[OperatingPoint, ...]] = {}
    for task in tasks:
        points = tuple(_pitch_points(task, context))
        hulls[task.pitch] = tuple(lower_convex_hull(points))
        items.append(
            KnapsackItem(
                key=str(task.pitch),
                weight=task.weight,
                points=points,
            ),
        )

    return tuple(items), hulls
