"""The zone cost model: every candidate zone's ``(representative, encoding)`` options and their cost.

A zone is a contiguous run of keys served by one stored sample. Any member can be the stored
representative and any encoding can store it; each choice reconstructs the whole zone with some
usage-weighted, summed distortion at some byte cost. This module enumerates all of them -- the "menu"
the partition+allocation DP in :mod:`optisample.optimize.grouping.solve` chooses from.

Folding representative selection into the RD options (rather than picking a medoid up front) is what
makes the choice budget-aware: at minimum byte pressure the best representative is exactly the
k-medoids/PAM medoid (the member minimizing within-zone distortion), but a tighter budget may prefer a
different member or a cheaper encoding, so representative and allocation are chosen jointly by the DP.

This is the expensive half of grouping (an encode plus a reconstruction score per representative,
encoding and covered pitch); the DP that consumes the menu is cheap.
"""

from __future__ import annotations

from collections.abc import Sequence

from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode
from optisample.music import semitone_ratio
from optisample.optimize.operating_points import lower_convex_hull, sweep_rates
from optisample.optimize.plans.grouped import ZoneOption
from optisample.optimize.tasks import EvalContext, PitchTask, score_reconstruction

_Range = tuple[int, int]  # half-open [i, j) index range into the ordered pitch tasks
_ZoneOptions = dict[_Range, tuple[ZoneOption, ...]]


def _zone_trim(range_tasks: Sequence[PitchTask], representative: int) -> float:
    """Stored duration the representative needs so every covered key's longest note plays in full.

    Playing key ``p`` from a sample rooted at ``representative`` runs it at ``2**((p - rep)/12)`` times
    speed, so a higher key consumes stored frames faster and needs a proportionally longer sample.
    """
    return max(task.max_duration_s * semitone_ratio(task.pitch - representative) for task in range_tasks)


def _evaluate_zone_option(
    rep_task: PitchTask, range_tasks: Sequence[PitchTask], params: EncodingParams, ctx: EvalContext
) -> ZoneOption:
    """Store ``rep_task``'s recording with ``params`` and score it reconstructing every key in the zone.

    The zone's total distortion is the usage-weighted sum of each covered key's reconstruction from this
    one stored sample (repitched to that key), so a distant key that the representative serves poorly
    costs the option here rather than being averaged away.
    """
    encode_ctx = EncodeContext(root_pitch=rep_task.pitch, config=ctx.encode, rng=ctx.rng)
    stored = encode(rep_task.representative, ctx.sample_rate, params, encode_ctx)
    distortion = sum(task.weight * score_reconstruction(stored, task, ctx) for task in range_tasks)
    return ZoneOption(rep_task.pitch, params, stored.stored_bytes, distortion, stored.frames)


def _zone_options(range_tasks: Sequence[PitchTask], ctx: EvalContext) -> list[ZoneOption]:
    """Every ``(representative, encoding)`` for one candidate zone, with its cost and total distortion.

    Enumerates the outer product of representative (each covered key's own recording, the k-medoids
    candidates) and encoding (loop x depth x rate); :func:`_evaluate_zone_option` scores each.
    """
    rates = sweep_rates(ctx.sweep, ctx.sample_rate)
    options: list[ZoneOption] = []
    for rep_task in range_tasks:
        trim_s = _zone_trim(range_tasks, rep_task.pitch)
        for loop in ctx.sweep.loops:
            for depth in ctx.sweep.depths:
                for rate in rates:
                    params = EncodingParams(rate, depth, trim_s, ctx.sweep.dither, ctx.sweep.noise_shaping, loop)
                    options.append(_evaluate_zone_option(rep_task, range_tasks, params, ctx))
    return options


def build_zone_options(tasks: Sequence[PitchTask], ctx: EvalContext) -> _ZoneOptions:
    """Score every contiguous pitch range ``[i, j)`` -- the menu the partition+allocation DP chooses from.

    This is the expensive step (an encode + reconstruction score per representative, encoding and
    covered pitch); the DP that consumes it is cheap.
    """
    count = len(tasks)
    options: _ZoneOptions = {}
    for i in range(count):
        for j in range(i + 1, count + 1):
            options[(i, j)] = tuple(_zone_options(tasks[i:j], ctx))
    return options


def zone_hull(options: Sequence[ZoneOption]) -> list[ZoneOption]:
    """A zone's byte-vs-distortion frontier over its ``(representative, encoding)`` options.

    Delegates to the shared :func:`optisample.optimize.operating_points.lower_convex_hull`; kept as a
    named entry point because this is where a zone's representative selection becomes visible (each
    hull vertex is the best member-plus-encoding at its byte level). Used for reporting.
    """
    return lower_convex_hull(options)
