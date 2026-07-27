from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final

from optisample.dsp.surrogate import EncodingParams
from optisample.music import semitone_ratio
from optisample.optimize.grouping.stores import (
    StoredEncoding,
    StoredKey,
    StoredScore,
    StoreRequest,
    score_stores,
)
from optisample.optimize.operating_points import lower_convex_hull
from optisample.optimize.plans.grouped import ZoneOption
from optisample.optimize.reduce.bandwidth import ClipDemand, ProxyGrid, narrowed_params, proxy_grid
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import EvalContext, PitchTask
from optisample.progress import ProgressSink

_Range = tuple[int, int]  # half-open [i, j) index range into the ordered pitch tasks
_ZoneOptions = dict[_Range, tuple[ZoneOption, ...]]
_ZoneKey = tuple[_Range, SampleKey]  # a candidate zone and the recording of one of its keys it would store
_GridKey = tuple[SampleKey, float]  # the recording stored, and the length a zone holds it for

_NO_TRANSPOSE: Final = 0  # a representative at the top of its zone plays every other key downward
_NARROW_LABEL: Final = "Narrowing pitch zones"


def _zone_trim(range_tasks: Sequence[PitchTask], representative: int) -> float:
    """Stored duration the representative needs so every covered key's longest note plays in full.

    Playing key ``p`` from a sample rooted at ``representative`` runs it at ``2**((p - rep)/12)`` times
    speed, so a higher key consumes stored frames faster and needs a proportionally longer sample.
    """
    return max(task.max_duration_s * semitone_ratio(task.pitch - representative) for task in range_tasks)


def _zone_delta(range_tasks: Sequence[PitchTask], representative: int) -> int:
    """Widest upward transpose the representative plays at -- the interval its stored band must survive.

    Keys below the representative play the stored sample slower, landing its content lower than it was
    recorded, so the top of the zone alone decides how much stored bandwidth stays audible.
    """
    return max(_NO_TRANSPOSE, max(task.pitch for task in range_tasks) - representative)


def _zone_demand(range_tasks: Sequence[PitchTask], representative: int, per_key_bytes: int) -> ClipDemand:
    """What a zone asks of the sample rooted at ``representative``: its length, transpose and budget.

    The keys the one sample stands for pool their shares of the budget, so a zone covering a dozen keys
    shortlists around a dozen times what a single key affords.
    """
    return ClipDemand(
        trim_s=_zone_trim(range_tasks, representative),
        delta_semitones=_zone_delta(range_tasks, representative),
        byte_target=per_key_bytes * len(range_tasks),
    )


def _capped_ranges(tasks: Sequence[PitchTask], max_semitones: int) -> Iterator[_Range]:
    """Every run of neighbouring keys spanning at most ``max_semitones``, as half-open index ranges.

    The cap states how far the material lets one recording reach: a wider zone asks a sample to stand in
    further from its root than repitching holds up over, and dropping those ranges is what keeps the
    candidate zones linear in the keyboard span. A single key spans nothing, so each key remains its own
    candidate zone and some partition always exists.
    """
    for start, lowest in enumerate(tasks):
        for stop in range(start + 1, len(tasks) + 1):
            if tasks[stop - 1].pitch - lowest.pitch > max_semitones:
                break

            yield start, stop


@dataclass
class _GridCache:
    """Recordings priced across the stored grid, kept for whichever zones hold one the same length.

    Pricing depends on the recording and the length it is held for, so the priced grid is kept under
    that pair. A zone's own transpose and key count then narrow it, which is arithmetic over points
    already measured.
    """

    context: EvalContext
    grids: dict[_GridKey, ProxyGrid] = field(default_factory=dict, init=False)

    def shortlist(self, rep_task: PitchTask, demand: ClipDemand) -> tuple[EncodingParams, ...]:
        """The encodings worth scoring for ``rep_task`` under ``demand``."""
        key = (rep_task.representative_key, demand.trim_s)
        if key not in self.grids:
            self.grids[key] = proxy_grid(rep_task.representative, demand.trim_s, self.context)

        return narrowed_params(self.grids[key], demand, self.context)


def zone_shortlists(
    tasks: Sequence[PitchTask],
    ranges: Sequence[_Range],
    context: EvalContext,
    progress: ProgressSink,
) -> dict[_ZoneKey, tuple[EncodingParams, ...]]:
    """The encodings each candidate zone would consider for each of its keys it might store.

    The bandwidth pre-pass runs here, ahead of any scoring, so the work the scoring stage is given is
    known before it starts. Many zones hold the same recording for the same length, and those share one
    priced grid (:class:`_GridCache`).
    """
    cache = _GridCache(context)
    shortlists: dict[_ZoneKey, tuple[EncodingParams, ...]] = {}
    for start, stop in progress.track(ranges, label=_NARROW_LABEL, total=len(ranges)):
        range_tasks = tasks[start:stop]
        for rep_task in range_tasks:
            demand = _zone_demand(range_tasks, rep_task.pitch, context.byte_target)
            shortlists[((start, stop), rep_task.representative_key)] = cache.shortlist(rep_task, demand)

    return shortlists


def store_requests(
    tasks: Sequence[PitchTask],
    shortlists: dict[_ZoneKey, tuple[EncodingParams, ...]],
) -> tuple[StoreRequest, ...]:
    """Gather the shortlists into one workload per stored recording: what to store, and who it serves.

    A candidate zone asks one of its keys' recordings for a shortlisted encoding and expects every key it
    covers reconstructed from it, and the zones sharing that recording overlap heavily in both. Pooling
    their asks per recording is what leaves each encode and each key's reconstruction stated once, so the
    scoring stage runs exactly the work the whole set of candidate zones needs.
    """
    by_pitch = {task.pitch: task for task in tasks}
    by_key = {task.representative_key: task for task in tasks}
    asked: dict[SampleKey, dict[EncodingParams, set[int]]] = {}
    for (span, stored_key), shortlist in shortlists.items():
        covered = [task.pitch for task in tasks[span[0] : span[1]]]
        wanted = asked.setdefault(stored_key, {})
        for params in shortlist:
            wanted.setdefault(params, set()).update(covered)

    return tuple(
        StoreRequest(
            representative=by_key[stored_key],
            served=tuple(by_pitch[pitch] for pitch in sorted({pitch for keys in wanted.values() for pitch in keys})),
            encodings=tuple(StoredEncoding(params=params, keys=tuple(sorted(keys))) for params, keys in wanted.items()),
        )
        for stored_key, wanted in sorted(asked.items())
    )


def _zone_option(
    range_tasks: Sequence[PitchTask],
    rep_task: PitchTask,
    params: EncodingParams,
    score: StoredScore,
) -> ZoneOption:
    """One way to realize a zone: ``rep_task``'s recording stored as ``params``, serving every key.

    The zone's total distortion is the usage-weighted sum of each covered key's reconstruction from this
    one stored sample (repitched to that key), so a distant key that the representative serves poorly
    costs the option here rather than being averaged away.
    """
    return ZoneOption(
        rep_task.pitch,
        params,
        score.stored_bytes,
        sum(task.weight * score.distortions[task.pitch] for task in range_tasks),
        score.frames,
    )


def _zone_options(
    range_tasks: Sequence[PitchTask],
    span: _Range,
    shortlists: dict[_ZoneKey, tuple[EncodingParams, ...]],
    scores: dict[StoredKey, StoredScore],
) -> tuple[ZoneOption, ...]:
    """Every ``(representative, encoding)`` for one candidate zone, with its cost and total distortion.

    Enumerates the outer product of representative (each covered key's own recording, the k-medoids
    candidates) and the encodings the bandwidth pre-pass left in the running for the zone's demand.
    """
    return tuple(
        _zone_option(range_tasks, rep_task, params, scores[(rep_task.representative_key, params)])
        for rep_task in range_tasks
        for params in shortlists[(span, rep_task.representative_key)]
    )


def build_zone_options(
    tasks: Sequence[PitchTask],
    context: EvalContext,
    *,
    workers: int,
    progress: ProgressSink,
) -> _ZoneOptions:
    """Score every candidate pitch zone -- the menu the partition+allocation DP chooses from.

    Runs in three passes, so the expensive one knows its work before it starts and each piece of it is
    computed once. :func:`zone_shortlists` narrows each candidate zone's stored grid;
    :func:`store_requests` pools those asks into one workload per representative; and
    :func:`~optisample.optimize.grouping.stores.score_stores` encodes and reconstructs them, sharing the
    representatives across processes. Assembling the zones from the results is then addition.
    """
    ranges = list(_capped_ranges(tasks, context.grouping.max_zone_semitones))
    shortlists = zone_shortlists(tasks, ranges, context, progress)
    scores = score_stores(store_requests(tasks, shortlists), context, workers=workers, progress=progress)
    return {span: _zone_options(tasks[span[0] : span[1]], span, shortlists, scores) for span in ranges}


def zone_starts(options: _ZoneOptions, count: int) -> list[tuple[int, ...]]:
    """Indexed by stop, the starts of the candidate zones ending there -- the ranges allocation may use.

    The span cap leaves the candidate ranges sparse, so the allocation DP walks the zones that were
    scored. Every key is its own candidate zone, so each stop is reachable from at least one start.
    """
    starts: list[list[int]] = [[] for _ in range(count + 1)]
    for start, stop in options:
        starts[stop].append(start)

    return [tuple(sorted(group)) for group in starts]


def zone_hull(options: Sequence[ZoneOption]) -> list[ZoneOption]:
    """A zone's byte-vs-distortion frontier over its ``(representative, encoding)`` options.

    Delegates to the shared :func:`optisample.optimize.operating_points.lower_convex_hull`; kept as a
    named entry point because this is where a zone's representative selection becomes visible (each
    hull vertex is the best member-plus-encoding at its byte level). Used for reporting.
    """
    return lower_convex_hull(options)
