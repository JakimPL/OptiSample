from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Final

from optisample.dsp.surrogate import EncodingParams
from optisample.frontier import lower_convex_hull, pareto_frontier
from optisample.keys import SampleKey
from optisample.music import semitone_ratio
from optisample.optimize.grouping.stores import (
    ServedKey,
    StoredEncoding,
    StoredKey,
    StoredScore,
    StoreRequest,
    score_stores,
)
from optisample.optimize.plans.grouped import ZoneOption
from optisample.optimize.reduce.bandwidth import (
    ClipDemand,
    carried_storages,
    clip_band_hz,
    format_from_band,
    stored_encodings,
)
from optisample.optimize.tasks import EvalContext, PitchTask
from optisample.progress import ProgressSink

_Range = tuple[int, int]  # half-open [i, j) index range into the ordered pitch tasks
_ZoneOptions = dict[_Range, tuple[ZoneOption, ...]]
_ZoneKey = tuple[_Range, SampleKey]  # a candidate zone and the recording of one of its keys it would store
_BandKey = tuple[SampleKey, float]  # the recording stored, and the length a zone holds it for

# One layer's stretch of the axis, in pitch order. Every candidate zone lies inside a single segment, so a
# partition of the axis splits at each boundary and one stored sample answers for keys of one layer alone.
ZoneSegment = tuple[PitchTask, ...]

_NO_TRANSPOSE: Final = 0  # a representative at the top of its zone plays every other key downward
_NARROW_LABEL: Final = "Narrowing pitch zones"


def segment_offsets(segments: Sequence[ZoneSegment]) -> tuple[int, ...]:
    """Where each segment's keys begin once the segments are laid end to end, the total last."""
    return tuple(accumulate((len(segment) for segment in segments), initial=0))


def axis_tasks(segments: Sequence[ZoneSegment]) -> tuple[PitchTask, ...]:
    """Every segment's keys laid end to end -- the ordered axis a partition DP walks."""
    return tuple(task for segment in segments for task in segment)


def combined_options(segments: Sequence[ZoneSegment], options: Sequence[_ZoneOptions]) -> _ZoneOptions:
    """Each segment's scored zones on one axis, their ranges offset to where the segment sits on it."""
    return {
        (offset + start, offset + stop): zone_options
        for offset, segment_options in zip(segment_offsets(segments), options)
        for (start, stop), zone_options in segment_options.items()
    }


@dataclass(frozen=True)
class _Candidate:
    """One candidate zone: where it sits on the axis, and the layer it belongs to."""

    span: _Range
    segment: int


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


def _zone_demand(range_tasks: Sequence[PitchTask], representative: int) -> ClipDemand:
    """What a zone asks of the sample rooted at ``representative``: how long it is held, and its transpose.

    Both follow from the keys the one sample stands for, so a wider zone asks for a longer stored stretch
    and a band that survives reaching further up the keyboard.
    """
    return ClipDemand(
        trim_s=_zone_trim(range_tasks, representative),
        delta_semitones=_zone_delta(range_tasks, representative),
    )


def _capped_ranges(tasks: Sequence[PitchTask], max_semitones: int) -> Iterator[_Range]:
    """Every run of neighbouring keys spanning at most ``max_semitones``, as half-open index ranges.

    The cap states how far the material lets one recording reach: a wider zone asks a sample to stand in
    further from its root than repitching holds up over, and dropping those ranges is what keeps the
    candidate zones linear in the keyboard span. A single key spans nothing, so each key remains its own
    candidate zone and some partition always exists -- which is also what a cap of
    :data:`~optisample.config.reduce.NO_GROUPING` leaves, since any second key already spans more.
    """
    for start, lowest in enumerate(tasks):
        for stop in range(start + 1, len(tasks) + 1):
            if tasks[stop - 1].pitch - lowest.pitch > max_semitones:
                break

            yield start, stop


@dataclass
class _BandCache:
    """The band each recording occupies, kept for whichever zones hold that recording the same length.

    A recording's band depends on the stretch stored of it and nothing else, so it is measured once per
    recording and length. A zone's own transpose then settles the format from it, which is arithmetic.

    Which storages the recording is worth being offered as is a property of the recording alone
    (:func:`~optisample.optimize.reduce.bandwidth.carried_storages`), so it is read once beside the band
    and every zone reaching the same recording is answered from it.
    """

    context: EvalContext
    bands: dict[_BandKey, float] = field(default_factory=dict, init=False)
    storages: dict[SampleKey, tuple[bool, ...]] = field(default_factory=dict, init=False)

    def encodings(self, rep_task: PitchTask, demand: ClipDemand) -> tuple[EncodingParams, ...]:
        """The encodings worth scoring for ``rep_task`` under ``demand``."""
        key = (rep_task.representative_key, demand.trim_s)
        if key not in self.bands:
            self.bands[key] = clip_band_hz(
                rep_task.representative,
                demand.trim_s,
                self.context.sample_rate,
                self.context.bandwidth,
            )

        if rep_task.representative_key not in self.storages:
            self.storages[rep_task.representative_key] = carried_storages(rep_task.representative, self.context)

        stored = format_from_band(
            self.bands[key],
            demand,
            self.context,
            self.storages[rep_task.representative_key],
        )
        return stored_encodings(
            stored,
            self.context.sweep,
            sample_rate=self.context.sample_rate,
            trim_s=demand.trim_s,
            loops=rep_task.offered_loops,
        )


def candidate_zones(segments: Sequence[ZoneSegment], max_semitones: int) -> list[_Candidate]:
    """Every candidate zone across the axis: each segment's capped runs, placed where the segment sits.

    Ranges are taken inside one segment at a time, so no candidate spans two layers and a partition of
    the axis is a partition of each layer's keys.
    """
    zones: list[_Candidate] = []
    for index, (segment, offset) in enumerate(zip(segments, segment_offsets(segments))):
        for start, stop in _capped_ranges(segment, max_semitones):
            zones.append(_Candidate(span=(offset + start, offset + stop), segment=index))

    return zones


def zone_encodings(
    axis: Sequence[PitchTask],
    zones: Sequence[_Candidate],
    context: EvalContext,
    progress: ProgressSink,
) -> dict[_ZoneKey, tuple[EncodingParams, ...]]:
    """The encodings each candidate zone would consider for each of its keys it might store.

    The bandwidth pre-pass runs here, ahead of any scoring, so the work the scoring stage is given is
    known before it starts. Many zones hold the same recording for the same length, and those read its
    band from one measurement (:class:`_BandCache`) whatever layer they belong to.
    """
    cache = _BandCache(context)
    encodings: dict[_ZoneKey, tuple[EncodingParams, ...]] = {}
    for zone in progress.track(zones, label=_NARROW_LABEL, total=len(zones)):
        range_tasks = axis[zone.span[0] : zone.span[1]]
        for rep_task in range_tasks:
            demand = _zone_demand(range_tasks, rep_task.pitch)
            encodings[(zone.span, rep_task.representative_key)] = cache.encodings(rep_task, demand)

    return encodings


def store_requests(
    axis: Sequence[PitchTask],
    encodings: dict[_ZoneKey, tuple[EncodingParams, ...]],
) -> tuple[StoreRequest, ...]:
    """Gather the zones' asks into one workload per stored recording: what to store, and who it serves.

    A candidate zone asks one of its keys' recordings for an encoding and expects every key it covers
    reconstructed from it, and the zones sharing that recording overlap heavily in both. Pooling their
    asks per recording is what leaves each encode and each key's reconstruction stated once, so the
    scoring stage runs exactly the work the whole set of candidate zones needs. Layers pool together too:
    two bands whose loudest dynamics reach the same recording store it once between them.
    """
    by_key = {task.representative_key: task for task in axis}
    asked: dict[SampleKey, dict[EncodingParams, set[int]]] = {}
    for (span, stored_key), zone_params in encodings.items():
        covered = range(span[0], span[1])
        wanted = asked.setdefault(stored_key, {})
        for params in zone_params:
            wanted.setdefault(params, set()).update(covered)

    return tuple(
        StoreRequest(
            representative=by_key[stored_key],
            served=tuple(
                ServedKey(position, axis[position])
                for position in sorted({position for served in wanted.values() for position in served})
            ),
            encodings=tuple(
                StoredEncoding(params=params, positions=tuple(sorted(served))) for params, served in wanted.items()
            ),
        )
        for stored_key, wanted in sorted(asked.items())
    )


def _zone_option(
    zone: _Candidate,
    axis: Sequence[PitchTask],
    rep_task: PitchTask,
    params: EncodingParams,
    score: StoredScore,
) -> ZoneOption:
    """One way to realize a zone: ``rep_task``'s recording stored as ``params``, serving every key.

    The zone's total distortion is the objective-weighted sum of each covered key's reconstruction from
    this one stored sample (repitched to that key), so a distant key that the representative serves
    poorly costs the option here rather than being averaged away.
    """
    start, stop = zone.span
    return ZoneOption(
        rep_task.pitch,
        params,
        score.stored_bytes,
        sum(axis[position].objective_weight * score.distortions[position] for position in range(start, stop)),
        score.frames,
    )


def _zone_options(
    zone: _Candidate,
    axis: Sequence[PitchTask],
    encodings: dict[_ZoneKey, tuple[EncodingParams, ...]],
    scores: dict[StoredKey, StoredScore],
) -> tuple[ZoneOption, ...]:
    """What one candidate zone offers the allocation: its byte-vs-distortion frontier, cheapest first.

    Enumerates the outer product of representative (each covered key's own recording, the k-medoids
    candidates) and the encodings the bandwidth pre-pass settled for the zone's demand, then keeps the
    ones an allocation may choose (:func:`~optisample.frontier.pareto_frontier`). A wide zone offers one
    pair per covered key per encoding and the allocation walks every option it is handed, so leaving each
    byte level to the representative that reads closest at it is what the partition DP is given.
    """
    return tuple(
        pareto_frontier(
            [
                _zone_option(zone, axis, rep_task, params, scores[(rep_task.representative_key, params)])
                for rep_task in axis[zone.span[0] : zone.span[1]]
                for params in encodings[(zone.span, rep_task.representative_key)]
            ]
        )
    )


def build_zone_options(
    segments: Sequence[ZoneSegment],
    context: EvalContext,
    *,
    workers: int,
    progress: ProgressSink,
) -> tuple[_ZoneOptions, ...]:
    """Score every candidate pitch zone -- the menu the partition+allocation DP chooses from.

    Runs in three passes, so the expensive one knows its work before it starts and each piece of it is
    computed once. :func:`zone_encodings` settles each candidate zone's stored format;
    :func:`store_requests` pools those asks into one workload per representative; and
    :func:`~optisample.optimize.grouping.stores.score_stores` encodes and reconstructs them, sharing the
    representatives across processes. Assembling the zones from the results is then addition.

    Answers one option table per segment, keyed by ranges into that segment's own keys, so a caller
    holding more segments than any one partition uses reads back exactly the layers it wants.
    """
    axis = axis_tasks(segments)
    zones = candidate_zones(segments, context.grouping.max_zone_semitones)
    encodings = zone_encodings(axis, zones, context, progress)
    scores = score_stores(store_requests(axis, encodings), context, workers=workers, progress=progress)
    tables: tuple[_ZoneOptions, ...] = tuple({} for _ in segments)
    offsets = segment_offsets(segments)
    for zone in zones:
        offset = offsets[zone.segment]
        tables[zone.segment][(zone.span[0] - offset, zone.span[1] - offset)] = _zone_options(
            zone, axis, encodings, scores
        )

    return tables


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

    Delegates to the shared :func:`optisample.frontier.lower_convex_hull`; kept as a
    named entry point because this is where a zone's representative selection becomes visible (each
    hull vertex is the best member-plus-encoding at its byte level). Used for reporting.
    """
    return lower_convex_hull(options)
