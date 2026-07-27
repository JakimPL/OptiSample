import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
from optisample.music import semitone_ratio
from optisample.optimize.operating_points import lower_convex_hull
from optisample.optimize.plans.grouped import ZoneOption
from optisample.optimize.reduce.bandwidth import ClipDemand, ProxyGrid, narrowed_params, proxy_grid
from optisample.optimize.tasks import EvalContext, PitchTask, score_reconstruction
from optisample.progress import ProgressSink

_Range = tuple[int, int]  # half-open [i, j) index range into the ordered pitch tasks
_ZoneOptions = dict[_Range, tuple[ZoneOption, ...]]
_GridKey = tuple[int, float]  # representative pitch, the stored length a zone holds its sample for
_StoredKey = tuple[int, EncodingParams]  # representative pitch, the encoding storing it
_ScoreKey = tuple[int, EncodingParams, int]  # representative pitch, encoding, the key reconstructed

_SEED_BYTES: Final = 8  # digest width the dither stream of one encode identity starts from
_NO_TRANSPOSE: Final = 0  # a representative at the top of its zone plays every other key downward
_ZONE_LABEL: Final = "Scoring pitch zones"


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
    """What a zone asks of the sample rooted at ``representative``: its length, transpose and reach."""
    return ClipDemand(
        trim_s=_zone_trim(range_tasks, representative),
        delta_semitones=_zone_delta(range_tasks, representative),
        key_count=len(range_tasks),
    )


@dataclass
class _ZoneScorer:
    """Prices and scores zone options for one run, reading back the work its candidate zones share.

    Pricing a representative's grid depends on that representative and the length its sample is held
    for, so the priced grid is kept for whichever other zones hold the same recording that long -- the
    transpose and the key count each zone adds are arithmetic over it. The encode and each covered key's
    reconstruction are kept under ``memoize``, which is also what makes them reusable: the dither then
    comes from a seed fixed by ``(representative, encoding)``, so one identity scores the same wherever
    the enumeration reaches it. Left off, every encode draws from the run's single shared stream in
    enumeration order, and each zone is scored on its own.
    """

    context: EvalContext
    grids: dict[_GridKey, ProxyGrid] = field(default_factory=dict, init=False)
    stored: dict[_StoredKey, StoredSample] = field(default_factory=dict, init=False)
    distortions: dict[_ScoreKey, float] = field(default_factory=dict, init=False)

    @property
    def memoize(self) -> bool:
        """Whether a scored encode is kept and read back in the other zones that share it."""
        return self.context.grouping.memoize

    def _dither(self, root_pitch: int, params: EncodingParams) -> np.random.Generator:
        """Where one encode draws its dither: a stream its own identity fixes, or the run's shared one."""
        if not self.memoize:
            return self.context.rng

        identity = repr((self.context.seed, root_pitch, params)).encode()
        digest = hashlib.blake2b(identity, digest_size=_SEED_BYTES).digest()
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def _encode(self, rep_task: PitchTask, params: EncodingParams) -> StoredSample:
        encode_context = EncodeContext(
            root_pitch=rep_task.pitch,
            config=self.context.encode,
            rng=self._dither(rep_task.pitch, params),
        )
        return encode(rep_task.representative, self.context.sample_rate, params, encode_context)

    def priced_grid(self, rep_task: PitchTask, trim_s: float) -> ProxyGrid:
        """``rep_task``'s recording priced across the stored grid at ``trim_s``, as every zone prices it.

        This is where the bandwidth pre-pass spends its time inside grouping: one encode and one
        composite evaluation per grid entry. What it measures follows from the recording and the length
        held, and a zone's stored length is the longest note it covers stretched by the transpose that
        reaches it, so the many zones landing on the same length share one priced grid.
        """
        key = (rep_task.pitch, trim_s)
        if key not in self.grids:
            self.grids[key] = proxy_grid(rep_task.representative, trim_s, self.context)

        return self.grids[key]

    def shortlist(self, rep_task: PitchTask, demand: ClipDemand) -> tuple[EncodingParams, ...]:
        """The encodings worth scoring for ``rep_task`` under ``demand``.

        Prices the grid for the length ``demand`` holds the sample (:meth:`priced_grid`), then lets the
        transpose and the key count narrow it, which is arithmetic over points already measured.
        """
        return narrowed_params(self.priced_grid(rep_task, demand.trim_s), demand, self.context)

    def stored_sample(self, rep_task: PitchTask, params: EncodingParams) -> StoredSample:
        """``rep_task``'s recording stored with ``params``, as every zone rooted there stores it."""
        if not self.memoize:
            return self._encode(rep_task, params)

        key = (rep_task.pitch, params)
        if key not in self.stored:
            self.stored[key] = self._encode(rep_task, params)

        return self.stored[key]

    def key_distortion(self, stored: StoredSample, params: EncodingParams, task: PitchTask) -> float:
        """How well ``task``'s notes reconstruct from ``stored``, as in every zone covering that key."""
        if not self.memoize:
            return score_reconstruction(stored, task, self.context)

        key = (stored.root_pitch, params, task.pitch)
        if key not in self.distortions:
            self.distortions[key] = score_reconstruction(stored, task, self.context)

        return self.distortions[key]

    def option(
        self,
        rep_task: PitchTask,
        range_tasks: Sequence[PitchTask],
        params: EncodingParams,
    ) -> ZoneOption:
        """Store ``rep_task``'s recording with ``params`` and score it reconstructing every key in the zone.

        The zone's total distortion is the usage-weighted sum of each covered key's reconstruction from
        this one stored sample (repitched to that key), so a distant key that the representative serves
        poorly costs the option here rather than being averaged away.
        """
        stored = self.stored_sample(rep_task, params)
        distortion = sum(task.weight * self.key_distortion(stored, params, task) for task in range_tasks)
        return ZoneOption(
            rep_task.pitch,
            params,
            self.context.storage.sample_bytes(frames=stored.frames, depth=stored.depth),
            distortion,
            stored.frames,
        )

    def zone_options(self, range_tasks: Sequence[PitchTask]) -> tuple[ZoneOption, ...]:
        """Every ``(representative, encoding)`` for one candidate zone, with its cost and total distortion.

        Enumerates the outer product of representative (each covered key's own recording, the k-medoids
        candidates) and the encodings the bandwidth pre-pass leaves in the running for the zone's demand.
        """
        return tuple(
            self.option(rep_task, range_tasks, params)
            for rep_task in range_tasks
            for params in self.shortlist(rep_task, _zone_demand(range_tasks, rep_task.pitch))
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


def build_zone_options(tasks: Sequence[PitchTask], context: EvalContext, progress: ProgressSink) -> _ZoneOptions:
    """Score every candidate pitch zone -- the menu the partition+allocation DP chooses from.

    This is the expensive step (an encode + reconstruction score per representative, encoding and
    covered pitch); the DP that consumes it is cheap. The candidates are the runs of keys inside
    ``reduce.grouping.max_zone_semitones`` of each other, each priced over the encodings the bandwidth
    pre-pass leaves it, and each scored once for every zone that shares the score.

    A wide zone costs more than a narrow one, so the reported progress runs ahead of the elapsed share
    early in each key's run of candidates and settles as the whole keyboard averages out.
    """
    scorer = _ZoneScorer(context)
    ranges = list(_capped_ranges(tasks, context.grouping.max_zone_semitones))
    return {
        (start, stop): scorer.zone_options(tasks[start:stop])
        for start, stop in progress.track(ranges, label=_ZONE_LABEL, total=len(ranges))
    }


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
