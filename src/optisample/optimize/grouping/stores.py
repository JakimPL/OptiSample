from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Final

import numpy as np

from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
from optisample.optimize.reduce.events import EventIdentity
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import EvalContext, Event, PitchTask, score_event, weighted_distortion
from optisample.parallel import map_workers
from optisample.progress import ProgressSink

StoredKey = tuple[SampleKey, EncodingParams]  # the recording stored, and the encoding it is stored under

_ClassKey = tuple[int, EventIdentity]  # the key a note class sounds at, and the class scored there

STORE_LABEL: Final = "Scoring stored samples"

_SEED_BYTES: Final = 8  # digest width the dither stream of one encode identity starts from


def dither(seed: int, key: SampleKey, params: EncodingParams) -> np.random.Generator:
    """The stream one stored sample draws its dither from: the one its own identity fixes.

    Deriving the seed from the run entropy, the recording stored and the encoding gives each stored
    sample a stream of its own, so the sample one candidate zone prices reads the same wherever another
    zone reaches it and whichever process carried the work.
    """
    identity = repr((seed, key, params)).encode()
    digest = hashlib.blake2b(identity, digest_size=_SEED_BYTES).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class StoredEncoding:
    """One encoding a recording is stored under, and the axis positions the zones asking for it route to it."""

    params: EncodingParams
    positions: tuple[int, ...]


@dataclass(frozen=True)
class ServedKey:
    """One key a stored sample answers for: where it sits on the axis, and the notes it plays there.

    A key is named by its position rather than its pitch because a layered axis sounds the same pitch
    once per velocity band, each time with the band's own notes to reconstruct.
    """

    position: int
    task: PitchTask


@dataclass(frozen=True)
class StoreRequest:
    """One representative's whole workload: every encoding asked of its recording, and who each serves.

    Collecting a representative's encodes and reconstructions into a single request is what lets each of
    them be computed exactly once however the work is shared out: a stored sample belongs to one
    representative and a reconstruction to one stored sample, so each lands in exactly one worker.
    ``served`` carries the pitch tasks the positions name, which is everything scoring them reads.
    """

    representative: PitchTask
    served: tuple[ServedKey, ...]
    encodings: tuple[StoredEncoding, ...]

    @property
    def stored_key(self) -> SampleKey:
        """The recording this request stores, which is the one its representative was chosen for."""
        return self.representative.representative_key

    @property
    def tasks_by_position(self) -> dict[int, PitchTask]:
        """The keys this request reconstructs, reachable by the position each encoding names them with."""
        return {key.position: key.task for key in self.served}


@dataclass(frozen=True)
class StoredScore:
    """What one stored sample costs to keep and how well each key it serves reconstructs from it.

    ``distortions`` holds the per-unit-weight reconstruction distortion of each key, keyed by its axis
    position, so a zone weights the keys it covers by its own material and leaves the rest to the zones
    that cover them.
    """

    stored_bytes: int
    frames: int
    distortions: dict[int, float]


@dataclass
class _StoredScorer:
    """One stored sample's reconstructions, each note class it is asked about measured exactly once.

    A class's score is settled by the stored sample, the key it sounds at and the class itself, so two
    positions on the axis asking for the same class read one measurement. That is what makes a layered
    axis affordable: the velocity bands sharing a stored sample overlap in the classes they cover, and
    the overlap is measured once between them.
    """

    stored: StoredSample
    context: EvalContext
    measured: dict[_ClassKey, float] = field(default_factory=dict, init=False)

    def _fidelity(self, task: PitchTask, event: Event) -> float:
        """How well one note class reconstructs at ``task``'s key, measured on first ask and kept."""
        identity = (task.pitch, event.identity)
        if identity not in self.measured:
            self.measured[identity] = score_event(self.stored, event, pitch=task.pitch, context=self.context).fidelity

        return self.measured[identity]

    def distortion(self, task: PitchTask) -> float:
        """``task``'s reconstruction distortion per unit of weight, weighted the way the objective is."""
        return weighted_distortion(task, partial(self._fidelity, task))


def _stored_sample(request: StoreRequest, params: EncodingParams, context: EvalContext) -> StoredSample:
    """``request``'s recording stored under ``params``, rooted at the pitch it was recorded at."""
    encode_context = EncodeContext(
        root_pitch=request.stored_key.pitch,
        config=context.encode,
        rng=dither(context.seed, request.stored_key, params),
    )
    return encode(request.representative.representative, context.sample_rate, params, encode_context)


def score_request(request: StoreRequest, context: EvalContext) -> dict[EncodingParams, StoredScore]:
    """Store ``request``'s recording under every encoding asked of it and score the keys each one serves.

    This is where pitch-zone grouping spends its time: one encode per encoding, and one reconstruction
    per note class that encoding is asked about. Both are settled by the recording, the encoding and the
    class alone, which is what makes a representative's workload a unit of work in its own right.
    """
    served = request.tasks_by_position
    scores: dict[EncodingParams, StoredScore] = {}
    for encoding in request.encodings:
        stored = _stored_sample(request, encoding.params, context)
        scorer = _StoredScorer(stored, context)
        scores[encoding.params] = StoredScore(
            stored_bytes=context.storage.sample_bytes(frames=stored.frames, depth=stored.depth),
            frames=stored.frames,
            distortions={position: scorer.distortion(served[position]) for position in encoding.positions},
        )

    return scores


def score_stores(
    requests: Sequence[StoreRequest],
    context: EvalContext,
    *,
    workers: int,
    progress: ProgressSink,
) -> dict[StoredKey, StoredScore]:
    """Score every representative's stored samples, sharing the representatives across ``workers``.

    Each request is answered from its own recording and the keys it serves, and every encode draws from
    the stream :func:`dither` fixes for it, so the scores read the same however many processes carried
    them. Answers come back under the ``(recording, encoding)`` each stored sample is identified by.
    """
    answered = map_workers(
        partial(score_request, context=context),
        requests,
        workers=workers,
        label=STORE_LABEL,
        progress=progress,
    )
    return {
        (request.stored_key, params): score
        for request, scores in zip(requests, answered)
        for params, score in scores.items()
    }
