from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final

import numpy as np

from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import EvalContext, PitchTask, score_reconstruction
from optisample.parallel import map_workers
from optisample.progress import ProgressSink

StoredKey = tuple[SampleKey, EncodingParams]  # the recording stored, and the encoding it is stored under

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
    """One encoding a recording is stored under, and the keys the zones asking for it route to it."""

    params: EncodingParams
    keys: tuple[int, ...]


@dataclass(frozen=True)
class StoreRequest:
    """One representative's whole workload: every encoding asked of its recording, and who each serves.

    Collecting a representative's encodes and reconstructions into a single request is what lets each of
    them be computed exactly once however the work is shared out: a stored sample belongs to one
    representative and a reconstruction to one stored sample, so each lands in exactly one worker.
    ``served`` carries the pitch tasks the keys name, which is everything scoring them reads.
    """

    representative: PitchTask
    served: tuple[PitchTask, ...]
    encodings: tuple[StoredEncoding, ...]

    @property
    def stored_key(self) -> SampleKey:
        """The recording this request stores, which is the one its representative was chosen for."""
        return self.representative.representative_key

    @property
    def tasks_by_pitch(self) -> dict[int, PitchTask]:
        """The keys this request reconstructs, reachable by the pitch each encoding names them with."""
        return {task.pitch: task for task in self.served}


@dataclass(frozen=True)
class StoredScore:
    """What one stored sample costs to keep and how well each key it serves reconstructs from it.

    ``distortions`` holds the per-unit-weight reconstruction distortion of each key, keyed by pitch, so
    a zone weights the keys it covers by its own material and leaves the rest to the zones that cover
    them.
    """

    stored_bytes: int
    frames: int
    distortions: dict[int, float]


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
    per key that encoding is asked about. Both are settled by the recording, the encoding and the key
    alone, which is what makes a representative's workload a unit of work in its own right.
    """
    served = request.tasks_by_pitch
    scores: dict[EncodingParams, StoredScore] = {}
    for encoding in request.encodings:
        stored = _stored_sample(request, encoding.params, context)
        scores[encoding.params] = StoredScore(
            stored_bytes=context.storage.sample_bytes(frames=stored.frames, depth=stored.depth),
            frames=stored.frames,
            distortions={pitch: score_reconstruction(stored, served[pitch], context) for pitch in encoding.keys},
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
