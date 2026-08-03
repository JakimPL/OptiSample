from __future__ import annotations

from collections.abc import Sequence

from optisample.artifacts.serialize import Frozen
from optisample.calibrate.ranking import ListeningPair, PairAxis, Rendition, Side


class RenditionRecord(Frozen):
    """One side of a blinded pair, decoded: the encoding it was stored under and what it earned.

    ``loop_index`` names the region of the recording the sample was stored around, and is absent where
    the encoding keeps the played span as it was recorded.
    """

    side: Side
    target_rate: int
    depth_bits: int
    compress: bool
    loop_index: int | None
    stored_bytes: int
    distortion: float


class ListeningPairRecord(Frozen):
    """One question a listening set asks, with the answer the composite already gives it.

    ``directory`` is where the audio sits, named so a listing reads in the order the pairs are met and
    carries nothing of what any of them is about. ``composite_side`` is the side the metric calls closer
    to the recording, which is the claim a label either confirms or overturns, and ``margin`` how far
    apart it puts the two -- a pair the metric all but ties is where a label is worth most.
    """

    directory: str
    axis: PairAxis
    pitch: int
    key: str
    velocity: int
    duration_s: float
    first: RenditionRecord
    second: RenditionRecord
    composite_side: Side
    margin: float


class RankingSetDocument(Frozen):
    """What a listening set is, held apart from the audio a listener meets.

    Reading this decodes the set: every pair, both sides, and the ranking the composite gives them. The
    labels are collected beside it, so what a listener answers stays independent of what the metric
    already claims.
    """

    instrument_id: str
    sample_rate: int
    seed: int
    priced_encodings: int
    pairs: list[ListeningPairRecord]


def _rendition_record(rendition: Rendition, side: Side) -> RenditionRecord:
    """One member of a pair as the manifest states it."""
    return RenditionRecord(
        side=side,
        target_rate=rendition.params.target_rate,
        depth_bits=rendition.params.depth_bits,
        compress=rendition.params.compress,
        loop_index=rendition.params.loop_index,
        stored_bytes=rendition.stored_bytes,
        distortion=rendition.distortion,
    )


def pair_directory(pair: ListeningPair, index: int) -> str:
    """Where one pair's audio sits: its place in the set, then the recording both sides were encoded from.

    The question the pair asks is left to the manifest, so a listener browsing the folders meets each
    pair knowing which recording it came from and nothing about which degradation it holds.
    """
    return f"{index:03d}_{pair.clip.key.label}"


def pair_record(pair: ListeningPair, index: int) -> ListeningPairRecord:
    """One pair as the manifest states it, decoding both sides and the composite's own call."""
    return ListeningPairRecord(
        directory=pair_directory(pair, index),
        axis=pair.axis,
        pitch=pair.clip.pitch,
        key=pair.clip.key.label,
        velocity=pair.clip.velocity,
        duration_s=pair.clip.event.duration_s,
        first=_rendition_record(pair.first, Side.A),
        second=_rendition_record(pair.second, Side.B),
        composite_side=pair.composite_side,
        margin=pair.margin,
    )


def ranking_document(
    pairs: Sequence[ListeningPair],
    *,
    instrument_id: str,
    sample_rate: int,
    seed: int,
    priced_encodings: int,
) -> RankingSetDocument:
    """The manifest written beside a listening set's audio."""
    return RankingSetDocument(
        instrument_id=instrument_id,
        sample_rate=sample_rate,
        seed=seed,
        priced_encodings=priced_encodings,
        pairs=[pair_record(pair, index) for index, pair in enumerate(pairs)],
    )
