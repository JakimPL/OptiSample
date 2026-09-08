from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from optisample.artifacts.serialize import Frozen
from optisample.carrier.instrument import CarrierInstrument
from optisample.carrier.store import StoredCarrier
from optisample.optimize.layers.bands import VelocityBand

MANIFEST_VERSION: Final = 1  # the manifest shape stated out loud, so a consumer reads the one it knows


class CarrierSampleRecord(Frozen):
    """One waveform a written instrument holds: the take it came from, and how it was stored.

    ``members`` is how many recordings the group this take stands for gathered, which says how much of the
    material one waveform answers for. ``gain`` is the step the format keeps beside it, which is where the
    balance between the takes of one instrument is restored.
    """

    key: str
    pitch: int
    velocity: int
    members: int
    frames: int
    rate: int
    depth: int
    gain: int
    looped: bool


class CarrierLayerRecord(Frozen):
    """One velocity band written as one instrument, and what sharing a single curve cost its takes.

    ``dispersion_db`` is how far the furthest take stands from the envelope they all play through, so a
    band reading high is one whose recordings decline at rates of their own and would be better told
    apart. ``files`` names the instrument written for each format, keyed by the extension it carries.
    """

    band: str
    lowest: int
    highest: int
    dispersion_db: float
    stored_bytes: int
    files: dict[str, str]
    samples: list[CarrierSampleRecord]


class ClusteredDocument(Frozen):
    """The manifest a clustered carrier run writes: which takes were chosen, and what they were written as.

    A tracker keymap names a key and nothing else, so the dynamics an instrument answers live here rather
    than in the instrument file, exactly as a bank states them. ``tempo`` is the clock the volume envelopes
    were fitted against, since an instrument file carries no clock of its own.
    """

    version: int
    instrument_id: str
    stage: str
    tempo: int
    groups: int
    bands: list[CarrierLayerRecord]


def _sample_record(carrier: StoredCarrier, gain: int, members: int) -> CarrierSampleRecord:
    source = carrier.source
    stored = carrier.stored
    return CarrierSampleRecord(
        key=source.key.label,
        pitch=source.root_pitch,
        velocity=source.key.velocity,
        members=members,
        frames=stored.frames,
        rate=stored.sample_rate,
        depth=stored.depth,
        gain=gain,
        looped=stored.loop is not None,
    )


def layer_record(
    band: VelocityBand,
    built: CarrierInstrument,
    *,
    members: Sequence[int],
    files: dict[str, str],
) -> CarrierLayerRecord:
    """One band's instrument as the manifest states it, in the order its unit numbers the samples."""
    return CarrierLayerRecord(
        band=band.label,
        lowest=band.lowest,
        highest=band.highest,
        dispersion_db=built.dispersion_db,
        stored_bytes=built.stored_bytes,
        files=files,
        samples=[
            _sample_record(carrier, gain, count) for carrier, gain, count in zip(built.stored, built.gains, members)
        ],
    )


def clustered_document(
    instrument_id: str,
    *,
    stage: str,
    tempo: int,
    groups: int,
    layers: Sequence[CarrierLayerRecord],
) -> ClusteredDocument:
    """The manifest one clustered carrier run is read back through."""
    return ClusteredDocument(
        version=MANIFEST_VERSION,
        instrument_id=instrument_id,
        stage=stage,
        tempo=tempo,
        groups=groups,
        bands=list(layers),
    )
