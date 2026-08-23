from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.carrier.compress import held_sources
from optisample.carrier.shape import WrittenShape, written_shape
from optisample.carrier.source import CarrierSource
from optisample.carrier.store import CarrierSettings, StoredCarrier, store_carrier
from optisample.io.tracker.loop import stored_loop
from optisample.io.tracker.target import ExportTarget, balanced_gains, sample_label
from optisample.music import sounded_note
from optisample.optimize.export.coverage import covered_routing
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, Keymap, routed_keymap
from trackmod.core.instruments.unit import InstrumentUnit
from trackmod.core.notes.pitch import Note
from trackmod.core.samples.sample import Sample

_BITS_PER_BYTE: Final = 8


@dataclass(frozen=True)
class CarrierInstrument:
    """A set of recordings written as one instrument: the waveforms, and what the shared curve cost them.

    ``unit`` is the instrument a format writes as a standalone file. ``stored`` and ``gains`` stand in the
    order the unit numbers its samples, so a caller reports each waveform beside the step written for it.
    """

    unit: InstrumentUnit
    stored: tuple[StoredCarrier, ...]
    gains: tuple[int, ...]
    shape: WrittenShape

    @property
    def dispersion_db(self) -> float:
        """How far the furthest source stands from the one envelope this instrument plays them through."""
        return self.shape.dispersion_db

    @property
    def stored_bytes(self) -> int:
        """What the waveforms occupy, which is what this many samples cost before any record is counted."""
        return sum(carrier.stored.frames * carrier.stored.depth_bits // _BITS_PER_BYTE for carrier in self.stored)


def _routing(sources: Sequence[CarrierSource], target: ExportTarget) -> dict[Note, KeyAssignment]:
    """Each source's own key routed to the waveform holding it, sounding the pitch that key is named for.

    A source states one key of its own, so what this builds is the keyboard as the set was recorded.
    Widening it to every key the format numbers is what
    :func:`~optisample.optimize.export.coverage.covered_routing` then does, handing each stretch between
    two recordings to whichever of them is nearer.
    """
    routing: dict[Note, KeyAssignment] = {}
    for sample, source in enumerate(sources):
        key = target.key(source.root_pitch)
        routing[key] = KeyAssignment(sample=sample, note=sounded_note(key, key))

    return routing


def carrier_keymap(sources: Sequence[CarrierSource], target: ExportTarget) -> Keymap:
    """Every key ``target`` numbers routed to the source nearest it, each sounding its own pitch."""
    return routed_keymap(covered_routing(_routing(sources, target), target))


def carrier_instrument(
    sources: Sequence[CarrierSource],
    *,
    instrument_id: str,
    name: str,
    settings: CarrierSettings,
) -> CarrierInstrument:
    """``sources`` written as one instrument whose samples hold timbre and whose envelope holds the level.

    The order the work runs in is the point of it. The split states each recording's level exactly, so a
    first shape is fitted from those levels (:func:`written_shape`) and states as much of them as a format
    carries. What it has no room for is read straight back off the flattened waveforms and held back at a
    ratio the format does carry (:func:`~optisample.carrier.compress.held_sources`), which leaves the set at
    a level a curve can follow. A second shape is fitted to that held-back level; each waveform is then the
    restated recording divided by what the second curve plays it down by
    (:func:`~optisample.carrier.store.store_carrier`); and the balance between the waveforms is restored
    last, on the step the format keeps beside each sample
    (:func:`~optisample.io.tracker.target.balanced_gains`). Two passes settle it -- the first states the
    level, the second holds back what was left over -- and nothing iterates past them.

    Sources are encoded in order from one seeded generator, so the bytes a set is written as reproduce.

    Raises:
        ValueError: when the sources were read at rates that differ, or when one is worth nothing to the
            instrument sharing the shape.
    """
    stated = written_shape(sources, target=settings.target, grid=settings.grid)
    held = held_sources(sources, stated.envelope, settings=settings)
    shape = written_shape(held, target=settings.target, grid=settings.grid)
    rng = np.random.default_rng(settings.seed)
    stored = tuple(store_carrier(source, shape.envelope, settings=settings, rng=rng) for source in held)
    gains = balanced_gains([carrier.playback_gain for carrier in stored], settings.target)
    samples = tuple(
        Sample(
            name=sample_label(instrument_id, pitch=carrier.source.root_pitch, velocity=carrier.source.key.velocity),
            pcm=carrier.stored.pcm,
            rate=carrier.stored.sample_rate,
            depth=carrier.stored.depth,
            gain=gain,
            loop=stored_loop(carrier.stored.loop),
        )
        for carrier, gain in zip(stored, gains)
    )
    return CarrierInstrument(
        unit=InstrumentUnit(
            instrument=Instrument(
                name=name,
                keymap=carrier_keymap(sources, settings.target),
                volume_envelope=shape.envelope,
            ),
            samples=samples,
        ),
        stored=stored,
        gains=gains,
        shape=shape,
    )
