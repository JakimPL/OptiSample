from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from optisample.artifacts.serialize import Frozen, VelocityMapDocument
from optisample.optimize.layers.slots import ONE_SLOT, SlotLayout

MANIFEST_VERSION: Final = 2  # the manifest shape stated out loud, so a consumer reads the one it knows
_VELOCITY_AXIS: Final = "velocity"  # the axis name a selector spells its velocity band under
_PITCH_AXIS: Final = "pitch"  # the axis name a selector spells its run of keys under


class BandRecord(Frozen):
    """A stretch of one axis a layer answers, with both ends inside it."""

    low: int
    high: int


class SourceRecord(Frozen):
    """Which entry of the bank one layer's instrument is read out of."""

    file: str


class LayerRecord(Frozen):
    """One instrument of the bank: where it is read from, the notes it answers, and how it reads them.

    ``select`` states a band per axis the layer is picked by, and a note reaches the layer whose every
    stated band covers it. ``velocity_map`` carries the measured velocity->volume table the layer's own
    samples were stored under, so a note sounds at the volume its dynamic was measured to reach and the
    table travels with the waveforms it was measured from.
    """

    source: SourceRecord
    select: dict[str, BandRecord]
    velocity_map: VelocityMapDocument


class BankDocument(Frozen):
    """The manifest a player loads a written plan through: what the bank calls itself and what it plays.

    Layers stand in band order and each states the velocities it answers, so a note picks its instrument
    by the dynamic it was struck at and the instrument's own keymap then picks the sample. Every entry a
    layer names is held in the bank itself, so the manifest and the instruments it points at travel as
    one unit however the bank is copied or renamed.
    """

    version: int
    name: str
    layers: list[LayerRecord]


def _selector(layout: SlotLayout, index: int) -> dict[str, BandRecord]:
    """The bands one written instrument is picked by: its dynamics, and its keys where a band was split.

    A velocity band written as one instrument owns the whole keyboard, so its dynamics name it on their
    own and the selector states them alone. A band the format had room for only in several instruments
    states the run of keys each of them owns as well, since every one of those files answers every key it
    was filled over and the keymaps alone no longer tell them apart.
    """
    slot = layout.slots[index]
    select = {_VELOCITY_AXIS: BandRecord(low=slot.band.lowest, high=slot.band.highest)}
    if len(layout.layer_slots(slot.layer)) > ONE_SLOT:
        keys = layout.key_band(index)
        select[_PITCH_AXIS] = BandRecord(low=keys.lowest, high=keys.highest)

    return select


def _layer_record(layout: SlotLayout, index: int, entry: str, velocity_map: VelocityMapDocument) -> LayerRecord:
    return LayerRecord(
        source=SourceRecord(file=entry),
        select=_selector(layout, index),
        velocity_map=velocity_map,
    )


def bank_document(
    name: str,
    layout: SlotLayout,
    entries: Sequence[str],
    velocity_map: VelocityMapDocument,
) -> BankDocument:
    """The bank ``layout`` is played through: one layer per written instrument, in the order they were written.

    ``entries`` name the instruments inside the bank, in the same order as the slots. Each layer states
    the dynamics and, where its band was split across several instruments, the keys it answers, so the
    manifest says out loud what the allocation decided and a note reaches the instrument meant for it.
    Every layer carries the map the whole plan measured, which is the dynamic each of its samples was
    stored at.
    """
    return BankDocument(
        version=MANIFEST_VERSION,
        name=name,
        layers=[_layer_record(layout, index, entry, velocity_map) for index, entry in enumerate(entries)],
    )
