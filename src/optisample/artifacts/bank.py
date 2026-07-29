from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from optisample.artifacts.paths import PlanPaths
from optisample.artifacts.serialize import Frozen, write_json
from optisample.optimize.layers.slots import ONE_SLOT, SlotLayout

MANIFEST_VERSION: Final = 1  # the manifest shape stated out loud, so a consumer reads the one it knows
_VELOCITY_AXIS: Final = "velocity"  # the axis name a selector spells its velocity band under
_PITCH_AXIS: Final = "pitch"  # the axis name a selector spells its run of keys under


class BandRecord(Frozen):
    """A stretch of one axis a layer answers, with both ends inside it."""

    low: int
    high: int


class SourceRecord(Frozen):
    """Which file one layer's instrument is read out of, named against the directory the manifest sits in."""

    file: str


class LayerRecord(Frozen):
    """One instrument of the bank: where it is read from, the notes it answers, and how it reads them.

    ``select`` states a band per axis the layer is picked by, and a note reaches the layer whose every
    stated band covers it. ``velocity_map`` names the measured velocity->volume table written beside the
    instruments, so a note sounds at the volume its dynamic was measured to reach.
    """

    source: SourceRecord
    select: dict[str, BandRecord]
    velocity_map: str


class BankDocument(Frozen):
    """The manifest a player loads a written plan through: what the bank calls itself and what it plays.

    Layers stand in band order and each states the velocities it answers, so a note picks its instrument
    by the dynamic it was struck at and the instrument's own keymap then picks the sample. Every path is
    read against the directory the manifest sits in, which is the strategy's own, so the tree plays
    wherever it is copied to.
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


def _layer_record(layout: SlotLayout, index: int, written: Path, paths: PlanPaths) -> LayerRecord:
    return LayerRecord(
        source=SourceRecord(file=paths.reference(written)),
        select=_selector(layout, index),
        velocity_map=paths.reference(paths.velocity_map_json),
    )


def bank_document(name: str, layout: SlotLayout, written: Sequence[Path], paths: PlanPaths) -> BankDocument:
    """The bank ``layout`` is played through: one layer per written instrument, in the order they were written.

    ``written`` are the files the instruments landed as, in the same order as the slots. Each states the
    dynamics and, where its band was split across several files, the keys it answers, so the manifest
    says out loud what the allocation decided and a note reaches the instrument meant for it.
    """
    return BankDocument(
        version=MANIFEST_VERSION,
        name=name,
        layers=[_layer_record(layout, index, path, paths) for index, path in enumerate(written)],
    )


def write_bank(name: str, layout: SlotLayout, written: Sequence[Path], paths: PlanPaths) -> None:
    """Write the manifest a player loads the whole plan through, beside the instruments it names."""
    write_json(paths.bank_json, bank_document(name, layout, written, paths))
