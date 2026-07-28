from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from optisample.artifacts.paths import PlanPaths
from optisample.artifacts.serialize import Frozen, write_json, write_text
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout

MANIFEST_VERSION: Final = 1  # the manifest shape stated out loud, so a consumer reads the one it knows
_VELOCITY_AXIS: Final = "velocity"  # the axis name a selector spells its velocity band under
_KEYS_TELL_APART: Final = (
    "{count} instruments written where the velocity split states {bands}: a note finds its own by the key "
    "it plays as well as by the dynamic it was struck at.\n"
    "A bank manifest picks its layer by dynamics, so these instruments are reached a file at a time under "
    "{directory}/.\n"
)


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


def _layer_record(slot: InstrumentSlot, written: Path, paths: PlanPaths) -> LayerRecord:
    return LayerRecord(
        source=SourceRecord(file=paths.reference(written)),
        select={_VELOCITY_AXIS: BandRecord(low=slot.band.lowest, high=slot.band.highest)},
        velocity_map=paths.reference(paths.velocity_map_json),
    )


def bank_document(name: str, layout: SlotLayout, written: Sequence[Path], paths: PlanPaths) -> BankDocument:
    """The bank ``layout`` is played through: one layer per written instrument, in band order.

    ``written`` are the files the instruments landed as, in the same order as the slots. Every band the
    plan split states its own velocities, so a note picks the layer the allocation meant for its dynamic
    and the manifest says out loud what the split decided.
    """
    return BankDocument(
        version=MANIFEST_VERSION,
        name=name,
        layers=[_layer_record(slot, path, paths) for slot, path in zip(layout.slots, written)],
    )


def _keys_tell_apart(layout: SlotLayout, paths: PlanPaths) -> str:
    """The note left beside instruments the keys tell apart, saying how many files are to be loaded."""
    return _KEYS_TELL_APART.format(
        count=layout.count,
        bands=layout.layers.count,
        directory=paths.reference(paths.instruments_dir),
    )


def write_bank(name: str, layout: SlotLayout, written: Sequence[Path], paths: PlanPaths) -> None:
    """Write the manifest a player loads the whole plan through, or the note saying how its files are reached.

    A bank picks its layer by the dynamic a note was struck at, so a plan written as one instrument per
    band is stated whole. A band the format had room for only in several instruments is reached by
    loading those files one at a time, which the note beside them says.
    """
    if layout.split_by_keys:
        write_text(paths.unbanked, _keys_tell_apart(layout, paths))
        return

    write_json(paths.bank_json, bank_document(name, layout, written, paths))
