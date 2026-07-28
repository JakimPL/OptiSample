from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Final

from optisample.io.tracker.target import ExportTarget
from optisample.music import note_name
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers
from optisample.optimize.plans.strategy import SampleUnit, StrategyPlan

ONE_SLOT: Final = 1  # instruments a band is written as while every sample it stores fits inside one
_NO_KEYS: Final = ""  # the stretch of keyboard a slot storing nothing owns


@dataclass(frozen=True)
class InstrumentSlot:
    """One written instrument: a velocity band's stored samples over the run of keys they answer for.

    A format numbering few samples inside one instrument writes a wide band as several slots, each owning
    an ascending run of the band's keys, so every zone the allocation paid for stays stored and a note's
    pitch names the slot within its band. ``samples`` gives the position each held sample takes in the
    song's own sample table, in the same order as ``units``, which is what a keymap routes its keys to.
    """

    layer: int
    band: VelocityBand
    samples: tuple[int, ...]
    units: tuple[SampleUnit, ...]

    @property
    def pitches(self) -> tuple[int, ...]:
        """Every key this slot's samples were stored for, ascending and named once."""
        return tuple(sorted({key for unit in self.units for key in unit.keys}))

    @property
    def keys(self) -> int:
        """How many keys the slot's samples were stored for."""
        return len(self.pitches)

    @property
    def span(self) -> str:
        """The stretch of keyboard the slot owns, named by its outer keys (``F1-B4``).

        A slot the allocation stored nothing in owns no key, and names an empty stretch.
        """
        pitches = self.pitches
        if not pitches:
            return _NO_KEYS

        return f"{note_name(pitches[0])}-{note_name(pitches[-1])}"

    @property
    def stored_bytes(self) -> int:
        """What the slot's recordings occupy, records included."""
        return sum(unit.stored_bytes for unit in self.units)

    @property
    def weight(self) -> float:
        """Playing time the material spends on the notes this slot answers."""
        return sum(unit.weight for unit in self.units)

    @property
    def objective_share(self) -> float:
        """What this slot's notes carry of the plan's objective, which is the reading a split is judged on."""
        return sum(unit.objective_share for unit in self.units)

    def answers(self, pitch: int) -> bool:
        """Whether ``pitch`` reaches no further than the top of the run this slot owns.

        Slots are asked in key order, so the first one answering a pitch is the one that owns it and a
        key below every stored run falls to the lowest slot, which is the instrument whose keymap was
        filled down to it.
        """
        pitches = self.pitches
        return bool(pitches) and pitch <= pitches[-1]


@dataclass(frozen=True)
class SlotLayout:
    """Every instrument a plan is written as, in the order the module numbers them.

    Each velocity band holds at least one slot and a band's slots own ascending runs of its keys, so a
    note's dynamic picks the band and its pitch the slot inside that band.
    """

    layers: VelocityLayers
    slots: tuple[InstrumentSlot, ...]

    @property
    def count(self) -> int:
        """How many instruments the module carries."""
        return len(self.slots)

    def layer_slots(self, layer: int) -> tuple[int, ...]:
        """Which instruments one velocity band is written as, in ascending key order."""
        return tuple(index for index, slot in enumerate(self.slots) if slot.layer == layer)

    def instrument(self, layer: int, pitch: int) -> int:
        """Which written instrument plays ``pitch`` at the dynamics ``layer`` answers for.

        A key above every run the band stores is played through its topmost slot, which is the instrument
        whose keymap was filled up to it.
        """
        written = self.layer_slots(layer)
        for index in written:
            if self.slots[index].answers(pitch):
                return index

        return written[-1]


def _runs(positions: Sequence[int], per_instrument: int) -> Iterator[tuple[int, ...]]:
    """``positions`` cut into runs of at most ``per_instrument``, holding nothing giving one empty run."""
    if not positions:
        yield ()
        return

    for start in range(0, len(positions), per_instrument):
        yield tuple(positions[start : start + per_instrument])


def pack_slots(units: Sequence[SampleUnit], layers: VelocityLayers, per_instrument: int) -> SlotLayout:
    """Cut each velocity band's stored samples into the instruments the format has room to write them in.

    A band stays one instrument for as long as the format numbers enough samples inside one, which is why
    a format numbering hundreds writes today's shape exactly: one instrument per band. Where a format
    numbers few, the band's samples are cut in key order into runs of at most ``per_instrument``, so every
    zone the allocation paid for stays stored and each instrument answers a contiguous stretch of the
    keyboard. Every band is written, so a band the allocation stored nothing in keeps the instrument its
    dynamics resolve to.
    """
    slots: list[InstrumentSlot] = []
    for layer, band in enumerate(layers.bands):
        held = sorted(
            (position for position, unit in enumerate(units) if unit.layer == layer),
            key=lambda position: units[position].keys,
        )
        slots += [
            InstrumentSlot(layer=layer, band=band, samples=run, units=tuple(units[position] for position in run))
            for run in _runs(held, per_instrument)
        ]

    return SlotLayout(layers=layers, slots=tuple(slots))


def plan_slots(plan: StrategyPlan, target: ExportTarget) -> SlotLayout:
    """The instruments ``plan`` is written as in ``target``, which is what every reader of them states.

    The exporter, the report and the plan document each answer for the same written instruments, so the
    plan's stored samples meet the format's own sample-per-instrument bound in one place.
    """
    return pack_slots(plan.sample_units(), plan.layers, target.max_samples_per_instrument)


def reserved_slots(keys: Iterable[int], per_instrument: int) -> int:
    """How many instruments a plan reserves before it is solved, from the keys each of its bands plays.

    A band stores at most one sample per key it plays, so cutting that many samples into runs of
    ``per_instrument`` is the most instruments it can come out as, and reserving that is what makes the
    byte budget honest while the allocation is still choosing what to store. Every band is written, so one
    storing nothing still reserves its instrument.
    """
    return sum(max(ONE_SLOT, ceil(count / per_instrument)) for count in keys)
