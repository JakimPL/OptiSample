from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from optisample.optimize.layers.bands import VelocityBand, VelocityLayers
from optisample.optimize.plans.strategy import SampleUnit


@dataclass(frozen=True)
class LayerTotals:
    """One velocity band of a plan, holding the stored samples written into its own instrument.

    ``layer`` is the position the band sits at, which is the instrument number a note of this band is
    played through, and ``units`` the samples the allocation gave it. Everything a reader prices the
    band on is read off those units, so the band states one set of numbers however it is displayed.
    """

    layer: int
    band: VelocityBand
    units: tuple[SampleUnit, ...]

    @property
    def samples(self) -> int:
        """Stored recordings this band's instrument owns."""
        return len(self.units)

    @property
    def keys(self) -> int:
        """Keys the band's samples reach between them, counted once each."""
        return sum(len(unit.keys) for unit in self.units)

    @property
    def stored_bytes(self) -> int:
        """What the band's recordings occupy, records included."""
        return sum(unit.stored_bytes for unit in self.units)

    @property
    def weight(self) -> float:
        """Playing time the material spends at the dynamics this band answers for."""
        return sum(unit.weight for unit in self.units)

    @property
    def objective_share(self) -> float:
        """What this band's notes carry of the plan's objective, which is the reading a split is judged on."""
        return sum(unit.objective_share for unit in self.units)


def layer_totals(layers: VelocityLayers, units: Sequence[SampleUnit]) -> tuple[LayerTotals, ...]:
    """One entry per band, in band order, holding the plan's units grouped by the layer each is written as.

    Splitting the plan's own samples along the axis it stores them on is what states the price of the
    vocabulary: what each band covers, how many recordings it took and what they cost. The entries
    account for every unit, so their stored bytes, material weight and objective shares add up to the
    plan's own and a reader checks the split against the totals it came out of.
    """
    return tuple(
        LayerTotals(layer, band, tuple(unit for unit in units if unit.layer == layer))
        for layer, band in enumerate(layers.bands)
    )
