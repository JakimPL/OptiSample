from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from optisample.io.tracker.target import ExportTarget
from optisample.model import InstrumentSpec
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.grouping.cost_model import ZoneSegment, _ZoneOptions, build_zone_options
from optisample.optimize.grouping.solve import solve_grouping
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers, partitions, velocity_cells
from optisample.optimize.layers.tasks import band_tasks
from optisample.optimize.orchestrate import RunInputs
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import BudgetBreakdown, Zone, per_key_bytes, split_budget
from optisample.optimize.tasks import PitchTask


@dataclass(frozen=True)
class LayeredAllocation:
    """The velocity split the search settled on, and the zones allocated across it.

    ``layers`` names the band each stored instrument answers for and ``budget`` the split those layers
    were priced against, so a plan built from this states both what it stores and what carrying that
    many instruments cost it before the first frame of audio.
    """

    layers: VelocityLayers
    budget: BudgetBreakdown
    zones: tuple[Zone, ...]
    total_bytes: int
    objective: float


@dataclass(frozen=True)
class _Layering:
    """What building one layer's stretch of the axis reads: the instrument, its recordings and the rules.

    Bundled because every band asks the same questions of the run, and the bands are asked for once per
    candidate split they appear in.
    """

    instrument: InstrumentSpec
    inputs: RunInputs
    settings: OptimizeSettings

    def budget(self, layers: int) -> BudgetBreakdown:
        """What the instrument may spend on samples once ``layers`` instrument records are reserved."""
        return split_budget(self.instrument.budget_kb, self.settings.target.storage, layers)

    def keys(self, band: VelocityBand) -> tuple[PitchTask, ...]:
        """The keys ``band`` plays and what each of them stores and scores, which the layer count leaves alone."""
        return tuple(
            band_tasks(
                self.instrument,
                self.inputs.audio,
                self.inputs.velocity_map,
                self.settings.reduce,
                band,
            )
        )

    def byte_target(self, split: VelocityLayers, keys: Mapping[VelocityBand, tuple[PitchTask, ...]]) -> int:
        """The share of the budget one stored sample of ``split`` may spend when every key spends the same.

        The keys of every layer share one budget, so what a sample can afford follows from how wide the
        whole split is: layers that each cover the keyboard leave every sample a fraction of what one
        layer would, while layers that divide the keyboard between them leave it very nearly the same.
        The instrument records the extra layers cost come off the budget first.
        """
        width = sum(len(keys[band]) for band in split.bands)
        return per_key_bytes(self.budget(split.count), width)


@dataclass(frozen=True)
class _Universe:
    """Every band any candidate split stores, scored once, with each split's bands placed inside it.

    A band's keys are the same wherever it appears, and two splits pricing it the same way ask exactly
    the same question of it, so the whole search is scored from one pass over these segments and each
    split reads back the layers it holds. ``placement`` gives, per split, the segment each of its bands
    came out as.
    """

    segments: tuple[ZoneSegment, ...]
    placement: tuple[tuple[int, ...], ...]


def _universe(layering: _Layering, splits: Sequence[VelocityLayers]) -> _Universe:
    """Collect the distinct ``(band, byte target)`` segments the splits ask for, and where each sits."""
    keys: dict[VelocityBand, tuple[PitchTask, ...]] = {}
    for split in splits:
        for band in split.bands:
            if band not in keys:
                keys[band] = layering.keys(band)

    segments: list[ZoneSegment] = []
    known: dict[tuple[VelocityBand, int], int] = {}
    placement: list[tuple[int, ...]] = []
    for split in splits:
        target = layering.byte_target(split, keys)
        places: list[int] = []
        for band in split.bands:
            asked = (band, target)
            if asked not in known:
                known[asked] = len(segments)
                segments.append(ZoneSegment(keys[band], target))

            places.append(known[asked])

        placement.append(tuple(places))

    return _Universe(tuple(segments), tuple(placement))


def preference(allocation: LayeredAllocation, min_gain: float) -> float:
    """The number a split is judged on: its distortion, marked up by what each extra layer must buy.

    A layer is worth storing when it improves the objective by more than the configured margin, so every
    layer past the first raises the mark-up by that margin and the comparison between two splits is then
    plain. A margin of zero takes whichever split scores best on the objective alone.
    """
    return allocation.objective * (1.0 + min_gain) ** (allocation.layers.count - 1)


def _allocate(
    layering: _Layering,
    universe: _Universe,
    scored: Sequence[_ZoneOptions],
    place: Sequence[int],
    split: VelocityLayers,
) -> LayeredAllocation:
    """Partition and allocate one candidate split over the layers it stores, sharing one budget.

    Raises:
        BudgetInfeasibleError: when the cheapest sample per key still overruns what the split can spend.
    """
    budget = layering.budget(split.count)
    result = solve_grouping(
        [universe.segments[segment] for segment in place],
        [scored[segment] for segment in place],
        budget.sample_bytes,
    )
    return LayeredAllocation(
        layers=split,
        budget=budget,
        zones=result.zones,
        total_bytes=result.total_bytes,
        objective=result.objective,
    )


def fits_format(allocation: LayeredAllocation, target: ExportTarget) -> bool:
    """Whether the target format numbers enough samples for every zone this split stores.

    A single layer keeps at most one sample per key the material plays, which every format numbers, so
    this is what velocity layers add: several layers each holding a zone per key can ask for more
    samples than the module has slots, and a split that does is one the exporter could not write.
    """
    return len(allocation.zones) <= target.max_samples


def _preferred(
    candidate: LayeredAllocation,
    incumbent: LayeredAllocation,
    settings: OptimizeSettings,
) -> bool:
    """Whether ``candidate`` is the split to keep: writable by the format, and worth the layers it holds."""
    return fits_format(candidate, settings.target) and preference(candidate, settings.layers.min_gain) < preference(
        incumbent, settings.layers.min_gain
    )


def _require_instrument_room(instrument: InstrumentSpec, settings: OptimizeSettings) -> None:
    """Check the layer cap against the instruments the target format numbers.

    Raises:
        ValueError: when more layers are asked for than the format has instrument slots to write them.
    """
    target = settings.target
    if settings.layers.max_layers > target.max_instruments:
        raise ValueError(
            f"instrument {instrument.id!r} asks for {settings.layers.max_layers} velocity layers, "
            f"more than the {target.max_instruments} instruments {target.format.upper()} numbers"
        )


def allocate_layers(
    instrument: InstrumentSpec,
    inputs: RunInputs,
    settings: OptimizeSettings,
) -> LayeredAllocation:
    """Choose how many velocity layers to store, and allocate the byte budget across them.

    Cuts the velocity axis into cells, scores every band any split of them into at most ``max_layers``
    would store, and solves the exact partition and allocation over each split in turn. The
    single-layer split is solved first, because it is the plan every richer one has to beat by
    ``min_gain`` (:func:`preference`) and the one whose feasibility is the run's own -- it keeps at most
    one sample per key, so any budget and any format that hold an instrument at all hold it. Richer
    splits the budget or the format cannot carry are passed over.

    Raises:
        BudgetInfeasibleError: when even a single layer of the cheapest samples overruns the budget.
        ValueError: when the layer cap asks for more instruments than the format numbers.
    """
    _require_instrument_room(instrument, settings)
    layering = _Layering(instrument, inputs, settings)
    cells = velocity_cells(instrument.material, settings.layers.nodes)
    splits = tuple(partitions(cells, settings.layers.max_layers))
    universe = _universe(layering, splits)
    scored = build_zone_options(
        universe.segments,
        inputs.context,
        workers=settings.workers,
        progress=settings.progress,
    )

    best = _allocate(layering, universe, scored, universe.placement[0], splits[0])
    for split, place in zip(splits[1:], universe.placement[1:]):
        try:
            candidate = _allocate(layering, universe, scored, place, split)
        except BudgetInfeasibleError:
            continue

        if _preferred(candidate, best, settings):
            best = candidate

    return best
