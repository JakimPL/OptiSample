from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from optisample.io.tracker.target import ExportTarget
from optisample.model import InstrumentSpec
from optisample.optimize.dp import AllocationInfeasibleError
from optisample.optimize.grouping.cost_model import ZoneSegment, _ZoneOptions, build_zone_options
from optisample.optimize.grouping.reserve import PROBES_PER_SOLVE, solve_within_cap
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers, partitions, velocity_cells
from optisample.optimize.layers.slots import reserved_slots
from optisample.optimize.layers.tasks import band_tasks
from optisample.optimize.orchestrate import RunInputs
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import BudgetBreakdown, SampleReserve, Zone, split_budget
from optisample.optimize.tasks import PitchTask
from optisample.progress import ProgressStep, counting

_ALLOCATE_LABEL: Final = "Allocating layers"


@dataclass(frozen=True)
class LayeredAllocation:
    """The velocity split the search settled on, and the zones allocated across it.

    ``layers`` names the band each stored instrument answers for and ``budget`` the split those layers
    were priced against, so a plan built from this states both what it stores and what carrying that
    many instruments cost it before the first frame of audio. ``reserve`` states the sample cap the
    zones were held to and the charge per stored sample that held them there.
    """

    layers: VelocityLayers
    budget: BudgetBreakdown
    zones: tuple[Zone, ...]
    total_bytes: int
    objective: float
    reserve: SampleReserve


@dataclass(frozen=True)
class _Layering:
    """What building one layer's stretch of the axis reads: the instrument, its recordings and the rules.

    Bundled because every band asks the same questions of the run, and the bands are asked for once per
    candidate split they appear in.
    """

    instrument: InstrumentSpec
    inputs: RunInputs
    settings: OptimizeSettings

    def budget(self, instruments: int) -> BudgetBreakdown:
        """What the instrument may spend on samples once ``instruments`` records are reserved."""
        return split_budget(self.instrument.budget_kb, self.settings.target.storage, instruments)

    def instruments(self, widths: Sequence[int]) -> int:
        """How many instrument records a split reserves, from the keys each of its bands plays.

        The zones a band comes out as are known only after the solve, so the reserve is taken at the most
        instruments those keys could fill -- a sample per key, cut into the runs the format writes.
        """
        return reserved_slots(widths, self.settings.target.max_samples_per_instrument)

    def keys(self, band: VelocityBand) -> tuple[PitchTask, ...]:
        """The keys ``band`` plays and what each of them stores and scores, which the layer count leaves alone."""
        return tuple(band_tasks(self.instrument, self.inputs.task_inputs, band))


@dataclass(frozen=True)
class _Universe:
    """Every band any candidate split stores, scored once, with each split's bands placed inside it.

    A band's keys are the same wherever it appears, and every split asks exactly the same question of it,
    so the whole search is scored from one pass over these segments and each split reads back the layers
    it holds. ``placement`` gives, per split, the segment each of its bands came out as, and ``scored``
    the zone options that segment offers, which is what a split is allocated from.
    """

    segments: tuple[ZoneSegment, ...]
    placement: tuple[tuple[int, ...], ...]
    scored: tuple[_ZoneOptions, ...]

    def bands(self, place: Sequence[int]) -> list[ZoneSegment]:
        """The keys each of one split's layers plays, in the order the split states them."""
        return [self.segments[segment] for segment in place]

    def options(self, place: Sequence[int]) -> list[_ZoneOptions]:
        """The scored zones each of one split's layers may be partitioned into."""
        return [self.scored[segment] for segment in place]


def _universe(layering: _Layering, splits: Sequence[VelocityLayers]) -> _Universe:
    """Collect the distinct bands the splits ask for, score each one, and place every split's bands."""
    segments: list[ZoneSegment] = []
    known: dict[VelocityBand, int] = {}
    placement: list[tuple[int, ...]] = []
    for split in splits:
        places: list[int] = []
        for band in split.bands:
            if band not in known:
                known[band] = len(segments)
                segments.append(layering.keys(band))

            places.append(known[band])

        placement.append(tuple(places))

    return _Universe(
        segments=tuple(segments),
        placement=tuple(placement),
        scored=build_zone_options(
            tuple(segments),
            layering.inputs.context,
            workers=layering.settings.workers,
            progress=layering.settings.progress,
        ),
    )


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
    place: Sequence[int],
    split: VelocityLayers,
    probe: ProgressStep,
) -> LayeredAllocation:
    """Partition and allocate one candidate split over the layers it stores, sharing one budget.

    The split is held to the run's sample cap, which every layer's zones are counted against together,
    so what the layers store stays inside one instrument's worth of samples however they divide the keys.

    Raises:
        BudgetInfeasibleError: when the cheapest sample per key still overruns what the split can spend.
        SampleCapInfeasibleError: when the charge meeting the cap leaves the budget carrying no partition.
    """
    bands = universe.bands(place)
    budget = layering.budget(layering.instruments([len(band) for band in bands]))
    capped = solve_within_cap(
        bands,
        universe.options(place),
        budget.sample_bytes,
        layering.settings.sample_cap,
        probe=probe,
    )
    return LayeredAllocation(
        layers=split,
        budget=budget,
        zones=capped.result.zones,
        total_bytes=capped.result.total_bytes,
        objective=capped.result.objective,
        reserve=capped.reserve,
    )


def fits_format(allocation: LayeredAllocation, target: ExportTarget) -> bool:
    """Whether the target format lists enough instruments for what this split reserves.

    A single layer keeps at most one sample per key the material plays and is written as the handful of
    instruments those samples fill, which every format holds, so this is what velocity layers add: each
    wide layer is cut to the samples one instrument owns, and several of them together can reserve more
    instruments than the module lists. The samples themselves are settled during the solve, which holds
    every split to the cap the format's own sample count bounds.
    """
    return allocation.budget.instruments <= target.max_instruments


def _preferred(
    candidate: LayeredAllocation,
    incumbent: LayeredAllocation,
    settings: OptimizeSettings,
) -> bool:
    """Whether ``candidate`` is the split to keep: writable by the format, and worth the layers it holds."""
    return fits_format(candidate, settings.target) and preference(candidate, settings.layers.min_gain) < preference(
        incumbent, settings.layers.min_gain
    )


def _best_split(
    layering: _Layering,
    universe: _Universe,
    splits: Sequence[VelocityLayers],
    probe: ProgressStep,
) -> LayeredAllocation:
    """The split worth storing, out of every one the layer cap allows.

    The single-layer split leads, because it is the plan every richer one has to beat by ``min_gain``
    (:func:`preference`) and the one whose feasibility is the run's own. Richer splits the budget or the
    format cannot carry are passed over, so the search answers with the best split any of them reached.

    Raises:
        BudgetInfeasibleError: when even a single layer of the cheapest samples overruns the budget.
        SampleCapInfeasibleError: when a single layer within the sample cap overruns the budget.
    """
    best = _allocate(layering, universe, universe.placement[0], splits[0], probe)
    for split, place in zip(splits[1:], universe.placement[1:]):
        try:
            candidate = _allocate(layering, universe, place, split, probe)
        except AllocationInfeasibleError:
            continue

        if _preferred(candidate, best, layering.settings):
            best = candidate

    return best


def _require_instrument_room(instrument: InstrumentSpec, settings: OptimizeSettings) -> None:
    """Check the layer cap against the instruments the target format numbers.

    Every velocity layer is written as at least one instrument, so a cap naming more layers than the
    format lists instruments leaves no split the exporter could write.

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
    would store, and solves the exact partition and allocation over each split in turn
    (:func:`_best_split`). Each of those solves walks the whole keyboard several times over, so the
    search reports against the walks every split it holds may spend
    (:data:`~optisample.optimize.grouping.reserve.PROBES_PER_SOLVE`), which is the longest stretch of a
    run that a budget alone decides the length of.

    Raises:
        BudgetInfeasibleError: when even a single layer of the cheapest samples overruns the budget.
        SampleCapInfeasibleError: when a single layer within the sample cap overruns the budget.
        ValueError: when the layer cap asks for more instruments than the format numbers.
    """
    _require_instrument_room(instrument, settings)
    layering = _Layering(instrument, inputs, settings)
    cells = velocity_cells(instrument.material, settings.layers.nodes)
    splits = tuple(partitions(cells, settings.layers.max_layers))
    universe = _universe(layering, splits)
    with counting(settings.progress, label=_ALLOCATE_LABEL, total=len(splits) * PROBES_PER_SOLVE) as probe:
        return _best_split(layering, universe, splits, probe)
