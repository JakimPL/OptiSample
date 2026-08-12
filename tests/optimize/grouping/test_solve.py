import math
from itertools import product
from typing import Final

import numpy as np
import pytest

from optisample.dsp.surrogate import EncodingParams
from optisample.keys import SampleKey
from optisample.optimize.dp import BudgetInfeasibleError, byte_grid
from optisample.optimize.grouping import solve_grouping
from optisample.optimize.grouping.cost_model import ZoneSegment, zone_starts
from optisample.optimize.grouping.solve import _cheapest_partition_steps
from optisample.optimize.plans import NO_RESERVE, ZoneOption
from optisample.optimize.tasks import Event, PitchTask

_ANY_BUDGET: Final = 10_000  # a budget large enough that the grid it names states every byte total
_COARSE: Final = 4  # steps a budget is resolved into where the walk is meant to skip byte totals
_TOLERANCE: Final = 1e-9  # slack the float sums of two walks over the same options are compared within


def _fake_layer(pitches: tuple[int, ...]) -> tuple[ZoneSegment, ...]:
    """One velocity layer covering ``pitches``, which is the axis a plain pitch grouping walks."""
    silence = np.zeros(4, dtype=np.float64)
    tasks = tuple(
        PitchTask(
            pitch,
            1.0,
            SampleKey(pitch, 100),
            silence,
            (SampleKey(pitch, 100),),
            (Event(SampleKey(pitch, 100), 100, 64, 1.0, 1.0, silence, 1.0),),
        )
        for pitch in pitches
    )
    return (tasks,)


def _fake_options(pitches: tuple[int, ...], seed: int) -> dict[tuple[int, int], tuple[ZoneOption, ...]]:
    rng = np.random.default_rng(seed)
    options: dict[tuple[int, int], tuple[ZoneOption, ...]] = {}
    count = len(pitches)
    for i in range(count):
        for j in range(i + 1, count + 1):
            built = []
            for k in range(3):
                rep = pitches[i + (k % (j - i))]
                nbytes = int(rng.integers(80, 400))
                built.append(ZoneOption(rep, EncodingParams(11_025, 16, 1.0), nbytes, float(rng.uniform(0.01, 5.0)), 1))
            options[(i, j)] = tuple(built)
    return options


def _brute_force(
    pitches: tuple[int, ...], options: dict[tuple[int, int], tuple[ZoneOption, ...]], budget: int
) -> float:
    """Minimum distortion over *all* contiguous partitions and per-zone option choices within budget."""
    count = len(pitches)
    best = math.inf
    for mask in range(1 << (count - 1)):
        cuts = [0] + [k + 1 for k in range(count - 1) if mask >> k & 1] + [count]
        segments = list(zip(cuts, cuts[1:]))
        for combo in product(*(options[segment] for segment in segments)):
            if sum(option.stored_bytes for option in combo) <= budget:
                best = min(best, sum(option.distortion for option in combo))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_solve_grouping_is_exact_against_brute_force(seed: int) -> None:
    pitches = (60, 62, 64, 66)
    layer = _fake_layer(pitches)
    options = _fake_options(pitches, seed)
    for budget in (200, 400, 700, 1100, 1600):
        expected = _brute_force(pitches, options, budget)
        if math.isinf(expected):
            with pytest.raises(BudgetInfeasibleError):
                solve_grouping(layer, [options], byte_grid(budget, resolution=None), reserve=NO_RESERVE)
            continue
        result = solve_grouping(layer, [options], byte_grid(budget, resolution=None), reserve=NO_RESERVE)
        assert result.objective == pytest.approx(expected)
        assert result.total_bytes <= budget
        assert [pitch for zone in result.zones for pitch in zone.pitches] == list(pitches)  # contiguous cover
        assert {zone.layer for zone in result.zones} == {0}  # one segment in, one layer out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_walk_skipping_byte_totals_still_answers_with_a_plan_the_budget_carries(seed: int) -> None:
    """The steps a coarse grid states are what the plan is charged at, so its bytes stay inside the budget."""
    pitches = (60, 62, 64, 66)
    layer = _fake_layer(pitches)
    options = _fake_options(pitches, seed)
    for budget in (700, 1100, 1600):
        grid = byte_grid(budget, _COARSE)
        assert grid.granularity > 1  # the grid this reads is one the walk truly skips totals on
        result = solve_grouping(layer, [options], grid, reserve=NO_RESERVE)
        assert result.total_bytes <= budget
        assert [pitch for zone in result.zones for pitch in zone.pitches] == list(pitches)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_walk_skipping_byte_totals_reads_no_better_than_the_one_that_states_them_all(seed: int) -> None:
    """A coarse grid states a subset of the totals, so every plan it reaches the exact walk reaches too."""
    pitches = (60, 62, 64, 66)
    layer = _fake_layer(pitches)
    options = _fake_options(pitches, seed)
    for budget in (700, 1100, 1600):
        coarse = solve_grouping(layer, [options], byte_grid(budget, _COARSE), reserve=NO_RESERVE)
        exact = solve_grouping(layer, [options], byte_grid(budget, resolution=None), reserve=NO_RESERVE)
        assert coarse.objective >= exact.objective - _TOLERANCE


def test_solve_grouping_raises_when_cheapest_partition_overflows() -> None:
    pitches = (60, 62)
    layer = _fake_layer(pitches)
    options = _fake_options(pitches, 0)
    starts = zone_starts(options, len(pitches))
    floor = _cheapest_partition_steps(options, starts, byte_grid(_ANY_BUDGET, resolution=None), NO_RESERVE)
    with pytest.raises(BudgetInfeasibleError):
        solve_grouping(layer, [options], byte_grid(floor - 1, resolution=None), reserve=NO_RESERVE)


def test_solve_grouping_with_no_pitches_is_empty() -> None:
    result = solve_grouping([], [], byte_grid(1_000, resolution=None), reserve=NO_RESERVE)
    assert result.zones == () and result.total_bytes == 0 and result.objective == 0.0


def test_solve_grouping_names_the_layer_each_zone_came_from() -> None:
    """Two layers laid end to end partition independently, and each zone states the one it belongs to."""
    pitches = (60, 62)
    quiet, loud = _fake_layer(pitches)[0], _fake_layer(pitches)[0]
    options = _fake_options(pitches, 0)
    result = solve_grouping([quiet, loud], [options, options], byte_grid(4_000, resolution=None), reserve=NO_RESERVE)
    assert [pitch for zone in result.zones for pitch in zone.pitches] == list(pitches) * 2
    layers = [zone.layer for zone in result.zones]
    assert layers == sorted(layers) and set(layers) == {0, 1}
