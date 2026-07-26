"""The contiguous-partition DP (``grouping/solve.py``), checked exactly against brute force."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from optisample.dsp.surrogate import EncodingParams
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.grouping import solve_grouping
from optisample.optimize.grouping.solve import _cheapest_partition_bytes
from optisample.optimize.plans import ZoneOption
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import Event, PitchTask


def _fake_tasks(pitches: tuple[int, ...]) -> list[PitchTask]:
    silence = np.zeros(4, dtype=np.float64)
    return [
        PitchTask(
            pitch, 1.0, SampleKey(pitch, 100), silence, (SampleKey(pitch, 100),), (Event(100, 64, 1.0, 1.0, silence),)
        )
        for pitch in pitches
    ]


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
    tasks = _fake_tasks(pitches)
    options = _fake_options(pitches, seed)
    for budget in (200, 400, 700, 1100, 1600):
        expected = _brute_force(pitches, options, budget)
        if math.isinf(expected):
            with pytest.raises(BudgetInfeasibleError):
                solve_grouping(tasks, options, budget)
            continue
        result = solve_grouping(tasks, options, budget)
        assert result.objective == pytest.approx(expected)
        assert result.total_bytes <= budget
        assert [pitch for zone in result.zones for pitch in zone.pitches] == list(pitches)  # contiguous cover


def test_solve_grouping_raises_when_cheapest_partition_overflows() -> None:
    pitches = (60, 62)
    tasks = _fake_tasks(pitches)
    options = _fake_options(pitches, 0)
    floor = _cheapest_partition_bytes(options, len(pitches))
    with pytest.raises(BudgetInfeasibleError):
        solve_grouping(tasks, options, floor - 1)


def test_solve_grouping_with_no_pitches_is_empty() -> None:
    result = solve_grouping([], {}, 1_000)
    assert result.zones == () and result.total_bytes == 0 and result.objective == 0.0
