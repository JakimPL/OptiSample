from dataclasses import dataclass
from typing import Final

import numpy as np
import pytest
from trackmod import BitDepth

from optisample.dsp.surrogate import EncodingParams
from optisample.keys import SampleKey
from optisample.optimize.dp import BudgetInfeasibleError, byte_grid
from optisample.optimize.grouping.cost_model import ZoneSegment
from optisample.optimize.grouping.reserve import (
    PROBES_PER_SOLVE,
    SampleCapInfeasibleError,
    _Search,
    solve_within_cap,
)
from optisample.optimize.plans import NO_RESERVE, ZoneOption
from optisample.optimize.tasks import Event, PitchTask

_PITCHES = (60, 61, 62, 63)
_PER_KEY_BYTES = 100  # what a zone's stored sample costs for each key it covers
_PER_KEY_DISTORTION = 1.0  # what covering one more key from the same sample costs the objective
_WIDE_ZONE_PENALTY = 4.0  # what a key served by a neighbor's recording costs beyond serving itself
_GENEROUS = 100_000
_CHEAP_KEY = 10  # what a key of a zone costs where the budget is small enough to search to the byte
_TIGHT = 254  # a budget whose charge window the refinements run all the way down
_EVERY_KEY = len(_PITCHES)
_ONE_SAMPLE = 1
_HALF = 2
_ONE_WALK = 1  # what a solve the budget settles on its own spends
_EXACT: Final = None  # the resolution that walks every byte total a budget holds
_GENEROUS_GRID: Final = byte_grid(_GENEROUS, _EXACT)
_TIGHT_GRID: Final = byte_grid(_TIGHT, _EXACT)


@dataclass(frozen=True)
class _Keyboard:
    """A grouping problem whose only choice is how coarsely to partition the keys.

    Each candidate zone has one option: it costs its width in samples' worth of bytes and distorts every
    key it covers, with the keys away from the representative distorting more. Storing a sample per key
    is the best plan the objective knows, so a cap below the key count is met by giving samples up.
    """

    pitches: tuple[int, ...]
    per_key_bytes: int = _PER_KEY_BYTES

    @property
    def segment(self) -> ZoneSegment:
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
            for pitch in self.pitches
        )
        return tasks

    @property
    def options(self) -> dict[tuple[int, int], tuple[ZoneOption, ...]]:
        options: dict[tuple[int, int], tuple[ZoneOption, ...]] = {}
        for start in range(len(self.pitches)):
            for stop in range(start + 1, len(self.pitches) + 1):
                width = stop - start
                options[(start, stop)] = (
                    ZoneOption(
                        self.pitches[start],
                        EncodingParams(11_025, BitDepth.SIXTEEN, 1.0),
                        self.per_key_bytes * width,
                        _PER_KEY_DISTORTION * width + _WIDE_ZONE_PENALTY * (width - 1),
                        1,
                    ),
                )

        return options


@dataclass
class _Probes:
    """A stand-in for the run's progress bar, holding the walks the solves reported to it."""

    taken: int = 0

    def __call__(self) -> None:
        self.taken += 1


@pytest.fixture
def keyboard() -> _Keyboard:
    return _Keyboard(_PITCHES)


@pytest.fixture
def probes() -> _Probes:
    return _Probes()


def test_a_cap_the_plan_already_meets_charges_nothing(keyboard: _Keyboard, probes: _Probes) -> None:
    """The budget alone decides the plan whenever it stores few enough samples, which costs one walk."""
    capped = solve_within_cap([keyboard.segment], [keyboard.options], _GENEROUS_GRID, _EVERY_KEY, probe=probes)
    assert len(capped.result.zones) == _EVERY_KEY
    assert not capped.reserve.binding
    assert capped.reserve.bytes_per_sample == NO_RESERVE
    assert capped.reserve.objective_uncapped == pytest.approx(capped.result.objective)
    assert probes.taken == _ONE_WALK


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_a_solve_spends_the_walks_the_bar_it_reports_to_is_drawn_against(
    keyboard: _Keyboard, cap: int, probes: _Probes
) -> None:
    """A search states its walks as it takes them, and takes at most the count a caller sizes its bar by."""
    solve_within_cap([keyboard.segment], [keyboard.options], _GENEROUS_GRID, cap, probe=probes)
    assert _ONE_WALK < probes.taken <= PROBES_PER_SOLVE


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_a_cap_below_what_the_budget_would_store_is_met(keyboard: _Keyboard, cap: int, probes: _Probes) -> None:
    """Charging each stored sample draws the walk into fewer, wider zones until the cap is met."""
    capped = solve_within_cap([keyboard.segment], [keyboard.options], _GENEROUS_GRID, cap, probe=probes)
    assert len(capped.result.zones) <= cap
    assert capped.reserve.binding and capped.reserve.cap == cap
    assert [pitch for zone in capped.result.zones for pitch in zone.pitches] == list(_PITCHES)


def test_a_met_cap_states_what_it_cost_and_what_it_truly_stores(keyboard: _Keyboard, probes: _Probes) -> None:
    """The charge shapes the partition; the bytes and the objective reported are the plan's own."""
    capped = solve_within_cap([keyboard.segment], [keyboard.options], _GENEROUS_GRID, _HALF, probe=probes)
    free = solve_within_cap([keyboard.segment], [keyboard.options], _GENEROUS_GRID, _EVERY_KEY, probe=probes)
    assert capped.reserve.objective_uncapped == pytest.approx(free.result.objective)
    assert capped.result.objective > free.result.objective
    assert capped.result.total_bytes == sum(zone.chosen.stored_bytes for zone in capped.result.zones)


def test_the_charge_settled_on_is_worked_down_from_the_one_that_prices_the_rest_out(
    keyboard: _Keyboard, probes: _Probes
) -> None:
    """A charge below the ceiling leaves more of the budget for the samples, so the search narrows toward it."""
    search = _Search(
        segments=[keyboard.segment], options=[keyboard.options], grid=_GENEROUS_GRID, cap=_HALF, probe=probes
    )
    capped = solve_within_cap([keyboard.segment], [keyboard.options], _GENEROUS_GRID, _HALF, probe=probes)
    charge = capped.reserve.bytes_per_sample
    assert NO_RESERVE < charge < search.ceiling
    assert search.within(search.walk(charge))  # the charge kept is one that holds the walk within the cap
    assert not search.within(search.walk(NO_RESERVE))  # and the charge is what holds it there


def test_a_window_the_refinements_close_leaves_the_smallest_charge_that_meets_the_cap(probes: _Probes) -> None:
    """A search that runs its window down to a byte states the exact price the cap was met at."""
    pair = _Keyboard(_PITCHES[:2], per_key_bytes=_CHEAP_KEY)
    search = _Search(segments=[pair.segment], options=[pair.options], grid=_TIGHT_GRID, cap=_ONE_SAMPLE, probe=probes)
    capped = solve_within_cap([pair.segment], [pair.options], _TIGHT_GRID, _ONE_SAMPLE, probe=probes)
    charge = capped.reserve.bytes_per_sample
    assert search.within(search.walk(charge))
    assert not search.within(search.walk(charge - 1))  # a byte cheaper and the walk holds two samples


def test_a_cap_below_the_widest_zone_the_keys_allow_is_refused(keyboard: _Keyboard, probes: _Probes) -> None:
    """One sample can cover a run of keys and no more, so a tighter cap has no partition to reach for."""
    pair = _Keyboard(_PITCHES[:2])
    with pytest.raises(SampleCapInfeasibleError, match="out of reach"):
        solve_within_cap(
            [pair.segment, pair.segment], [pair.options, pair.options], _GENEROUS_GRID, _ONE_SAMPLE, probe=probes
        )


def test_a_budget_carrying_nothing_fails_before_a_charge_is_added(keyboard: _Keyboard, probes: _Probes) -> None:
    """The budget is answered for first, so a run too small for the cheapest partition says exactly that."""
    with pytest.raises(BudgetInfeasibleError):
        solve_within_cap(
            [keyboard.segment], [keyboard.options], byte_grid(_PER_KEY_BYTES, _EXACT), _EVERY_KEY, probe=probes
        )
