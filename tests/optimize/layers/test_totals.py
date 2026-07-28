from optisample.dsp.surrogate.params import EncodingParams
from optisample.music import MIDI_MAX_VELOCITY
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers
from optisample.optimize.layers.totals import layer_totals
from optisample.optimize.plans import SampleUnit
from optisample.optimize.reduce.keys import SampleKey

_SPLIT = VelocityLayers((VelocityBand(0, 50), VelocityBand(51, MIDI_MAX_VELOCITY)))
_VELOCITY = 100


def _unit(layer: int, keys: tuple[int, ...], *, stored_bytes: int, share: float, weight: float) -> SampleUnit:
    return SampleUnit(
        label=f"layer{layer}_rep{keys[0]:03d}",
        representative_key=SampleKey(keys[0], _VELOCITY),
        layer=layer,
        keys=keys,
        params=EncodingParams(target_rate=22_050, depth_bits=16),
        frames=2400,
        stored_bytes=stored_bytes,
        distortion=share,
        objective_share=share,
        hull_size=1,
        weight=weight,
    )


_QUIET = _unit(0, (60, 61), stored_bytes=4000, share=1.5, weight=3.0)
_LOUD = _unit(1, (60,), stored_bytes=6000, share=0.5, weight=2.0)
_ALSO_LOUD = _unit(1, (61,), stored_bytes=5000, share=0.25, weight=1.0)


def test_one_row_comes_back_per_band_in_the_order_the_instruments_are_written() -> None:
    rows = layer_totals(_SPLIT, (_QUIET, _LOUD, _ALSO_LOUD))
    assert [row.layer for row in rows] == [0, 1]
    assert [row.band.label for row in rows] == ["v000-v050", "v051-v127"]


def test_a_band_sums_only_the_samples_written_into_its_own_instrument() -> None:
    quiet, loud = layer_totals(_SPLIT, (_QUIET, _LOUD, _ALSO_LOUD))
    assert (quiet.samples, quiet.keys, quiet.stored_bytes) == (1, 2, 4000)
    assert (loud.samples, loud.keys, loud.stored_bytes) == (2, 2, 11_000)


def test_the_rows_add_up_to_what_the_whole_plan_stores() -> None:
    """A reader checks a split against the plan it came out of, so the bands account for all of it."""
    units = (_QUIET, _LOUD, _ALSO_LOUD)
    rows = layer_totals(_SPLIT, units)
    assert sum(row.stored_bytes for row in rows) == sum(unit.stored_bytes for unit in units)
    assert sum(row.weight for row in rows) == sum(unit.weight for unit in units)
    assert sum(row.objective_share for row in rows) == sum(unit.objective_share for unit in units)
    assert sum(row.samples for row in rows) == len(units)


def test_a_band_storing_nothing_still_reports_the_dynamics_it_answers_for() -> None:
    """A layer the allocation left empty keeps its place, since the instrument it names is still written."""
    quiet, loud = layer_totals(_SPLIT, (_LOUD, _ALSO_LOUD))
    assert (quiet.samples, quiet.keys, quiet.stored_bytes) == (0, 0, 0)
    assert (quiet.weight, quiet.objective_share) == (0.0, 0.0)
    assert quiet.band.label == "v000-v050"
    assert loud.samples == 2


def test_an_unlayered_plan_reports_its_one_band_holding_everything() -> None:
    (whole,) = layer_totals(UNSPLIT, (_unit(0, (60, 61, 62), stored_bytes=9000, share=2.0, weight=5.0),))
    assert whole.band.label == "v000-v127"
    assert (whole.samples, whole.keys, whole.stored_bytes) == (1, 3, 9000)
