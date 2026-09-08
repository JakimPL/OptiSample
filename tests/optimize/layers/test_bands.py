from math import comb

import pytest

from optisample.model import NoteEvent
from optisample.music import MIDI_MAX_VELOCITY
from optisample.optimize.layers.bands import (
    VelocityBand,
    VelocityCells,
    VelocityLayers,
    partitions,
    velocity_cells,
)

VELOCITIES = tuple(range(10, 130, 15))  # eight dynamics spread over the axis: 10, 25, ..., 115


def _material(*played: tuple[int, float]) -> list[NoteEvent]:
    """One note per ``(velocity, playing time)`` pair, all at one pitch."""
    return [NoteEvent(pitch=60, velocity=velocity, duration_s=seconds, count=1) for velocity, seconds in played]


def _even_material() -> list[NoteEvent]:
    """Eight dynamics each played for the same length, so the quantiles land two velocities apart."""
    return _material(*((velocity, 1.0) for velocity in VELOCITIES))


# --- cells ------------------------------------------------------------------------------------------


def test_cells_tile_the_whole_velocity_axis() -> None:
    """Every dynamic resolves to a cell, including the ones the material leaves unplayed."""
    cells = velocity_cells(_even_material(), 4).cells
    assert cells[0].lowest == 0
    assert cells[-1].highest == MIDI_MAX_VELOCITY
    for below, above in zip(cells, cells[1:]):
        assert above.lowest == below.highest + 1


def test_cells_hold_equal_shares_of_playing_time() -> None:
    cells = velocity_cells(_even_material(), 4)
    assert cells.count == 4
    for cell in cells.cells:
        assert len([velocity for velocity in VELOCITIES if cell.covers(velocity)]) == 2


def test_a_cut_lands_on_a_velocity_the_material_plays() -> None:
    cells = velocity_cells(_even_material(), 4)
    assert [cell.highest for cell in cells.cells[:-1]] == [25, 55, 85]


def test_cells_are_finer_where_the_material_spends_its_time() -> None:
    """One dynamic holding most of the playing time earns a cell of its own."""
    cells = velocity_cells(_material((10, 4.0), (100, 1.0), (110, 1.0), (120, 1.0)), 2)
    assert [(cell.lowest, cell.highest) for cell in cells.cells] == [(0, 10), (11, MIDI_MAX_VELOCITY)]


def test_fewer_dynamics_than_nodes_yield_fewer_cells() -> None:
    """A cut needs two played dynamics to sit between, so the material bounds how fine the axis gets."""
    cells = velocity_cells(_material((30, 1.0), (90, 1.0)), 6)
    assert cells.count == 2


def test_one_node_leaves_the_whole_axis_in_one_cell() -> None:
    cells = velocity_cells(_even_material(), 1)
    assert cells.cells == (VelocityBand(0, MIDI_MAX_VELOCITY),)


def test_a_band_spans_a_run_of_cells_from_its_floor_to_its_ceiling() -> None:
    cells = velocity_cells(_even_material(), 4)
    assert cells.band(1, 3) == VelocityBand(cells.cells[1].lowest, cells.cells[2].highest)


# --- bands ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("velocity", "covered"), [(0, True), (63, True), (64, False), (127, False)])
def test_a_band_covers_the_velocities_between_its_bounds(velocity: int, covered: bool) -> None:
    assert VelocityBand(0, 63).covers(velocity) is covered


def test_a_band_is_labeled_by_the_span_it_answers_for() -> None:
    assert VelocityBand(0, 63).label == "v000-v063"


# --- splits -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_layers", [1, 2, 3, 4])
def test_every_split_of_the_cells_is_offered_once(max_layers: int) -> None:
    cells = velocity_cells(_even_material(), 4)
    splits = list(partitions(cells, max_layers))
    expected = sum(comb(cells.count - 1, count - 1) for count in range(1, min(max_layers, cells.count) + 1))
    assert len(splits) == expected
    assert len({split.bands for split in splits}) == expected


def test_splits_are_offered_fewest_layers_first() -> None:
    """The single-layer plan comes first, so it is the one every richer split has to beat."""
    counts = [split.count for split in partitions(velocity_cells(_even_material(), 4), 3)]
    assert counts[0] == 1
    assert counts == sorted(counts)


def test_one_layer_is_a_single_band_over_the_whole_axis() -> None:
    (split,) = partitions(velocity_cells(_even_material(), 4), 1)
    assert split.bands == (VelocityBand(0, MIDI_MAX_VELOCITY),)


def test_asking_for_more_layers_than_cells_stops_at_the_cells() -> None:
    cells = velocity_cells(_material((30, 1.0), (90, 1.0)), 6)
    assert max(split.count for split in partitions(cells, 5)) == cells.count


def test_every_split_tiles_the_axis_in_ascending_bands() -> None:
    for split in partitions(velocity_cells(_even_material(), 4), 3):
        assert split.bands[0].lowest == 0
        assert split.bands[-1].highest == MIDI_MAX_VELOCITY
        for below, above in zip(split.bands, split.bands[1:]):
            assert above.lowest == below.highest + 1


# --- routing a note to its layer --------------------------------------------------------------------


def test_every_velocity_resolves_to_the_layer_that_covers_it() -> None:
    for split in partitions(velocity_cells(_even_material(), 4), 3):
        for velocity in range(MIDI_MAX_VELOCITY + 1):
            assert split.bands[split.band_index(velocity)].covers(velocity)


def test_a_velocity_above_the_topmost_layer_is_rejected() -> None:
    layers = VelocityLayers((VelocityBand(0, 63),))
    with pytest.raises(ValueError, match="above the topmost layer v000-v063"):
        layers.band_index(64)


def test_a_split_states_how_many_instruments_it_asks_for() -> None:
    cells = VelocityCells((VelocityBand(0, 63), VelocityBand(64, MIDI_MAX_VELOCITY)))
    assert [split.count for split in partitions(cells, 2)] == [1, 2]
