from pathlib import Path
from typing import Final

import pytest

from optisample.config.dynamic_axis import AS_WRITTEN, DynamicAxis, DynamicAxisConfig
from optisample.dynamic_axis import dynamic_of, on_axis
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample

_WHEEL: Final = 1
_EXPRESSION: Final = 11
_PITCH: Final = 60
_HELD_S: Final = 1.0
_BUDGET_KB: Final = 64.0
_PLAYED: Final = 100  # the velocity every note of the fixture was struck at, which carries no dynamic
_POSITIONS: Final = (18, 67, 104)  # where the wheel stood on each note, which is what does carry one
_HALF: Final = 66.5  # a wheel average landing between two positions, as a swept note leaves one


def _wheel(controller: int) -> DynamicAxisConfig:
    return DynamicAxisConfig(axis=DynamicAxis.CONTROLLER, controller=controller)


def _manifest(positions: tuple[float, ...]) -> Manifest:
    """One instrument played at a single velocity, its dynamics spelled by the wheel alone."""
    return Manifest(
        project=ProjectSpec(name="song"),
        instruments=[
            InstrumentSpec(
                id="Strings",
                budget_kb=_BUDGET_KB,
                samples=[
                    SourceSample(
                        file=Path(f"{index:04d}.wav"),
                        pitch=_PITCH,
                        velocity=_PLAYED,
                        cc_averages={_WHEEL: position},
                    )
                    for index, position in enumerate(positions)
                ],
                material=[
                    NoteEvent(
                        pitch=_PITCH,
                        velocity=_PLAYED,
                        cc_averages={_WHEEL: position},
                        duration_s=_HELD_S,
                    )
                    for position in positions
                ],
            )
        ],
    )


@pytest.fixture(name="swept")
def _swept() -> Manifest:
    return _manifest(tuple(float(position) for position in _POSITIONS))


def test_a_velocity_axis_reads_the_velocity_a_note_was_struck_at() -> None:
    note = SourceSample(file=Path("0.wav"), pitch=_PITCH, velocity=_PLAYED, cc_averages={_WHEEL: 12.0})

    assert dynamic_of(note, AS_WRITTEN) == _PLAYED


def test_a_controller_axis_reads_the_position_that_controller_stood_at() -> None:
    note = SourceSample(file=Path("0.wav"), pitch=_PITCH, velocity=_PLAYED, cc_averages={_WHEEL: 67.0})

    assert dynamic_of(note, _wheel(_WHEEL)) == 67


def test_a_position_between_two_is_read_as_the_nearer_of_them() -> None:
    """A tracker writes one of 128 positions, so a swept note's average settles on the nearest."""
    note = SourceSample(file=Path("0.wav"), pitch=_PITCH, velocity=_PLAYED, cc_averages={_WHEEL: _HALF})

    assert dynamic_of(note, _wheel(_WHEEL)) == round(_HALF)


def test_a_controller_the_material_never_carried_is_refused() -> None:
    """Naming the wrong axis is a configuration error, and it states which controllers were recorded."""
    note = SourceSample(file=Path("0.wav"), pitch=_PITCH, velocity=_PLAYED, cc_averages={_WHEEL: 67.0})

    with pytest.raises(ValueError, match=f"controller {_EXPRESSION}"):
        dynamic_of(note, _wheel(_EXPRESSION))


def test_an_axis_the_library_holds_still_is_refused() -> None:
    """A library tracking a controller it never plays would hand every key one volume, silently."""
    held = _manifest((0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="carries no dynamics"):
        on_axis(held, _wheel(_WHEEL))


def test_a_velocity_axis_is_left_alone_where_every_note_shares_a_dynamic() -> None:
    """The guard asks about the axis a run chose, so material read as played is read whatever it holds."""
    held = _manifest((0.0, 0.0, 0.0))

    assert on_axis(held, AS_WRITTEN) == held


def test_a_velocity_axis_leaves_a_manifest_as_it_stands(swept: Manifest) -> None:
    assert on_axis(swept, AS_WRITTEN) == swept


def test_a_controller_axis_carries_each_recording_at_the_position_it_was_played(swept: Manifest) -> None:
    (instrument,) = on_axis(swept, _wheel(_WHEEL)).instruments

    assert tuple(sample.velocity for sample in instrument.samples) == _POSITIONS


def test_the_material_is_read_the_same_way_the_recordings_are(swept: Manifest) -> None:
    """A played note is routed to the recording nearest it in dynamic, so the two sides share one axis."""
    (instrument,) = on_axis(swept, _wheel(_WHEEL)).instruments

    assert tuple(event.velocity for event in instrument.material) == _POSITIONS


def test_the_position_a_note_was_recorded_at_stays_on_the_note(swept: Manifest) -> None:
    """The reading survives the run, so a dataset a stage writes states the axis it was keyed on."""
    (instrument,) = on_axis(swept, _wheel(_WHEEL)).instruments

    assert [sample.cc_averages[_WHEEL] for sample in instrument.samples] == list(_POSITIONS)


def test_reading_a_keyed_manifest_on_its_own_axis_again_answers_the_same_way(swept: Manifest) -> None:
    """Stages chain, each writing a dataset the next reads, so the reading has to hold across the join."""
    once = on_axis(swept, _wheel(_WHEEL))

    assert on_axis(once, _wheel(_WHEEL)) == once


def test_a_dataset_a_stage_wrote_reads_back_at_the_dynamic_it_was_keyed_on(swept: Manifest) -> None:
    """A written dataset states its dynamic as velocity, which is the axis a later stage reads it on."""
    written = on_axis(swept, _wheel(_WHEEL))

    assert on_axis(written, AS_WRITTEN) == written
