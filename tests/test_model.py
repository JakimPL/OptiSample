from __future__ import annotations

import pytest
from pydantic import ValidationError

from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample


def _sample(**overrides: object) -> SourceSample:
    data: dict[str, object] = {"file": "a.wav", "pitch": 60, "velocity": 100}
    data.update(overrides)
    return SourceSample(**data)  # type: ignore[arg-type]


def test_note_event_weight() -> None:
    event = NoteEvent(pitch=60, velocity=100, duration_s=2.5, count=4)
    assert event.weight == pytest.approx(10.0)


def test_default_count_is_one() -> None:
    assert NoteEvent(pitch=60, velocity=100, duration_s=1.0).weight == pytest.approx(1.0)


@pytest.mark.parametrize("pitch", [-1, 128, 999])
def test_pitch_out_of_range_rejected(pitch: int) -> None:
    with pytest.raises(ValidationError):
        _sample(pitch=pitch)


@pytest.mark.parametrize("velocity", [-1, 128])
def test_velocity_out_of_range_rejected(velocity: int) -> None:
    with pytest.raises(ValidationError):
        _sample(velocity=velocity)


def test_extra_keys_forbidden() -> None:
    with pytest.raises(ValidationError):
        _sample(unexpected="x")


def test_duration_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        NoteEvent(pitch=60, velocity=100, duration_s=0.0)


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        InstrumentSpec(
            id="x", budget_kb=0.0, samples=[_sample()], material=[NoteEvent(pitch=60, velocity=1, duration_s=1.0)]
        )


def test_instrument_requires_at_least_one_sample() -> None:
    with pytest.raises(ValidationError):
        InstrumentSpec(id="x", budget_kb=1.0, samples=[], material=[NoteEvent(pitch=60, velocity=1, duration_s=1.0)])


def test_exactly_one_material_source_required() -> None:
    event = NoteEvent(pitch=60, velocity=100, duration_s=1.0)
    # Neither material nor material_midi.
    with pytest.raises(ValidationError):
        InstrumentSpec(id="x", budget_kb=1.0, samples=[_sample()])
    # Both provided.
    with pytest.raises(ValidationError):
        InstrumentSpec(id="x", budget_kb=1.0, samples=[_sample()], material=[event], material_midi="m.mid")


def test_valid_manifest_round_trips_through_python() -> None:
    manifest = Manifest(
        project=ProjectSpec(name="song"),
        instruments=[
            InstrumentSpec(
                id="strings",
                budget_kb=128.0,
                samples=[_sample()],
                material=[NoteEvent(pitch=60, velocity=100, duration_s=2.0, count=3)],
            )
        ],
    )
    assert manifest.instruments[0].material is not None
    assert manifest.instruments[0].material[0].weight == pytest.approx(6.0)  # count 3 x 2.0 s
