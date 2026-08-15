from typing import Any

import pytest
from pydantic import ValidationError

from optisample.config.reduce import ReduceConfig


def raw(**sections: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A valid reduce payload with the named sections' fields overridden."""
    base: dict[str, dict[str, Any]] = {
        "dedupe": {
            "key": "pitch_velocity",
            "cc_quantum": 1.0,
            "transposition_headroom_semitones": 12,
            "representatives": "nearest_loudest",
        },
        "trim": {"max_length_s": 10.0, "tail_floor": 1.0e-4, "silence_floor": 1.0e-3},
        "events": {"duration_bucket_ratio": 1.25},
        "bandwidth": {"ceiling_hz": 16_000.0, "content_floor_db": 60.0, "content_band_hz": 200.0},
        "grouping": {"max_zone_semitones": 12},
    }
    return {name: {**fields, **sections.get(name, {})} for name, fields in base.items()}


def test_bundled_shape_validates() -> None:
    config = ReduceConfig.model_validate(raw())
    assert config.dedupe.key.includes_velocity
    assert not config.dedupe.key.includes_cc


@pytest.mark.parametrize(
    ("section", "overrides"),
    [
        ("dedupe", {"key": "pitch_and_articulation"}),
        ("dedupe", {"cc_quantum": 0.0}),
        ("dedupe", {"transposition_headroom_semitones": -1}),
        ("dedupe", {"representatives": "loudest"}),
        ("trim", {"max_length_s": 0.0}),
        ("trim", {"tail_floor": 0.0}),
        ("trim", {"silence_floor": 0.0}),
        ("trim", {"silence_floor": 1.0e-5}),  # under tail_floor, so the trim would empty a kept recording
        ("events", {"duration_bucket_ratio": 0.9}),
        ("bandwidth", {"ceiling_hz": 0.0}),
        ("bandwidth", {"content_floor_db": 0.0}),
        ("bandwidth", {"content_band_hz": 0.0}),
        ("grouping", {"max_zone_semitones": 0}),
    ],
)
def test_out_of_range_values_are_rejected(section: str, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ReduceConfig.model_validate(raw(**{section: overrides}))


def test_a_ratio_of_one_is_the_lower_bound_that_keeps_every_duration() -> None:
    assert ReduceConfig.model_validate(raw(events={"duration_bucket_ratio": 1.0})).events.duration_bucket_ratio == 1.0


def test_the_two_floors_may_meet() -> None:
    """One floor for both questions admits exactly the recordings the tail trim leaves a frame of."""
    trim = ReduceConfig.model_validate(raw(trim={"silence_floor": 1.0e-4})).trim
    assert trim.silence_floor == trim.tail_floor
