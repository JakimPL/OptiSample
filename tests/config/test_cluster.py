from typing import Any

import pytest
from pydantic import ValidationError

from optisample.config.cluster import ClusterConfig


def raw(**sections: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A valid cluster payload with the named sections' fields overridden."""
    base: dict[str, dict[str, Any]] = {
        "descriptor": {
            "frequency_basis": "relative",
            "anchor_depths_db": [0.0, 6.0, 12.0],
            "anchor_span_s": 0.08,
            "harmonics": 16,
            "harmonic_band_share": 0.5,
            "harmonic_range_db": 60.0,
            "cepstral_coefficients": 12,
            "envelope_nodes": 6,
        },
        "space": {
            "onset_weight": 1.0,
            "sustain_weight": 1.0,
            "movement_weight": 0.5,
            "envelope_weight": 0.25,
            "min_reached_share": 0.8,
        },
        "partition": {
            "algorithm": "hierarchical",
            "linkage": "ward",
            "groups": 8,
            "max_groups": 16,
            "representative": "medoid",
            "min_duration_s": 0.25,
        },
        "instrument": {
            "layers": 2,
            "rate": 22_050,
            "depth": 8,
        },
    }
    return {name: {**fields, **sections.get(name, {})} for name, fields in base.items()}


def test_bundled_shape_validates() -> None:
    config = ClusterConfig.model_validate(raw())

    assert config.descriptor.anchor_depths_db == (0.0, 6.0, 12.0)
    assert config.partition.max_groups >= config.partition.groups


@pytest.mark.parametrize(
    ("section", "overrides"),
    [
        ("descriptor", {"frequency_basis": "harmonic"}),
        ("descriptor", {"anchor_depths_db": []}),
        ("descriptor", {"anchor_depths_db": [0.0, 12.0, 6.0]}),  # a depth under the one before it
        ("descriptor", {"anchor_depths_db": [6.0, 6.0]}),  # a depth repeating the one before it
        ("descriptor", {"anchor_depths_db": [-1.0, 6.0]}),
        ("descriptor", {"anchor_span_s": 0.0}),
        ("descriptor", {"harmonics": 0}),
        ("descriptor", {"harmonic_band_share": 0.0}),
        ("descriptor", {"harmonic_band_share": 0.6}),  # wider than half a harmonic, so two bands overlap
        ("descriptor", {"harmonic_range_db": 0.0}),
        ("descriptor", {"cepstral_coefficients": -1}),
        ("descriptor", {"envelope_nodes": 1}),
        ("space", {"onset_weight": -0.1}),
        ("space", {"sustain_weight": -0.1}),
        ("space", {"movement_weight": -0.1}),
        ("space", {"envelope_weight": -0.1}),
        ("space", {"min_reached_share": 0.0}),
        ("space", {"min_reached_share": 1.1}),
        ("partition", {"algorithm": "spectral"}),
        ("partition", {"linkage": "single"}),
        ("partition", {"groups": 1}),
        ("partition", {"max_groups": 1}),
        ("partition", {"groups": 20}),  # past the ceiling the sweep climbs to
        ("partition", {"representative": "loudest"}),
        ("partition", {"min_duration_s": -0.1}),
    ],
)
def test_out_of_range_values_are_rejected(section: str, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ClusterConfig.model_validate(raw(**{section: overrides}))


def test_a_single_depth_is_enough_to_read_a_decline_at() -> None:
    """One anchor states one moment of a note's fall, which is the shortest stack there is."""
    assert ClusterConfig.model_validate(raw(descriptor={"anchor_depths_db": [0.0]})).descriptor.anchor_depths_db == (
        0.0,
    )


def test_the_sweep_may_stop_exactly_at_the_count_the_space_is_cut_at() -> None:
    config = ClusterConfig.model_validate(raw(partition={"groups": 16, "max_groups": 16}))

    assert config.partition.max_groups == config.partition.groups


def test_keeping_no_coefficients_reads_the_bands_as_they_stand() -> None:
    assert (
        ClusterConfig.model_validate(raw(descriptor={"cepstral_coefficients": 0})).descriptor.cepstral_coefficients == 0
    )
