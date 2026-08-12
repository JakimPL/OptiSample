from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.cluster.descriptor import SampleDescriptor, describe
from optisample.config.cluster import DescriptorConfig, FrequencyBasis
from optisample.config.loop import FeatureConfig
from tests.cluster.conftest import SR, Signal

ROOT_HZ = 220.0

_LOUD_GAIN = 4.0  # the gain a copy of a take is captured at, a power of two so the scaling stays exact
_ODD_GAIN = 0.3  # a gain that rounds at every step, which is what a real difference in capture level does
_SCALARS = 4  # the readings the contour states ahead of its anchor times


def _timbre(descriptor: SampleDescriptor) -> Signal:
    """The whole of what a descriptor says about the sound, gathered as one vector."""
    return np.concatenate((descriptor.onset, descriptor.sustain.ravel()))


def _describe(
    signal: Signal, config: DescriptorConfig, features: FeatureConfig, root_hz: float = ROOT_HZ
) -> SampleDescriptor:
    return describe(signal, SR, root_hz=root_hz, features=features, config=config)


def test_a_descriptor_states_one_row_of_shape_per_fall_depth(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    descriptor = _describe(ringing_note(ROOT_HZ), descriptor_config, features_config)

    assert descriptor.depths == len(descriptor_config.anchor_depths_db)
    assert descriptor.columns == descriptor_config.cepstral_coefficients
    assert descriptor.sustain.shape == (descriptor.depths, descriptor.columns)
    assert descriptor.reached.shape == (descriptor.depths,)
    assert descriptor.movement.values.shape == (2,)
    assert descriptor.envelope.values.shape == (_SCALARS + descriptor.depths,)


def test_keeping_no_coefficients_reads_the_partials_as_they_stand(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    config = descriptor_config.model_copy(update={"cepstral_coefficients": 0})

    assert _describe(ringing_note(ROOT_HZ), config, features_config).columns == config.harmonics


def test_the_absolute_basis_reads_the_mel_bands_the_composite_scores_with(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    config = descriptor_config.model_copy(
        update={"frequency_basis": FrequencyBasis.ABSOLUTE, "cepstral_coefficients": 0}
    )

    assert _describe(ringing_note(ROOT_HZ), config, features_config).columns == features_config.bands


@pytest.mark.parametrize("gain", [_LOUD_GAIN, _ODD_GAIN])
@pytest.mark.parametrize("basis", list(FrequencyBasis))
def test_the_level_a_take_was_captured_at_leaves_the_reading_where_it_stands(
    gain: float,
    basis: FrequencyBasis,
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    """The whole point of the space: two captures of one sound stand together whatever level each was at."""
    config = descriptor_config.model_copy(update={"frequency_basis": basis})
    signal = ringing_note(ROOT_HZ)
    quiet = _describe(signal, config, features_config)
    loud = _describe(gain * signal, config, features_config)

    assert np.allclose(_timbre(loud), _timbre(quiet), atol=1e-9)
    assert np.allclose(loud.movement.values, quiet.movement.values, atol=1e-9)
    assert np.allclose(loud.envelope.values, quiet.envelope.values, atol=1e-9)


def test_truncating_a_ringing_note_leaves_every_depth_it_still_reaches_where_it_stood(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    """A short take and a long take of one note are the same sound, and the anchors are what say so."""
    signal = ringing_note(ROOT_HZ, duration_s=3.0, decay_db_per_s=18.0)
    whole = _describe(signal, descriptor_config, features_config)
    cut = _describe(signal[:SR], descriptor_config, features_config)
    shared = whole.reached & cut.reached

    assert shared.any() and not cut.reached.all()
    assert np.allclose(whole.sustain[shared], cut.sustain[shared])
    assert np.allclose(whole.onset, cut.onset)


def test_two_notes_an_octave_apart_sharing_a_balance_sit_together_on_the_relative_basis(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
    dull_balance: tuple[float, ...],
) -> None:
    """Reading the note's own partials is what makes a group a sound rather than a register."""
    low = _describe(ringing_note(ROOT_HZ), descriptor_config, features_config)
    high = _describe(ringing_note(2.0 * ROOT_HZ), descriptor_config, features_config, root_hz=2.0 * ROOT_HZ)
    dull = _describe(ringing_note(ROOT_HZ, balance=dull_balance), descriptor_config, features_config)

    across_registers = float(np.linalg.norm(_timbre(high) - _timbre(low)))
    across_balances = float(np.linalg.norm(_timbre(dull) - _timbre(low)))

    assert across_registers < across_balances


def test_the_absolute_basis_tells_the_registers_apart(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    """A group on this basis is a set of recordings that compare alike under the metric the optimizer uses."""
    config = descriptor_config.model_copy(update={"frequency_basis": FrequencyBasis.ABSOLUTE})
    low = _describe(ringing_note(ROOT_HZ), config, features_config)
    high = _describe(ringing_note(2.0 * ROOT_HZ), config, features_config, root_hz=2.0 * ROOT_HZ)

    assert float(np.linalg.norm(_timbre(high) - _timbre(low))) > 1.0


def test_the_contour_states_the_rate_a_note_falls_at(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    fast = _describe(ringing_note(ROOT_HZ, decay_db_per_s=24.0), descriptor_config, features_config)
    slow = _describe(ringing_note(ROOT_HZ, decay_db_per_s=6.0), descriptor_config, features_config)

    assert fast.envelope.decay_db_per_s == pytest.approx(-24.0, abs=1.0)
    assert slow.envelope.decay_db_per_s == pytest.approx(-6.0, abs=1.0)
    assert fast.envelope.curvature_db == pytest.approx(0.0, abs=1.0)  # a steady fall runs straight


def test_a_note_departing_from_a_straight_fall_states_how_far(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    """A double decay -- fast, then slow -- is what a real string does, and the curvature is what says so."""
    signal = np.concatenate(
        (ringing_note(ROOT_HZ, duration_s=0.5, decay_db_per_s=48.0), ringing_note(ROOT_HZ, duration_s=1.5) * 1e-2)
    )
    bent = _describe(signal, descriptor_config, features_config)
    straight = _describe(ringing_note(ROOT_HZ, duration_s=2.0, decay_db_per_s=18.0), descriptor_config, features_config)

    assert bent.envelope.curvature_db > straight.envelope.curvature_db


def test_the_anchor_times_state_how_long_the_note_took_to_reach_each_depth(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    """A note falling twice as fast reaches every depth it shares in half the time, which is 0.3 log-seconds."""
    fast = _describe(ringing_note(ROOT_HZ, duration_s=3.0, decay_db_per_s=24.0), descriptor_config, features_config)
    slow = _describe(ringing_note(ROOT_HZ, duration_s=3.0, decay_db_per_s=12.0), descriptor_config, features_config)
    reached = fast.reached & slow.reached & (np.asarray(descriptor_config.anchor_depths_db) > 0.0)

    assert reached.any()
    assert np.allclose(
        slow.envelope.anchor_log_s[reached] - fast.envelope.anchor_log_s[reached], np.log10(2.0), atol=0.1
    )


def test_a_take_too_short_for_a_line_reads_a_straight_level_fall(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    descriptor = _describe(ringing_note(ROOT_HZ, duration_s=0.05), descriptor_config, features_config)

    assert descriptor.envelope.decay_db_per_s == 0.0
    assert descriptor.envelope.curvature_db == 0.0


def test_a_take_of_one_frame_still_reads_an_attack_and_a_movement(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    """Material holding a single frame states the sound it opened on, which is all the evidence it carries."""
    descriptor = _describe(ringing_note(ROOT_HZ, duration_s=0.02), descriptor_config, features_config)

    assert descriptor.onset.shape == (descriptor.columns,)
    assert descriptor.movement.travel_db_per_s == 0.0
    assert descriptor.movement.change_db_per_s == 0.0


def test_a_note_whose_timbre_keeps_moving_reads_a_wider_travel(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    times = np.arange(int(1.5 * SR), dtype=np.float64) / SR
    climbing = ROOT_HZ + (8.0 * ROOT_HZ - ROOT_HZ) * times / times[-1]
    sweep = np.sin(2.0 * np.pi * np.cumsum(climbing) / SR)
    moving = _describe(sweep, descriptor_config, features_config)
    holding = _describe(ringing_note(ROOT_HZ), descriptor_config, features_config)

    assert moving.movement.travel_db_per_s > holding.movement.travel_db_per_s
    assert moving.movement.change_db_per_s > holding.movement.change_db_per_s
