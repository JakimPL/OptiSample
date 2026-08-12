from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from optisample.cluster.descriptor import SampleDescriptor, describe
from optisample.cluster.space import (
    Block,
    block_weight,
    mean_pairwise_square,
    sample_space,
    scale_to_unit_spread,
    standardize,
)
from optisample.config.cluster import DescriptorConfig, SpaceConfig
from optisample.config.loop import FeatureConfig
from tests.cluster.conftest import SR, Signal

_REGISTERS = (110.0, 165.0, 220.0, 330.0)  # the pitches every sound in the corpus is taken at
_BRIGHT = (1.0, 0.6, 0.4, 0.25, 0.15)  # a take ringing with its upper partials well up
_MELLOW = (1.0, 0.1, 0.03, 0.01, 0.004)  # the same note with those partials taken away
_UNSCALED = (1.0, 100.0, 0.01, 5.0)  # columns stated in wildly different units, as the raw blocks are


@pytest.fixture
def corpus(
    descriptor_config: DescriptorConfig,
    features_config: FeatureConfig,
    ringing_note: Callable[..., Signal],
) -> tuple[SampleDescriptor, ...]:
    """Two sounds, one bright and one mellow, each taken at four registers and two rates of decay."""
    return tuple(
        describe(
            ringing_note(root_hz, balance=balance, decay_db_per_s=decay_db_per_s),
            SR,
            root_hz=root_hz,
            features=features_config,
            config=descriptor_config,
        )
        for balance in (_BRIGHT, _MELLOW)
        for root_hz in _REGISTERS
        for decay_db_per_s in (9.0, 18.0)
    )


def test_standardize_states_every_column_in_spreads_of_its_own() -> None:
    values = np.random.default_rng(11).normal(size=(30, 4)) * np.asarray(_UNSCALED)

    standardized = standardize(values)

    assert np.allclose(standardized.mean(axis=0), 0.0)
    assert np.allclose(standardized.std(axis=0), 1.0)


def test_a_column_holding_one_value_across_the_corpus_reads_zero() -> None:
    values = np.column_stack((np.arange(6.0), np.full(6, 3.0)))

    assert np.allclose(standardize(values)[:, 1], 0.0)


def test_the_mean_pairwise_square_matches_the_distances_it_stands_for() -> None:
    values = np.random.default_rng(5).normal(size=(12, 3))

    assert mean_pairwise_square(values) == pytest.approx(float(np.square(pdist(values)).mean()))


def test_a_corpus_of_one_recording_holds_no_pair_to_read() -> None:
    assert mean_pairwise_square(np.zeros((1, 4))) == 0.0


def test_scaling_leaves_a_corpus_standing_at_one_point_where_it_is() -> None:
    values = np.full((5, 3), 2.0)

    assert np.array_equal(scale_to_unit_spread(values), values)


def test_scaling_brings_any_units_to_the_same_pairwise_spread() -> None:
    values = np.random.default_rng(7).normal(size=(20, 4)) * np.asarray(_UNSCALED)

    assert mean_pairwise_square(scale_to_unit_spread(standardize(values))) == pytest.approx(1.0)


def test_every_block_carries_the_say_its_weight_gives_it(
    corpus: tuple[SampleDescriptor, ...], space_config: SpaceConfig
) -> None:
    """A weight is what a block's spread comes to, which is what makes one weight mean one thing anywhere."""
    space = sample_space(corpus, space_config)

    for block in Block:
        assert mean_pairwise_square(space.block(block)) == pytest.approx(block_weight(block, space_config))


def test_a_block_weighed_at_nothing_stands_at_the_origin(
    corpus: tuple[SampleDescriptor, ...], space_config: SpaceConfig
) -> None:
    space = sample_space(corpus, space_config.model_copy(update={"envelope_weight": 0.0}))

    assert np.array_equal(space.block(Block.ENVELOPE), np.zeros_like(space.block(Block.ENVELOPE)))


def test_the_distance_is_the_weighted_sum_of_the_blocks_own_distances(
    corpus: tuple[SampleDescriptor, ...], space_config: SpaceConfig
) -> None:
    """Weighing the contour at nothing reproduces the pure-timbre distance, and adds back exactly its share."""
    timbre = sample_space(corpus, space_config.model_copy(update={"envelope_weight": 0.0}))
    whole = sample_space(corpus, space_config)

    assert np.allclose(
        np.square(pdist(whole.coordinates)),
        np.square(pdist(timbre.coordinates)) + np.square(pdist(whole.block(Block.ENVELOPE))),
    )


def test_the_spans_lay_the_blocks_side_by_side(corpus: tuple[SampleDescriptor, ...], space_config: SpaceConfig) -> None:
    space = sample_space(corpus, space_config)
    spans = [space.spans[block] for block in Block]

    assert spans[0].start == 0
    assert all(later.start == earlier.stop for earlier, later in zip(spans, spans[1:]))
    assert spans[-1].stop == space.dimensions
    assert sum(span.columns for span in spans) == space.dimensions


def test_the_sustain_is_read_over_the_depths_the_corpus_shares(
    corpus: tuple[SampleDescriptor, ...], space_config: SpaceConfig
) -> None:
    """Asking every recording to arrive keeps the depths they all reached, and asking less keeps more."""
    demanding = sample_space(corpus, space_config.model_copy(update={"min_reached_share": 1.0}))
    lenient = sample_space(corpus, space_config.model_copy(update={"min_reached_share": 0.1}))

    assert int(demanding.depths.sum()) <= int(lenient.depths.sum())
    assert demanding.spans[Block.SUSTAIN].columns == int(demanding.depths.sum()) * corpus[0].columns
    assert lenient.spans[Block.SUSTAIN].columns == int(lenient.depths.sum()) * corpus[0].columns


def test_the_space_holds_one_point_per_recording(
    corpus: tuple[SampleDescriptor, ...], space_config: SpaceConfig
) -> None:
    space = sample_space(corpus, space_config)

    assert space.samples == len(corpus)
    assert space.coordinates.shape == (space.samples, space.dimensions)


def test_a_space_is_placed_from_a_recording_at_the_least(space_config: SpaceConfig) -> None:
    with pytest.raises(ValueError, match="at least one recording"):
        sample_space((), space_config)


def test_two_takes_of_one_sound_stand_closer_than_two_sounds(
    corpus: tuple[SampleDescriptor, ...], space_config: SpaceConfig
) -> None:
    """The whole point of the space, read on material: a sound gathers and two sounds part company."""
    space = sample_space(corpus, space_config)
    takes = len(corpus) // 2
    distances = np.asarray(
        [[float(np.linalg.norm(one - other)) for other in space.coordinates] for one in space.coordinates]
    )
    within = np.mean([distances[:takes, :takes].mean(), distances[takes:, takes:].mean()])

    assert within < distances[:takes, takes:].mean()
