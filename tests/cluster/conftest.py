from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import OptiConfig
from optisample.config.cluster import DescriptorConfig, PartitionConfig, SpaceConfig
from optisample.config.loop import FeatureConfig

SR = 44_100

Signal = NDArray[np.float64]

_NOISE_FLOOR = 1e-5  # the floor a synthesized take stands on, so every bin carries content to read
_BALANCE = (1.0, 0.5, 0.25, 0.12, 0.06)  # the amplitude each partial of a bright take rings at
_DULL_BALANCE = (1.0, 0.05, 0.01, 0.005, 0.002)  # the same note with its upper partials taken away


@pytest.fixture
def descriptor_config(config: OptiConfig) -> DescriptorConfig:
    """How a recording is read into the sample space, taken off the shipped config."""
    return config.cluster.descriptor


@pytest.fixture
def features_config(config: OptiConfig) -> FeatureConfig:
    """How a recording is read as a series of timbre frames, taken off the shipped config."""
    return config.loop.features


@pytest.fixture
def space_config(config: OptiConfig) -> SpaceConfig:
    """How the blocks are scaled and weighed into one space, taken off the shipped config."""
    return config.cluster.space


@pytest.fixture
def partition_config(config: OptiConfig) -> PartitionConfig:
    """How a space is cut into groups and which member stands for each, taken off the shipped config."""
    return config.cluster.partition


@pytest.fixture
def ringing_note() -> Callable[..., Signal]:
    """Factory: a harmonic tone decaying at a steady number of decibels a second, on a noise floor.

    Stating the take as a balance of partials over one pitch is what lets a test move exactly one property
    at a time -- the register, the balance, the rate it falls at, or the level it was captured at -- and
    read what the descriptor makes of that move.
    """

    def _note(
        root_hz: float,
        *,
        balance: Sequence[float] = _BALANCE,
        duration_s: float = 1.5,
        decay_db_per_s: float = 12.0,
        seed: int = 3,
    ) -> Signal:
        times = np.arange(int(duration_s * SR), dtype=np.float64) / SR
        partials = sum(
            amplitude * np.sin(2.0 * np.pi * root_hz * (index + 1) * times) for index, amplitude in enumerate(balance)
        )
        falling = partials * 10.0 ** (-decay_db_per_s * times / 20.0)
        return np.asarray(
            falling + np.random.default_rng(seed).standard_normal(times.size) * _NOISE_FLOOR, dtype=np.float64
        )

    return _note


@pytest.fixture
def dull_balance() -> tuple[float, ...]:
    """The balance a take of the same note rings with once its upper partials are taken away."""
    return _DULL_BALANCE
