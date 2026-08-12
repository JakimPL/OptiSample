from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from notebooks.utils import loading, viz
from optisample.cluster.corpus import DescribedCorpus, describe_corpus
from optisample.cluster.partition import partition
from optisample.cluster.representative import Group, grouping
from optisample.cluster.space import Coordinates, SampleSpace, pairwise_distances
from optisample.cluster.stages import Stage, StageCorpus, StageRecording
from optisample.config import OptiConfig
from optisample.config.cluster import PartitionConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.spectral import SpectralConfig
from optisample.keys import SampleKey
from optisample.model import Manifest
from optisample.music import midi_to_freq
from optisample.progress import NO_PROGRESS

SR = 44_100

Demo = tuple[Path, Manifest]

_PITCHES = (48, 52, 60, 64, 72, 76)  # takes the fixture corpus holds, two registers of one triad apiece
_VELOCITY = 96
_BALANCE = (1.0, 0.5, 0.25, 0.12)  # the amplitude each partial of a fixture take rings at
_DECAY_DB_PER_S = 12.0
_DURATION_S = 1.2
_NOISE_FLOOR = 1e-5
_GROUPS = 3
_ALONE = 1  # the caller's own process, which carries a fixture's reading as it stands


def _ringing(root_hz: float, seed: int) -> NDArray[np.float64]:
    """A harmonic take of ``root_hz`` falling at a steady number of decibels a second, on a noise floor."""
    times = np.arange(int(_DURATION_S * SR), dtype=np.float64) / SR
    partials = sum(
        amplitude * np.sin(2.0 * np.pi * root_hz * (index + 1) * times) for index, amplitude in enumerate(_BALANCE)
    )
    falling = partials * 10.0 ** (-_DECAY_DB_PER_S * times / 20.0)
    return np.asarray(
        falling + np.random.default_rng(seed).standard_normal(times.size) * _NOISE_FLOOR, dtype=np.float64
    )


@dataclass(frozen=True)
class Clustered:
    """A corpus read, placed and cut, which is the state every cluster panel draws from."""

    described: DescribedCorpus
    space: SampleSpace
    groups: tuple[Group, ...]
    distances: Coordinates
    cutting: PartitionConfig


@pytest.fixture
def clustered(config: OptiConfig) -> Clustered:
    """A handful of synthesised takes carried the whole way a notebook carries them, ready to draw."""
    corpus = StageCorpus(
        stage=Stage.SUBSET,
        instrument_id="Piano",
        recordings=tuple(
            StageRecording(
                file=Path(f"{pitch:03d}.wav"),
                key=SampleKey(pitch=pitch, velocity=_VELOCITY),
                signal=_ringing(midi_to_freq(pitch), index),
                sample_rate=SR,
                weight=float(index + 1),
            )
            for index, pitch in enumerate(_PITCHES)
        ),
    )
    described = describe_corpus(
        corpus,
        features=config.loop.features,
        config=config.cluster.descriptor,
        workers=_ALONE,
        progress=NO_PROGRESS,
    )
    space = described.space(config.cluster.space)
    cutting = config.cluster.partition.model_copy(update={"groups": _GROUPS})
    cut_now = partition(space.coordinates, groups=_GROUPS, config=cutting)
    return Clustered(
        described=described,
        space=space,
        groups=grouping(space.coordinates, cut_now.labels, described.readings, config=cutting),
        distances=pairwise_distances(space.coordinates),
        cutting=cutting,
    )


@pytest.fixture
def spectrogram_style(spectral_config: SpectralConfig, metrics_config: MetricsConfig) -> viz.SpectrogramStyle:
    """The reading every spectrogram in these tests is drawn under, taken off the shipped config."""
    return viz.SpectrogramStyle(
        params=spectral_config.stft, dynamic_range_db=metrics_config.preprocess.dynamic_range_db
    )


@pytest.fixture
def tone() -> Callable[..., NDArray[np.float64]]:
    """Factory: a pure sine at ``SR`` (default 440 Hz, 1 s, 0.5 amplitude)."""

    def _tone(freq: float = 440.0, duration: float = 1.0, amplitude: float = 0.5) -> NDArray[np.float64]:
        times = np.arange(int(duration * SR)) / SR
        return amplitude * np.sin(2.0 * np.pi * freq * times)

    return _tone


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory, config: OptiConfig) -> Demo:
    root = tmp_path_factory.mktemp("demo")
    manifest = loading.load(loading.ensure_demo(root, config.synth))
    return root, manifest
