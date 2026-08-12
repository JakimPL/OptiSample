from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from optisample.cluster.corpus import DescribedCorpus, describe_corpus
from optisample.cluster.space import Block
from optisample.cluster.stages import Stage, StageCorpus, StageRecording
from optisample.config.cluster import DescriptorConfig, SpaceConfig
from optisample.config.loop import FeatureConfig
from optisample.keys import SampleKey
from optisample.music import midi_to_freq
from optisample.progress import NO_PROGRESS
from tests.cluster.conftest import SR, Signal

_INSTRUMENT = "Piano"
_PITCHES = (48, 55, 60, 67)
_VELOCITY = 96
_SHARED = 2  # processes the fan-out is read across, which is what puts the work through a pickle
_ALONE = 1  # the caller's own process, which carries the work as it stands
_SILENT = 0.0  # the say a block left out of the space carries


def _recording(pitch: int, signal: Signal, *, weight: float) -> StageRecording:
    """One take of ``pitch`` as the corpus lists it, carrying the playing time its notes ask of it."""
    return StageRecording(
        file=Path(f"{pitch:03d}.wav"),
        key=SampleKey(pitch=pitch, velocity=_VELOCITY),
        signal=signal,
        sample_rate=SR,
        weight=weight,
    )


def _corpus(ringing_note: Callable[..., Signal], pitches: Sequence[int] = _PITCHES) -> StageCorpus:
    """A stage's worth of takes, one per pitch, each carrying a playing time of its own."""
    return StageCorpus(
        stage=Stage.SUBSET,
        instrument_id=_INSTRUMENT,
        recordings=tuple(
            _recording(pitch, ringing_note(midi_to_freq(pitch)), weight=float(index + 1))
            for index, pitch in enumerate(pitches)
        ),
    )


def _described(
    corpus: StageCorpus,
    features_config: FeatureConfig,
    descriptor_config: DescriptorConfig,
    *,
    workers: int = _ALONE,
) -> DescribedCorpus:
    """The corpus read into its blocks, which is what every reading below is taken off."""
    return describe_corpus(
        corpus,
        features=features_config,
        config=descriptor_config,
        workers=workers,
        progress=NO_PROGRESS,
    )


def test_every_recording_answers_one_descriptor_in_the_order_the_corpus_lists_it(
    ringing_note: Callable[..., Signal],
    features_config: FeatureConfig,
    descriptor_config: DescriptorConfig,
) -> None:
    """A place in the corpus is a place in the descriptors, which is what lets a point name its own take."""
    corpus = _corpus(ringing_note)
    described = _described(corpus, features_config, descriptor_config)

    assert described.size == len(_PITCHES)
    assert described.recordings == corpus.recordings
    assert len({descriptor.columns for descriptor in described.descriptors}) == 1


def test_a_recording_is_read_at_the_pitch_its_key_says_it_was_played_at(
    ringing_note: Callable[..., Signal],
    features_config: FeatureConfig,
    descriptor_config: DescriptorConfig,
) -> None:
    """The register a take is read on comes off its own key, so two notes are told apart by their balance.

    A note and the same balance an octave up sit close on the relative basis exactly because each was read
    over its own partials, which is the reading the corpus hands every recording.
    """
    corpus = _corpus(ringing_note, pitches=(48, 60))
    described = _described(corpus, features_config, descriptor_config)
    apart = float(np.abs(described.descriptors[0].onset - described.descriptors[1].onset).max())

    assert apart < 1.0


def test_sharing_the_work_across_processes_answers_what_one_process_answers(
    ringing_note: Callable[..., Signal],
    features_config: FeatureConfig,
    descriptor_config: DescriptorConfig,
) -> None:
    """The fan-out states a reading of the recording alone, so a pool answers what the caller would have."""
    corpus = _corpus(ringing_note)
    carried = _described(corpus, features_config, descriptor_config, workers=_ALONE)
    shared = _described(corpus, features_config, descriptor_config, workers=_SHARED)

    for alone, across in zip(carried.descriptors, shared.descriptors, strict=True):
        assert alone.onset == pytest.approx(across.onset)
        assert alone.sustain == pytest.approx(across.sustain)
        assert alone.envelope.values == pytest.approx(across.envelope.values)


def test_each_recording_carries_the_playing_time_the_material_gives_it(
    ringing_note: Callable[..., Signal],
    features_config: FeatureConfig,
    descriptor_config: DescriptorConfig,
) -> None:
    """The weights stand where their recordings do, which is what a weighted medoid reads them by."""
    described = _described(_corpus(ringing_note), features_config, descriptor_config)

    assert described.weights == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_the_space_places_one_point_per_recording(
    ringing_note: Callable[..., Signal],
    features_config: FeatureConfig,
    descriptor_config: DescriptorConfig,
    space_config: SpaceConfig,
) -> None:
    """Reading the corpus into a space leaves the corpus's own order standing, point for recording."""
    described = _described(_corpus(ringing_note), features_config, descriptor_config)
    space = described.space(space_config)

    assert space.samples == described.size


def test_moving_a_weight_re_reads_the_space_off_the_blocks_already_read(
    ringing_note: Callable[..., Signal],
    features_config: FeatureConfig,
    descriptor_config: DescriptorConfig,
    space_config: SpaceConfig,
) -> None:
    """One reading of the recordings serves any weighing of it, which is what makes a weight worth moving.

    A block given no say lays down columns holding zero, so the space narrows to what the other blocks
    carry while the descriptors behind it stay exactly as they were read.
    """
    described = _described(_corpus(ringing_note), features_config, descriptor_config)
    weighed = described.space(space_config)
    without = described.space(space_config.model_copy(update={"envelope_weight": _SILENT}))

    assert weighed.dimensions == without.dimensions
    assert np.abs(without.block(Block.ENVELOPE)).max() == pytest.approx(_SILENT)
    assert np.abs(weighed.block(Block.ENVELOPE)).max() > _SILENT
