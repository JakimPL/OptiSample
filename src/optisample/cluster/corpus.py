from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Final

from optisample.cluster.descriptor import SampleDescriptor, describe
from optisample.cluster.representative import MemberReadings
from optisample.cluster.space import SampleSpace, sample_space
from optisample.cluster.stages import RecordingCorpus, StageRecording
from optisample.config.cluster import DescriptorConfig, SpaceConfig
from optisample.config.loop import FeatureConfig
from optisample.parallel import map_workers
from optisample.progress import ProgressSink

_DESCRIBE_LABEL: Final = "Reading recordings into blocks"


def _described(recording: StageRecording, *, features: FeatureConfig, config: DescriptorConfig) -> SampleDescriptor:
    """One recording read into its blocks, at the pitch its key says it was played at.

    Stated as a module-level function of the recording alone so a pool of workers is handed something it
    can pickle, with the two run-wide readings bound by the caller.
    """
    return describe(
        recording.signal,
        recording.sample_rate,
        root_hz=recording.root_hz,
        features=features,
        config=config,
    )


@dataclass(frozen=True)
class DescribedCorpus:
    """A gathered corpus beside the blocks each of its recordings was read into, held in one order.

    ``recordings`` and ``descriptors`` stand at matching positions, so an index answers in both: a point
    picked out of a space names the take it came from, the file that take is stored in, and the playing
    time the material gives it. The reading behind the descriptors is the expensive half of the work and
    is settled once here, which lets :meth:`space` be re-read as often as a caller cares to move the
    weights.
    """

    corpus: RecordingCorpus
    descriptors: tuple[SampleDescriptor, ...]

    @property
    def recordings(self) -> tuple[StageRecording, ...]:
        """The takes the descriptors were read from, in the order the space places them."""
        return self.corpus.recordings

    @property
    def size(self) -> int:
        """How many recordings the corpus holds."""
        return len(self.descriptors)

    @property
    def readings(self) -> MemberReadings:
        """What each recording carries beyond its place in the space: the say it holds and how long it rings."""
        return self.corpus.readings

    def space(self, config: SpaceConfig) -> SampleSpace:
        """The corpus placed as points whose plain distance is the weighted distance across the blocks.

        The blocks are already read, so a caller moving a weight pays for the scaling and the concatenation
        alone and sees the geometry every group and representative is settled in shift as it moves.
        """
        return sample_space(self.descriptors, config)


def describe_corpus(
    corpus: RecordingCorpus,
    *,
    features: FeatureConfig,
    config: DescriptorConfig,
    workers: int,
    progress: ProgressSink,
) -> DescribedCorpus:
    """Every recording of ``corpus`` read into its blocks, shared across ``workers`` processes.

    Each recording is read from itself and the two configs alone, so the work splits cleanly and comes back
    in the order the corpus lists it however the pool happened to reach it.
    """
    descriptors = map_workers(
        partial(_described, features=features, config=config),
        corpus.recordings,
        workers=workers,
        label=_DESCRIBE_LABEL,
        progress=progress,
    )
    return DescribedCorpus(corpus=corpus, descriptors=tuple(descriptors))
