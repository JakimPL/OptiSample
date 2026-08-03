from __future__ import annotations

from dataclasses import dataclass

from optisample.calibrate.ranking.grid import RankingGrid, widened_encodings
from optisample.calibrate.ranking.pairs import ListeningPair, PairQuota, listening_pairs
from optisample.calibrate.ranking.renditions import ClipRenditions, clip_renditions
from optisample.optimize.orchestrate import RunInputs
from optisample.optimize.tasks import EvalContext, PitchTask
from optisample.progress import ProgressSink

_PRICE_LABEL = "Pricing encodings"


@dataclass(frozen=True)
class RankingSettings:
    """What a listening set is built to: the grid it prices, the questions it asks, and the draw blinding it.

    ``byte_tolerance`` is how near two encodings must come in size for the pair to be asked as a trade,
    as a share of the larger. ``seed`` settles both the order the pairs are met in and the side each
    member takes, so one seed reproduces the whole set.
    """

    grid: RankingGrid
    quota: PairQuota
    byte_tolerance: float
    seed: int


@dataclass(frozen=True)
class RankingSet:
    """A listening set as it stands before anything is written: what was priced, and what will be asked.

    ``context`` is what every member was priced under and what rebuilds any of them
    (:func:`~optisample.calibrate.ranking.renditions.rendered`), so the set carries the whole of what a
    writer needs to put the audio a listener judges on disk.
    """

    clips: tuple[ClipRenditions, ...]
    pairs: tuple[ListeningPair, ...]
    context: EvalContext

    @property
    def priced(self) -> int:
        """How many encodings were rendered and scored to choose the set's pairs from."""
        return sum(len(clip.renditions) for clip in self.clips)


def _clip(task: PitchTask, inputs: RunInputs, grid: RankingGrid) -> ClipRenditions:
    """One pitch priced under the sweep's own encodings and the rungs ``grid`` widens them by."""
    context = inputs.context
    encodings = widened_encodings(
        inputs.reduction.encodings()[task.pitch],
        grid,
        context.sweep,
        sample_rate=context.sample_rate,
    )
    return clip_renditions(task, encodings, context)


def assemble_ranking(inputs: RunInputs, settings: RankingSettings, progress: ProgressSink) -> RankingSet:
    """Price every played pitch's encodings and choose the pairs a listener is asked to rank.

    The run is the one the allocation itself prepares, so the operating points in the set are the very
    points the sweep prices and a label reaches the plan as it stands. Pricing runs pitch by pitch and
    keeps only each encoding's bytes and distortion, which holds a whole instrument's grid in the memory
    one reconstruction takes.
    """
    tracked = progress.track(inputs.tasks, label=_PRICE_LABEL, total=len(inputs.tasks))
    clips = tuple(_clip(task, inputs, settings.grid) for task in tracked)
    return RankingSet(
        clips=clips,
        pairs=listening_pairs(
            clips,
            settings.quota,
            byte_tolerance=settings.byte_tolerance,
            seed=settings.seed,
        ),
        context=inputs.context,
    )
