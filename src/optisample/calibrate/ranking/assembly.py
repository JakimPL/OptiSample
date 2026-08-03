from __future__ import annotations

from collections.abc import Sequence
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
    as a share of the larger. ``min_duration_s`` is how long a note class must play to be worth asking
    about, since a degradation shows itself over a decay and a note that stops first leaves a listener
    guessing. ``repeats`` is how many of the chosen questions are put a second time, which is what reads
    a listener's own consistency and gives the agreement a ceiling to be read against. ``seed`` settles
    the order the pairs are met in, where the repeats fall, and the side each member takes, so one seed
    reproduces the whole set.
    """

    grid: RankingGrid
    quota: PairQuota
    byte_tolerance: float
    min_duration_s: float
    repeats: int
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

    @property
    def questions(self) -> int:
        """How many distinct comparisons the set puts, counting one that is asked twice once."""
        return len({pair.question_id for pair in self.pairs})

    @property
    def repeats(self) -> int:
        """How many of the pairs put a question the listener has already been asked."""
        return len(self.pairs) - self.questions


def _audible(tasks: Sequence[PitchTask], min_duration_s: float) -> tuple[PitchTask, ...]:
    """The pitches whose representative note plays long enough for a degradation to show itself.

    Leaving the short notes unpriced spends the listening on questions that can be answered, at the cost
    of a set weighted toward the longer material -- which the report states, so what the labels speak for
    stays clear.
    """
    return tuple(task for task in tasks if task.representative_event.duration_s >= min_duration_s)


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
    keeps only each encoding's bytes, level and distortion, which holds a whole instrument's grid in the
    memory one reconstruction takes.
    """
    audible = _audible(inputs.tasks, settings.min_duration_s)
    tracked = progress.track(audible, label=_PRICE_LABEL, total=len(audible))
    clips = tuple(_clip(task, inputs, settings.grid) for task in tracked)
    return RankingSet(
        clips=clips,
        pairs=listening_pairs(
            clips,
            settings.quota,
            byte_tolerance=settings.byte_tolerance,
            repeats=settings.repeats,
            seed=settings.seed,
        ),
        context=inputs.context,
    )
