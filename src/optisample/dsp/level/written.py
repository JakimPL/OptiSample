from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from optisample.dsp.level.level import UNITY_DB, Level


@dataclass(frozen=True)
class WrittenLevel:
    """A level as a format states it: a shape that only attenuates, beside the level its unity stands for.

    A tracker's volume envelope holds steps from silence up to a full one and multiplies them onto a voice,
    so the loudest moment it can state is the level of the material under it. Writing a level therefore
    splits in two: ``shape`` states the decline against that full step and never rises above it, and
    ``reference_db`` records what the full step is worth, which is what the levels written beside the
    envelope are staged against.

    Stating every instrument of one plan against a single reference is what keeps the balance between them:
    two shapes each taken against its own loudest moment would stand exactly the difference between those
    moments apart. :func:`loudest_db` is that shared reference.
    """

    shape: Level
    reference_db: float

    @property
    def absolute(self) -> Level:
        """The level this states outright, which is the shape stood back up at its own reference."""
        return self.shape.scaled_db(self.reference_db)


def written_level(level: Level, *, reference_db: float) -> WrittenLevel:
    """``level`` restated as a shape against ``reference_db``, which is what a format has room to write.

    Raises:
        ValueError: when the level rises above the reference, which a shape that only attenuates has no
            step to state.
    """
    if level.peak_db > reference_db:
        raise ValueError(f"a written shape attenuates, against the {level.peak_db - reference_db} dB above unity")

    return WrittenLevel(shape=level.scaled_db(-reference_db), reference_db=reference_db)


def loudest_db(levels: Sequence[Level]) -> float:
    """The loudest moment any of ``levels`` reaches, which is the reference they are all written against.

    A set holding nothing names unity itself, there being no material to stand under.
    """
    return max((level.peak_db for level in levels), default=UNITY_DB)
