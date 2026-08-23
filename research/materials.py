from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

import yaml

from optisample.io.note_extractor import NOTES_SUFFIX, read_manifest

CATALOGUE: Final = Path(__file__).with_name("corpus.yaml")
UNCLASSIFIED: Final = "unclassified"
_FAST_NOTES: Final = 70  # notes a whole sweep is run over, since a stage answers in seconds at this size
_MEDIUM_NOTES: Final = 250  # notes a contrast is measured over, where a stage answers in minutes


@unique
class Tier(StrEnum):
    """How much of a sweep one instrument is worth, read off the notes it carries.

    Cost follows the note count through every stage, so the tier is what says whether an instrument
    belongs in every cell of a grid, in the material contrasts alone, or only in a confirming run.
    """

    FAST = "fast"
    MEDIUM = "medium"
    HEAVY = "heavy"

    @classmethod
    def of(cls, notes: int) -> Tier:
        """The tier an instrument of ``notes`` notes falls in."""
        if notes <= _FAST_NOTES:
            return cls.FAST

        if notes <= _MEDIUM_NOTES:
            return cls.MEDIUM

        return cls.HEAVY


@dataclass(frozen=True)
class Material:
    """One instrument as the research reads it: where its dataset is, and what kind of sound it holds."""

    instrument_id: str
    notes_json: Path
    samples_dir: Path
    notes: int
    material_class: str

    @property
    def tier(self) -> Tier:
        """How much of a sweep this instrument is worth."""
        return Tier.of(self.notes)


@dataclass(frozen=True)
class Corpus:
    """The catalogue a run reads its materials from: the source datasets and the stage already reduced."""

    manifests: Path
    samples: Path
    reduced: Path
    classes: dict[str, str]

    @classmethod
    def load(cls, catalogue: Path = CATALOGUE) -> Corpus:
        """The corpus ``catalogue`` describes."""
        payload = yaml.safe_load(catalogue.read_text(encoding="utf-8"))
        return cls(
            manifests=Path(payload["manifests"]),
            samples=Path(payload["samples"]),
            reduced=Path(payload["reduced"]),
            classes=dict(payload["classes"]),
        )

    def sources(self) -> tuple[Material, ...]:
        """Every source dataset the catalogue points at, in the order a note count puts them."""
        return self._gathered(self.manifests, lambda name: self.samples / name)

    def reduced_sources(self) -> tuple[Material, ...]:
        """Every already-reduced dataset, whose recordings sit beside their manifest."""
        return self._gathered(self.reduced, lambda name: self.reduced / name)

    def _gathered(self, root: Path, samples_of: Callable[[str], Path]) -> tuple[Material, ...]:
        gathered = [
            Material(
                instrument_id=name,
                notes_json=manifest,
                samples_dir=samples_of(name),
                notes=len(read_manifest(manifest).notes),
                material_class=self.classes.get(name, UNCLASSIFIED),
            )
            for manifest, name in (
                (path, path.name[: -len(NOTES_SUFFIX)]) for path in sorted(root.glob(f"*{NOTES_SUFFIX}"))
            )
        ]
        return tuple(sorted(gathered, key=lambda material: material.notes))
