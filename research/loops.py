from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from research.materials import Corpus, Material
from research.runner import Cell, Source, Stage

_BUDGET_KB: Final = 256.0  # a loop-only run reads no budget, so one value stands for every cell

CLASS_SPAN: Final = (
    "Big Band - Staccato",
    "Ensemble - Strings C",
    "Big Band - Horn A",
    "Drums - Secondary Kit",
    "Ensemble - Solo",
    "Big Band - Flute",
    "Big Band - Long",
    "Big Band - Trumpet",
    "Bass",
)
CHEAP: Final = CLASS_SPAN[:5]
DEEP: Final = ("Bass", "Piano")


def chosen(corpus: Corpus, wanted: Sequence[str]) -> tuple[Material, ...]:
    """The materials ``wanted`` names, in the order the catalogue puts them.

    Raises:
        KeyError: when a name matches no dataset, so a grid states its materials once and fails before a
            run is spent rather than quietly measuring fewer than it asked for.
    """
    listed = {material.instrument_id: material for material in corpus.sources()}
    missing = [name for name in wanted if name not in listed]
    if missing:
        raise KeyError(f"the corpus holds no dataset named {missing}")

    return tuple(material for material in corpus.sources() if material.instrument_id in set(wanted))


def _grid(family: str, knob: str, settings: Sequence[Any], materials: Sequence[Material]) -> tuple[Cell, ...]:
    """One cell per setting of ``knob`` per material, which is how a family answers for every class."""
    return tuple(
        Cell(
            family=family,
            label=f"{setting:g}" if isinstance(setting, float) else str(setting),
            material=material,
            stage=Stage.LOOP,
            budget_kb=_BUDGET_KB,
            overrides={knob: setting},
        )
        for setting in settings
        for material in materials
    )


_WHOLE_CORPUS: Final = ("wrap_gate", "octaves_below")  # families the struck material answers for too

_SPAN_KNOBS: Final[Mapping[str, tuple[str, tuple[Any, ...]]]] = {
    "wrap_gate": ("loop.frontier.max_wrap_distance_db", (1.5, 3.0, 6.0, 9.0, 15.0, 30.0)),
    "settle": ("loop.features.settle_db_per_s", (2.0, 5.0, 10.0, 15.0, 30.0, 60.0)),
    "seam_gate": ("loop.quality.max_seam_step", (0.5, 1.0, 2.0, 4.0, 8.0, 32.0)),
    "octaves_below": ("loop.geometry.octaves_below", (0, 1, 2, 3)),
}

_CHEAP_KNOBS: Final[Mapping[str, tuple[str, tuple[Any, ...]]]] = {
    "offers": ("loop.frontier.max_offers", (1, 2, 3, 5, 8, 12)),
    "reach": ("loop.frontier.max_reach_s", (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)),
    "spectral_gate": ("loop.quality.max_spectral_distance_db", (3.0, 6.0, 12.0, 18.0, 30.0, 60.0)),
    "min_loop_s": ("loop.geometry.min_loop_s", (0.01, 0.03, 0.1, 0.3, 1.0)),
    "octave_margin": ("loop.geometry.octave_margin", (0.5, 0.7, 0.85, 0.95, 1.0)),
    "attack_room": ("loop.geometry.max_attack_s", (0.05, 0.15, 0.5, 1.0, 2.0)),
    "correlation": ("loop.geometry.min_correlation", (0.05, 0.15, 0.3, 0.5, 0.8)),
}

_DEEP_KNOBS: Final[Mapping[str, tuple[str, tuple[Any, ...]]]] = {
    "min_periods": ("loop.geometry.min_periods", (1, 4, 8, 16, 32, 64)),
}


def cells(corpus: Corpus) -> tuple[Cell, ...]:
    """Every loop-stage cell the sweep runs, widest grids on the cheapest material.

    Three rings share the cost. The gates deciding how much material loops at all run across every
    acoustic class, since that is where a class is expected to part company; the geometry and the frontier
    knobs run on the cheap end of the corpus, where a threshold reads as clearly for a tenth of the time;
    and the floors on loop length run on the low material they are expected to bind on. The two families
    whose answers turn on how a note's own period reads additionally run on the piano, which is both
    the material the shipped defaults were written against and the one whose strings ring widest of
    any whole multiple of their pitch.
    """
    span = chosen(corpus, CLASS_SPAN)
    whole = chosen(corpus, (*CLASS_SPAN, "Piano"))
    cheap = chosen(corpus, CHEAP)
    deep = chosen(corpus, (*CHEAP, *DEEP))
    gathered = [
        *(
            cell
            for family, (knob, settings) in _SPAN_KNOBS.items()
            for cell in _grid(family, knob, settings, whole if family in _WHOLE_CORPUS else span)
        ),
        *(cell for family, (knob, settings) in _CHEAP_KNOBS.items() for cell in _grid(family, knob, settings, cheap)),
        *(cell for family, (knob, settings) in _DEEP_KNOBS.items() for cell in _grid(family, knob, settings, deep)),
    ]
    return tuple(gathered)


KITS: Final = ("Drums - Supplementary Kit", "Drums - Secondary Kit")
SUSTAINED: Final = ("Ensemble - Strings C",)
_ONE_SHOT_BUDGETS: Final = (512.0, 2048.0)  # budgets a kit stored key by key has a plan inside at all
_NO_GROUPING: Final = {"reduce.grouping.max_zone_semitones": 0}
_UNLOOPED: Final = ("--no-loop",)


def one_shot(corpus: Corpus) -> tuple[Cell, ...]:
    """Whether storing loops is worth it, asked of the material the one-shot recipe is used on.

    Every kit runs with its keys left standing on their own, since a percussion map holds one instrument
    per key and merging them measures the merge rather than the loop; the sustained instrument beside them
    runs at the shipped width, which is what the same question looks like on material a loop was meant for.
    Each pair of cells differs by the loop stage alone, so the objective between them is what loops bought.
    """
    return tuple(
        Cell(
            family="one_shot",
            label=f"{'unlooped' if unlooped else 'looped'}@{budget_kb:g}",
            material=material,
            stage=Stage.OPTIMIZE,
            budget_kb=budget_kb,
            overrides=_NO_GROUPING if material.instrument_id in KITS else {},
            flags=_UNLOOPED if unlooped else (),
            source=Source.REDUCED,
        )
        for budget_kb in _ONE_SHOT_BUDGETS
        for unlooped in (False, True)
        for material in chosen(corpus, (*KITS, *SUSTAINED))
    )


_WORTH: Final = ("Big Band - Staccato", "Ensemble - Strings C", "Big Band - Horn A")
_PREFIX_BUDGET_KB: Final = 512.0
_ONE_STORAGE: Final[Mapping[str, Any]] = {"optimize.sweep.carriers": [False]}
_PREFIXES: Final[tuple[tuple[str, Mapping[str, Any]], ...]] = (
    ("shipped", {}),
    ("settle30", {"loop.features.settle_db_per_s": 30.0}),
    ("settle60", {"loop.features.settle_db_per_s": 60.0}),
    ("attack0.15", {"loop.geometry.max_attack_s": 0.15}),
    ("settle30_timbre6", {"loop.features.settle_db_per_s": 30.0, "loop.quality.max_spectral_distance_db": 6.0}),
)


def prefix_worth(corpus: Corpus) -> tuple[Cell, ...]:
    """What an allocation makes of the shorter prefixes an earlier-opening loop window offers it.

    The loop stage says the material wraps well several times earlier than the shipped settings let it be
    stored from, and that no gate turns the result down; whether the bytes that frees are worth what the
    earlier wrap sounds like is a question only a budget answers. Each setting reaches the same place from
    a different side -- the settling threshold reads it off the material, the attack cap imposes it -- and
    the last pairs the loosest threshold with a timbre gate tight enough to still say no, which is what
    says whether the two knobs belong together.

    Every cell stores its recordings as they were played, which holds the storage axis still while the
    prefix moves: what separates two cells is then the loop window alone, and each of them prices half the
    encodings a run offering both storages would.
    """
    return tuple(
        Cell(
            family="prefix_worth",
            label=name,
            material=material,
            stage=Stage.OPTIMIZE,
            budget_kb=_PREFIX_BUDGET_KB,
            overrides={**_ONE_STORAGE, **overrides},
            source=Source.REDUCED,
        )
        for name, overrides in _PREFIXES
        for material in chosen(corpus, _WORTH)
    )
