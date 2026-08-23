from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from research.config_tree import bundled_root

_SUFFIX: Final = ".yaml"
_PACKAGE: Final = Path("src/optisample")
_SOURCES: Final = "*.py"


@dataclass(frozen=True)
class Knob:
    """One tunable the shipped tree states, and where the package reads the name it is stated under."""

    dotted: str
    leaf: str
    value: Any
    readers: tuple[str, ...]

    @property
    def read(self) -> bool:
        """Whether any module of the package names this knob, which is what makes it reachable at all."""
        return bool(self.readers)


def _leaves(payload: Any, prefix: str) -> Iterator[tuple[str, str, Any]]:
    """Every leaf of ``payload``, as the dotted path reaching it, its own name, and the value it holds."""
    if not isinstance(payload, dict):
        yield prefix, prefix.rsplit(".", maxsplit=1)[-1], payload
        return

    for name, value in payload.items():
        yield from _leaves(value, f"{prefix}.{name}" if prefix else str(name))


def knobs(root: Path | None = None) -> tuple[Knob, ...]:
    """Every knob the shipped tree states, beside the modules naming it.

    A knob is looked up by the name it is stated under, which is how a config field reaches the code, so
    a name no module carries is one nothing can act on. A name several groups share reads as carried by
    every module naming any of them, so this states the census a knob belongs to rather than proving one
    group's field is read.
    """
    tree = root or bundled_root()
    sources = {path: path.read_text(encoding="utf-8") for path in _PACKAGE.rglob(_SOURCES)}
    gathered: list[Knob] = []
    for path in sorted(tree.rglob(f"*{_SUFFIX}")):
        group = ".".join([*path.relative_to(tree).parts[:-1], path.stem])
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for dotted, leaf, value in _leaves(payload, ""):
            pattern = re.compile(rf"\b{re.escape(leaf)}\b")
            readers = tuple(str(source).replace("\\", "/") for source, text in sources.items() if pattern.search(text))
            gathered.append(Knob(dotted=f"{group}.{dotted}", leaf=leaf, value=value, readers=readers))

    return tuple(gathered)


def unread(inventory: Mapping[str, Knob]) -> tuple[str, ...]:
    """The knobs no module of the package names, which are the ones a run cannot act on."""
    return tuple(dotted for dotted, knob in inventory.items() if not knob.read)
