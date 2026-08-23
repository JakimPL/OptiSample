from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from shutil import copytree
from typing import Any, Final

import yaml

from optisample.config.loader import load_config

_PACKAGE: Final = "opticonfig"
_SUFFIX: Final = ".yaml"
_STAGE_DEPTH: Final = 2  # a stage group is addressed as "<stage>.<group>", which is a file two names deep

Overrides = Mapping[str, Any]


def bundled_root() -> Path:
    """Where the shipped config tree sits on disk, which every variant is copied from."""
    return Path(str(resources.files(_PACKAGE)))


def locate(root: Path, dotted: str) -> tuple[Path, tuple[str, ...]]:
    """The file ``dotted`` addresses under ``root``, and the key path reaching the value inside it.

    A group of a stage lives two names deep (``reduce.trim.max_length_s`` -> ``reduce/trim.yaml``) and a
    standalone group one (``subset.min_duration_s`` -> ``subset.yaml``), so the longer layout is tried
    first and whatever the file does not account for addresses the value within it.

    Raises:
        KeyError: when neither layout names a file that exists.
    """
    parts = dotted.split(".")
    if len(parts) >= _STAGE_DEPTH:
        staged = root / parts[0] / f"{parts[1]}{_SUFFIX}"
        if staged.is_file():
            return staged, tuple(parts[_STAGE_DEPTH:])

    flat = root / f"{parts[0]}{_SUFFIX}"
    if flat.is_file():
        return flat, tuple(parts[1:])

    raise KeyError(f"no config file under {root} is addressed by {dotted!r}")


def _assigned(payload: Any, keys: Sequence[str], value: Any) -> Any:
    """``payload`` with ``keys`` set to ``value``, the mapping rebuilt down to the key it reaches.

    Raises:
        KeyError: when a key names nothing the payload already holds, which is how a typo is caught
            before a run is spent on it.
    """
    if not keys:
        return value

    if not isinstance(payload, dict) or keys[0] not in payload:
        raise KeyError(f"{keys[0]!r} names nothing in {payload!r}")

    return {**payload, keys[0]: _assigned(payload[keys[0]], keys[1:], value)}


def apply(root: Path, overrides: Overrides) -> None:
    """Set each dotted key of ``overrides`` in the config tree at ``root``, in place."""
    for dotted, value in overrides.items():
        path, keys = locate(root, dotted)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(yaml.safe_dump(_assigned(payload, keys, value), sort_keys=False), encoding="utf-8")


def _only_yaml(directory: str, names: list[str]) -> set[str]:
    """The entries of ``directory`` a config copy leaves behind: everything but its stages and their files."""
    root = Path(directory)
    return {name for name in names if not (root.joinpath(name).is_dir() or name.endswith(_SUFFIX))}


def variant(directory: Path, overrides: Overrides) -> Path:
    """A copy of the shipped tree at ``directory`` carrying ``overrides``, validated before it is returned.

    Loading the result is what makes a mistyped key or an out-of-bounds value fail here rather than
    inside the run that would have spent minutes on it.
    """
    if directory.exists():
        return directory

    copytree(bundled_root(), directory, ignore=_only_yaml)
    apply(directory, overrides)
    load_config(directory)
    return directory
