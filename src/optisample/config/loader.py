"""Load an :class:`OptiConfig` from a directory of per-group YAML files.

Each ``OptiConfig`` field is read from ``<field>.yaml``; the whole set is validated in one pass, so a
missing file, a mistyped key (``extra="forbid"``) or a wrong type fails loudly at load. With no
argument the bundled ``opticonfig`` package is used; pass a directory to load a tuned copy -- this is
the entry point for tweaking the algorithm without editing code.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from optisample.config.root import OptiConfig


def _read_yaml_text(directory: Path | None, name: str) -> str:
    """Text of ``<name>.yaml`` from ``directory`` (or the bundled ``opticonfig`` package if ``None``)."""
    filename = f"{name}.yaml"
    if directory is None:
        return (resources.files("opticonfig") / filename).read_text(encoding="utf-8")
    return (directory / filename).read_text(encoding="utf-8")


def load_config(directory: Path | None = None) -> OptiConfig:
    """Assemble and validate an :class:`OptiConfig` from the per-group YAML files."""
    raw: dict[str, Any] = {}
    # model_fields is a dict keyed by field name; pylint-pydantic does not model it as iterable.
    for name in OptiConfig.model_fields:  # pylint: disable=not-an-iterable
        raw[name] = yaml.safe_load(_read_yaml_text(directory, name))
    return OptiConfig.model_validate(raw)
