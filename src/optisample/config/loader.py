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
    for name in OptiConfig.model_fields:  # pylint: disable=not-an-iterable
        raw[name] = yaml.safe_load(_read_yaml_text(directory, name))

    return OptiConfig.model_validate(raw)
