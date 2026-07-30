from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from optisample.config.root import OptiConfig
from optisample.config.stage import StageConfig

_PACKAGE = "opticonfig"


def _read_yaml(directory: Path | None, *parts: str) -> Any:
    """The parsed contents of ``parts`` under ``directory``, or under the bundled ``opticonfig`` package."""
    filename = f"{parts[-1]}.yaml"
    if directory is None:
        resource = resources.files(_PACKAGE)
        for part in parts[:-1]:
            resource = resource.joinpath(part)

        return yaml.safe_load(resource.joinpath(filename).read_text(encoding="utf-8"))

    return yaml.safe_load(directory.joinpath(*parts[:-1], filename).read_text(encoding="utf-8"))


def _stage_type(annotation: type[Any] | None) -> type[StageConfig] | None:
    """``annotation`` where it names a stage, which is what puts its groups in a directory of their own."""
    if isinstance(annotation, type) and issubclass(annotation, StageConfig):
        return annotation

    return None


def _group_payload(directory: Path | None, name: str, annotation: type[Any] | None) -> Any:
    """The raw settings one field of :class:`OptiConfig` is built from.

    A stage reads one file per group from the directory it names, so ``loop.seam`` comes from
    ``loop/seam.yaml``; every other group is a file of its own name.
    """
    stage = _stage_type(annotation)
    if stage is None:
        return _read_yaml(directory, name)

    return {group: _read_yaml(directory, name, group) for group in stage.model_fields}


def load_config(directory: Path | None = None) -> OptiConfig:
    """Assemble and validate an :class:`OptiConfig` from the per-stage YAML layout."""
    raw = {
        name: _group_payload(directory, name, field.annotation)
        for name, field in OptiConfig.model_fields.items()  # pylint: disable=no-member
    }
    return OptiConfig.model_validate(raw)
