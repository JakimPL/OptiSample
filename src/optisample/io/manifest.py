"""Load and dump the manifest YAML: validate it against the model and resolve relative paths.

:func:`load_manifest` reads the YAML, validates it into a :class:`optisample.model.Manifest`, and
rewrites each sample/material path relative to the manifest's own directory; :func:`dump_manifest`
writes a manifest back out with unset and ``None`` fields omitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from optisample.model import Manifest


def load_manifest(path: Path | str) -> Manifest:
    """Load and validate a manifest YAML, resolving relative paths against its directory."""
    path = Path(path)
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest {path} must be a mapping at the top level")
    manifest = Manifest.model_validate(raw)
    base_dir = path.resolve().parent
    for instrument in manifest.instruments:
        for sample in instrument.samples:
            if not sample.file.is_absolute():
                sample.file = (base_dir / sample.file).resolve()
        if instrument.material_midi is not None and not instrument.material_midi.is_absolute():
            instrument.material_midi = (base_dir / instrument.material_midi).resolve()
    return manifest


def dump_manifest(manifest: Manifest, path: Path | str) -> None:
    """Serialize a manifest to YAML (unset/None fields omitted)."""
    data = manifest.model_dump(mode="json", exclude_none=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
