import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Base for every artifact document: immutable and rejecting unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _json_safe(value: Any) -> Any:
    """Recursively coerce numpy scalars to Python and non-finite floats (silence -> -inf) to null."""
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def json_text(document: BaseModel) -> str:
    """A document as pretty JSON, coercing numpy/non-finite values to JSON-safe Python.

    A document stored inside an archive is written from its text rather than from a file, so a manifest
    travelling with the instruments it names reads exactly as one written beside them.
    """
    return json.dumps(_json_safe(document.model_dump()), indent=2, allow_nan=False) + "\n"


def write_json(path: Path, document: BaseModel) -> None:
    """Serialize a document to pretty JSON at ``path`` (see :func:`json_text`)."""
    path.write_text(json_text(document), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write ``text`` verbatim as UTF-8 (the human-readable report and the infeasibility note)."""
    path.write_text(text, encoding="utf-8")
