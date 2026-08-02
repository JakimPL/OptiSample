import json
import math
from pathlib import Path
from typing import Any, TypeVar

import msgpack
import numpy as np
from pydantic import BaseModel, ConfigDict

DocumentT = TypeVar("DocumentT", bound=BaseModel)


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


def _native(value: object) -> int | float:
    """A numpy scalar as the Python number msgpack stores, which is what the packer asks this hook for.

    Raises:
        TypeError: when the value is of a kind msgpack states no representation for.
    """
    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    raise TypeError(f"{type(value).__name__} has no msgpack representation")


def msgpack_bytes(document: BaseModel) -> bytes:
    """A document as msgpack: binary fields stored as binary, every float kept as the double it holds.

    This is the carrier a document with a payload travels on. A level curve or a block of PCM is written
    as the bytes it already is, where the text carrier spends a dozen characters a number on it, and a
    reading of ``-inf`` arrives as the float it was packed from, so a document read back states exactly
    the values it was built with.
    """
    packed: bytes = msgpack.packb(document.model_dump(), default=_native)
    return packed


def write_msgpack(path: Path, document: BaseModel) -> None:
    """Serialize a document to msgpack at ``path`` (see :func:`msgpack_bytes`)."""
    path.write_bytes(msgpack_bytes(document))


def read_msgpack(path: Path, document: type[DocumentT]) -> DocumentT:
    """The msgpack document at ``path``, validated against the shape ``document`` states."""
    return document.model_validate(msgpack.unpackb(path.read_bytes()))
