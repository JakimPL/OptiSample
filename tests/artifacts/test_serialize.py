from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optisample.artifacts.serialize import (
    Frozen,
    _json_safe,
    _native,
    msgpack_bytes,
    read_msgpack,
    write_msgpack,
)


class _Curve(Frozen):
    payload: bytes
    readings: list[float]


class _Held(Frozen):
    name: str
    frames: int
    curve: _Curve


@pytest.fixture
def held() -> _Held:
    return _Held(name="C-4", frames=3, curve=_Curve(payload=b"\x00\x01\xfe\xff", readings=[1.5, -math.inf]))


def test_json_safe_coerces_numpy_and_non_finite() -> None:
    out = _json_safe({"a": np.float64(1.5), "b": np.int64(3), "c": math.inf, "d": [-math.inf, 2.0]})
    assert out == {"a": 1.5, "b": 3, "c": None, "d": [None, 2.0]}
    assert isinstance(out["b"], int) and not isinstance(out["b"], np.integer)


def test_a_document_read_back_from_msgpack_states_what_it_was_written_with(held: _Held, tmp_path: Path) -> None:
    """The carrier is a round trip, which is what lets a container be the unit a later stage works from."""
    path = tmp_path / "held.sample"
    write_msgpack(path, held)

    assert read_msgpack(path, _Held) == held


def test_msgpack_stores_a_payload_as_the_bytes_it_already_is(held: _Held) -> None:
    """Binary reaching the file as binary is the whole reason a payload travels on this carrier."""
    assert held.curve.payload in msgpack_bytes(held)


def test_msgpack_keeps_a_reading_no_finite_value_stands_behind(held: _Held, tmp_path: Path) -> None:
    """The text carrier states such a reading as null; this one hands back the float it was packed from."""
    path = tmp_path / "held.sample"
    write_msgpack(path, held)

    assert read_msgpack(path, _Held).curve.readings == [1.5, -math.inf]


def test_numpy_scalars_reach_msgpack_as_the_python_numbers_they_stand_for() -> None:
    assert (_native(np.float64(1.5)), _native(np.int64(3))) == (1.5, 3)


def test_a_value_msgpack_states_no_representation_for_is_refused() -> None:
    with pytest.raises(TypeError, match="has no msgpack representation"):
        _native(object())
