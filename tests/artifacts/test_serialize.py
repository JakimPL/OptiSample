from __future__ import annotations

import math

import numpy as np

from optisample.artifacts.serialize import _json_safe


def test_json_safe_coerces_numpy_and_non_finite() -> None:
    out = _json_safe({"a": np.float64(1.5), "b": np.int64(3), "c": math.inf, "d": [-math.inf, 2.0]})
    assert out == {"a": 1.5, "b": 3, "c": None, "d": [None, 2.0]}
    assert isinstance(out["b"], int) and not isinstance(out["b"], np.integer)
