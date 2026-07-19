"""Base model for every configuration group.

Config is loaded from YAML (see :mod:`optisample.config.loader`) and then never mutated, so the base
is ``frozen`` (immutable + hashable -- config value-objects such as :class:`StftParams` live inside
tuples and are compared) and ``extra="forbid"`` (a mistyped YAML key fails loudly at load rather than
being silently ignored). Config fields deliberately carry **no defaults** for tunable parameters: the
single source of truth is the bundled YAML, not a value hidden in Python.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Frozen, strict base: immutable, hashable, and rejecting unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")
