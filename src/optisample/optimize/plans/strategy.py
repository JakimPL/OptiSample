"""The strategy-agnostic surface both instrument plans share.

The ungrouped and grouped plans differ in their internals -- one stored sample per key versus one per
pitch zone -- but every consumer downstream (the IT exporter, the artifact dumper, the JSON serializer)
needs the same handful of facts from either: the ordered stored :class:`SampleUnit`s, which strategy
produced the plan, and its byte/objective summary. Capturing that as the :class:`StrategyPlan` protocol
lets those consumers work off one shape instead of branching on the concrete plan type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from optisample.dsp.surrogate import EncodingParams
from optisample.optimize.velocity_map import VelocityVolumeMap

Strategy = Literal["ungrouped", "grouped"]  # which allocation the plan came from.
Method = Literal["exact", "lagrangian"]  # the MCKP solver the ungrouped strategy ran.


@dataclass(frozen=True)
class SampleUnit:  # pylint: disable=too-many-instance-attributes
    """One stored sample a plan kept: the recording it re-encodes and every key that sample serves.

    Ungrouped, a unit is a single key playing its own sample; grouped, it is a whole pitch zone routed
    to one repitched representative. The exporter, dumper and serializer all read this normalized view,
    so none of them has to know which strategy built the plan. ``label`` is the unit's stable name (the
    dumper's per-sample WAV filename and ``served_by`` reference).
    """

    label: str
    representative: int
    representative_velocity: int
    keys: tuple[int, ...]
    params: EncodingParams
    frames: int
    stored_bytes: int
    distortion: float
    hull_size: int
    weight: float


class StrategyPlan(Protocol):
    """What both instrument plans expose so their consumers stay strategy-agnostic.

    Read-only by construction: the concrete plans are frozen. ``sample_units`` yields the kept samples in
    the plan's own order. Strategy-only facts (the ungrouped solver ``method``) stay on the concrete
    plan, reached by narrowing on :attr:`strategy` where a serializer genuinely needs them.
    """

    @property
    def instrument_id(self) -> str: ...

    @property
    def velocity_map(self) -> VelocityVolumeMap: ...

    @property
    def strategy(self) -> Strategy: ...

    @property
    def objective(self) -> float: ...

    @property
    def used_bytes(self) -> int: ...

    @property
    def module_budget_bytes(self) -> int: ...

    @property
    def sample_budget_bytes(self) -> int: ...

    @property
    def module_bytes(self) -> int: ...

    def sample_units(self) -> tuple[SampleUnit, ...]: ...
