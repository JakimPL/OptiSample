import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod import BitDepth, TrackerModule

from optisample.config.optimize import SweepConfig
from optisample.config.tracker import TrackerFormat
from optisample.io.tracker.target import ExportTarget
from optisample.keys import SampleKey
from optisample.model import NoteEvent
from optisample.optimize.export import build_module
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.material import Voicing
from optisample.optimize.grouping.optimize import optimize_instrument_grouped
from optisample.optimize.layers.bands import UNSPLIT
from optisample.optimize.layers.slots import pack_slots
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.tasks import StoredRecordings
from optisample.optimize.velocity_map import VelocityVolumeMap
from tests.optimize.export.demo import (
    DEFAULT_BUDGET_KB,
    PITCHES,
    SR,
    VELOCITIES,
    demo_instrument,
    demo_material,
)

Recordings = Callable[..., StoredRecordings]

_GROUPED_BUDGET_KB = 3.0
_LAYERED_BUDGET_KB = 96.0  # room for a sample per key of every band, so a velocity split is affordable


@pytest.fixture
def as_format(
    export_context: ExportContext, retarget: Callable[[TrackerFormat], ExportTarget]
) -> Callable[[TrackerFormat | None], ExportContext]:
    """Factory: the bundled exporter context aimed at another format, so one plan writes both ways."""

    def _as_format(tracker_format: TrackerFormat | None) -> ExportContext:
        if tracker_format is None:
            return export_context

        return dataclasses.replace(export_context, target=retarget(tracker_format))

    return _as_format


@pytest.fixture
def plain_voicing(flat_velocity_map: VelocityVolumeMap, target: ExportTarget) -> Voicing:
    """How an instrument written whole voices a note: one instrument, answering every dynamic and key."""
    return Voicing(
        layout=pack_slots((), UNSPLIT, target.max_samples_per_instrument),
        velocity_map=flat_velocity_map,
    )


@pytest.fixture
def demo_audio(piano_note: Callable[..., NDArray[np.float64]]) -> dict[SampleKey, NDArray[np.float64]]:
    """The 2x2 demo audio grid (piano notes are fixture-independent test-signal data)."""
    return {
        SampleKey(pitch, velocity): piano_note(pitch, velocity, seed=pitch * 200 + velocity)
        for pitch in PITCHES
        for velocity in VELOCITIES
    }


@pytest.fixture
def build(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    as_format: Callable[[TrackerFormat | None], ExportContext],
    demo_audio: dict[SampleKey, NDArray[np.float64]],
    recordings: Recordings,
) -> Callable[..., tuple[InstrumentPlan, TrackerModule]]:
    """Optimize the demo instrument over a 2x2 encoding grid and export it to a module.

    ``tracker_format`` writes the same plan as another format, which is how the cross-format tests get
    two modules that differ only in how they spell the one song.
    """

    def _build(
        material: list[NoteEvent] | None = None,
        tracker_format: TrackerFormat | None = None,
        budget_kb: float = DEFAULT_BUDGET_KB,
    ) -> tuple[InstrumentPlan, TrackerModule]:
        settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depth=BitDepth.SIXTEEN))
        plan = optimize_instrument(demo_instrument(budget_kb), recordings(demo_audio, SR), settings)
        module = build_module(
            plan,
            recordings(demo_audio, SR),
            material if material is not None else demo_material(),
            as_format(tracker_format),
        )
        return plan, module

    return _build


@pytest.fixture
def grouped_build(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    export_context: ExportContext,
    demo_audio: dict[SampleKey, NDArray[np.float64]],
    recordings: Recordings,
) -> Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]]:
    """A tight-budget grouped build: one cheap operating point forces both keys into one shared zone."""

    def _grouped_build(budget_kb: float = _GROUPED_BUDGET_KB) -> tuple[GroupedInstrumentPlan, TrackerModule]:
        settings = optimize_settings(sweep=sweep(rates=(11_025,), depth=BitDepth.EIGHT, dither=False))
        plan = optimize_instrument_grouped(demo_instrument(budget_kb), recordings(demo_audio, SR), settings)
        module = build_module(plan, recordings(demo_audio, SR), demo_material(), export_context)
        return plan, module

    return _grouped_build


@pytest.fixture
def layered_build(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    export_context: ExportContext,
    demo_audio: dict[SampleKey, NDArray[np.float64]],
    recordings: Recordings,
) -> Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]]:
    """A generous grouped build: pitch 60 is played at both dynamics, so a velocity split can pay."""

    def _layered_build(budget_kb: float = _LAYERED_BUDGET_KB) -> tuple[GroupedInstrumentPlan, TrackerModule]:
        settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depth=BitDepth.SIXTEEN))
        plan = optimize_instrument_grouped(demo_instrument(budget_kb), recordings(demo_audio, SR), settings)
        module = build_module(plan, recordings(demo_audio, SR), demo_material(), export_context)
        return plan, module

    return _layered_build
