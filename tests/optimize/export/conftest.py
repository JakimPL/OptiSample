"""Shared fixtures for the exporter tests: the demo audio grid and the builds both strategies produce."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.config.tracker import TrackerFormat
from optisample.io.tracker.target import ExportTarget
from optisample.model import NoteEvent
from optisample.optimize.export import build_module
from optisample.optimize.export.context import ExportContext
from optisample.optimize.grouping.optimize import optimize_instrument_grouped
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.reduce.keys import SampleKey
from tests.optimize.export.demo import PITCHES, SR, VELOCITIES, demo_instrument, demo_material
from trackmod.module.protocol import TrackerModule

_GROUPED_BUDGET_KB = 8.0


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
) -> Callable[..., tuple[InstrumentPlan, TrackerModule]]:
    """Optimize the demo instrument over a 2x2 encoding grid and export it to a module.

    ``tracker_format`` writes the same plan as another format, which is how the cross-format tests get
    two modules that differ only in how they spell the one song.
    """

    def _build(
        material: list[NoteEvent] | None = None,
        tracker_format: TrackerFormat | None = None,
    ) -> tuple[InstrumentPlan, TrackerModule]:
        settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depths=(16, 8)))
        plan = optimize_instrument(demo_instrument(), demo_audio, SR, settings)
        module = build_module(
            plan,
            demo_audio,
            SR,
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
) -> Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]]:
    """A tight-budget grouped build: one cheap operating point forces both keys into one shared zone."""

    def _grouped_build(budget_kb: float = _GROUPED_BUDGET_KB) -> tuple[GroupedInstrumentPlan, TrackerModule]:
        settings = optimize_settings(sweep=sweep(rates=(11_025,), depths=(8,), dither=False))
        plan = optimize_instrument_grouped(demo_instrument(budget_kb), demo_audio, SR, settings)
        module = build_module(plan, demo_audio, SR, demo_material(), export_context)
        return plan, module

    return _grouped_build
