import dataclasses
from collections.abc import Callable, Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import OptiConfig, load_config
from optisample.config.dsp import (
    EncodeConfig,
    LoopConfig,
    QuantizeConfig,
    SpectralConfig,
)
from optisample.config.dynamics import DynamicsConfig
from optisample.config.layers import LayersConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import (
    Method,
    OptimizeConfig,
    SweepConfig,
    VelocityConfig,
)
from optisample.config.reduce import ReduceConfig
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.synth import SynthConfig
from optisample.config.tracker import TrackerFormat
from optisample.dsp.surrogate import EncodeContext, EncodingParams
from optisample.io.tracker.target import ExportTarget, export_target
from optisample.metrics import CompositeFidelity, build_composite
from optisample.optimize.export.context import ExportContext
from optisample.optimize.operating_points import SweepContext
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.grids import NarrowedGrid
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.reduce.summary import KeptRecording, ReductionSummary
from optisample.optimize.tasks import AudioMap, TaskInputs
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap
from optisample.synth import NoteSpec, synthesize
from trackmod.module.storage import Storage
from trackmod.spec.levels import MAX_VOLUME

_NOTE_SR = 44_100
_MIDI_VELOCITIES = 128
_ANCHORS = (VelocityAnchor(100, -10.0, MAX_VOLUME),)


@pytest.fixture(scope="session")
def config() -> OptiConfig:
    """The bundled configuration, loaded once for the whole test session."""
    return load_config()


@pytest.fixture
def loop_config(config: OptiConfig) -> LoopConfig:
    return config.loop


@pytest.fixture
def quantize_config(config: OptiConfig) -> QuantizeConfig:
    return config.quantize


@pytest.fixture
def encode_config(config: OptiConfig) -> EncodeConfig:
    return config.encode


@pytest.fixture
def dynamics_config(config: OptiConfig) -> DynamicsConfig:
    return config.dynamics


@pytest.fixture
def spectral_config(config: OptiConfig) -> SpectralConfig:
    return config.spectral


@pytest.fixture
def metrics_config(config: OptiConfig) -> MetricsConfig:
    return config.metrics


@pytest.fixture
def composite(config: OptiConfig) -> CompositeFidelity:
    """The composite fidelity built once from the bundled metrics config."""
    return build_composite(config.metrics)


@pytest.fixture
def sweep_config(config: OptiConfig) -> SweepConfig:
    return config.sweep


@pytest.fixture
def optimize_config(config: OptiConfig) -> OptimizeConfig:
    return config.optimize


@pytest.fixture
def reduce_config(config: OptiConfig) -> ReduceConfig:
    return config.reduce


@pytest.fixture
def velocity_config(config: OptiConfig) -> VelocityConfig:
    return config.velocity


@pytest.fixture
def render_config(config: OptiConfig) -> RenderConfig:
    return config.render


@pytest.fixture
def playback_config(config: OptiConfig) -> PlaybackConfig:
    return config.playback


@pytest.fixture
def target(config: OptiConfig) -> ExportTarget:
    """The export target built from the bundled tracker config (format, compliance, format settings)."""
    return export_target(config.tracker)


@pytest.fixture
def retarget(target: ExportTarget) -> Callable[[TrackerFormat], ExportTarget]:
    """Factory: the bundled export target rewritten to another format, for the cross-format tests."""

    def _retarget(tracker_format: TrackerFormat) -> ExportTarget:
        return dataclasses.replace(target, format=tracker_format)

    return _retarget


@pytest.fixture
def storage(target: ExportTarget) -> Storage:
    """The target format's record cost table, which prices every stored sample."""
    return target.storage


@pytest.fixture
def sweep_context(config: OptiConfig, composite: CompositeFidelity, storage: Storage) -> SweepContext:
    """The scoring context an encoding sweep runs under, built from the bundled config."""
    return SweepContext(composite=composite, encode=config.encode, storage=storage)


@pytest.fixture
def export_context(config: OptiConfig, target: ExportTarget) -> ExportContext:
    """The exporter context (encode + playback + target) built from the bundled config, seed 0."""
    return ExportContext(encode=config.encode, playback=config.playback, target=target)


@pytest.fixture
def synth_config(config: OptiConfig) -> SynthConfig:
    return config.synth


@pytest.fixture
def sweep(config: OptiConfig) -> Callable[..., SweepConfig]:
    """Factory: the bundled sweep config with the given fields overridden (re-validated)."""

    def _build(**overrides: object) -> SweepConfig:
        return SweepConfig.model_validate({**config.sweep.model_dump(), **overrides})

    return _build


@pytest.fixture
def reduce(config: OptiConfig) -> Callable[..., ReduceConfig]:
    """Factory: the bundled reduce config with the named sections' fields overridden (re-validated).

    Sections are merged field-by-field (``reduce(dedupe={"key": DedupeKey.PITCH})``), so a test states
    only the knob it varies and every other value stays the bundled one.
    """

    def _build(**sections: Mapping[str, object]) -> ReduceConfig:
        raw: dict[str, dict[str, object]] = config.reduce.model_dump()
        merged = {name: {**fields, **sections.get(name, {})} for name, fields in raw.items()}
        return ReduceConfig.model_validate(merged)

    return _build


@pytest.fixture
def layers(config: OptiConfig) -> Callable[..., LayersConfig]:
    """Factory: the bundled velocity layering with the given fields overridden (re-validated)."""

    def _build(**overrides: object) -> LayersConfig:
        return LayersConfig.model_validate({**config.layers.model_dump(), **overrides})

    return _build


@pytest.fixture
def task_inputs(config: OptiConfig) -> Callable[..., TaskInputs]:
    """Factory: the bundle a pitch task is built from, over the bundled reduction and cost weighting.

    ``audio`` and ``velocity_map`` are what a test varies; ``reduce``, ``sample_rate`` and
    ``energy_exponent`` fall back to the bundled values, so a test states only the knob it is about.
    """

    def _build(
        audio: AudioMap,
        velocity_map: VelocityVolumeMap,
        *,
        reduce: ReduceConfig | None = None,
        sample_rate: int = _NOTE_SR,
        energy_exponent: float | None = None,
    ) -> TaskInputs:
        return TaskInputs(
            audio=audio,
            velocity_map=velocity_map,
            reduce=reduce if reduce is not None else config.reduce,
            sample_rate=sample_rate,
            energy_exponent=config.optimize.energy_exponent if energy_exponent is None else energy_exponent,
        )

    return _build


@pytest.fixture
def flat_velocity_map() -> VelocityVolumeMap:
    """A map sending every velocity to full volume, so a test can look past the loudness axis."""
    return VelocityVolumeMap(tuple(MAX_VOLUME for _ in range(_MIDI_VELOCITIES)), _ANCHORS)


@pytest.fixture
def graded_velocity_map() -> VelocityVolumeMap:
    """A map giving every velocity its own volume, so only genuinely equal dynamics share one."""
    volumes = tuple(min(velocity, MAX_VOLUME) for velocity in range(_MIDI_VELOCITIES))
    return VelocityVolumeMap(volumes, _ANCHORS)


@pytest.fixture
def optimize_settings(config: OptiConfig, target: ExportTarget) -> Callable[..., OptimizeSettings]:
    """Factory: an ``OptimizeSettings`` from the bundled config, overriding the swept grid/method/seed.

    ``sweep`` (a ``SweepConfig``, usually built via the ``sweep`` factory), ``reduce`` (a
    ``ReduceConfig``, usually built via the ``reduce`` factory) and ``layers`` (a ``LayersConfig``,
    usually built via the ``layers`` factory) are the knobs the optimize tests vary; ``encode``,
    ``metrics``, ``velocity`` and ``target`` come from the bundled config, and ``max_samples`` caps what
    a grouped plan may store. Every run stays in the calling process, so the suite reads a stage's own
    timing rather than a pool's startup.
    """

    def _build(
        *,
        sweep: SweepConfig,
        reduce: ReduceConfig | None = None,
        layers: LayersConfig | None = None,
        method: Method | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> OptimizeSettings:
        return OptimizeSettings(
            sweep=sweep,
            reduce=reduce if reduce is not None else config.reduce,
            layers=layers if layers is not None else config.layers,
            encode=config.encode,
            metrics=config.metrics,
            velocity=config.velocity,
            method=method if method is not None else config.optimize.method,
            energy_exponent=config.optimize.energy_exponent,
            max_samples=config.optimize.max_samples if max_samples is None else max_samples,
            target=target,
            seed=seed,
        )

    return _build


@pytest.fixture
def make_encode_ctx(config: OptiConfig) -> Callable[..., EncodeContext]:
    """Factory: an ``EncodeContext`` at ``root_pitch`` using the bundled encode config.

    ``seed`` (when given) seeds the dither RNG; the default leaves it ``None`` so encoding uses the
    surrogate's own fixed-seed fallback -- matching the pre-config call sites. ``release_fade_s``
    overrides the ramp closing a stored span, which is what a test isolating the codec alone sets to zero.
    """

    def _build(root_pitch: int, *, seed: int | None = None, release_fade_s: float | None = None) -> EncodeContext:
        rng = np.random.default_rng(seed) if seed is not None else None
        encode = (
            config.encode
            if release_fade_s is None
            else config.encode.model_copy(update={"release_fade_s": release_fade_s})
        )
        return EncodeContext(root_pitch=root_pitch, config=encode, rng=rng)

    return _build


@pytest.fixture(scope="session")
def piano_note(config: OptiConfig) -> Callable[..., NDArray[np.float64]]:
    """Factory: render one piano note (a test-signal generator, fixture-independent test data).

    ``seed`` is explicit so each call site keeps its own recorded-sample identity.
    """

    def _piano(pitch: int, velocity: int = 100, dur: float = 0.6, *, seed: int) -> NDArray[np.float64]:
        spec = NoteSpec(pitch, velocity, 0.0, dur, _NOTE_SR)
        return synthesize("piano", spec, np.random.default_rng(seed), config.synth)

    return _piano


@pytest.fixture
def reduction() -> ReductionSummary:
    """A stand-in pre-optimization summary, for tests that build a plan by hand.

    Every count differs from the others so a report or document asserting one of them pins that field
    rather than any field that happens to hold the same number.
    """
    return ReductionSummary(
        listed_recordings=9,
        played_notes=8,
        scored_classes=3,
        grid_size=12,
        recordings=(
            KeptRecording(SampleKey(60, 100), duration_s=1.0, required_duration_s=0.8),
            KeptRecording(SampleKey(67, 100), duration_s=0.4, required_duration_s=0.8),
        ),
        grids=(NarrowedGrid(60, 11_025.0, (EncodingParams(target_rate=11_025, depth_bits=16),)),),
    )
