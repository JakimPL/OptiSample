import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import OptiConfig, load_config
from optisample.config.codec import EncodeConfig, QuantizeConfig
from optisample.config.dynamics import DynamicsConfig
from optisample.config.layers import LayersConfig
from optisample.config.loop import (
    EnvelopeConfig,
    FeatureConfig,
    FrontierConfig,
    GeometryConfig,
    LoopConfig,
    QualityConfig,
    SeamConfig,
)
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import (
    BudgetConfig,
    Method,
    SweepConfig,
    VelocityConfig,
)
from optisample.config.reduce import ReduceConfig
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.spectral import SpectralConfig
from optisample.config.synth import SynthConfig
from optisample.config.tracker import TrackerFormat
from optisample.dsp.surrogate import NO_LOOPS, EncodeContext, EncodingParams, SettledLoops
from optisample.io.note_extractor import IngestSettings
from optisample.io.tracker.target import ExportTarget, export_target
from optisample.keys import SampleKey
from optisample.loop.settle import settle_loop
from optisample.metrics import CompositeFidelity, build_composite
from optisample.metrics.base import Signal
from optisample.model import ProjectSpec
from optisample.music import midi_to_freq
from optisample.optimize.export.context import ExportContext
from optisample.optimize.operating_points import SweepContext
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.bandwidth import StoredFormat
from optisample.optimize.reduce.grids import NarrowedGrid
from optisample.optimize.reduce.summary import KeptRecording, ReductionSummary
from optisample.optimize.tasks import (
    AudioMap,
    Event,
    LoopMap,
    StoredRecordings,
    TaskInputs,
)
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap
from optisample.synth import NoteSpec, synthesize
from trackmod.module.storage import Storage
from trackmod.spec.levels import MAX_VOLUME

TEST_CONFIG_DIR: Final = Path(__file__).parent / "opticonfig"
"""The settings the suite runs under, held in the repository beside the tests that read them.

Every value a test reads is stated here, so a run reproduces the same numbers whatever the bundled
``opticonfig`` currently ships. A test needing another value states it through the group's own model
(``SweepConfig.model_validate({**config.optimize.sweep.model_dump(), ...})``) or through the flag the
command offers, which keeps the knob it is about beside the assertion it makes.
"""

_NOTE_SR = 44_100
_MIDI_VELOCITIES = 128
_ANCHORS = (VelocityAnchor(100, -10.0, MAX_VOLUME),)
_INGEST_BUDGET_KB = 64.0
_NOISE_FLOOR = 1e-4  # -80 dB of a full-scale tone: a floor a recording holds and arithmetic stays under
_SCORED_S = 0.5
_SCORED_FRAMES = int(_SCORED_S * _NOTE_SR)


def recorded(signal: NDArray[np.float64], seed: int = 0) -> NDArray[np.float64]:
    """``signal`` over the noise floor a recording of it would carry, which is content in every bin.

    A tone written straight from :func:`numpy.sin` leaves most of its spectrum holding rounding alone, some
    160 dB down, and a distance read over spectra normalized to unit sum weighs those bins as heavily as the
    ones carrying the note -- so two stretches of the same tone can measure tens of decibels apart on
    arithmetic. Material that stands on a floor is what such a reading is stated against, so a test
    measuring timbre puts one there and reads the note.
    """
    floor = np.random.default_rng(seed).standard_normal(signal.size) * _NOISE_FLOOR
    return np.asarray(signal + floor, dtype=np.float64)


@pytest.fixture(scope="session")
def config() -> OptiConfig:
    """The suite's own configuration (:data:`TEST_CONFIG_DIR`), loaded once for the whole session."""
    return load_config(TEST_CONFIG_DIR)


@pytest.fixture
def ingest_settings() -> Callable[..., IngestSettings]:
    """Factory: what an ingest takes from its caller, for a run stating no padding flags of its own.

    ``instrument_id`` names the instrument and the project it belongs to; the padding a directory of
    recordings holds and ``keep_tail`` are what the ingest tests vary, so each states only its own knob.
    """

    def _build(
        instrument_id: str,
        *,
        project_name: str | None = None,
        budget_kb: float = _INGEST_BUDGET_KB,
        pre_roll_s: float = 0.0,
        post_roll_s: float = 0.0,
        keep_tail: bool = False,
    ) -> IngestSettings:
        return IngestSettings(
            instrument_id=instrument_id,
            budget_kb=budget_kb,
            project=ProjectSpec(name=project_name if project_name is not None else instrument_id),
            pre_roll_s=pre_roll_s,
            post_roll_s=post_roll_s,
            keep_tail=keep_tail,
        )

    return _build


@pytest.fixture
def loop_config(config: OptiConfig) -> LoopConfig:
    return config.loop


@pytest.fixture
def geometry_config(config: OptiConfig) -> GeometryConfig:
    return config.loop.geometry


@pytest.fixture
def features_config(config: OptiConfig) -> FeatureConfig:
    return config.loop.features


@pytest.fixture
def loop_frontier_config(config: OptiConfig) -> FrontierConfig:
    return config.loop.frontier


@pytest.fixture
def seam_config(config: OptiConfig) -> SeamConfig:
    return config.loop.seam


@pytest.fixture
def envelope_config(config: OptiConfig) -> EnvelopeConfig:
    return config.loop.envelope


@pytest.fixture
def quality_config(config: OptiConfig) -> QualityConfig:
    return config.loop.quality


@pytest.fixture
def loop_reserve_s(config: OptiConfig) -> float:
    """The span the reduce stage reserves for a loop search: the onset, the loop, the wrap reading, the tail.

    A test reading what a kept recording is asked to hold states its lengths against this, so it reads off
    the shipped config rather than a number that happened to sit on the right side of it.
    """
    geometry = config.loop.geometry
    return geometry.max_attack_s + geometry.min_loop_s + config.loop.features.change_span_s + geometry.tail_skip_s


@pytest.fixture
def loop_room_s(config: OptiConfig) -> float:
    """The span a recording holding one sound needs for a loop to fit inside it.

    Material that settles at once opens its window where a wrap has material to blend into, so what a loop
    asks of such a recording is that blend, the shortest loop accepted, and the tail left alone. A test
    wanting material a loop fits inside states a length above this, and one wanting material stored over
    the span it plays states a length under it.
    """
    return config.loop.seam.min_fade_s + config.loop.geometry.min_loop_s + config.loop.geometry.tail_skip_s


@pytest.fixture
def loop(config: OptiConfig) -> Callable[..., LoopConfig]:
    """Factory: the bundled loop stage with the given groups overridden (re-validated)."""

    def _build(**overrides: object) -> LoopConfig:
        return LoopConfig.model_validate({**config.loop.model_dump(), **overrides})

    return _build


@pytest.fixture
def quantize_config(config: OptiConfig) -> QuantizeConfig:
    return config.codec.quantize


@pytest.fixture
def encode_config(config: OptiConfig) -> EncodeConfig:
    return config.encode


@pytest.fixture
def dynamics_config(config: OptiConfig) -> DynamicsConfig:
    return config.codec.dynamics


@pytest.fixture
def spectral_config(config: OptiConfig) -> SpectralConfig:
    return config.analysis.spectral


@pytest.fixture
def metrics_config(config: OptiConfig) -> MetricsConfig:
    return config.analysis.metrics


@pytest.fixture
def composite(config: OptiConfig) -> CompositeFidelity:
    """The composite fidelity built once from the bundled metrics config."""
    return build_composite(config.analysis.metrics)


@pytest.fixture
def sweep_config(config: OptiConfig) -> SweepConfig:
    return config.optimize.sweep


@pytest.fixture
def budget_config(config: OptiConfig) -> BudgetConfig:
    return config.optimize.budget


@pytest.fixture
def reduce_config(config: OptiConfig) -> ReduceConfig:
    return config.reduce


@pytest.fixture
def velocity_config(config: OptiConfig) -> VelocityConfig:
    return config.optimize.velocity


@pytest.fixture
def render_config(config: OptiConfig) -> RenderConfig:
    return config.export.render


@pytest.fixture
def playback_config(config: OptiConfig) -> PlaybackConfig:
    return config.export.playback


@pytest.fixture
def target(config: OptiConfig) -> ExportTarget:
    """The export target built from the bundled tracker config (format, compliance, format settings)."""
    return export_target(config.export.tracker)


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
    """The exporter context (encode + playback + target + envelope) built from the bundled config, seed 0."""
    return ExportContext(
        encode=config.encode,
        playback=config.export.playback,
        target=target,
        envelope=config.export.envelope,
        carrier=config.optimize.sweep.carrier,
    )


@pytest.fixture
def synth_config(config: OptiConfig) -> SynthConfig:
    return config.synth


@pytest.fixture
def sweep(config: OptiConfig) -> Callable[..., SweepConfig]:
    """Factory: the bundled sweep config with the given fields overridden (re-validated)."""

    def _build(**overrides: object) -> SweepConfig:
        return SweepConfig.model_validate({**config.optimize.sweep.model_dump(), **overrides})

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
        return LayersConfig.model_validate({**config.optimize.layers.model_dump(), **overrides})

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
        settled: LoopMap | None = None,
    ) -> TaskInputs:
        return TaskInputs(
            audio=audio,
            settled={} if settled is None else settled,
            velocity_map=velocity_map,
            reduce=reduce if reduce is not None else config.reduce,
            sample_rate=sample_rate,
            energy_exponent=config.optimize.budget.energy_exponent if energy_exponent is None else energy_exponent,
        )

    return _build


@pytest.fixture
def scored_event() -> Event:
    """One scored note class built by hand: the recording it stands for, its dynamic, and its length.

    For the tests that read a class rather than build one -- what a listening pair names, what a document
    states -- so they state the class they are about instead of running a reduction to reach one.
    """
    key = SampleKey(60, 100)
    reference = np.linspace(1.0, 0.0, _SCORED_FRAMES, dtype=np.float64)
    return Event(key, 100, MAX_VOLUME, _SCORED_S, 1.0, reference, 1.0)


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
        loop: LoopConfig | None = None,
        reduce: ReduceConfig | None = None,
        layers: LayersConfig | None = None,
        method: Method | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> OptimizeSettings:
        return OptimizeSettings(
            sweep=sweep,
            loop=loop if loop is not None else config.loop,
            reduce=reduce if reduce is not None else config.reduce,
            layers=layers if layers is not None else config.optimize.layers,
            encode=config.encode,
            metrics=config.analysis.metrics,
            velocity=config.optimize.velocity,
            method=method if method is not None else config.optimize.budget.method,
            energy_exponent=config.optimize.budget.energy_exponent,
            max_samples=config.optimize.budget.max_samples if max_samples is None else max_samples,
            resolution=config.optimize.budget.resolution,
            target=target,
            playback=config.export.playback,
            envelope=config.export.envelope,
            seed=seed,
        )

    return _build


@pytest.fixture
def make_encode_ctx(config: OptiConfig) -> Callable[..., EncodeContext]:
    """Factory: an ``EncodeContext`` at ``root_pitch`` using the bundled encode config.

    ``seed`` (when given) seeds the dither RNG; the default leaves it ``None`` so encoding uses the
    surrogate's own fixed-seed fallback -- matching the pre-config call sites. ``release_fade_s``
    overrides the ramp closing a stored span, which is what a test isolating the codec alone sets to zero.
    ``settled`` holds the loops the clip offers, which a test asking for a looped span supplies.
    """

    def _build(
        root_pitch: int,
        *,
        seed: int | None = None,
        release_fade_s: float | None = None,
        settled: SettledLoops = NO_LOOPS,
    ) -> EncodeContext:
        rng = np.random.default_rng(seed) if seed is not None else None
        encode = (
            config.encode
            if release_fade_s is None
            else config.encode.model_copy(update={"release_fade_s": release_fade_s})
        )
        return EncodeContext(root_pitch=root_pitch, config=encode, settled=settled, rng=rng)

    return _build


@pytest.fixture(scope="session")
def recordings(config: OptiConfig) -> Callable[..., StoredRecordings]:
    """Factory: the recordings a run encodes from, with the loops each one offers settled as the stage would.

    Candidates are searched over the whole of each recording, so a test gets the loops the material supports
    rather than bounds it picked. ``loops=False`` leaves every recording unlooped, which is what a test about
    the trimmed span alone asks for.
    """

    def _build(audio: AudioMap, sample_rate: int, *, loops: bool = True) -> StoredRecordings:
        settled = {
            key: _settled(signal, sample_rate, config.loop, midi_to_freq(key.pitch)) if loops else NO_LOOPS
            for key, signal in audio.items()
        }
        return StoredRecordings(audio=audio, settled=settled, sample_rate=sample_rate)

    return _build


def _settled(signal: Signal, sample_rate: int, config: LoopConfig, root_hz: float) -> SettledLoops:
    """The loops the stage offers over the whole of ``signal``, which is what an encode is handed."""
    settlement = settle_loop(signal, sample_rate, config, root_hz=root_hz, search_s=signal.size / sample_rate)
    return settlement.settled


@pytest.fixture
def settle(config: OptiConfig) -> Callable[..., SettledLoops]:
    """Factory: the loops the stage offers for a signal, which is what an encode is handed.

    Runs the real settlement, so a test encoding a looped span is stored around a loop the stage would
    have offered for that recording rather than around bounds a test picked. ``root_hz`` is the pitch the
    material was played at, which the period searched and the level read are both taken over.
    """

    def _build(
        signal: Signal,
        sample_rate: int,
        *,
        root_hz: float,
        search_s: float,
        loop: LoopConfig | None = None,
    ) -> SettledLoops:
        settlement = settle_loop(
            signal,
            sample_rate,
            loop if loop is not None else config.loop,
            root_hz=root_hz,
            search_s=search_s,
        )
        return settlement.settled

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
        recordings=(
            KeptRecording(SampleKey(60, 100), duration_s=1.0, required_duration_s=0.8),
            KeptRecording(SampleKey(67, 100), duration_s=0.4, required_duration_s=0.8),
        ),
        grids=(
            NarrowedGrid(
                pitch=60,
                useful_rate_hz=10_500.0,
                stored=StoredFormat(target_rate=11_025, depths=(16,)),
                encodings=(EncodingParams(target_rate=11_025, depth_bits=16),),
            ),
        ),
    )
