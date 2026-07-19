"""Shared config fixtures.

The algorithm's parameters live in the bundled ``opticonfig`` YAML (see :mod:`optisample.config`), so
tests obtain a fully-populated config from here rather than relying on constructor defaults (there are
none for tunables). ``config`` loads the bundled values once per session; the derived fixtures expose
each group; the factory fixtures (``sweep``) build tweaked configs for tests that need specific values.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.config import OptiConfig, load_config
from optisample.config.dsp import EncodeConfig, LoopConfig, QuantizeConfig, SpectralConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import Method, OptimizeConfig, SweepConfig, VelocityConfig
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.synth import SynthConfig
from optisample.dsp.surrogate import EncodeContext
from optisample.io.it_writer import ITPlayback, it_playback
from optisample.metrics import CompositeFidelity, build_composite
from optisample.optimize.orchestrate import OptimizeSettings


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
def velocity_config(config: OptiConfig) -> VelocityConfig:
    return config.velocity


@pytest.fixture
def render_config(config: OptiConfig) -> RenderConfig:
    return config.render


@pytest.fixture
def playback_config(config: OptiConfig) -> PlaybackConfig:
    return config.playback


@pytest.fixture
def playback(playback_config: PlaybackConfig) -> ITPlayback:
    """The IT playback value-object built from the bundled config (for constructing ITModules in tests)."""
    return it_playback(playback_config)


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
def optimize_settings(config: OptiConfig, composite: CompositeFidelity) -> Callable[..., OptimizeSettings]:
    """Factory: an ``OptimizeSettings`` from the bundled config, overriding the swept grid/method/seed.

    ``sweep`` (a ``SweepConfig``, usually built via the ``sweep`` factory) is the only knob the
    optimize tests vary; ``encode``, ``composite`` and ``velocity`` come from the bundled config.
    """

    def _build(*, sweep: SweepConfig, method: Method | None = None, seed: int = 0) -> OptimizeSettings:
        return OptimizeSettings(
            sweep=sweep,
            encode=config.encode,
            composite=composite,
            velocity=config.velocity,
            method=method if method is not None else config.optimize.method,
            seed=seed,
        )

    return _build


@pytest.fixture
def make_encode_ctx(config: OptiConfig) -> Callable[..., EncodeContext]:
    """Factory: an ``EncodeContext`` at ``root_pitch`` using the bundled encode config.

    ``seed`` (when given) seeds the dither RNG; the default leaves it ``None`` so encoding uses the
    surrogate's own fixed-seed fallback -- matching the pre-config call sites.
    """

    def _build(root_pitch: int, *, seed: int | None = None) -> EncodeContext:
        rng = np.random.default_rng(seed) if seed is not None else None
        return EncodeContext(root_pitch=root_pitch, config=config.encode, rng=rng)

    return _build
