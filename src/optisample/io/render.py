"""Render ``.IT`` modules to audio with ``openmpt123`` -- the ground-truth playback engine.

The optimizer's inner loop uses the fast numpy surrogate (:mod:`optisample.dsp.surrogate`); this
wrapper drives the *real* tracker so we can calibrate that surrogate against reality (P4) and
validate exported modules. ``openmpt123`` is an external system binary, not a Python dependency, so
this module degrades gracefully: :func:`openmpt123_available` reports whether it is installed and the
render functions raise a clear, actionable error when it is not.

We always render **mono float** at a fixed interpolation so the ground truth is reproducible -- the
IT format does not store the interpolation filter, it is a player setting (see the plan), and the
default here is 8-tap sinc/polyphase, OpenMPT's highest-quality mode. Rendering happens in a
temporary directory (``openmpt123 --render`` writes ``<input>.wav`` next to its input), so the
caller's files are never touched.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from optisample.config.render import Interpolation, RenderConfig
from optisample.io.audio import read_wav
from optisample.io.it_writer import ITModule, write_it

_BINARY = "openmpt123"
# openmpt123 --filter takes interpolation *taps*; more taps = higher-quality (sinc) interpolation.
_INTERPOLATION_TAPS: dict[Interpolation, int] = {"none": 1, "linear": 2, "cubic": 4, "sinc": 8}


def filter_taps(config: RenderConfig) -> int:
    """Map the interpolation name to the ``--filter`` tap count openmpt123 expects.

    ``RenderConfig.interpolation`` is a validated literal, so every value is a key of the map.
    """
    return _INTERPOLATION_TAPS[config.interpolation]


def openmpt123_available() -> bool:
    """Return whether the ``openmpt123`` binary is discoverable on ``PATH``."""
    return shutil.which(_BINARY) is not None


def _require_binary() -> None:
    if not openmpt123_available():
        raise RuntimeError(
            f"{_BINARY!r} not found on PATH; install it (e.g. `apt install openmpt123`) to render .IT files"
        )


def _render_in_place(it_path: Path, config: RenderConfig) -> tuple[NDArray[np.float64], int]:
    """Run ``openmpt123 --render`` on ``it_path`` and read back the ``<it_path>.wav`` it produces."""
    command = [
        _BINARY,
        "--render",
        "--samplerate",
        str(config.sample_rate),
        "--channels",
        "1",
        "--filter",
        str(filter_taps(config)),
        "--gain",
        str(config.gain_db),
        "--force",
        "--quiet",
        str(it_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{_BINARY} render failed (exit {result.returncode}): {result.stderr.strip()}")
    rendered, rate = read_wav(it_path.with_name(it_path.name + ".wav"))
    return np.asarray(rendered, dtype=np.float64).ravel(), rate


def render_it(path: Path | str, config: RenderConfig) -> tuple[NDArray[np.float64], int]:
    """Render an existing ``.IT`` file to mono float PCM, returning ``(samples, sample_rate)``.

    Raises :class:`RuntimeError` if ``openmpt123`` is missing or the render fails.
    """
    _require_binary()
    source = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / source.name
        local.write_bytes(source.read_bytes())
        return _render_in_place(local, config)


def render_module(module: ITModule, config: RenderConfig) -> tuple[NDArray[np.float64], int]:
    """Write ``module`` to a temporary ``.IT`` file and render it (see :func:`render_it`)."""
    _require_binary()
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "module.it"
        write_it(local, module)
        return _render_in_place(local, config)
