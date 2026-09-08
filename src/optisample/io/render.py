import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from trackmod import TrackerModule

from optisample.config.render import Interpolation, RenderConfig
from optisample.io.audio import read_wav

_BINARY: Final = "openmpt123"
_MODULE_STEM: Final = "module"
_INTERPOLATION_TAPS: Final[dict[Interpolation, int]] = {"none": 1, "linear": 2, "cubic": 4, "sinc": 8}


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
            f"{_BINARY!r} not found on PATH; install it (e.g. `apt install openmpt123`) to render modules"
        )


def _render_in_place(module_path: Path, config: RenderConfig) -> tuple[NDArray[np.float64], int]:
    """Run ``openmpt123 --render`` on ``module_path`` and read back the ``<module_path>.wav`` it produces."""
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
        str(module_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{_BINARY} render failed (exit {result.returncode}): {result.stderr.strip()}")

    rendered, rate = read_wav(module_path.with_name(module_path.name + ".wav"))
    return np.asarray(rendered, dtype=np.float64).ravel(), rate


def render_file(
    path: Path | str,
    config: RenderConfig,
) -> tuple[NDArray[np.float64], int]:
    """Render an existing module file to mono float PCM, returning ``(samples, sample_rate)``.

    Raises :class:`RuntimeError` if ``openmpt123`` is missing or the render fails.
    """
    _require_binary()
    source = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / source.name
        local.write_bytes(source.read_bytes())
        return _render_in_place(local, config)


def render_module(
    module: TrackerModule,
    config: RenderConfig,
) -> tuple[NDArray[np.float64], int]:
    """Write ``module`` to a temporary file in its own format and render it (see :func:`render_file`)."""
    _require_binary()
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / f"{_MODULE_STEM}{module.extension}"
        module.save(local)
        return _render_in_place(local, config)
