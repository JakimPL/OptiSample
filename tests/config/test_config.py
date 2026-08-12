from collections.abc import Callable
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from pydantic import ValidationError

from optisample.config import OptiConfig, load_config
from optisample.config.spectral import StftParams

_YAML_SUFFIX: Final = ".yaml"
_PRIVATE_PREFIX: Final = "_"  # dunder directories the package carries beside its settings


def _copy_tree(source: Traversable, destination: Path) -> None:
    """Copy the YAML tree under ``source`` into ``destination``, keeping the shape it was laid out in."""
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name.startswith(_PRIVATE_PREFIX):
            continue

        if entry.is_dir():
            _copy_tree(entry, destination / entry.name)
        elif entry.name.endswith(_YAML_SUFFIX):
            (destination / entry.name).write_text(entry.read_text(encoding="utf-8"), encoding="utf-8")


def _bundled_copy(directory: Path) -> None:
    """The bundled settings laid out under ``directory``, where a single group can then be retuned."""
    _copy_tree(resources.files("opticonfig"), directory)


def _retune(directory: Path, *parts: str, **changes: Any) -> None:
    """Rewrite one copied group file with ``changes`` over it, which is what tuning a run does to it."""
    path = directory.joinpath(*parts[:-1], f"{parts[-1]}{_YAML_SUFFIX}")
    settings = {**yaml.safe_load(path.read_text(encoding="utf-8")), **changes}
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")


def test_yaml_lists_coerce_to_tuples() -> None:
    cfg = load_config()
    assert isinstance(cfg.optimize.sweep.rates, tuple)
    assert isinstance(cfg.analysis.metrics.mrstft.resolutions, tuple)
    assert isinstance(cfg.analysis.metrics.mrstft.resolutions[0], StftParams)
    assert isinstance(cfg.synth.presets[0].material, tuple)


def test_encode_is_derived_from_the_stages_that_own_its_parts() -> None:
    """The seam comes from the loop stage and the shaping from the codec stage, gathered into one bundle."""
    cfg = load_config()
    assert cfg.encode.seam == cfg.loop.seam
    assert cfg.encode.dynamics == cfg.codec.dynamics
    assert cfg.encode.headroom_db == cfg.codec.quantize.headroom_db


def test_a_freshly_loaded_encode_config_names_no_instrument_to_normalize_against() -> None:
    """The reference is a per-instrument measurement, so config alone leaves each clip on its own peak."""
    assert load_config().encode.peak_reference is None


@pytest.mark.parametrize(
    ("parts", "setting", "value", "read"),
    [
        pytest.param(
            ("loop", "geometry"),
            "min_loop_s",
            0.75,
            lambda cfg: cfg.loop.geometry.min_loop_s,
            id="a stage reads each of its groups from a file in the directory naming the stage",
        ),
        pytest.param(
            ("runtime",),
            "workers",
            3,
            lambda cfg: cfg.runtime.workers,
            id="a group standing on its own reads the one file that names it",
        ),
        pytest.param(
            ("subsonic",),
            "cutoff_hz",
            25.0,
            lambda cfg: cfg.subsonic.cutoff_hz,
            id="the band a run works in is retuned where it stands, beside the stages that read past it",
        ),
    ],
)
def test_a_retuned_directory_is_read_in_place_of_the_bundled_settings(
    tmp_path: Path,
    parts: tuple[str, ...],
    setting: str,
    value: object,
    read: Callable[[OptiConfig], object],
) -> None:
    _bundled_copy(tmp_path)
    _retune(tmp_path, *parts, **{setting: value})

    assert read(load_config(tmp_path)) == value


def test_missing_group_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path)  # empty directory: the first group file is absent


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    _bundled_copy(tmp_path)
    _retune(tmp_path, "loop", "geometry", bogus_key=1)

    with pytest.raises(ValidationError):
        load_config(tmp_path)
