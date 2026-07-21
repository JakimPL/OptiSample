from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from optisample.config import OptiConfig, load_config
from optisample.config.dsp import StftParams


def _write_bundled_copy(directory: Path) -> None:
    """Copy the bundled YAML group files into ``directory`` (so a single one can then be tweaked)."""
    for name in OptiConfig.model_fields:
        directory.joinpath(f"{name}.yaml").write_text(yaml.safe_dump(load_group(name)), encoding="utf-8")


def load_group(name: str) -> object:
    from importlib import resources  # local import keeps the helper self-contained

    return yaml.safe_load((resources.files("opticonfig") / f"{name}.yaml").read_text(encoding="utf-8"))


def test_yaml_lists_coerce_to_tuples() -> None:
    cfg = load_config()
    assert isinstance(cfg.sweep.depths, tuple)
    assert isinstance(cfg.metrics.mrstft.resolutions, tuple)
    assert isinstance(cfg.metrics.mrstft.resolutions[0], StftParams)
    assert isinstance(cfg.synth.presets[0].material, tuple)


def test_encode_is_derived_from_loop_and_quantize() -> None:
    cfg = load_config()
    assert cfg.encode.loop == cfg.loop
    assert cfg.encode.target_peak == cfg.quantize.target_peak


def test_custom_directory_overrides_bundled(tmp_path: Path) -> None:
    _write_bundled_copy(tmp_path)
    loop = yaml.safe_load((tmp_path / "loop.yaml").read_text())
    loop["min_loop_s"] = 0.75
    (tmp_path / "loop.yaml").write_text(yaml.safe_dump(loop), encoding="utf-8")
    assert load_config(tmp_path).loop.min_loop_s == 0.75


def test_missing_group_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path)  # empty directory: the first group file is absent


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    _write_bundled_copy(tmp_path)
    loop = yaml.safe_load((tmp_path / "loop.yaml").read_text())
    loop["bogus_key"] = 1
    (tmp_path / "loop.yaml").write_text(yaml.safe_dump(loop), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(tmp_path)
