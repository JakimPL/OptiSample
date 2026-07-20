from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from optisample.io.manifest import dump_manifest, load_manifest
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample


def _manifest() -> Manifest:
    return Manifest(
        project=ProjectSpec(name="song"),
        instruments=[
            InstrumentSpec(
                id="strings",
                budget_kb=128.0,
                samples=[SourceSample(file=Path("strings/p60.wav"), pitch=60, velocity=100)],
                material=[NoteEvent(pitch=60, velocity=100, duration_s=2.0, count=3)],
            )
        ],
    )


def test_dump_then_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    dump_manifest(_manifest(), path)
    loaded = load_manifest(path)

    assert loaded.project.name == "song"
    assert loaded.instruments[0].id == "strings"
    assert loaded.instruments[0].material is not None


def test_relative_sample_paths_resolved_against_manifest_dir(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    dump_manifest(_manifest(), path)
    loaded = load_manifest(path)

    resolved = loaded.instruments[0].samples[0].file
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "strings" / "p60.wav").resolve()


def test_none_fields_omitted_in_dump(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    dump_manifest(_manifest(), path)
    raw = yaml.safe_load(path.read_text())
    assert "material_midi" not in raw["instruments"][0]
    assert "articulation" not in raw["instruments"][0]["samples"][0]


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)
