import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from optisample.artifacts.container import (
    MANIFEST_NAME,
    BankContents,
    bank_contents,
    write_container,
)
from optisample.artifacts.documents.bank import MANIFEST_VERSION
from optisample.artifacts.documents.velocity import VelocityMapDocument
from optisample.artifacts.serialize import json_text
from optisample.config.tracker import TrackerFormat
from optisample.io.tracker.target import ExportTarget
from optisample.io.tracker.voices import instrument_voices, routed_voices
from optisample.music import MIDI_MAX_VELOCITY
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, routed_keymap
from trackmod.core.instruments.transfer import extract
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.trackers.it.instrument_file import ITInstrumentFile

_NAME = "piano"
_KEYS = (Note(60), Note(72))
_CHANNELS = len(_KEYS)  # a channel per voice, so the shortest pattern either format accepts holds them both
_FRAMES = 64
_RATE = 22_050
_VOLUME = 64
_LAYERS = VelocityLayers((VelocityBand(0, 50), VelocityBand(51, MIDI_MAX_VELOCITY)))
_VOLUMES = [velocity // 2 for velocity in range(MIDI_MAX_VELOCITY + 1)]
_ENTRY_MODE = 0o644
_INSTRUMENT_EXTENSIONS = ((TrackerFormat.IT, ".iti"), (TrackerFormat.XM, ".xi"))


def _song(rows: int) -> Song:
    """A song numbering one instrument per velocity band, which is the shape a two-dynamic plan is written as."""
    builder = PatternBuilder(rows=rows, channels=_CHANNELS)
    for index, key in enumerate(_KEYS):
        builder.place(0, index, Cell(note=key, instrument=index, volume=_VOLUME))

    return Song(
        name=_NAME,
        channels=_CHANNELS,
        patterns=(builder.build(),),
        order=OrderList.sequential(1),
        voices=instrument_voices(
            tuple(
                Instrument(name=f"{_NAME} {index}", keymap=routed_keymap({key: KeyAssignment(sample=index, note=key)}))
                for index, key in enumerate(_KEYS)
            ),
            tuple(
                Sample(name=f"s{index}", pcm=np.full(_FRAMES, 0.5), rate=_RATE, depth=BitDepth.SIXTEEN)
                for index in range(len(_KEYS))
            ),
        ),
        playback=Playback(speed=6, tempo=125),
    )


def _layout() -> SlotLayout:
    """One instrument per band, each holding the one sample its own dynamics were stored at."""
    return SlotLayout(
        layers=_LAYERS,
        slots=tuple(
            InstrumentSlot(layer=layer, band=band, samples=(layer,), units=())
            for layer, band in enumerate(_LAYERS.bands)
        ),
    )


def _velocity_map() -> VelocityMapDocument:
    return VelocityMapDocument(reference_volume=max(_VOLUMES), anchors=[], volumes=_VOLUMES)


def _contents(target: ExportTarget) -> BankContents:
    return bank_contents(_NAME, _song(target.min_rows), _layout(), _velocity_map(), target)


@pytest.fixture
def contents(target: ExportTarget) -> BankContents:
    """The bank a two-band plan is played through, held in memory as the manifest and its instruments."""
    return _contents(target)


@pytest.fixture
def container(tmp_path: Path, contents: BankContents) -> Path:
    path = tmp_path / f"{_NAME}.bank"
    write_container(path, contents)
    return path


def test_a_bank_holds_its_manifest_and_every_instrument_it_names(container: Path) -> None:
    """One file carries the whole plan, so a bank plays wherever it is copied, renamed or handed on."""
    with zipfile.ZipFile(container) as archive:
        assert archive.namelist() == [
            MANIFEST_NAME,
            "instruments/v000-v050.iti",
            "instruments/v051-v127.iti",
        ]


def test_every_entry_the_manifest_names_is_stored_in_the_bank(container: Path, contents: BankContents) -> None:
    with zipfile.ZipFile(container) as archive:
        held = set(archive.namelist())

    assert {layer.source.file for layer in contents.document.layers} <= held


def test_an_entry_loads_back_as_the_voice_the_song_numbers(
    container: Path, contents: BankContents, target: ExportTarget
) -> None:
    """An entry stands alone: it holds the instrument's own keymap and its samples' PCM, renumbered."""
    song = _song(target.min_rows)
    with zipfile.ZipFile(container) as archive:
        for index, layer in enumerate(contents.document.layers):
            loaded = ITInstrumentFile.parse(archive.read(layer.source.file)).unit
            held = extract(routed_voices(song), index)
            assert loaded.instrument.keymap == held.instrument.keymap
            assert all(np.array_equal(one.pcm, other.pcm) for one, other in zip(loaded.samples, held.samples))


def test_the_manifest_states_the_version_and_the_map_a_consumer_reads(container: Path) -> None:
    with zipfile.ZipFile(container) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME))

    assert manifest["version"] == MANIFEST_VERSION
    assert manifest["name"] == _NAME
    assert [layer["velocity_map"]["volumes"] for layer in manifest["layers"]] == [_VOLUMES, _VOLUMES]


def test_the_stored_manifest_reads_as_the_document_the_bank_states(container: Path, contents: BankContents) -> None:
    """The archived manifest and the document it was written from state one bank, character for character."""
    with zipfile.ZipFile(container) as archive:
        assert archive.read(MANIFEST_NAME).decode("utf-8") == json_text(contents.document)


def test_one_plan_writes_one_archive(tmp_path: Path, target: ExportTarget) -> None:
    """Every entry carries a fixed stamp, so a run reproduces the bank it wrote byte for byte."""
    first, second = tmp_path / "first.bank", tmp_path / "second.bank"
    write_container(first, _contents(target))
    write_container(second, _contents(target))
    assert first.read_bytes() == second.read_bytes()


def test_an_unpacked_bank_is_readable_where_it_lands(container: Path) -> None:
    with zipfile.ZipFile(container) as archive:
        modes = {entry.external_attr >> 16 for entry in archive.infolist()}

    assert modes == {_ENTRY_MODE}


@pytest.mark.parametrize(("tracker_format", "extension"), _INSTRUMENT_EXTENSIONS)
def test_an_entry_is_stored_as_the_file_its_own_format_writes(
    tracker_format: TrackerFormat,
    extension: str,
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    """The format is the run's, so a bank ships the instrument files the module was written beside."""
    contents = _contents(retarget(tracker_format))
    assert [entry.name for entry in contents.entries] == [
        f"instruments/v000-v050{extension}",
        f"instruments/v051-v127{extension}",
    ]


def test_the_bank_is_written_where_it_is_asked_for(tmp_path: Path, contents: BankContents) -> None:
    """A bank names its own directory into being, so a caller states where it goes and nothing else."""
    path = tmp_path / "shipped" / f"{_NAME}.bank"
    write_container(path, contents)
    assert zipfile.is_zipfile(path)
