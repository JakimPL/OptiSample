import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.artifacts.bank import BankDocument, bank_document
from optisample.artifacts.serialize import VelocityMapDocument, json_text
from optisample.io.tracker.target import ExportTarget
from optisample.optimize.layers.slots import SlotLayout
from trackmod.core.instruments.transfer import extract
from trackmod.core.songs.song import Song

MANIFEST_NAME: Final = "bank.json"
INSTRUMENTS_ENTRY: Final = "instruments"
_ENTRY_DATE: Final = (1980, 1, 1, 0, 0, 0)  # the zip epoch, which is the earliest stamp the format states
_ENTRY_MODE: Final = 0o644 << 16  # zip keeps unix permissions in the high half of the external attributes


@dataclass(frozen=True)
class BankEntry:
    """One file a bank holds: the name it is stored under, and the bytes stored there."""

    name: str
    data: bytes


@dataclass(frozen=True)
class BankContents:
    """A whole bank in memory: the manifest and every instrument it names.

    A producer measures the dynamics a layer reads against the very waveforms it stores, which makes the
    manifest and the instruments one calibrated unit. Holding them together is what lets the same bank be
    written as one archive and spread over a directory from a single description.
    """

    document: BankDocument
    entries: tuple[BankEntry, ...]

    @property
    def manifest(self) -> BankEntry:
        """The manifest as an entry of its own, which is where a reader opens the bank."""
        return BankEntry(name=MANIFEST_NAME, data=json_text(self.document).encode("utf-8"))

    @property
    def stored(self) -> tuple[BankEntry, ...]:
        """Every entry the archive holds: the manifest, then the instruments in the order it names them."""
        return (self.manifest, *self.entries)


def _instrument_entry(song: Song, layout: SlotLayout, index: int, target: ExportTarget) -> BankEntry:
    """The entry one written instrument is stored as, taken out of the song that numbers it.

    Extraction renumbers a slot's samples into a table of its own, which is what lets the entry be loaded
    into a song knowing nothing of the one it was written beside. The name states the keys and the
    dynamics the instrument answers, so a bank reads as the map of which voice plays what.
    """
    written = target.instrument_file(extract(song, index))
    return BankEntry(
        name=f"{INSTRUMENTS_ENTRY}/{layout.slots[index].file_label}{written.extension}",
        data=written.to_bytes(),
    )


def bank_contents(
    name: str,
    song: Song,
    layout: SlotLayout,
    velocity_map: VelocityMapDocument,
    target: ExportTarget,
) -> BankContents:
    """The bank a written plan is played through: every voice the song numbers, and the manifest naming them.

    The entries stand in the order the module numbers its instruments, which is the order the manifest
    states its layers in, so a note's dynamic and pitch resolve to the entry the allocation stored them
    in. Each instrument is serialised once here and the caller decides where those bytes land.
    """
    entries = tuple(_instrument_entry(song, layout, index, target) for index in range(layout.count))
    return BankContents(
        document=bank_document(name, layout, [entry.name for entry in entries], velocity_map),
        entries=entries,
    )


def write_container(path: Path, contents: BankContents) -> None:
    """Write a bank as one archive holding its manifest and every instrument the manifest names.

    A bank shipped as a single file keeps the manifest with the waveforms it was measured against however
    it is copied, renamed or handed on. Every entry carries a fixed stamp and the permissions a readable
    file keeps, so one plan writes one archive byte for byte and an unpacked bank is readable where it
    lands.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as container:
        for entry in contents.stored:
            info = zipfile.ZipInfo(filename=entry.name, date_time=_ENTRY_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = _ENTRY_MODE
            container.writestr(info, entry.data)
