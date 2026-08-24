from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.loader import load_config
from optisample.config.root import OptiConfig
from optisample.dsp.surrogate import UNLOOPED, EncodeContext, EncodingParams, render
from optisample.io.audio import write_wav
from optisample.io.tracker.envelope import NO_ENVELOPE
from optisample.loop.settle import settle_loop
from optisample.music import midi_to_freq
from optisample.optimize.carrier import PlayedCurve, stored_carrier
from research.bench import Recording
from research.config_tree import variant

Signal = NDArray[np.float64]

_TEMPO: Final = 125
_SEED: Final = 137
_DEPTH: Final = 16  # the grid a pair is stored on, deep enough that the knob is what a listener hears
_HELD_S: Final = 3.0  # how long the note is held, which is several rounds of even the longest loop
_SIDES: Final = ("a", "b")
_ANSWERS: Final = "answers.json"
_REFERENCE: Final = "reference.wav"
_CHEAPEST: Final = 0  # the offer a budget under pressure reaches for, which is the one worth hearing
_FRONTIER: Final = "frontier"  # where the offers one config gives a take are written, cheapest first


@dataclass(frozen=True)
class Candidate:
    """One default a pair is built to decide: what it changes, and what a listener is being asked.

    ``asks`` is the question in the listener's own terms, written into the answer sheet beside the pair so
    what was being judged is readable once the sides are unblinded.
    """

    name: str
    overrides: Mapping[str, Any]
    asks: str


def held_note(take: Recording, config: OptiConfig) -> Signal:
    """``take`` stored the way ``config`` decides and played back for longer than it was recorded.

    The whole of what a loop knob does is heard here: the stage settles which rounds the recording offers,
    the cheapest of them is what a budget under pressure stores, and holding the note past the stored span
    is what makes the wrap sound. A recording the config leaves unlooped plays out and stops, which is
    itself the answer where the knob decides whether anything loops at all.
    """
    settlement = _settled(take, config)
    return _stored(take, config, settlement.settled, _CHEAPEST if settlement.offered else UNLOOPED)


def _stored(take: Recording, config: OptiConfig, settled: Any, offer: int | None) -> Signal:
    """``take`` stored around ``offer`` of the loops ``settled`` holds, and held past its own end."""
    params = EncodingParams(
        target_rate=take.sample_rate,
        depth_bits=_DEPTH,
        trim_s=None,
        dither=config.optimize.sweep.dither,
        loop_index=offer,
    )
    context = EncodeContext(
        root_pitch=take.key.pitch,
        config=config.encode,
        settled=settled,
        rng=np.random.default_rng(_SEED),
    )
    stored = stored_carrier(take.signal, take.sample_rate, params, context, PlayedCurve(NO_ENVELOPE, _TEMPO))
    return render(stored, take.sample_rate, pitch=take.key.pitch, duration_s=_HELD_S)


def _settled(take: Recording, config: OptiConfig) -> Any:
    """The loops ``config`` settles for ``take``, measured over the whole of what the recording holds."""
    return settle_loop(
        take.signal,
        take.sample_rate,
        config.loop,
        root_hz=midi_to_freq(take.key.pitch),
        search_s=take.signal.size / take.sample_rate,
    )


def write_frontier(take: Recording, configs: Path, out: Path) -> dict[str, float]:
    """Write every round the shipped config offers ``take``, cheapest first, and how long each one stores.

    This is the question ``max_offers`` asks and a pair cannot: the cheapest and the dearest offer are
    always among those priced, so what the ones between them buy is whether the middle of the frontier
    sounds like anything the two ends do not.
    """
    settlement = _settled(take, load_config(variant(configs / "shipped", {})))
    folder = out / _FRONTIER / take.label
    folder.mkdir(parents=True, exist_ok=True)
    write_wav(folder / _REFERENCE, take.signal, take.sample_rate)
    stored: dict[str, float] = {}
    config = load_config(variant(configs / "shipped", {}))
    for offer, held in enumerate(settlement.offered):
        write_wav(folder / f"offer{offer}.wav", _stored(take, config, settlement.settled, offer), take.sample_rate)
        stored[f"offer{offer}"] = held.loop.end / take.sample_rate

    return stored


def _sides(random: Random) -> tuple[str, str]:
    """Which side each rendering is written as, drawn so a listener meets the two in no fixed order."""
    drawn = list(_SIDES)
    random.shuffle(drawn)
    return drawn[0], drawn[1]


def write_pair(take: Recording, candidate: Candidate, configs: Path, out: Path, random: Random) -> dict[str, str]:
    """Write one blinded pair and the recording behind it, and answer which side held which config.

    The shipped tree and the tree carrying the candidate's overrides are both built and validated before
    either is heard, so a pair that reaches the folder is one both configs actually produced.
    """
    shipped = load_config(variant(configs / "shipped", {}))
    changed = load_config(variant(configs / candidate.name, dict(candidate.overrides)))
    folder = out / candidate.name / take.label
    folder.mkdir(parents=True, exist_ok=True)
    write_wav(folder / _REFERENCE, take.signal, take.sample_rate)
    left, right = _sides(random)
    write_wav(folder / f"{left}.wav", held_note(take, shipped), take.sample_rate)
    write_wav(folder / f"{right}.wav", held_note(take, changed), take.sample_rate)
    return {left: "shipped", right: candidate.name}


def write_set(takes: Sequence[Recording], candidates: Sequence[Candidate], configs: Path, out: Path) -> Path:
    """Write every pair of every candidate, blinded, and the answer sheet naming the sides.

    The answers land in one file at the top rather than beside each pair, so a listener works through the
    folders without meeting them. Sides are drawn from one seeded stream, so the same set is written
    twice over the same takes and a note taken on one reading holds for the other.
    """
    random = Random(_SEED)
    answers: dict[str, Any] = {}
    for candidate in candidates:
        answers[candidate.name] = {
            "asks": candidate.asks,
            "overrides": dict(candidate.overrides),
            "pairs": {take.label: write_pair(take, candidate, configs, out, random) for take in takes},
        }

    out.mkdir(parents=True, exist_ok=True)
    sheet = out / _ANSWERS
    sheet.write_text(json.dumps(answers, indent=1), encoding="utf-8")
    return sheet
