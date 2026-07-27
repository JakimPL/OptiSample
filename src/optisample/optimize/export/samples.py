from collections.abc import Iterator, Sequence
from typing import Final

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.dsp.surrogate import EncodeContext, StoredSample, encode
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.base import Signal
from optisample.music import note_name, sounded_note
from optisample.optimize.export.context import ExportContext
from optisample.optimize.layers.bands import VelocityLayers
from optisample.optimize.plans import SampleUnit, StrategyPlan
from optisample.optimize.tasks import AudioMap
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.core.instruments.keymap import KeyAssignment, Keymap, routed_keymap
from trackmod.core.notes.pitch import Note
from trackmod.core.samples.loop import Loop
from trackmod.core.samples.sample import Sample
from trackmod.spec.levels import MAX_VOLUME

_SAMPLE_LABEL_CHARS: Final = 13  # instrument-id chars kept before the " <note> v<velocity>" suffix, XM's 22.
_UNIT_GAIN: Final = 1.0  # what a sample stored without scaling plays back at
_QUIETEST_GAIN: Final = 1  # the softest step that still sounds, so a quiet sample is heard rather than dropped


def encode_plan_units(
    units: Sequence[SampleUnit],
    audio: AudioMap,
    sample_rate: int,
    encode_config: EncodeConfig,
    seed: int,
) -> Iterator[tuple[SampleUnit, StoredSample]]:
    """Re-encode each unit's representative recording in plan order from one seeded RNG.

    The exporter and the artifact dumper share this loop so the decoded PCM stays byte-identical between
    the written module and the inspection WAVs. One RNG advances once per unit in iteration order, so
    every stored sample's dither is reproducible from ``seed``.
    """
    rng = np.random.default_rng(seed)
    for unit in units:
        representative: Signal = audio[unit.representative_key]
        encode_context = EncodeContext(
            root_pitch=unit.representative,
            config=encode_config,
            rng=rng,
        )
        yield unit, encode(
            representative,
            sample_rate,
            unit.params,
            encode_context,
        )


def _stored_loop(stored: StoredSample) -> Loop | None:
    """The stored sample's loop as the half-open frame range a tracker repeats."""
    return None if stored.loop is None else Loop(begin=stored.loop.start, end=stored.loop.end)


def _makeup(unit: SampleUnit, stored: StoredSample, velocity_map: VelocityVolumeMap) -> float:
    """How loud one sample has to play for its notes to sound at the level they were recorded at.

    Two multipliers meet on every note: this one, and the note volume the velocity map writes into the
    pattern. That volume already states the dynamic of the velocity this sample was recorded at, so what
    is left here is the rest -- undoing how hot the sample was stored, then dividing out the dynamic the
    pattern will apply again. What survives the division is the balance between the recordings
    themselves, which is the part the velocity axis never carried. A representative whose velocity maps
    to silence leaves its own scaling to stand, since the pattern silences the note either way.
    """
    written = velocity_map.volume(unit.representative_key.velocity)
    if written <= 0:
        return stored.playback_gain

    return stored.playback_gain * MAX_VOLUME / written


def sample_gains(
    encoded: Sequence[tuple[SampleUnit, StoredSample]],
    velocity_map: VelocityVolumeMap,
    target: ExportTarget,
) -> tuple[int, ...]:
    """Each sample's playback multiplier, scaled so the one needing most of it takes the top step.

    Every sample is stored as hot as its own depth allows, which spends the whole grid on one recording
    and leaves the instrument flat -- a naturally quiet key comes back as loud as a bright one. The 0-64
    gain the format keeps per sample is where that balance is restored (see :func:`_makeup`), stated
    relative to the sample asking for the most so the whole set fits the steps available. A format
    pinning the gain to full scale carries the balance in the PCM instead, so every sample there reports
    the same top step.
    """
    if not target.stores_sample_gain:
        return tuple(MAX_VOLUME for _ in encoded)

    makeups = [_makeup(unit, stored, velocity_map) for unit, stored in encoded]
    loudest = max(makeups, default=_UNIT_GAIN)
    return tuple(max(_QUIETEST_GAIN, round(MAX_VOLUME * makeup / loudest)) for makeup in makeups)


def sample_name(instrument_id: str, unit: SampleUnit) -> str:
    """The stored sample's display name: the instrument, shortened, plus the recording it holds.

    Naming the recording rather than the key tells the samples of a layered instrument apart, since one
    key stores a recording per velocity band and the tracker lists them side by side. The whole name fits
    the narrowest field a target format keeps for it.
    """
    key = unit.representative_key
    return f"{instrument_id[:_SAMPLE_LABEL_CHARS]} {note_name(key.pitch)} v{key.velocity}"


def _unit_assignments(unit: SampleUnit, sample: int, target: ExportTarget) -> dict[Note, KeyAssignment]:
    """Route every key the unit serves to ``sample``, transposed from the unit's own recorded pitch."""
    root_key = target.key(unit.representative)
    keys = (target.key(pitch) for pitch in unit.keys)
    return {key: KeyAssignment(sample=sample, note=sounded_note(key, root_key)) for key in keys}


def _layer_keymaps(units: Sequence[SampleUnit], layers: VelocityLayers, target: ExportTarget) -> tuple[Keymap, ...]:
    """One key routing per velocity layer, each holding only the samples its own band stores.

    A layer is written as its own instrument, and a keymap is keyed by note alone, so the velocity axis
    lives in the choice of instrument the pattern names. Splitting the routings here is what lets one key
    play a soft recording at one dynamic and a loud one at another. Samples are numbered across the whole
    plan, so a routing names its samples by their position in the song's single sample list.
    """
    assignments: tuple[dict[Note, KeyAssignment], ...] = tuple({} for _ in layers.bands)
    for index, unit in enumerate(units):
        assignments[unit.layer].update(_unit_assignments(unit, index, target))

    return tuple(routed_keymap(routing) for routing in assignments)


def plan_samples(
    plan: StrategyPlan,
    audio: AudioMap,
    sample_rate: int,
    context: ExportContext,
) -> tuple[tuple[Sample, ...], tuple[Keymap, ...]]:
    """Re-encode each unit's representative and map every key it serves onto the resulting sample.

    Units are encoded in order from one seeded RNG, so the byte layout reproduces the plan exactly. The
    whole set is encoded before any sample is built, because each one's gain is stated against the
    sample asking for the most of it (:func:`sample_gains`), and read alongside the velocity map the
    same plan writes into the patterns. The routings come back one per velocity layer, in band order, so
    the caller writes one instrument for each.
    """
    encoded = list(
        encode_plan_units(
            plan.sample_units(),
            audio,
            sample_rate,
            context.encode,
            context.seed,
        )
    )
    gains = sample_gains(encoded, plan.velocity_map, context.target)
    samples = tuple(
        Sample(
            name=sample_name(plan.instrument_id, unit),
            pcm=stored.pcm,
            rate=stored.sample_rate,
            depth=stored.depth,
            gain=gain,
            loop=_stored_loop(stored),
        )
        for (unit, stored), gain in zip(encoded, gains)
    )
    keymaps = _layer_keymaps([unit for unit, _ in encoded], plan.layers, context.target)
    return samples, keymaps
