from collections.abc import Iterator, Sequence
from typing import Final

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.dsp.surrogate import EncodeContext, StoredSample, encode
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.base import Signal
from optisample.music import note_name, sounded_note
from optisample.optimize.export.context import ExportContext
from optisample.optimize.plans import SampleUnit
from optisample.optimize.tasks import AudioMap
from trackmod.core.instruments.keymap import KeyAssignment, Keymap, routed_keymap
from trackmod.core.notes.pitch import Note
from trackmod.core.samples.loop import Loop
from trackmod.core.samples.sample import Sample

_SAMPLE_LABEL_CHARS: Final = 18  # instrument-id chars kept before the " <note>" suffix in a sample name.


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
        representative: Signal = audio[(unit.representative, unit.representative_velocity)]
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


def sample_name(instrument_id: str, unit: SampleUnit) -> str:
    """The stored sample's display name: the instrument, shortened, plus the note it was recorded at."""
    return f"{instrument_id[:_SAMPLE_LABEL_CHARS]} {note_name(unit.representative)}"


def _unit_assignments(unit: SampleUnit, sample: int, target: ExportTarget) -> dict[Note, KeyAssignment]:
    """Route every key the unit serves to ``sample``, transposed from the unit's own recorded pitch."""
    root_key = target.key(unit.representative)
    keys = (target.key(pitch) for pitch in unit.keys)
    return {key: KeyAssignment(sample=sample, note=sounded_note(key, root_key)) for key in keys}


def plan_samples(
    instrument_id: str,
    units: Sequence[SampleUnit],
    audio: AudioMap,
    sample_rate: int,
    context: ExportContext,
) -> tuple[tuple[Sample, ...], Keymap]:
    """Re-encode each unit's representative and map every key it serves onto the resulting sample.

    Units are encoded in order from one seeded RNG, so the byte layout reproduces the plan exactly.
    """
    samples: list[Sample] = []
    assignments: dict[Note, KeyAssignment] = {}
    encoded = encode_plan_units(
        units,
        audio,
        sample_rate,
        context.encode,
        context.seed,
    )
    for index, (unit, stored) in enumerate(encoded):
        samples.append(
            Sample(
                name=sample_name(instrument_id, unit),
                pcm=stored.pcm,
                rate=stored.sample_rate,
                depth=stored.depth,
                loop=_stored_loop(stored),
            )
        )
        assignments.update(_unit_assignments(unit, index, context.target))

    return tuple(samples), routed_keymap(assignments)
