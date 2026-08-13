from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from optisample.config.codec import EncodeConfig
from optisample.dsp.surrogate import NO_LOOPS, EncodeContext, StoredSample, encode
from optisample.io.tracker.envelope import NO_ENVELOPE, sounding_gain
from optisample.io.tracker.target import ExportTarget, balanced_gains, sample_label
from optisample.metrics.base import Signal
from optisample.music import sounded_note
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.coverage import covered_routing
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout
from optisample.optimize.plans import SampleUnit, StrategyPlan
from optisample.optimize.tasks import StoredRecordings
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.instruments.keymap import KeyAssignment, Keymap, routed_keymap
from trackmod.core.notes.pitch import Note
from trackmod.core.samples.loop import Loop
from trackmod.core.samples.sample import Sample
from trackmod.spec.levels import MAX_VOLUME


@dataclass(frozen=True)
class PlannedSamples:
    """What a plan is written as: the stored waveforms, the keys routed onto them, and how each was encoded.

    ``samples`` and ``keymaps`` are what the song carries; ``stored`` is the encoder's own answer for each
    sample and ``gains`` the 0-64 multiplier written beside it, both in the same order. Keeping the four
    together lets one instrument's envelope be fitted from the very samples it starts, at the levels the
    module actually plays them at.
    """

    samples: tuple[Sample, ...]
    keymaps: tuple[Keymap, ...]
    stored: tuple[StoredSample, ...]
    gains: tuple[int, ...]


def carried_signal(
    representative: Signal,
    envelope: Envelope | None,
    *,
    tempo: int,
    sample_rate: int,
) -> Signal:
    """``representative`` divided by the gain the instrument's own curve plays it down by.

    This is what a stored carrier holds: the timbre, with the level handed to the envelope above it, so a
    shallow grid spends itself on sound rather than on a decline the curve states anyway. A slot carrying
    no curve leaves its recording as it stands.
    """
    if envelope is NO_ENVELOPE:
        return representative

    gain = sounding_gain(envelope, tempo=tempo, frames=int(representative.size), sample_rate=sample_rate)
    return np.asarray(representative / gain, dtype=np.float64)


@dataclass(frozen=True)
class EncodeOrder:
    """What re-encoding one plan's units is carried out with, beyond the units and the recordings.

    ``envelopes`` names the curve the instrument each unit belongs to plays it down by, one per unit in
    plan order, and ``tempo`` the clock those curves were written on -- the pair a run storing carriers
    divides its recordings by. ``seed`` starts the one generator every unit's dither is drawn from, so the
    byte layout reproduces the plan exactly.
    """

    config: EncodeConfig
    envelopes: Sequence[Envelope | None]
    tempo: int
    seed: int


def encode_plan_units(
    units: Sequence[SampleUnit],
    recordings: StoredRecordings,
    order: EncodeOrder,
) -> Iterator[tuple[SampleUnit, StoredSample]]:
    """Re-encode each unit's representative recording in plan order from one seeded RNG.

    The exporter and the artifact dumper share this loop so the decoded PCM stays byte-identical between
    the written module and the inspection WAVs. One RNG advances once per unit in iteration order, so
    every stored sample's dither is reproducible from ``seed``. ``recordings`` supplies the loop each one was
    offers, of which a unit asking to be stored looped names the one it was priced against.

    ``envelopes`` names the curve the instrument each unit belongs to will play it down by, one per unit in
    the same order, so a run storing carriers hands the encoder the recording already divided by it. A run
    storing recordings passes no curves and the signal reaches the encoder as it was played.
    """
    rng = np.random.default_rng(order.seed)
    for position, unit in enumerate(units):
        representative: Signal = carried_signal(
            recordings.audio[unit.representative_key],
            order.envelopes[position],
            tempo=order.tempo,
            sample_rate=recordings.sample_rate,
        )
        encode_context = EncodeContext(
            root_pitch=unit.representative,
            config=order.config,
            settled=recordings.settled.get(unit.representative_key, NO_LOOPS),
            rng=rng,
        )
        yield unit, encode(
            representative,
            recordings.sample_rate,
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

    What each sample asks for is read against the velocity map the same plan writes into the patterns
    (:func:`_makeup`), and the set is then stated across the steps the format keeps
    (:func:`~optisample.io.tracker.target.balanced_gains`).
    """
    return balanced_gains([_makeup(unit, stored, velocity_map) for unit, stored in encoded], target)


def sample_name(instrument_id: str, unit: SampleUnit) -> str:
    """The stored sample's display name, taken from the recording the unit re-encodes."""
    key = unit.representative_key
    return sample_label(instrument_id, pitch=key.pitch, velocity=key.velocity)


def _unit_assignments(unit: SampleUnit, sample: int, target: ExportTarget) -> dict[Note, KeyAssignment]:
    """Route every key the unit serves to ``sample``, transposed from the unit's own recorded pitch."""
    root_key = target.key(unit.representative)
    keys = (target.key(pitch) for pitch in unit.keys)
    return {key: KeyAssignment(sample=sample, note=sounded_note(key, root_key)) for key in keys}


def _slot_routing(slot: InstrumentSlot, target: ExportTarget) -> dict[Note, KeyAssignment]:
    """Every key one written instrument was given a recording for, routed to the sample holding it.

    Samples are numbered across the whole plan, so a routing names them by the position each takes in the
    song's single sample list, which is what the slot carries beside the units themselves.
    """
    routing: dict[Note, KeyAssignment] = {}
    for sample, unit in zip(slot.samples, slot.units):
        routing.update(_unit_assignments(unit, sample, target))

    return routing


def _slot_keymaps(layout: SlotLayout, target: ExportTarget) -> tuple[Keymap, ...]:
    """One key routing per written instrument, each holding only the samples that instrument owns.

    A keymap is keyed by note alone, so both axes a plan splits live in the choice of instrument the
    pattern names: the velocity band, and the stretch of keyboard a format numbering few samples per
    instrument cuts that band into. Splitting the routings here is what lets one key play a soft
    recording at one dynamic and a loud one at another. Each routing is then widened to the whole
    keyboard (:func:`~optisample.optimize.export.coverage.covered_routing`) from the samples that
    instrument holds, so every written instrument answers every key the format numbers on its own.
    """
    return tuple(routed_keymap(covered_routing(_slot_routing(slot, target), target)) for slot in layout.slots)


def unit_envelopes(layout: SlotLayout, envelopes: Sequence[Envelope | None], units: int) -> tuple[Envelope | None, ...]:
    """The curve each stored sample's own instrument plays it down by, one per unit in plan order."""
    by_sample: dict[int, Envelope | None] = {}
    for index, slot in enumerate(layout.slots):
        for sample in slot.samples:
            by_sample[sample] = envelopes[index]

    return tuple(by_sample.get(sample, NO_ENVELOPE) for sample in range(units))


def plan_samples(
    plan: StrategyPlan,
    layout: SlotLayout,
    recordings: StoredRecordings,
    context: ExportContext,
    *,
    envelopes: Sequence[Envelope | None],
) -> PlannedSamples:
    """Re-encode each unit's representative and map every key it serves onto the resulting sample.

    Units are encoded in order from one seeded RNG, so the byte layout reproduces the plan exactly. The
    whole set is encoded before any sample is built, because each one's gain is stated against the
    sample asking for the most of it (:func:`sample_gains`), and read alongside the velocity map the
    same plan writes into the patterns. The routings come back one per slot of ``layout``, in the order
    the module numbers its instruments, so the caller writes one instrument for each.
    """
    units = plan.sample_units()
    encoded = list(
        encode_plan_units(
            units,
            recordings,
            EncodeOrder(
                config=context.encode,
                envelopes=unit_envelopes(layout, envelopes, len(units)),
                tempo=context.envelope_grid.tempo,
                seed=context.seed,
            ),
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
    return PlannedSamples(
        samples=samples,
        keymaps=_slot_keymaps(layout, context.target),
        stored=tuple(stored for _, stored in encoded),
        gains=gains,
    )
