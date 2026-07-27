from __future__ import annotations

from optisample.config.reduce import ReduceConfig
from optisample.model import InstrumentSpec
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers
from optisample.optimize.tasks import AudioMap, PitchTask, build_tasks
from optisample.optimize.velocity_map import VelocityVolumeMap


def band_instrument(instrument: InstrumentSpec, band: VelocityBand) -> InstrumentSpec:
    """``instrument`` as one layer sees it: every recording it has, playing the notes ``band`` answers for.

    The recordings stay whole because a layer still scores each note against the recording nearest that
    note's own velocity, and may still store any survivor at a key. Narrowing the material is what makes a
    layer's representative the recording nearest the loudest dynamic *it* covers, which is the one move
    velocity layering adds.
    """
    played = [event for event in instrument.material if band.covers(event.velocity)]
    return instrument.model_copy(update={"material": played})


def band_tasks(
    instrument: InstrumentSpec,
    audio: AudioMap,
    velocity_map: VelocityVolumeMap,
    reduce: ReduceConfig,
    band: VelocityBand,
) -> list[PitchTask]:
    """One layer's ordered pitch tasks: the keys ``band`` plays, and what each of them stores and scores.

    The whole-instrument rule (:func:`~optisample.optimize.tasks.build_tasks`) runs on the band's own
    material, so one band covering the entire velocity axis yields exactly the tasks a single-layer plan
    is built from, key for key.
    """
    return build_tasks(band_instrument(instrument, band), audio, velocity_map, reduce)


def layered_tasks(
    instrument: InstrumentSpec,
    audio: AudioMap,
    velocity_map: VelocityVolumeMap,
    reduce: ReduceConfig,
    layers: VelocityLayers,
) -> tuple[tuple[PitchTask, ...], ...]:
    """Each layer's tasks in band order -- the ordered axis a layered partition DP segments over.

    A layer covers the keys its own dynamics are played at, so the lists differ in length and a quiet
    layer holds only the keys the material plays quietly. Every note lands in exactly one layer, so the
    weights across the lists add up to the material's own.
    """
    return tuple(tuple(band_tasks(instrument, audio, velocity_map, reduce, band)) for band in layers.bands)
