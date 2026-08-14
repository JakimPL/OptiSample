from collections.abc import Sequence

from optisample.artifacts import (
    DumpResult,
    ListeningSet,
    LoopedInstrumentArtifacts,
    ReducedInstrument,
    SlicedDataset,
    WrittenInstruments,
)
from optisample.calibrate.ranking import MetricAgreement, PairAxis, RankingReport
from optisample.cluster.instruments import ClusteredArtifacts
from optisample.io.dataset import SubsetDataset
from optisample.optimize.reduce.trim import RecordingScreen


def print_screen(screen: RecordingScreen) -> None:
    """State what the silence screen left out, on the runs where it left anything out."""
    if screen.admitted_everything:
        return

    print(
        f"  {len(screen.silenced)} recordings carried no signal, "
        f"dropping {screen.dropped_notes} notes at {len(screen.unplayable)} pitches"
    )


def print_instruments(written: WrittenInstruments) -> None:
    """State how many standalone instruments a stage wrote, and what a format had no room to state."""
    directories = ", ".join(directory.name for directory in written.directories)
    print(f"  {written.files} instruments -> {directories}")
    if written.unreachable:
        print(f"  {len(written.unreachable)} recordings play at a key one format leaves out")

    if written.understated:
        print(f"  {len(written.understated)} sound under the level of the audio they were written from")


def print_clustered(written: ClusteredArtifacts) -> None:
    """State what each velocity band was written as, and what sharing one envelope cost its takes."""
    print(f"  {written.recordings} recordings read, {written.calibrated} of them through a .sample container")
    for band in written.bands:
        looped = sum(1 for sample in band.samples if sample.looped)
        print(
            f"  {band.band}: {len(band.samples)} samples ({looped} looped), "
            f"{band.stored_bytes / 1024:.1f} KiB, dispersion {band.dispersion_db:.2f} dB -> {band.files}"
        )

    print(f"  {written.auditions} auditions, manifest {written.manifest}")


def print_brief(dataset: SubsetDataset) -> None:
    """State what the length floor held out, on the sources where it held anything out."""
    if not dataset.brief_notes:
        return

    print(f"  {dataset.brief_notes} of {dataset.source_notes} notes sounded too briefly to slice")


def print_subset(sliced: SlicedDataset) -> None:
    """State where a slice landed and how much of its source it holds."""
    dataset = sliced.dataset
    print(f"{dataset.source.path}")
    print(f"  {dataset.kept_notes} of {dataset.source_notes} notes, {dataset.recordings} recordings")
    print_brief(dataset)
    print(
        f"  pitches {dataset.pitches[0]}-{dataset.pitches[1]}, velocities {dataset.velocities[0]}-{dataset.velocities[1]}"
    )
    print(f"  samples -> {dataset.source.recordings_dir}")
    print_instruments(sliced.instruments)


def print_looped(result: LoopedInstrumentArtifacts) -> None:
    """State where one instrument's looped dataset landed and how many of its recordings offer a loop."""
    print(f"{result.instrument_id}: {result.paths.notes_json}  [{result.elapsed_s:.1f}s]")
    print(f"  {result.looped} of {result.recordings} recordings offer a loop -> {result.paths.samples_dir}")
    print_instruments(result.instruments)
    print(f"  {result.auditions} auditions -> {result.paths.auditions_dir}")
    print(f"  loops -> {result.paths.loops_json}")


def print_reduced(result: ReducedInstrument) -> None:
    """State where one instrument's reduced dataset landed and what the stage left it holding."""
    print(f"{result.instrument_id}: {result.paths.notes_json}  [{result.elapsed_s:.1f}s]")
    print(f"  {result.survivors} samples, {result.notes} notes -> {result.paths.samples_dir}")
    print(f"  {result.looped} of them offer a loop the written span holds")
    print_screen(result.screen)
    print_instruments(result.instruments)
    print(f"  {result.auditions} auditions -> {result.paths.auditions_dir}")
    print(f"  reduction -> {result.paths.reduction_json}")


def print_listening(result: ListeningSet) -> None:
    """State where one instrument's listening set landed and how much listening it asks for."""
    print(f"{result.instrument_id}: {result.paths.pairs_dir}  [{result.elapsed_s:.1f}s]")
    print(f"  {result.questions} questions chosen from {result.priced} priced encodings")
    print(f"  {result.repeats} of them asked twice -> {result.pairs} pairs to hear")
    print(f"  answer sheet -> {result.paths.labels_csv}")


def _reading(value: float | None) -> str:
    """One figure as the ranking table prints it, dashed where there was nothing to read."""
    return f"{value:.3f}" if value is not None else "--"


def _calls(matched: int, decided: int) -> str:
    """How many of the calls a listener made were made the same way by whoever is being read against them."""
    return f"{matched}/{decided}"


def _axis_calls(metric: MetricAgreement, axis: PairAxis) -> str:
    """How many of the listener's calls on one question a metric makes the same way."""
    read = metric.by_axis.get(axis)
    return "--" if read is None else _calls(read.matched, read.decided)


def _ranking_row(metric: MetricAgreement) -> str:
    """One metric's whole standing as a line of the ranking table."""
    axes = "".join(f"{_axis_calls(metric, axis):>9}" for axis in PairAxis)
    confirmed = _calls(metric.overall.matched, metric.overall.decided)
    margins = f"{_reading(metric.separation):>7}{_reading(metric.headroom):>7}"
    return f"  {metric.name:<16}{_reading(metric.overall.tau):>7}{confirmed:>11}{margins}{axes}"


def print_ranking(report: RankingReport) -> None:
    """State how every metric fared against one answer sheet, with the ceiling and the confound above it."""
    ceiling, level = report.ceiling, report.level
    print(f"{report.instrument_id}: {report.answered} answered, {report.outstanding} open")
    print(
        f"  ceiling: the listener repeats {_calls(ceiling.matched, ceiling.decided)} of their own calls"
        f" over {ceiling.repeated} questions asked twice  (tau {_reading(ceiling.tau)})"
    )
    print(
        f"  level:   {_calls(level.louder, level.gapped)} of the calls with an audible gap named the louder"
        f" side  (tau {_reading(level.tau)})"
    )
    heading = "".join(f"{axis:>9}" for axis in PairAxis)
    print(f"  {'metric':<16}{'tau':>7}{'confirmed':>11}{'sep':>7}{'head':>7}{heading}")
    for metric in report.metrics:
        print(_ranking_row(metric))


def print_plans(result: DumpResult) -> None:
    """State how each strategy fared for one instrument, and where all of them landed."""
    print(f"{result.instrument_id}: {result.directory}")
    for plan in result.plans:
        timing = f"[{plan.elapsed_s:.1f}s]"
        if not plan.feasible:
            print(f"  {plan.name:>9}: infeasible ({plan.reason})  {timing}")
            continue

        rendered = "rendered" if plan.rendered else "no render"
        print(f"  {plan.name:>9}: objective {plan.objective:.4f}, {plan.used_bytes} B used, {rendered}  {timing}")
        if plan.instruments is not None:
            print_instruments(plan.instruments)


def plan_total(results: Sequence[DumpResult]) -> float:
    """The wall-clock every strategy of every instrument took together, which closes an allocating run."""
    return sum(plan.elapsed_s for result in results for plan in result.plans)
