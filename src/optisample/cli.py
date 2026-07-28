import argparse
import cProfile
import pstats
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, get_args

from optisample.artifacts import DumpSettings, dump_project, reduce_project
from optisample.config import OptiConfig, load_config
from optisample.config.layers import LayersConfig
from optisample.config.optimize import OptimizeConfig, SweepConfig
from optisample.config.reduce import DedupeKey, ReduceConfig
from optisample.config.render import Interpolation
from optisample.config.tracker import TrackerConfig, TrackerFormat
from optisample.io.note_extractor import NOTES_SUFFIX, IngestSettings, load_notes
from optisample.io.subset import write_subset
from optisample.io.tracker.target import ExportTarget, export_target
from optisample.model import ProjectSpec
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.trim import RecordingScreen
from optisample.progress import ProgressSink, bars_are_watchable, progress_sink
from optisample.seed import DEFAULT_SEED
from optisample.synth import DemoSettings, generate_demo

_PROFILE_TOP_FUNCTIONS: Final = 20
_MS_PER_S: Final = 1000.0
_INTERPOLATIONS: Final = get_args(Interpolation)
_FORMATS: Final = tuple(TrackerFormat)
_DEDUPE_KEYS: Final = tuple(DedupeKey)
_ARTIFACTS_OUT: Final = Path("artifacts")
_REDUCED_OUT: Final = Path("reduced")
_SUBSET_OUT: Final = Path("subset")


def _config_parser() -> argparse.ArgumentParser:
    """The flag every subcommand reads: which directory a run loads its configured values from."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config directory to load (default: bundled)",
    )
    return parser


def _runtime_parser() -> argparse.ArgumentParser:
    """The flags a command with stages reads: how far they fan out, and whether they draw their bars."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processes sharing the stages that fan out; 0 uses every core, 1 keeps the run in-process",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Keep stderr clear of stage progress bars (they are drawn when it is a terminal)",
    )
    return parser


def _ingest_parser() -> argparse.ArgumentParser:
    """The flags ``optimize`` and ``reduce`` share: which recordings to read and how to narrow them.

    Both commands run the same ingest and the same pre-optimization stage, so the notes file, the budget
    the shortlist is priced against, the trimmer padding and every reduction knob are declared once and
    read identically whichever command was asked for.
    """
    ingest = argparse.ArgumentParser(add_help=False)
    ingest.add_argument(
        "notes_json",
        type=Path,
        help="Path to a NoteExtractor .notes.json manifest",
    )
    ingest.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Per-note WAV directory (default: notes_json's sibling <name>/)",
    )
    ingest.add_argument(
        "--budget-kb",
        type=float,
        required=True,
        help="Byte budget for the instrument (KiB)",
    )
    ingest.add_argument(
        "--instrument-id",
        default=None,
        help="Instrument id (default: the .notes.json base name)",
    )
    ingest.add_argument(
        "--format",
        choices=_FORMATS,
        default=None,
        help="Tracker format to write (default: the config's)",
    )
    ingest.add_argument(
        "--interpolation",
        choices=_INTERPOLATIONS,
        default=None,
        help="Playback interpolation (default: sinc)",
    )
    ingest.add_argument(
        "--pre-roll-ms",
        type=float,
        default=0.0,
        help="Pre-roll padding trimmed as lead-in (ms)",
    )
    ingest.add_argument(
        "--post-roll-ms",
        type=float,
        default=0.0,
        help="Post-roll padding recorded for provenance (ms)",
    )
    ingest.add_argument(
        "--rate",
        type=int,
        action="append",
        dest="rates",
        help="Sample rate to sweep (repeatable)",
    )
    ingest.add_argument(
        "--depth",
        type=int,
        action="append",
        dest="depths",
        help="Bit depth to sweep (repeatable)",
    )
    ingest.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable looping (store full-length samples)",
    )
    ingest.add_argument(
        "--dedupe-key",
        choices=_DEDUPE_KEYS,
        default=None,
        help="Identity one recording is kept per (default: the config's)",
    )
    ingest.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Stored encodings shortlisted per sample; at or above the grid size keeps every one",
    )
    ingest.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for reproducible encoding",
    )
    ingest.add_argument(
        "--profile",
        action="store_true",
        help="Run under cProfile and print the hottest functions to stderr",
    )
    return ingest


def _describe_synth(parser: argparse.ArgumentParser) -> None:
    """Add what generating a demo dataset asks for beyond the shared flags."""
    parser.add_argument(
        "outdir",
        type=Path,
        help="Directory to write each preset's samples dir and .notes.json into",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Render sample rate (Hz); defaults to the config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for reproducible output",
    )


def _describe_optimize(parser: argparse.ArgumentParser) -> None:
    """Add what allocating a budget asks for beyond the shared ingest flags."""
    parser.add_argument(
        "--out",
        type=Path,
        default=_ARTIFACTS_OUT,
        help="Artifact output directory",
    )
    parser.add_argument(
        "--strategy",
        choices=("both", "grouped", "ungrouped"),
        default="both",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Velocity bands a key may store, one written instrument each (default: the config's)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Samples a grouped plan may store, met by wider zones; 0 keeps what the format numbers",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip openmpt123 ground-truth renders",
    )


def _describe_subset(parser: argparse.ArgumentParser) -> None:
    """Add what carving a smaller dataset out of a larger one asks for."""
    parser.add_argument(
        "notes_json",
        type=Path,
        help="Path to the NoteExtractor .notes.json manifest to take a subset of",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Share of the notes to keep, in (0, 1]",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Per-note WAV directory (default: notes_json's sibling <name>/)",
    )
    parser.add_argument(
        "--instrument-id",
        default=None,
        help="Instrument id naming the written dataset (default: the .notes.json base name)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_SUBSET_OUT,
        help="Directory to write the subset dataset into",
    )


def _describe_reduce(parser: argparse.ArgumentParser) -> None:
    """Add what reducing alone asks for beyond the shared ingest flags."""
    parser.add_argument(
        "--out",
        type=Path,
        default=_REDUCED_OUT,
        help="Directory to write the reduced dataset, its reduction report and its auditions into",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optisample",
        description="Tracker module sample optimizer",
    )
    configured = _config_parser()
    staged = [configured, _runtime_parser()]
    ingest = _ingest_parser()
    sub = parser.add_subparsers(dest="command", required=True)
    _describe_synth(
        sub.add_parser("synth", parents=staged, help="Generate a synthetic demo dataset (.notes.json + WAVs)")
    )
    _describe_optimize(
        sub.add_parser(
            "optimize",
            parents=[*staged, ingest],
            help="Optimize a .notes.json and dump inspectable artifacts",
        )
    )
    _describe_reduce(
        sub.add_parser(
            "reduce",
            parents=[*staged, ingest],
            help="Run the pre-optimization stage alone and write the reduced dataset it decided on",
        )
    )
    _describe_subset(
        sub.add_parser(
            "subset",
            parents=[configured],
            help="Write the share of a dataset that spans its pitch and velocity ranges",
        )
    )
    return parser


def _export_target(config: OptiConfig, args: argparse.Namespace) -> ExportTarget:
    """The target the module is written through, with ``--format`` overriding the configured format."""
    if args.format is None:
        return export_target(config.tracker)

    return export_target(TrackerConfig.model_validate({**config.tracker.model_dump(), "format": args.format}))


def _reduce_config(config: OptiConfig, args: argparse.Namespace) -> ReduceConfig:
    """The reduction config with ``--dedupe-key`` and ``--candidates`` applied over the loaded values.

    Both flags reach nested sections, so the override goes through a dump-and-revalidate: the schema
    settles what a key or a candidate count may be, in one place, whichever side supplied it.
    """
    data = config.reduce.model_dump()
    if args.dedupe_key is not None:
        data["dedupe"]["key"] = args.dedupe_key

    if args.candidates is not None:
        data["bandwidth"]["candidates"] = args.candidates

    return ReduceConfig.model_validate(data)


def _layers_config(config: OptiConfig, args: argparse.Namespace) -> LayersConfig:
    """The layering config with ``--max-layers`` applied over the loaded values.

    The override goes through a dump-and-revalidate so the schema settles what a layer count may be in
    one place, whichever side supplied it. Allocating is the only stage that reads it, which is why the
    flag sits on ``optimize`` alone.
    """
    if args.max_layers is None:
        return config.layers

    return LayersConfig.model_validate({**config.layers.model_dump(), "max_layers": args.max_layers})


def _optimize_config(config: OptiConfig, args: argparse.Namespace) -> OptimizeConfig:
    """The solver config with ``--max-samples`` applied over the loaded values.

    The override goes through a dump-and-revalidate so the schema settles what a sample cap may be in one
    place, whichever side supplied it. Allocating is the only stage that reads it, which is why the flag
    sits on ``optimize`` alone.
    """
    if args.max_samples is None:
        return config.optimize

    return OptimizeConfig.model_validate({**config.optimize.model_dump(), "max_samples": args.max_samples})


def _workers(config: OptiConfig, args: argparse.Namespace) -> int:
    """How many processes the run's fanned-out stages share: ``--workers`` over the configured count."""
    if args.workers is None:
        return config.runtime.workers

    return int(args.workers)


def _progress(args: argparse.Namespace) -> ProgressSink:
    """Where the run reports its stages: bars on stderr when a terminal is there to redraw them.

    Redirected output keeps every redraw as literal text, so a piped or logged run reports nothing and
    ``--no-progress`` asks for the same silence at a terminal.
    """
    return progress_sink(sys.stderr, enabled=not args.no_progress and bars_are_watchable(sys.stderr))


def _optimize_settings(
    config: OptiConfig,
    args: argparse.Namespace,
    layers: LayersConfig,
    optimize: OptimizeConfig,
) -> OptimizeSettings:
    """Build the optimization settings from ``config``, applying the sweep/reduce/format/seed overrides.

    ``layers`` and ``optimize`` arrive from the caller because only the allocating command declares
    ``--max-layers`` and ``--max-samples``, so reducing alone states the configured split and cap and
    optimizing states the ones the flags asked for.
    """
    grid = SweepConfig.model_validate(
        {
            **config.sweep.model_dump(),
            "rates": tuple(args.rates) if args.rates else config.sweep.rates,
            "depths": tuple(args.depths) if args.depths else config.sweep.depths,
            "loops": (False,) if args.no_loop else config.sweep.loops,
        }
    )
    return OptimizeSettings(
        sweep=grid,
        reduce=_reduce_config(config, args),
        layers=layers,
        encode=config.encode,
        metrics=config.metrics,
        velocity=config.velocity,
        method=optimize.method,
        energy_exponent=optimize.energy_exponent,
        max_samples=optimize.max_samples,
        target=_export_target(config, args),
        seed=args.seed,
        workers=_workers(config, args),
        progress=_progress(args),
    )


def _dump_settings(
    config: OptiConfig,
    args: argparse.Namespace,
) -> DumpSettings:
    """Assemble the artifact-dump settings from ``config`` and the CLI flags."""
    return DumpSettings(
        optimize=_optimize_settings(config, args, _layers_config(config, args), _optimize_config(config, args)),
        render=config.render,
        playback=config.playback,
        render_ground_truth=not args.no_render,
        grouped=args.strategy in ("both", "grouped"),
        ungrouped=args.strategy in ("both", "ungrouped"),
    )


def _instrument_base(notes_json: Path) -> str:
    """The instrument name behind a ``.notes.json`` path (its ``.notes.json`` suffix stripped)."""
    name = notes_json.name
    if name.endswith(NOTES_SUFFIX):
        return name[: -len(NOTES_SUFFIX)]

    return notes_json.stem


def _samples_dir(args: argparse.Namespace) -> Path:
    """The per-note WAV directory: the ``--samples-dir`` override, else the notes file's sibling ``<name>/``."""
    if args.samples_dir is not None:
        return Path(args.samples_dir)

    return Path(args.notes_json.parent / _instrument_base(args.notes_json))


def _project(args: argparse.Namespace) -> ProjectSpec:
    """Build the project settings from the flags; ``--interpolation`` overrides the ``ProjectSpec`` default."""
    name = _instrument_base(args.notes_json)
    if args.interpolation is None:
        return ProjectSpec(name=name)

    return ProjectSpec(name=name, interpolation=args.interpolation)


def _ingest_settings(args: argparse.Namespace) -> IngestSettings:
    """The manifest fields the notes file leaves to the caller, read off the shared ingest flags."""
    return IngestSettings(
        instrument_id=args.instrument_id or _instrument_base(args.notes_json),
        budget_kb=args.budget_kb,
        project=_project(args),
        pre_roll_s=args.pre_roll_ms / _MS_PER_S,
        post_roll_s=args.post_roll_ms / _MS_PER_S,
    )


def _print_screen(screen: RecordingScreen) -> None:
    """State what the silence screen left out, on the runs where it left anything out."""
    if screen.admitted_everything:
        return

    print(
        f"  {len(screen.silenced)} recordings carried no signal, "
        f"dropping {screen.dropped_notes} notes at {len(screen.unplayable)} pitches"
    )


def _run_reduce(config: OptiConfig, args: argparse.Namespace) -> None:
    manifest = load_notes(args.notes_json, _samples_dir(args), _ingest_settings(args))
    results = reduce_project(manifest, args.out, _optimize_settings(config, args, config.layers, config.optimize))
    for result in results:
        print(f"{result.instrument_id}: {result.paths.notes_json}  [{result.elapsed_s:.1f}s]")
        print(f"  {result.survivors} samples, {result.notes} notes -> {result.paths.samples_dir}")
        _print_screen(result.screen)
        print(f"  {result.auditions} auditions -> {result.paths.auditions_dir}")
        print(f"  reduction -> {result.paths.reduction_json}")


def _run_subset(args: argparse.Namespace) -> None:
    dataset = write_subset(
        args.notes_json,
        _samples_dir(args),
        args.out,
        instrument_id=args.instrument_id or _instrument_base(args.notes_json),
        fraction=args.fraction,
    )
    print(f"{dataset.notes_json}")
    print(f"  {dataset.kept_notes} of {dataset.source_notes} notes, {dataset.recordings} recordings")
    print(
        f"  pitches {dataset.pitches[0]}-{dataset.pitches[1]}, velocities {dataset.velocities[0]}-{dataset.velocities[1]}"
    )
    print(f"  samples -> {dataset.samples_dir}")


def _run_optimize(config: OptiConfig, args: argparse.Namespace) -> None:
    manifest = load_notes(args.notes_json, _samples_dir(args), _ingest_settings(args))
    results = dump_project(manifest, args.out, _dump_settings(config, args))
    total_s = 0.0
    for result in results:
        print(f"{result.instrument_id}: {result.directory}")
        for plan in result.plans:
            total_s += plan.elapsed_s
            timing = f"[{plan.elapsed_s:.1f}s]"
            if not plan.feasible:
                print(f"  {plan.name:>9}: infeasible ({plan.reason})  {timing}")
                continue

            rendered = "rendered" if plan.rendered else "no render"
            print(f"  {plan.name:>9}: objective {plan.objective:.4f}, {plan.used_bytes} B used, {rendered}  {timing}")

    print(f"total: {total_s:.1f}s")


def _run_profiled(
    run: Callable[[], None],
    *,
    top: int = _PROFILE_TOP_FUNCTIONS,
) -> None:
    """Run ``run`` under cProfile and print the ``top`` functions by cumulative time to stderr."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run()
    finally:
        profiler.disable()
        pstats.Stats(profiler, stream=sys.stderr).sort_stats("cumulative").print_stats(top)


def _demo_settings(config: OptiConfig, args: argparse.Namespace) -> DemoSettings:
    """How the demo dataset is produced, with ``--sample-rate`` overriding the configured render rate."""
    return DemoSettings(
        sample_rate=config.synth.sample_rate if args.sample_rate is None else args.sample_rate,
        seed=args.seed,
        workers=_workers(config, args),
        progress=_progress(args),
    )


def _run_synth(config: OptiConfig, args: argparse.Namespace) -> None:
    for notes_json, samples_dir in generate_demo(args.outdir, config.synth, _demo_settings(config, args)):
        print(f"Wrote {notes_json} (samples: {samples_dir})")


def _dispatch(run: Callable[[], None], args: argparse.Namespace) -> None:
    """Carry out a command, under cProfile when ``--profile`` asked to see where the time went."""
    if args.profile:
        _run_profiled(run)
    else:
        run()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    match args.command:
        case "synth":
            _run_synth(config, args)
        case "subset":
            _run_subset(args)
        case "reduce":
            _dispatch(lambda: _run_reduce(config, args), args)
        case "optimize":
            _dispatch(lambda: _run_optimize(config, args), args)
