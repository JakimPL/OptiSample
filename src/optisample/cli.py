import argparse
import cProfile
import pstats
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from optisample.artifacts import DumpSettings, dump_project
from optisample.config import OptiConfig, load_config
from optisample.config.optimize import SweepConfig
from optisample.io.note_extractor import IngestSettings, load_notes
from optisample.io.tracker.target import export_target
from optisample.metrics import build_composite
from optisample.model import ProjectSpec
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.synth import generate_demo

DEFAULT_SEED: Final = 0
_PROFILE_TOP_FUNCTIONS: Final = 20
_MS_PER_S: Final = 1000.0
_NOTES_SUFFIX: Final = ".notes.json"
_INTERPOLATIONS: Final = ("none", "linear", "cubic", "sinc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optisample",
        description="Impulse Tracker sample optimizer",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config directory to load (default: bundled)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser(
        "synth",
        parents=[common],
        help="Generate a synthetic demo dataset (.notes.json + WAVs)",
    )
    synth.add_argument(
        "outdir",
        type=Path,
        help="Directory to write each preset's samples dir and .notes.json into",
    )
    synth.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Render sample rate (Hz); defaults to the config",
    )
    synth.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for reproducible output",
    )

    optimize = sub.add_parser(
        "optimize",
        parents=[common],
        help="Optimize a .notes.json and dump inspectable artifacts",
    )
    optimize.add_argument(
        "notes_json",
        type=Path,
        help="Path to a NoteExtractor .notes.json manifest",
    )
    optimize.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Per-note WAV directory (default: notes_json's sibling <name>/)",
    )
    optimize.add_argument(
        "--budget-kb",
        type=float,
        required=True,
        help="Byte budget for the instrument (KiB)",
    )
    optimize.add_argument(
        "--instrument-id",
        default=None,
        help="Instrument id (default: the .notes.json base name)",
    )
    optimize.add_argument(
        "--interpolation",
        choices=_INTERPOLATIONS,
        default=None,
        help="Playback interpolation (default: sinc)",
    )
    optimize.add_argument(
        "--pre-roll-ms",
        type=float,
        default=0.0,
        help="Pre-roll padding trimmed as lead-in (ms)",
    )
    optimize.add_argument(
        "--post-roll-ms",
        type=float,
        default=0.0,
        help="Post-roll padding recorded for provenance (ms)",
    )
    optimize.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts"),
        help="Artifact output directory",
    )
    optimize.add_argument(
        "--strategy",
        choices=("both", "grouped", "ungrouped"),
        default="both",
    )
    optimize.add_argument(
        "--no-render",
        action="store_true",
        help="Skip openmpt123 ground-truth renders",
    )
    optimize.add_argument(
        "--rate",
        type=int,
        action="append",
        dest="rates",
        help="Sample rate to sweep (repeatable)",
    )
    optimize.add_argument(
        "--depth",
        type=int,
        action="append",
        dest="depths",
        help="Bit depth to sweep (repeatable)",
    )
    optimize.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable looping (store full-length samples)",
    )
    optimize.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for reproducible encoding",
    )
    optimize.add_argument(
        "--profile",
        action="store_true",
        help="Run under cProfile and print the hottest functions to stderr",
    )
    return parser


def _optimize_settings(
    config: OptiConfig,
    args: argparse.Namespace,
) -> OptimizeSettings:
    """Build the optimization settings from ``config``, applying the sweep/seed CLI overrides."""
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
        encode=config.encode,
        composite=build_composite(config.metrics),
        velocity=config.velocity,
        method=config.optimize.method,
        target=export_target(config.tracker),
        seed=args.seed,
    )


def _dump_settings(
    config: OptiConfig,
    args: argparse.Namespace,
) -> DumpSettings:
    """Assemble the artifact-dump settings from ``config`` and the CLI flags."""
    return DumpSettings(
        optimize=_optimize_settings(config, args),
        render=config.render,
        playback=config.playback,
        render_ground_truth=not args.no_render,
        grouped=args.strategy in ("both", "grouped"),
        ungrouped=args.strategy in ("both", "ungrouped"),
    )


def _instrument_base(notes_json: Path) -> str:
    """The instrument name behind a ``.notes.json`` path (its ``.notes.json`` suffix stripped)."""
    name = notes_json.name
    if name.endswith(_NOTES_SUFFIX):
        return name[: -len(_NOTES_SUFFIX)]

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


def _run_optimize(config: OptiConfig, args: argparse.Namespace) -> None:
    settings = IngestSettings(
        instrument_id=args.instrument_id or _instrument_base(args.notes_json),
        budget_kb=args.budget_kb,
        project=_project(args),
        pre_roll_s=args.pre_roll_ms / _MS_PER_S,
        post_roll_s=args.post_roll_ms / _MS_PER_S,
    )
    manifest = load_notes(args.notes_json, _samples_dir(args), settings)
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "synth":
        outputs = generate_demo(
            args.outdir,
            config.synth,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )
        for notes_json, samples_dir in outputs:
            print(f"Wrote {notes_json} (samples: {samples_dir})")
    elif args.command == "optimize":
        if args.profile:
            _run_profiled(lambda: _run_optimize(config, args))
        else:
            _run_optimize(config, args)
