from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from optisample.artifacts import DumpSettings, dump_project
from optisample.config.optimize import SweepConfig
from optisample.io.manifest import load_manifest
from optisample.optimize.orchestrate import default_optimize_settings
from optisample.synth import default_synth_config, generate_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optisample", description="Impulse Tracker sample optimizer")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="Generate a synthetic demo dataset + manifest")
    synth.add_argument("outdir", type=Path, help="Directory to write samples and manifest.yaml into")
    # transitional: phase 9 sources the default rate from the --config directory the user tunes.
    synth.add_argument(
        "--sample-rate", type=int, default=default_synth_config().sample_rate, help="Render sample rate (Hz)"
    )
    synth.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible output")

    optimize = sub.add_parser("optimize", help="Optimize a manifest and dump inspectable artifacts")
    optimize.add_argument("manifest", type=Path, help="Path to manifest.yaml")
    optimize.add_argument("--out", type=Path, default=Path("artifacts"), help="Artifact output directory")
    optimize.add_argument("--strategy", choices=("both", "grouped", "ungrouped"), default="both")
    optimize.add_argument("--no-render", action="store_true", help="Skip openmpt123 ground-truth renders")
    optimize.add_argument("--rate", type=int, action="append", dest="rates", help="Sample rate to sweep (repeatable)")
    optimize.add_argument("--depth", type=int, action="append", dest="depths", help="Bit depth to sweep (repeatable)")
    optimize.add_argument("--no-loop", action="store_true", help="Disable looping (store full-length samples)")
    optimize.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible encoding")
    return parser


def _dump_settings(args: argparse.Namespace) -> DumpSettings:
    # transitional: phase 9 replaces the bundled default with a --config directory the user tunes.
    base = default_optimize_settings()
    grid = SweepConfig.model_validate(
        {
            **base.sweep.model_dump(),
            "rates": tuple(args.rates) if args.rates else base.sweep.rates,
            "depths": tuple(args.depths) if args.depths else base.sweep.depths,
            "loops": (False,) if args.no_loop else base.sweep.loops,
        }
    )
    return DumpSettings(
        optimize=replace(base, sweep=grid, seed=args.seed),
        render_ground_truth=not args.no_render,
        grouped=args.strategy in ("both", "grouped"),
        ungrouped=args.strategy in ("both", "ungrouped"),
    )


def _run_optimize(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    results = dump_project(manifest, args.out, _dump_settings(args))
    for result in results:
        print(f"{result.instrument_id}: {result.directory}")
        for plan in result.plans:
            if not plan.feasible:
                print(f"  {plan.name:>9}: infeasible ({plan.reason})")
                continue
            rendered = "rendered" if plan.rendered else "no render"
            print(f"  {plan.name:>9}: objective {plan.objective:.4f}, {plan.used_bytes} B used, {rendered}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "synth":
        manifest_path = generate_demo(args.outdir, default_synth_config(), sample_rate=args.sample_rate, seed=args.seed)
        print(f"Wrote demo dataset and manifest to {manifest_path}")
    elif args.command == "optimize":
        _run_optimize(args)
