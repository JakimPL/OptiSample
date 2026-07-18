from __future__ import annotations

import argparse
from pathlib import Path

from optisample.synth import SAMPLE_RATE, generate_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optisample", description="Impulse Tracker sample optimizer")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="Generate a synthetic demo dataset + manifest")
    synth.add_argument("outdir", type=Path, help="Directory to write samples and manifest.yaml into")
    synth.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="Render sample rate (Hz)")
    synth.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "synth":
        manifest_path = generate_demo(args.outdir, sample_rate=args.sample_rate, seed=args.seed)
        print(f"Wrote demo dataset and manifest to {manifest_path}")
