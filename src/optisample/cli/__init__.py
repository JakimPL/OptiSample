import argparse
from collections.abc import Callable

from optisample.cli.commands import (
    _run_cluster,
    _run_instruments,
    _run_listen,
    _run_loop,
    _run_optimize,
    _run_pipeline,
    _run_profiled,
    _run_rank,
    _run_reduce,
    _run_subset,
    _run_synth,
)
from optisample.cli.parsers import build_parser
from optisample.config import load_config


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
            _run_subset(config, args)
        case "cluster":
            _run_cluster(config, args)

        case "instruments":
            _run_instruments(config, args)
        case "rank":
            _run_rank(config, args)
        case "loop":
            _dispatch(lambda: _run_loop(config, args), args)
        case "reduce":
            _dispatch(lambda: _run_reduce(config, args), args)
        case "listen":
            _dispatch(lambda: _run_listen(config, args), args)
        case "optimize":
            _dispatch(lambda: _run_optimize(config, args), args)
        case "pipeline":
            _dispatch(lambda: _run_pipeline(config, args), args)
