import argparse
import cProfile
import pstats
import sys
from collections.abc import Callable
from typing import Final

from optisample.artifacts import (
    dump_project,
    loop_project,
    rank_listening_set,
    ranking_project,
    reduce_project,
    run_pipeline,
    write_dataset_instruments,
    write_slice,
)
from optisample.cli.settings import (
    _cluster_settings,
    _demo_settings,
    _dump_settings,
    _ingest_settings,
    _instrument_settings,
    _optimize_settings,
    _pipeline_settings,
    _progress,
    _ranking_settings,
    _slice_settings,
    _source,
)
from optisample.cluster.instruments import write_clustered
from optisample.config import OptiConfig
from optisample.console import (
    plan_total,
    print_clustered,
    print_instruments,
    print_listening,
    print_looped,
    print_plans,
    print_ranking,
    print_reduced,
    print_subset,
)
from optisample.io.source import load_source
from optisample.metrics.composite import build_composite
from optisample.synth import generate_demo

_PROFILE_TOP_FUNCTIONS: Final = 20


def _run_loop(config: OptiConfig, args: argparse.Namespace) -> None:
    manifest = load_source(_source(args), _ingest_settings(args))
    for result in loop_project(
        manifest,
        args.out,
        _optimize_settings(config, args, config.optimize.layers, config.optimize.budget),
        _instrument_settings(config, args),
    ):
        print_looped(result)


def _run_reduce(config: OptiConfig, args: argparse.Namespace) -> None:
    manifest = load_source(_source(args), _ingest_settings(args))
    for result in reduce_project(
        manifest,
        args.out,
        _optimize_settings(config, args, config.optimize.layers, config.optimize.budget),
        _instrument_settings(config, args),
    ):
        print_reduced(result)


def _run_listen(config: OptiConfig, args: argparse.Namespace) -> None:
    manifest = load_source(_source(args), _ingest_settings(args))
    for result in ranking_project(
        manifest,
        args.out,
        _optimize_settings(config, args, config.optimize.layers, config.optimize.budget),
        _ranking_settings(config, args),
    ):
        print_listening(result)


def _run_rank(config: OptiConfig, args: argparse.Namespace) -> None:
    print_ranking(rank_listening_set(args.listening_set, build_composite(config.analysis.metrics), _progress(args)))


def _run_subset(config: OptiConfig, args: argparse.Namespace) -> None:
    print_subset(write_slice(_source(args), args.out, _slice_settings(config, args)))


def _run_cluster(config: OptiConfig, args: argparse.Namespace) -> None:
    written = write_clustered(args.out, _cluster_settings(config, args))
    print_clustered(written)


def _run_instruments(config: OptiConfig, args: argparse.Namespace) -> None:
    source = _source(args)
    print(f"{source.path}")
    print_instruments(write_dataset_instruments(source, settings=_instrument_settings(config, args)))


def _run_optimize(config: OptiConfig, args: argparse.Namespace) -> None:
    manifest = load_source(_source(args), _ingest_settings(args))
    results = dump_project(manifest, args.out, _dump_settings(config, args))
    for result in results:
        print_plans(result)

    print(f"total: {plan_total(results):.1f}s")


def _run_pipeline(config: OptiConfig, args: argparse.Namespace) -> None:
    run = run_pipeline(_source(args), args.out, _pipeline_settings(config, args))
    if run.subset is not None:
        print_subset(run.subset)

    print_looped(run.looped)
    print_reduced(run.reduced)
    print_plans(run.optimized)
    print(f"total: {run.elapsed_s:.1f}s")


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


def _run_synth(config: OptiConfig, args: argparse.Namespace) -> None:
    for notes_json, samples_dir in generate_demo(args.outdir, config.synth, _demo_settings(config, args)):
        print(f"Wrote {notes_json} (samples: {samples_dir})")
