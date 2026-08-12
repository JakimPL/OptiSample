import argparse
import cProfile
import pstats
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, get_args

from optisample.artifacts import (
    DumpSettings,
    InstrumentSettings,
    PipelineSettings,
    SliceSettings,
    dump_project,
    loop_project,
    rank_listening_set,
    ranking_project,
    reduce_project,
    run_pipeline,
    write_dataset_instruments,
    write_slice,
)
from optisample.calibrate.ranking import (
    PairQuota,
    RankingGrid,
    RankingSettings,
)
from optisample.cluster.instruments import ClusterSettings, write_clustered
from optisample.cluster.stages import Stage, StageSettings, available_instruments
from optisample.config import OptiConfig, load_config
from optisample.config.cluster import ClusterConfig
from optisample.config.layers import LayersConfig
from optisample.config.optimize import BudgetConfig, SweepConfig
from optisample.config.ranking import RankingQuotaConfig
from optisample.config.reduce import DedupeKey, ReduceConfig
from optisample.config.render import Interpolation
from optisample.config.subset import IntakeConfig
from optisample.config.tracker import TrackerConfig, TrackerFormat
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
from optisample.io.dataset import SourceDataset, instrument_name
from optisample.io.note_extractor import IngestSettings
from optisample.io.source import load_source
from optisample.io.tracker.target import ExportTarget, export_target
from optisample.metrics.composite import build_composite
from optisample.model import ProjectSpec
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.progress import ProgressSink, bars_are_watchable, progress_sink
from optisample.seed import DEFAULT_SEED
from optisample.synth import DemoSettings, generate_demo

_PROFILE_TOP_FUNCTIONS: Final = 20
_MS_PER_S: Final = 1000.0
_INTERPOLATIONS: Final = get_args(Interpolation)
_FORMATS: Final = tuple(TrackerFormat)
_DEDUPE_KEYS: Final = tuple(DedupeKey)
_ARTIFACTS_OUT: Final = Path("artifacts")
_LOOPED_OUT: Final = Path("looped")
_REDUCED_OUT: Final = Path("reduced")
_SUBSET_OUT: Final = Path("subset")
_LISTENING_OUT: Final = _ARTIFACTS_OUT / "listening"
_CLUSTERED_OUT: Final = _ARTIFACTS_OUT / "clustered"
_DATASET_STRATEGY: Final = "grouped"  # a dataset stage names no allocation, so the strategy stands unread


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


def _progress_parser() -> argparse.ArgumentParser:
    """The flag every command reporting its stages reads: whether it draws their bars."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Keep stderr clear of stage progress bars (they are drawn when it is a terminal)",
    )
    return parser


def _runtime_parser() -> argparse.ArgumentParser:
    """The flags a command whose stages fan out reads: how far they do, and whether they draw their bars."""
    parser = argparse.ArgumentParser(add_help=False, parents=[_progress_parser()])
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processes sharing the stages that fan out; 0 uses every core, 1 keeps the run in-process",
    )
    return parser


def _ingest_parser() -> argparse.ArgumentParser:
    """The flags ``optimize`` and ``reduce`` share: which recordings to read and how to narrow them.

    Both commands run the same ingest and the same pre-optimization stage, so the notes file, the budget a
    plan is held to, the padding a directory of recordings holds and every reduction knob are declared
    once and read identically whichever command was asked for.
    """
    ingest = argparse.ArgumentParser(add_help=False)
    ingest.add_argument(
        "source",
        type=Path,
        help="A NoteExtractor .notes.json manifest, or a directory of recordings named by what they hold",
    )
    ingest.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Per-note WAV directory for a manifest source (default: the manifest's sibling <name>/)",
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
        help="Instrument id (default: the manifest's base name, or the directory's own name)",
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
        help="Pre-roll padding a directory of recordings holds before each onset (ms); a manifest states its own",
    )
    ingest.add_argument(
        "--post-roll-ms",
        type=float,
        default=0.0,
        help="Post-roll padding a directory of recordings holds past each release (ms); a manifest states its own",
    )
    ingest.add_argument(
        "--keep-tail",
        action="store_true",
        help="Store each recording through its post-roll padding, instead of ending it at the note's release",
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
        default=None,
        help="Bit depth every stored sample keeps (default: the config's)",
    )
    ingest.add_argument(
        "--no-loop",
        action="store_true",
        help="Store every sample over the span its material plays, leaving the loop stage off",
    )
    ingest.add_argument(
        "--dedupe-key",
        choices=_DEDUPE_KEYS,
        default=None,
        help="Identity one recording is kept per (default: the config's)",
    )
    ingest.add_argument(
        "--content-floor-db",
        type=float,
        default=None,
        help="How far under its loudest band a recording still carries content, which decides its stored rate",
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


def _allocation_parser() -> argparse.ArgumentParser:
    """The flags the allocating stage reads: which strategies to walk, the caps they allocate under, and rendering.

    ``optimize`` and ``pipeline`` reach the same stage, so both declare these once here and a cap means
    the same thing to either.
    """
    parser = argparse.ArgumentParser(add_help=False)
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
    return parser


def _describe_optimize(parser: argparse.ArgumentParser) -> None:
    """Add where allocating a budget writes, beyond the shared ingest and allocation flags."""
    parser.add_argument(
        "--out",
        type=Path,
        default=_ARTIFACTS_OUT,
        help="Artifact output directory",
    )


def _describe_admission(parser: argparse.ArgumentParser) -> None:
    """Add the length a note sounds for to be drawn on, shared by every command that takes a slice."""
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=None,
        help="Shortest a note may sound and still be sliced, in seconds (default: the configured floor)",
    )


def _describe_pipeline(parser: argparse.ArgumentParser) -> None:
    """Add what chaining the stages asks for beyond the shared ingest and allocation flags."""
    parser.add_argument(
        "--out",
        type=Path,
        default=_ARTIFACTS_OUT,
        help="Directory the run's numbered stage directories land under",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Share of the source notes to slice out first, in (0, 1]; naming none reduces the source itself",
    )
    _describe_admission(parser)


def _describe_subset(parser: argparse.ArgumentParser) -> None:
    """Add what carving a smaller dataset out of a larger one asks for."""
    parser.add_argument(
        "source",
        type=Path,
        help="The .notes.json manifest or directory of recordings to take a subset of",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Share of the notes to keep, in (0, 1], counted against the source before its short notes leave",
    )
    _describe_admission(parser)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Per-note WAV directory for a manifest source (default: the manifest's sibling <name>/)",
    )
    parser.add_argument(
        "--instrument-id",
        default=None,
        help="Instrument id naming the written dataset (default: the source's own name)",
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


def _describe_listen(parser: argparse.ArgumentParser) -> None:
    """Add what building a listening set asks for beyond the shared ingest flags."""
    parser.add_argument(
        "--out",
        type=Path,
        default=_LISTENING_OUT,
        help="Directory to write the blinded pairs, the manifest decoding them and the answer sheet into",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=None,
        help="Scale the configured quota to about this many pairs in total",
    )


def _describe_rank(parser: argparse.ArgumentParser) -> None:
    """Add what ranking the metric against a filled-in answer sheet asks for."""
    parser.add_argument(
        "listening_set",
        type=Path,
        help="Directory of one written listening set: its pairs, the manifest decoding them and the answers",
    )


def _describe_instruments(parser: argparse.ArgumentParser) -> None:
    """Add what carrying a written dataset's recordings as instruments asks for."""
    parser.add_argument(
        "source",
        type=Path,
        help="The .notes.json manifest or directory of recordings to write instruments from",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Per-note WAV directory for a manifest source (default: the manifest's sibling <name>/)",
    )


def _describe_cluster(parser: argparse.ArgumentParser) -> None:
    """Add what writing a run's recordings as clustered carrier instruments asks for."""
    parser.add_argument(
        "run_root",
        type=Path,
        help="The run root a chained pipeline wrote its stages under (the parent of 0_subset, 1_looped, ...)",
    )
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in Stage if stage.is_dataset],
        default=Stage.LOOPED.value,
        help="Which stage's recordings to cut, the looped one carrying the loops and the split a carrier needs",
    )
    parser.add_argument(
        "--instrument-id",
        default=None,
        help="Instrument to read (default: the only one the run left under its stages)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=None,
        help="Groups each velocity band's space is cut into, one stored sample standing for each",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Velocity bands the corpus is cut into, one written instrument answering each",
    )
    parser.add_argument("--rate", type=int, default=None, help="Rate every stored carrier keeps, in Hz")
    parser.add_argument("--depth", type=int, default=None, help="Bit depth every stored carrier keeps")
    parser.add_argument(
        "--format",
        choices=[fmt.value for fmt in TrackerFormat],
        default=None,
        help="Tracker format the instruments are written as (default: the configured one)",
    )
    parser.add_argument("--keep-tail", action="store_true", help="Read each recording past its note's release")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Dither seed the stored waveforms draw from")
    parser.add_argument(
        "--out",
        type=Path,
        default=_CLUSTERED_OUT,
        help="Directory to write the instruments, their auditions and the manifest naming them into",
    )


def _describe_loop(parser: argparse.ArgumentParser) -> None:
    """Add what settling loops alone asks for beyond the shared ingest flags."""
    parser.add_argument(
        "--out",
        type=Path,
        default=_LOOPED_OUT,
        help="Directory to write the looped dataset, its loop decisions and its auditions into",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optisample",
        description="Tracker module sample optimizer",
    )
    configured = _config_parser()
    staged = [configured, _runtime_parser()]
    ingest = _ingest_parser()
    allocation = _allocation_parser()
    sub = parser.add_subparsers(dest="command", required=True)
    _describe_synth(
        sub.add_parser("synth", parents=staged, help="Generate a synthetic demo dataset (.notes.json + WAVs)")
    )
    _describe_optimize(
        sub.add_parser(
            "optimize",
            parents=[*staged, ingest, allocation],
            help="Optimize a .notes.json and dump inspectable artifacts",
        )
    )
    _describe_pipeline(
        sub.add_parser(
            "pipeline",
            parents=[*staged, ingest, allocation],
            help="Slice, reduce and optimize a .notes.json in a row, each stage under its own directory",
        )
    )
    _describe_loop(
        sub.add_parser(
            "loop",
            parents=[*staged, ingest],
            help="Settle the loop each recording is stored around and write what it decided",
        )
    )
    _describe_reduce(
        sub.add_parser(
            "reduce",
            parents=[*staged, ingest],
            help="Run the pre-optimization stage alone and write the reduced dataset it decided on",
        )
    )
    _describe_listen(
        sub.add_parser(
            "listen",
            parents=[*staged, ingest],
            help="Write the blinded listening set the fidelity metric is ranked against",
        )
    )
    _describe_rank(
        sub.add_parser(
            "rank",
            parents=[configured, _progress_parser()],
            help="Rank the fidelity metric and its components against a filled-in answer sheet",
        )
    )
    _describe_subset(
        sub.add_parser(
            "subset",
            parents=[configured, _progress_parser()],
            help="Write the share of a dataset that spans its pitch and velocity ranges",
        )
    )
    _describe_cluster(
        sub.add_parser(
            "cluster",
            parents=staged,
            help="Write a run's recordings as carrier instruments, one per velocity band, cut into groups",
        )
    )
    _describe_instruments(
        sub.add_parser(
            "instruments",
            parents=[configured, _progress_parser()],
            help="Carry every recording of a written dataset as a standalone .iti and .xi instrument",
        )
    )
    return parser


def _export_target(config: OptiConfig, args: argparse.Namespace) -> ExportTarget:
    """The target the module is written through, with ``--format`` overriding the configured format."""
    if args.format is None:
        return export_target(config.export.tracker)

    return export_target(TrackerConfig.model_validate({**config.export.tracker.model_dump(), "format": args.format}))


def _reduce_config(config: OptiConfig, args: argparse.Namespace) -> ReduceConfig:
    """The reduction config with the flags that vary per run applied over the loaded values.

    Each flag reaches a nested section, so the override goes through a dump-and-revalidate: the schema
    settles what a key or a content floor may be, in one place, whichever side supplied it.
    """
    data = config.reduce.model_dump()
    if args.dedupe_key is not None:
        data["dedupe"]["key"] = args.dedupe_key

    if args.content_floor_db is not None:
        data["bandwidth"]["content_floor_db"] = args.content_floor_db

    return ReduceConfig.model_validate(data)


def _layers_config(config: OptiConfig, args: argparse.Namespace) -> LayersConfig:
    """The layering config with ``--max-layers`` applied over the loaded values.

    The override goes through a dump-and-revalidate so the schema settles what a layer count may be in
    one place, whichever side supplied it. Allocating is the only stage that reads it, which is why the
    flag sits on ``optimize`` alone.
    """
    if args.max_layers is None:
        return config.optimize.layers

    return LayersConfig.model_validate({**config.optimize.layers.model_dump(), "max_layers": args.max_layers})


def _budget_config(config: OptiConfig, args: argparse.Namespace) -> BudgetConfig:
    """The solver config with ``--max-samples`` applied over the loaded values.

    The override goes through a dump-and-revalidate so the schema settles what a sample cap may be in one
    place, whichever side supplied it. Allocating is the only stage that reads it, which is why the flag
    sits on ``optimize`` alone.
    """
    if args.max_samples is None:
        return config.optimize.budget

    return BudgetConfig.model_validate({**config.optimize.budget.model_dump(), "max_samples": args.max_samples})


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
    budget: BudgetConfig,
) -> OptimizeSettings:
    """Build the optimization settings from ``config``, applying the sweep/reduce/format/seed overrides.

    ``layers`` and ``budget`` arrive from the caller because only the allocating command declares
    ``--max-layers`` and ``--max-samples``, so reducing alone states the configured split and cap and
    optimizing states the ones the flags asked for.
    """
    grid = SweepConfig.model_validate(
        {
            **config.optimize.sweep.model_dump(),
            "rates": tuple(args.rates) if args.rates else config.optimize.sweep.rates,
            "depth": args.depth if args.depth is not None else config.optimize.sweep.depth,
        }
    )
    return OptimizeSettings(
        sweep=grid,
        loop=config.loop,
        reduce=_reduce_config(config, args),
        layers=layers,
        encode=config.encode,
        loops=not args.no_loop,
        metrics=config.analysis.metrics,
        velocity=config.optimize.velocity,
        method=budget.method,
        energy_exponent=budget.energy_exponent,
        max_samples=budget.max_samples,
        resolution=budget.resolution,
        target=_export_target(config, args),
        seed=args.seed,
        workers=_workers(config, args),
        progress=_progress(args),
    )


def _instrument_settings(config: OptiConfig, args: argparse.Namespace) -> InstrumentSettings:
    """What a stage carries its recordings as standalone instruments with, off the loaded config.

    Every format is written whichever one the run exports its module as, so the target here supplies what
    each format's files are held to and the configured clock stands for the datasets that state none of
    their own.
    """
    return InstrumentSettings(
        encode=config.encode,
        target=export_target(config.export.tracker),
        release_s=config.export.envelope.release_s,
        configured_tempo_bpm=config.export.playback.tempo,
        progress=_progress(args),
    )


def _configured_quota(quota: RankingQuotaConfig) -> PairQuota:
    """The quota as the config states it, before any request for a shorter set is applied."""
    return PairQuota(
        loop=quota.loop,
        rate=quota.rate,
        depth=quota.depth,
        compress=quota.compress,
        trade=quota.trade,
    )


def _listening_scale(configured: PairQuota, pairs: int | None) -> float:
    """How far the configured listening is stretched to reach about ``pairs`` questions.

    Asking for a total is how a listener states the time they have, which is the scarce input here; one
    scale over the whole set holds the balance between the four questions and the share of them put twice.
    """
    if pairs is None or configured.total == 0:
        return 1.0

    return pairs / configured.total


def _scaled_quota(configured: PairQuota, scale: float) -> PairQuota:
    """``configured`` stretched by ``scale``, which covers the same ground in proportionally less time."""
    return PairQuota(
        loop=round(configured.loop * scale),
        rate=round(configured.rate * scale),
        depth=round(configured.depth * scale),
        compress=round(configured.compress * scale),
        trade=round(configured.trade * scale),
    )


def _ranking_settings(config: OptiConfig, args: argparse.Namespace) -> RankingSettings:
    """Build the listening set's settings from ``config``, applying the ``--pairs`` override."""
    ranking = config.analysis.ranking
    configured = _configured_quota(ranking.quota)
    scale = _listening_scale(configured, args.pairs)
    return RankingSettings(
        grid=RankingGrid(depths=ranking.depths, rate_steps=ranking.rate_steps),
        quota=_scaled_quota(configured, scale),
        byte_tolerance=ranking.byte_tolerance,
        min_duration_s=ranking.min_duration_s,
        repeats=round(ranking.repeats * scale),
        seed=ranking.seed,
    )


def _dump_settings(
    config: OptiConfig,
    args: argparse.Namespace,
) -> DumpSettings:
    """Assemble the artifact-dump settings from ``config`` and the CLI flags."""
    return DumpSettings(
        optimize=_optimize_settings(config, args, _layers_config(config, args), _budget_config(config, args)),
        render=config.export.render,
        playback=config.export.playback,
        envelope=config.export.envelope,
        render_ground_truth=not args.no_render,
        grouped=args.strategy in ("both", "grouped"),
        ungrouped=args.strategy in ("both", "ungrouped"),
    )


def _source(args: argparse.Namespace) -> SourceDataset:
    """Where the run reads its recordings: the positional source, with ``--samples-dir`` where it is given."""
    return SourceDataset(path=args.source, samples_dir=args.samples_dir)


def _project(args: argparse.Namespace) -> ProjectSpec:
    """Build the project settings from the flags; ``--interpolation`` overrides the ``ProjectSpec`` default."""
    name = instrument_name(args.source)
    if args.interpolation is None:
        return ProjectSpec(name=name)

    return ProjectSpec(name=name, interpolation=args.interpolation)


def _ingest_settings(args: argparse.Namespace) -> IngestSettings:
    """The manifest fields a source leaves to the caller, read off the shared ingest flags."""
    return IngestSettings(
        instrument_id=args.instrument_id or instrument_name(args.source),
        budget_kb=args.budget_kb,
        project=_project(args),
        pre_roll_s=args.pre_roll_ms / _MS_PER_S,
        post_roll_s=args.post_roll_ms / _MS_PER_S,
        keep_tail=args.keep_tail,
    )


def _pipeline_settings(config: OptiConfig, args: argparse.Namespace) -> PipelineSettings:
    """What each stage of a chained run is carried out with, off the flags the whole chain shares.

    The reduction is settled at the configured velocity split and sample cap, the way ``reduce`` settles
    it, so ``--max-layers`` and ``--max-samples`` reach the allocation the run ends in and the dataset
    between them stays the one any allocation reads back.
    """
    return PipelineSettings(
        ingest=_ingest_settings(args),
        reduce=_optimize_settings(config, args, config.optimize.layers, config.optimize.budget),
        dump=_dump_settings(config, args),
        instruments=_instrument_settings(config, args),
        intake=_intake(config, args),
        fraction=args.fraction,
    )


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


def _intake(config: OptiConfig, args: argparse.Namespace) -> IntakeConfig:
    """What the way in does to a source, the length flag standing in for the configured floor."""
    return IntakeConfig(
        min_duration_s=config.subset.min_duration_s if args.min_duration_s is None else args.min_duration_s,
        subsonic=config.subsonic,
    )


def _slice_settings(config: OptiConfig, args: argparse.Namespace) -> SliceSettings:
    """What taking the share of a source a run begins from is carried out with, off the loaded config."""
    return SliceSettings(
        instrument_id=args.instrument_id or instrument_name(args.source),
        fraction=args.fraction,
        intake=_intake(config, args),
        instruments=_instrument_settings(config, args),
    )


def _run_subset(config: OptiConfig, args: argparse.Namespace) -> None:
    print_subset(write_slice(_source(args), args.out, _slice_settings(config, args)))


def _cluster_config(config: OptiConfig, args: argparse.Namespace) -> ClusterConfig:
    """The clustering config with the flags that vary per run applied over the loaded values.

    Each flag reaches a nested section, so the override goes through a dump-and-revalidate: the schema
    settles what a group count, a rate or a depth may be, in one place, whichever side supplied it.
    """
    data = config.cluster.model_dump()
    for flag, section, key in (
        (args.groups, "partition", "groups"),
        (args.layers, "instrument", "layers"),
        (args.rate, "instrument", "rate"),
        (args.depth, "instrument", "depth"),
    ):
        if flag is not None:
            data[section][key] = flag

    if args.groups is not None:
        data["partition"]["max_groups"] = max(data["partition"]["max_groups"], args.groups)

    return ClusterConfig.model_validate(data)


def _clustered_instrument(args: argparse.Namespace) -> str:
    """Which instrument the run is read for: the flag where one was given, the only one there is otherwise.

    Raises:
        ValueError: when the run holds no instrument, or holds several and none was named.
    """
    if args.instrument_id is not None:
        return str(args.instrument_id)

    found = available_instruments(args.run_root)
    if len(found) != 1:
        raise ValueError(f"name the instrument to read with --instrument-id; the run holds {list(found)}")

    return found[0]


def _cluster_settings(config: OptiConfig, args: argparse.Namespace) -> ClusterSettings:
    """What writing a run's recordings as clustered carrier instruments is carried out with."""
    progress = _progress(args)
    return ClusterSettings(
        stage=Stage(args.stage),
        reading=StageSettings(
            instrument_id=_clustered_instrument(args),
            strategy=_DATASET_STRATEGY,
            dedupe=config.reduce.dedupe,
            trim=config.reduce.trim,
            keep_tail=args.keep_tail,
            progress=progress,
        ),
        cluster=_cluster_config(config, args),
        features=config.loop.features,
        encode=config.encode,
        target=_export_target(config, args),
        release_s=config.export.envelope.release_s,
        tempo_bpm=config.export.playback.tempo,
        seed=args.seed,
        workers=_workers(config, args),
        progress=progress,
    )


def _run_cluster(config: OptiConfig, args: argparse.Namespace) -> None:
    written = write_clustered(args.run_root, args.out, _cluster_settings(config, args))
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
