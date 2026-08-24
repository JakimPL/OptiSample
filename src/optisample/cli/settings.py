import argparse
import re
import sys
from typing import Final

from optisample.artifacts import DumpSettings, InstrumentSettings, PipelineSettings, PipelineStage, SliceSettings
from optisample.calibrate.ranking import PairQuota, RankingGrid, RankingSettings
from optisample.cluster.instruments import ClusterSettings
from optisample.cluster.stages import ReadingSettings, RecordingSource, Stage, available_instruments
from optisample.config import OptiConfig
from optisample.config.cluster import ClusterConfig
from optisample.config.dynamic_axis import DynamicAxis, DynamicAxisConfig
from optisample.config.export import InstrumentsConfig
from optisample.config.layers import LayersConfig
from optisample.config.optimize import BudgetConfig, SweepConfig
from optisample.config.ranking import RankingQuotaConfig
from optisample.config.reduce import ReduceConfig
from optisample.config.subset import IntakeConfig
from optisample.config.tracker import TrackerConfig
from optisample.io.dataset import SourceDataset, instrument_name
from optisample.io.note_extractor import IngestSettings
from optisample.io.tracker.target import ExportTarget, export_target
from optisample.model import ProjectSpec
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.progress import ProgressSink, bars_are_watchable, progress_sink
from optisample.synth import DemoSettings

_MS_PER_S: Final = 1000.0
_DATASET_STRATEGY: Final = "grouped"  # a dataset stage names no allocation, so the strategy stands unread
_CONTROLLER_AXIS: Final = re.compile(r"^cc(\d+)$")  # how --dynamics spells a controller, as a sample name does


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

    if args.discard_penalty is not None:
        data["bandwidth"]["discard_penalty"] = args.discard_penalty

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
            "depths": tuple(args.depths) if args.depths else config.optimize.sweep.depths,
            "rate_headroom": (
                args.rate_headroom if args.rate_headroom is not None else config.optimize.sweep.rate_headroom
            ),
            "carriers": (False,) if args.no_carrier else config.optimize.sweep.carriers,
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
        playback=config.export.playback,
        envelope=config.export.envelope,
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
        post_loop=_post_loop(config, args),
        render_ground_truth=not args.no_render,
        grouped=args.strategy in ("both", "grouped"),
        ungrouped=args.strategy in ("both", "ungrouped"),
    )


def _post_loop(config: OptiConfig, args: argparse.Namespace) -> bool:
    """Whether the standalone instrument files keep what a recording makes past the loop it wraps on.

    The config states what a run writes by default and ``--no-post-loop`` asks for the region alone, which
    is the shorter file a tracker plays identically.
    """
    return config.export.instruments.post_loop and not args.no_post_loop


def _instruments_config(config: OptiConfig, args: argparse.Namespace) -> InstrumentsConfig:
    """How this run writes its standalone instruments, with ``--no-post-loop`` applied over the config."""
    return InstrumentsConfig.model_validate(
        {**config.export.instruments.model_dump(), "post_loop": _post_loop(config, args)}
    )


def _source(args: argparse.Namespace) -> SourceDataset:
    """Where the run reads its recordings: the positional source, with ``--samples-dir`` where it is given."""
    return SourceDataset(path=args.source, samples_dir=args.samples_dir)


def _project(args: argparse.Namespace) -> ProjectSpec:
    """The project one ingest states, which is the name every manifest it writes is filed under."""
    return ProjectSpec(name=instrument_name(args.source))


def _dynamic_axis(config: OptiConfig, args: argparse.Namespace) -> DynamicAxisConfig:
    """The axis this run reads its dynamics off: ``--dynamics`` where it was given, the configured one otherwise.

    The flag is per invocation because the axis is per instrument and a run carries one instrument, so a
    corpus of mixed libraries is driven by naming the axis beside each instrument's budget.

    Raises:
        ValueError: when the flag spells neither ``velocity`` nor a controller, which would otherwise
            leave the run reading an axis nobody asked for.
    """
    if args.dynamics is None:
        return config.dynamic_axis

    if args.dynamics == DynamicAxis.VELOCITY:
        return DynamicAxisConfig(axis=DynamicAxis.VELOCITY, controller=config.dynamic_axis.controller)

    named = _CONTROLLER_AXIS.match(args.dynamics)
    if named is None:
        raise ValueError(f"--dynamics {args.dynamics!r} names neither 'velocity' nor a controller such as 'cc1'")

    return DynamicAxisConfig(axis=DynamicAxis.CONTROLLER, controller=int(named.group(1)))


def _ingest_settings(config: OptiConfig, args: argparse.Namespace) -> IngestSettings:
    """The manifest fields a source leaves to the caller, read off the shared ingest flags."""
    return IngestSettings(
        instrument_id=args.instrument_id or instrument_name(args.source),
        budget_kb=args.budget_kb,
        project=_project(args),
        pre_roll_s=args.pre_roll_ms / _MS_PER_S,
        post_roll_s=args.post_roll_ms / _MS_PER_S,
        keep_tail=args.keep_tail,
        dynamic_axis=_dynamic_axis(config, args),
    )


def _pipeline_settings(config: OptiConfig, args: argparse.Namespace) -> PipelineSettings:
    """What each stage of a chained run is carried out with, off the flags the whole chain shares.

    The reduction is settled at the configured velocity split and sample cap, the way ``reduce`` settles
    it, so ``--max-layers`` and ``--max-samples`` reach the allocation the run ends in and the dataset
    between them stays the one any allocation reads back.
    """
    return PipelineSettings(
        ingest=_ingest_settings(config, args),
        reduce=_optimize_settings(config, args, config.optimize.layers, config.optimize.budget),
        dump=_dump_settings(config, args),
        instruments=_instrument_settings(config, args),
        intake=_intake(config, args),
        fraction=args.fraction,
        skip=PipelineStage(args.skip) if args.skip is not None else None,
    )


def _intake(config: OptiConfig, args: argparse.Namespace) -> IntakeConfig:
    """What the way in does to a source, the length flag standing in for the configured floor."""
    return IntakeConfig(
        min_duration_s=config.subset.min_duration_s if args.min_duration_s is None else args.min_duration_s,
        subsonic=config.subsonic,
        dynamic_axis=_dynamic_axis(config, args),
    )


def _slice_settings(config: OptiConfig, args: argparse.Namespace) -> SliceSettings:
    """What taking the share of a source a run begins from is carried out with, off the loaded config."""
    return SliceSettings(
        instrument_id=args.instrument_id or instrument_name(args.source),
        fraction=args.fraction,
        intake=_intake(config, args),
        instruments=_instrument_settings(config, args),
    )


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
        source=RecordingSource(
            root=args.run_root,
            instrument_id=_clustered_instrument(args),
            stage=Stage(args.stage),
        ),
        reading=ReadingSettings(
            strategy=_DATASET_STRATEGY,
            dedupe=config.reduce.dedupe,
            trim=config.reduce.trim,
            keep_tail=args.keep_tail,
            progress=progress,
        ),
        cluster=_cluster_config(config, args),
        features=config.loop.features,
        encode=config.encode,
        instruments=_instruments_config(config, args),
        target=_export_target(config, args),
        release_s=config.export.envelope.release_s,
        tempo_bpm=config.export.playback.tempo,
        seed=args.seed,
        workers=_workers(config, args),
        progress=progress,
    )


def _demo_settings(config: OptiConfig, args: argparse.Namespace) -> DemoSettings:
    """How the demo dataset is produced, with ``--sample-rate`` overriding the configured render rate."""
    return DemoSettings(
        sample_rate=config.synth.sample_rate if args.sample_rate is None else args.sample_rate,
        seed=args.seed,
        workers=_workers(config, args),
        progress=_progress(args),
    )
