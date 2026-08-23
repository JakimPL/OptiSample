import argparse
from pathlib import Path
from typing import Final, get_args

from optisample.artifacts.pipeline import PipelineStage
from optisample.cluster.stages import Stage
from optisample.config.reduce import DedupeKey
from optisample.config.render import Interpolation
from optisample.config.tracker import TrackerFormat
from optisample.seed import DEFAULT_SEED

_INTERPOLATIONS: Final = get_args(Interpolation)
_FORMATS: Final = tuple(TrackerFormat)
_DEDUPE_KEYS: Final = tuple(DedupeKey)
_ARTIFACTS_OUT: Final = Path("artifacts")
_LOOPED_OUT: Final = Path("looped")
_REDUCED_OUT: Final = Path("reduced")
_SUBSET_OUT: Final = Path("subset")
_LISTENING_OUT: Final = _ARTIFACTS_OUT / "listening"
_CLUSTERED_OUT: Final = _ARTIFACTS_OUT / "clustered"


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
        "--rate-headroom",
        type=int,
        default=None,
        help="Rungs above the rate a recording's own content asks for that each stored span is also offered at",
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
        "--discard-penalty",
        type=float,
        default=None,
        help="Distortion charged per octave of band a stored rate leaves out, which is what buys a wider one",
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


def _describe_post_loop(parser: argparse.ArgumentParser) -> None:
    """Add what a standalone instrument file stores past the loop, shared by every command writing one."""
    parser.add_argument(
        "--no-post-loop",
        action="store_true",
        help="End each written instrument at the loop it wraps on, dropping the rest of the recording",
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
    _describe_post_loop(parser)
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
    parser.add_argument(
        "--skip",
        choices=[stage.value for stage in PipelineStage],
        default=None,
        help="Stop the chain before this stage, dropping it and every stage after it",
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
    _describe_post_loop(parser)
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
