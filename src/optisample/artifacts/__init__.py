"""Inspectable-artifact dump: optimize an instrument and serialize the evidence behind its objective.

The public surface is the dump entry points plus their settings/result types; the typed documents,
JSON/text writers, unit-building and per-instrument context internals live in the sibling modules
(:mod:`.serialize`, :mod:`.units`, :mod:`.context`, :mod:`.dump`).
"""

from optisample.artifacts.context import DumpResult, DumpSettings, PlanArtifacts
from optisample.artifacts.dump import dump_instrument, dump_project

__all__ = ["DumpResult", "DumpSettings", "PlanArtifacts", "dump_instrument", "dump_project"]
