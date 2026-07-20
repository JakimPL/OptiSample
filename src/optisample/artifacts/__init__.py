"""Inspectable-artifact dump: optimize an instrument and serialize the evidence behind its objective.

The public surface is the dump entry points plus their settings/result types; the JSON, unit-building
and per-instrument context internals live in the sibling modules (:mod:`.serialize`, :mod:`.units`,
:mod:`.context`, :mod:`.dump`).
"""

from optisample.artifacts.context import DumpSettings
from optisample.artifacts.dump import DumpResult, PlanArtifacts, dump_instrument, dump_project

__all__ = ["DumpSettings", "DumpResult", "PlanArtifacts", "dump_instrument", "dump_project"]
