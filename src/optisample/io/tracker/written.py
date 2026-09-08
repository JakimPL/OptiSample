from dataclasses import dataclass
from typing import Final

from trackmod import Compliance, TrackerModule
from trackmod.limits.violation import Violation
from trackmod.module.size import SizeReport

UNSTATED_WRITER: Final = None  # what a format leaving its writer unnamed states
BEYOND_STORAGE: Final = "beyond storage"  # how a report names a module no record layout holds


@dataclass(frozen=True)
class WrittenModule:
    """What a written module states about itself: what it occupies, how far it reaches, and what wrote it.

    ``size`` is where the bytes went. ``reach`` is the strictest level the module fits inside, which is
    the question a byte budget leaves open: a plan graded at one compliance level may hold only values a
    wider one allows, and the level it reaches says which players open the file. ``exceeded`` names the
    ceilings it passed to get there, so a report states why it reaches where it does. ``provenance`` is
    the writer the file's own header names.
    """

    size: SizeReport
    reach: Compliance | None
    exceeded: tuple[Violation, ...]
    provenance: str | None

    @property
    def reach_label(self) -> str:
        """The level this module reaches, as a report names it.

        A module carrying a value no record layout holds fits none of the three levels, which is stated
        in words rather than left blank.
        """
        return BEYOND_STORAGE if self.reach is None else self.reach.value


def written_module(module: TrackerModule) -> WrittenModule:
    """Read what ``module`` states about itself, once, for the report and the plan document alike.

    Both accounts of a written module answer for the same file, so they read it here rather than each
    asking the binding on its own.
    """
    provenance = module.provenance
    return WrittenModule(
        size=module.size(),
        reach=module.reach,
        exceeded=module.exceeded(),
        provenance=UNSTATED_WRITER if provenance is None else str(provenance),
    )
