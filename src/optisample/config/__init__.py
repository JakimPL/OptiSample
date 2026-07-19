"""Configuration schema (pydantic) and loader.

The tunable values themselves live in the ``opticonfig`` YAML package, one file per group; this
package holds only the schema that validates them and :func:`load_config` that reads them.
"""

from optisample.config.loader import load_config
from optisample.config.root import OptiConfig

__all__ = ["OptiConfig", "load_config"]
