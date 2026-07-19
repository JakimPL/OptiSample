"""Bundled default configuration values, one YAML file per :class:`optisample.config.OptiConfig` group.

This is a data package (not code): it exists so ``importlib.resources.files("opticonfig")`` resolves
the YAML files both in a source checkout and in a built wheel. Edit these files to retune the
algorithm, or copy the directory and point ``optisample --config`` at your copy.
"""
