from optisample.config.base import ConfigModel


class StageConfig(ConfigModel):
    """A pipeline stage's settings, whose groups live one YAML file each in a directory of its name.

    Marking a group this way is what tells :func:`~optisample.config.loader.load_config` to read
    ``<stage>/<group>.yaml`` for every field, so the layout on disk follows the model and a reader
    looking for one knob starts from the stage that acts on it.
    """
