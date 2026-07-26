from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Frozen, strict base: immutable, hashable, and rejecting unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")
