# Minimal top-level package to mimic torchdrug structure used by the repo.
from . import core, data, layers, utils
from .core import Registry as R

__all__ = ["core", "data", "layers", "utils", "R"]
