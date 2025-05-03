from importlib.metadata import version

from lamindb import settings
from loguru import logger

from . import ln, pl, pp, tl

__all__ = ["ln", "pl", "pp", "tl"]

__version__ = version("nbl")


settings.sync_git_repo = ln.settings.git_repo

logger.add("logs/lamindb.log")
