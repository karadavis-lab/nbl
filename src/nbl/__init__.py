from importlib.metadata import version

from lamindb import settings
from loguru import logger

from . import ln, pl, pp, tl

__all__ = ["ln", "pl", "pp", "sync_git_repo", "tl"]

__version__ = version("nbl")


def sync_git_repo():
    """Sync LaminDB settings with the project's git repository."""
    settings.sync_git_repo = ln.settings.git_repo


logger.add("logs/lamindb.log")
