#!python

# Package level logger
import logging

logger = logging.getLogger("pocean")
logger.addHandler(logging.NullHandler())

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
