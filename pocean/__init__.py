#!python
# coding=utf-8

# Package level logger
import logging
logger = logging.getLogger("pocean")
logger.addHandler(logging.NullHandler())

__version__ = "1.2.1"
