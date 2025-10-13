#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata
import logging


__version__ = importlib.metadata.version(__package__ or __name__)
__author__ = "nielstron"
__author_email__ = "niels@opshin.dev"
__copyright__ = "Copyright (C) 2025 nielstron"
__license__ = "MIT"
__url__ = "https://github.com/opshin/uplc"

try:
    from .tools import parse, eval, dumps, UPLCDialect, flatten, unflatten, apply

except ImportError as e:
    logging.error(
        "Error, trying to import dependencies. Should only occur upon package installation",
        exc_info=e,
    )
