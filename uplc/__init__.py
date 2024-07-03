#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging


__version__ = "0.6.10"
__author__ = "nielstron"
__author_email__ = "n.muendler@web.de"
__copyright__ = "Copyright (C) 2023 nielstron"
__license__ = "MIT"
__url__ = "https://github.com/opshin/uplc"

try:
    from .tools import parse, eval, dumps, UPLCDialect, flatten, unflatten

except ImportError as e:
    logging.error(
        "Error, trying to import dependencies. Should only occur upon package installation",
        exc_info=e,
    )
