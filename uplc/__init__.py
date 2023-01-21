#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging

try:
    from .ast import *
    from .machine import *
    from .lexer import *
    from .parser import *

    def parse(s: str):
        l = Lexer().get_lexer()
        p = Parser().get_parser()
        tks = l.lex(s)
        program = p.parse(tks)
        return program

    def eval(u: AST):
        m = Machine(u)
        return m.eval()

    def dumps(u: AST):
        return u.dumps()

except ImportError as e:
    logging.error(
        "Error, trying to import dependencies. Should only occur upon package installation",
        exc_info=e,
    )

VERSION = (0, 3, 0)

__version__ = ".".join([str(i) for i in VERSION])
__author__ = "nielstron"
__author_email__ = "n.muendler@web.de"
__copyright__ = "Copyright (C) 2019 nielstron"
__license__ = "MIT"
__url__ = "https://github.com/imperatorlang/uplc"
