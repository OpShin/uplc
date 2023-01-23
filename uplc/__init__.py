#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging

import rply.errors

try:
    from .ast import *
    from .machine import *
    from .lexer import *
    from .parser import *

    def parse(s: str, filename=None):
        s = strip_comments(s)
        l = Lexer().get_lexer()
        p = Parser().get_parser()
        try:
            tks = l.lex(s)
            program = p.parse(tks)
        except rply.errors.LexingError as e:
            source = s.splitlines()[e.source_pos.lineno - 1]
            raise SyntaxError(
                "Lexing failed, invalid token",
                (filename, e.source_pos.lineno, e.source_pos.colno, source),
            )
        except rply.errors.ParsingError as e:
            source = s.splitlines()[e.source_pos.lineno - 1]
            raise SyntaxError(
                "Parsing failed, invalid token",
                (filename, e.source_pos.lineno, e.source_pos.colno, source),
            )
        return program

    def eval(u: AST):
        m = Machine(u)
        return m.eval()

    def dumps(u: AST, dialect=UPLCDialect.Aiken):
        return u.dumps(dialect)

except ImportError as e:
    logging.error(
        "Error, trying to import dependencies. Should only occur upon package installation",
        exc_info=e,
    )

VERSION = (0, 4, 3)

__version__ = ".".join([str(i) for i in VERSION])
__author__ = "nielstron"
__author_email__ = "n.muendler@web.de"
__copyright__ = "Copyright (C) 2019 nielstron"
__license__ = "MIT"
__url__ = "https://github.com/imperatorlang/uplc"
