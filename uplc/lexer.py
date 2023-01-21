from rply import LexerGenerator
import re


TOKENS = {
    "PROGRAM": "program",
    "LAMBDA": "lam",
    "FORCE": "force",
    "DELAY": "delay",
    "ERROR": "error",
    "CON": "con",
    "PAREN_OPEN": "\(",
    "PAREN_CLOSE": "\)",
    "BRACK_OPEN": "\[",
    "BRACK_CLOSE": "\]",
    "CARET_OPEN": "\<",
    "CARET_CLOSE": "\>",
    "TEXT": r'"[^\r\n"]*"',
    "COMMA": ",",
    "DOT": "\.",
    "NUMBER": "\d+",
    "NAME": "[\w_~][\w\d_~!#]*",
    "HEX": "#([\dabcdefABCDEF][\dabcdefABCDEF])+",
}


class Lexer:
    def __init__(self):
        self.lexer = LexerGenerator()

    def _add_tokens(self):
        for k, v in TOKENS.items():
            self.lexer.add(k, v)
        self.lexer.ignore("\s+")

    def get_lexer(self):
        self._add_tokens()
        return self.lexer.build()
