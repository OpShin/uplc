import rply.errors

from .lexer import strip_comments, Lexer
from .parser import Parser
from .machine import Machine
from .ast import AST, UPLCDialect


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
        ) from None
    except rply.errors.ParsingError as e:
        source = s.splitlines()[e.source_pos.lineno - 1] if s else ""
        raise SyntaxError(
            "Parsing failed, invalid production",
            (filename, e.source_pos.lineno, e.source_pos.colno, source),
        ) from None
    return program


def eval(u: AST):
    m = Machine(u)
    return m.eval()


def dumps(u: AST, dialect=UPLCDialect.Aiken):
    return u.dumps(dialect)
