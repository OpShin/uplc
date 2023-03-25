import copy

import rply.errors

from .lexer import strip_comments, Lexer
from .parser import Parser
from .machine import Machine
from .ast import AST, UPLCDialect, Program, plutus_cbor_dumps, PlutusByteString
from .flat_encoder import FlatEncodingVisitor
from .transformer.debrujin_variables import DeBrujinVariableTransformer


def flatten(x: Program):
    """Returns the properly CBOR wrapped program"""
    x_debrujin = DeBrujinVariableTransformer().visit(copy.deepcopy(x))
    x_flattener = FlatEncodingVisitor()
    x_flattener.visit(x_debrujin)
    x_flattened = x_flattener.bit_writer.finalize()
    x_flattened_cbor = plutus_cbor_dumps(PlutusByteString(x_flattened))
    return x_flattened_cbor


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
            f"Lexing failed, invalid token: {e.message}",
            (filename, e.source_pos.lineno, e.source_pos.colno, source),
        ) from None
    except rply.errors.ParsingError as e:
        source = s.splitlines()[e.source_pos.lineno - 1] if s else ""
        raise SyntaxError(
            f"Parsing failed, invalid production: {e.message}",
            (filename, e.source_pos.lineno, e.source_pos.colno, source),
        ) from None
    return program


def eval(u: AST):
    m = Machine(u)
    return m.eval()


def dumps(u: AST, dialect=UPLCDialect.Aiken):
    return u.dumps(dialect)
