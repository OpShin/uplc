import cbor2
import rply.errors

from .lexer import strip_comments, Lexer
from .parser import Parser
from .machine import Machine
from .ast import AST, UPLCDialect, Program, plutus_cbor_dumps, PlutusByteString
from .flat_encoder import FlatEncodingVisitor
from .flat_decoder import UplcDeserializer
from .transformer.debrujin_variables import DeBrujinVariableTransformer
from .transformer.undebrujin_variables import UnDeBrujinVariableTransformer


def flatten(x: Program) -> bytes:
    """Returns the properly CBOR wrapped program"""
    x_debrujin = DeBrujinVariableTransformer().visit(x)
    flattener = FlatEncodingVisitor()
    flattener.visit(x_debrujin)
    x_flattened = flattener.bit_writer.finalize()
    return cbor2.dumps(x_flattened)


def unflatten(x_cbor: bytes) -> Program:
    """Returns the program from a singly-CBOR wrapped flat encoding"""
    x = cbor2.loads(x_cbor)
    x_bin = "".join(f"{i:08b}" for i in x)
    reader = UplcDeserializer(x_bin)
    x_debrujin = reader.read_program()
    x_uplc = UnDeBrujinVariableTransformer().visit(x_debrujin)
    return x_uplc


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
