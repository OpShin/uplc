import copy

from rply import ParserGenerator
from . import lexer, ast

PRECEDENCE = [
    ("left", ["PLUS", "MINUS"]),
    ("left", ["MUL"]),
]


class Parser:
    def __init__(self):
        self.pg = ParserGenerator(lexer.TOKENS.keys())

        @self.pg.production(
            "program : PAREN_OPEN PROGRAM version expression PAREN_CLOSE"
        )
        def program(p):
            return ast.Program(p[2], p[3])

        @self.pg.production("version : NUMBER DOT NUMBER DOT NUMBER")
        def program(p):
            return f"{int(p[0].value)}.{int(p[2].value)}.{int(p[4].value)}"

        @self.pg.production(
            "expression : PAREN_OPEN LAMBDA NAME expression PAREN_CLOSE"
        )
        def expression(p):
            return ast.Lambda(p[2].value, p[3])

        @self.pg.production("expression : NAME")
        def expression(p):
            return ast.Variable(p[0].value)

        @self.pg.production("expression : PAREN_OPEN FORCE expression PAREN_CLOSE")
        def force(p):
            return ast.Force(p[2])

        @self.pg.production("expression : PAREN_OPEN DELAY expression PAREN_CLOSE")
        def delay(p):
            return ast.Delay(p[2])

        @self.pg.production("expression : PAREN_OPEN ERROR PAREN_CLOSE")
        def error(p):
            return ast.Error()

        @self.pg.production("expression : PAREN_OPEN BUILTIN NAME PAREN_CLOSE")
        def builtin(p):
            bfn = p[2].value.lower()
            correct_bfn = None
            for e in ast.BuiltInFun:
                if e.name.lower() == bfn:
                    correct_bfn = e
            if correct_bfn is None:
                raise SyntaxError(f"Unknown builtin function {bfn}")
            return ast.BuiltIn(correct_bfn)

        @self.pg.production("expression : BRACK_OPEN expression expression BRACK_CLOSE")
        def delay(p):
            return ast.Apply(p[1], p[2])

        @self.pg.production("builtintype : NAME")
        def builtintype(p):
            name = p[0].value
            if name == "integer":
                return ast.BuiltinInteger(0)
            if name == "bytestring":
                return ast.BuiltinByteString(b"")
            if name == "string":
                return ast.BuiltinString("")
            if name == "bool":
                return ast.BuiltinBool(False)
            if name == "unit":
                return ast.BuiltinUnit()
            if name == "data":
                return ast.PlutusData()
            raise SyntaxError(f"Unknown builtin type {name}")

        @self.pg.production("builtintype : NAME CARET_OPEN builtintype CARET_CLOSE")
        def builtintype(p):
            name = p[0].value
            if name == "list":
                return ast.BuiltinList([], p[2])
            raise SyntaxError(f"Unknown builtin type {name}")

        @self.pg.production(
            "builtintype : NAME CARET_OPEN builtintype COMMA builtintype CARET_CLOSE"
        )
        def builtintype(p):
            name = p[0].value
            if name == "pair":
                return ast.BuiltinPair(p[2], p[4])
            raise SyntaxError(f"Unknown builtin type {name}")

        @self.pg.production(
            "expression : PAREN_OPEN CON builtintype builtinvalue PAREN_CLOSE"
        )
        def constant(p):
            typ = p[2]
            val = p[3]
            return wrap_builtin_type(typ, val)

        @self.pg.production("builtinvalue : HEX")
        def expression(p):
            return bytes.fromhex(p[0].value[1:])

        @self.pg.production("builtinvalue : NUMBER")
        def expression(p):
            return int(p[0].value)

        @self.pg.production("builtinvalue : TEXT")
        def expression(p):
            return p[0].value[1:-1]

        @self.pg.production("builtinvalue : PAREN_OPEN PAREN_CLOSE")
        def expression(p):
            return None

        @self.pg.production("builtinvalue : NAME")
        def expression(p):
            assert p[0].value in ("True", "False"), f"Invalid boolean constant {p}"
            return p[0].value == "True"

        @self.pg.production("builtinvaluelist : builtinvalue COMMA builtinvaluelist ")
        def expression(p):
            return [p[0]] + p[2]

        @self.pg.production("builtinvaluelist : builtinvalue builtinvaluelist ")
        def expression(p):
            return [p[0]] + p[1]

        @self.pg.production("builtinvaluelist : BRACK_CLOSE ")
        def expression(p):
            return []

        @self.pg.production("builtinvalue : BRACK_OPEN builtinvaluelist")
        def expression(p):
            return p[1]

        @self.pg.error
        def error_handle(token):
            raise ValueError(token)

    def get_parser(self):
        return self.pg.build()


def wrap_builtin_type(typ: ast.Constant, val):
    """Hmmmmmmm wraps ...."""
    if isinstance(typ, ast.PlutusData):
        return ast.data_from_cbor(val)
    if isinstance(typ, ast.BuiltinList):
        return ast.BuiltinList(
            [wrap_builtin_type(typ.sample_value, v) for v in val], typ.sample_value
        )
    if isinstance(typ, ast.BuiltinPair):
        return ast.BuiltinPair(
            wrap_builtin_type(typ.l_value, val[0]),
            wrap_builtin_type(typ.r_value, val[1]),
        )
    if isinstance(typ, ast.BuiltinUnit):
        return ast.BuiltinUnit()
    return typ.__class__(val)
