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

        @self.pg.production("expression : PAREN_OPEN CON NAME HEX PAREN_CLOSE")
        def expression(p):
            b = bytes.fromhex(p[3].value[1:])
            if p[2].value == "bytestring":
                return ast.BuiltinByteString(b)
            if p[2].value == "data":
                return ast.data_from_cbor(b)
            raise SyntaxError(f"Unknown constructor value combination {p}")

        @self.pg.production("expression : PAREN_OPEN CON NAME NUMBER PAREN_CLOSE")
        def expression(p):
            if p[2].value == "integer":
                return ast.BuiltinInteger(int(p[3].value))
            raise SyntaxError(f"Unknown constructor value combination {p}")

        @self.pg.production("expression : PAREN_OPEN CON NAME TEXT PAREN_CLOSE")
        def expression(p):
            if p[2].value == "string":
                return ast.BuiltinString(p[3].value[1:-1])
            raise SyntaxError(f"Unknown constructor value combination {p}")

        @self.pg.production(
            "expression : PAREN_OPEN CON NAME PAREN_OPEN PAREN_CLOSE PAREN_CLOSE"
        )
        def expression(p):
            if p[2].value == "unit":
                return ast.BuiltinUnit()
            raise SyntaxError(f"Unknown constructor value combination {p}")

        @self.pg.production("expression : PAREN_OPEN CON NAME NAME PAREN_CLOSE")
        def expression(p):
            if p[2].value == "bool":
                assert p[3].value in ("True", "False"), f"Invalid boolean constant {p}"
                return ast.BuiltinBool(p[3].value == "True")
            raise SyntaxError(f"Unknown constructor value combination {p}")

        @self.pg.error
        def error_handle(token):
            raise ValueError(token)

    def get_parser(self):
        return self.pg.build()
