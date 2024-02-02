import ast as python_ast
import re

from rply import ParserGenerator
import rply
from . import lexer, ast
from .ast import (
    PlutusData,
    PlutusConstr,
    PlutusByteString,
    PlutusInteger,
    PlutusList,
    PlutusMap,
)


class Parser:
    def __init__(self):
        self.pg = ParserGenerator(lexer.TOKENS.keys())

        @self.pg.production(
            "program : PAREN_OPEN PROGRAM version expression PAREN_CLOSE"
        )
        def program(p):
            return ast.Program(tuple(map(int, p[2].split("."))), p[3])

        @self.pg.production("version : NUMBER DOT NUMBER DOT NUMBER")
        def version(p):
            return f"{int(p[0].value)}.{int(p[2].value)}.{int(p[4].value)}"

        @self.pg.production("name : NAME_NON_SPECIAL")
        @self.pg.production("name : I")
        @self.pg.production("name : B")
        @self.pg.production("name : LIST")
        @self.pg.production("name : MAP")
        @self.pg.production("name : CONSTR")
        @self.pg.production("name : BOOL")
        @self.pg.production("name : PROGRAM")
        @self.pg.production("name : LAMBDA")
        @self.pg.production("name : FORCE")
        @self.pg.production("name : DELAY")
        @self.pg.production("name : BUILTIN")
        @self.pg.production("name : CON")
        @self.pg.production("name : ERROR")
        def name(p):
            return p[0]

        @self.pg.production(
            "expression : PAREN_OPEN LAMBDA name expression PAREN_CLOSE"
        )
        def expression(p):
            return ast.Lambda(p[2].value, p[3])

        @self.pg.production("expression : name")
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

        @self.pg.production("expression : PAREN_OPEN BUILTIN name PAREN_CLOSE")
        def builtin(p):
            bfn = p[2].value.lower()
            correct_bfn = None
            for e in ast.BuiltInFun:
                if e.name.lower() == bfn:
                    correct_bfn = e
            if bfn == "verifysignature":
                correct_bfn = ast.BuiltInFun.VerifyEd25519Signature
            if correct_bfn is None:
                raise SyntaxError(f"Unknown builtin function {bfn}")
            return ast.BuiltIn(correct_bfn)

        @self.pg.production(
            "expression : BRACK_OPEN expression expression_list BRACK_CLOSE"
        )
        def delay(p):
            res = p[1]
            for e in p[2]:
                res = ast.Apply(res, e)
            return res

        @self.pg.production("expression_list : expression")
        def delay(p):
            return [p[0]]

        @self.pg.production("expression_list : expression expression_list")
        def delay(p):
            return [p[0]] + p[1]

        @self.pg.production("constanttype : name")
        def constanttype(p):
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

        @self.pg.production("constanttype : name CARET_OPEN constanttype CARET_CLOSE")
        def constanttype(p):
            # the Aiken dialect
            name = p[0].value
            if name == "list":
                return ast.BuiltinList([], p[2])
            raise SyntaxError(f"Unknown builtin type {name}")

        @self.pg.production("constanttype : PAREN_OPEN name constanttype PAREN_CLOSE")
        def constanttype(p):
            # the Plutus dialect
            name = p[1].value
            if name == "list":
                return ast.BuiltinList([], p[2])
            raise SyntaxError(f"Unknown builtin type {name}")

        @self.pg.production(
            "constanttype : name CARET_OPEN constanttype COMMA constanttype CARET_CLOSE"
        )
        def constanttype(p):
            # the Aiken dialect
            name = p[0].value
            if name == "pair":
                return ast.BuiltinPair(p[2], p[4])
            raise SyntaxError(f"Unknown builtin type {name}")

        @self.pg.production(
            "constanttype : PAREN_OPEN name constanttype constanttype PAREN_CLOSE"
        )
        def constanttype(p):
            # the Plutus dialect
            name = p[1].value
            if name == "pair":
                return ast.BuiltinPair(p[2], p[3])
            raise SyntaxError(f"Unknown builtin type {name}")

        @self.pg.production(
            "expression : PAREN_OPEN CON constanttype constantvalue PAREN_CLOSE"
        )
        def constant(p):
            typ = p[2]
            val = p[3]
            return wrap_builtin_type(typ, val)

        @self.pg.production("constantvalue : builtinvalue")
        def expression(p):
            return p[0]

        @self.pg.production("builtinvalue : HEX")
        def expression(p):
            return bytes.fromhex(p[0].value[1:])

        @self.pg.production("builtinvalue : NUMBER")
        def expression(p):
            return int(p[0].value)

        @self.pg.production("builtinvalue : TEXT")
        def expression(p):
            s = p[0].value
            return python_ast.literal_eval(s)

        @self.pg.production("builtinvalue : PAREN_OPEN PAREN_CLOSE")
        def expression(p):
            return None

        @self.pg.production("builtinvalue : BOOL")
        def expression(p):
            assert p[0].value in ("True", "False"), f"Invalid boolean constant {p}"
            return p[0].value == "True"

        @self.pg.production(
            "constantvaluelist : nestedconstantvalue COMMA constantvaluelist "
        )
        def expression(p):
            return [p[0]] + p[2]

        @self.pg.production(
            "constantvaluelist : nestedconstantvalue constantvaluelist "
        )
        def expression(p):
            return [p[0]] + p[1]

        @self.pg.production("constantvaluelist : BRACK_CLOSE ")
        def expression(p):
            return []

        @self.pg.production("builtinvalue : BRACK_OPEN constantvaluelist")
        def expression(p):
            # the Aiken dialect for pair and list values
            # and the Plutus dialect for list values
            return p[1]

        @self.pg.production(
            "builtinvalue : PAREN_OPEN nestedconstantvalue COMMA nestedconstantvalue PAREN_CLOSE"
        )
        def expression(p):
            # and the Plutus dialect for pairs
            return (p[1], p[3])

        @self.pg.production("constantvalue : PAREN_OPEN plutusvalue PAREN_CLOSE")
        def expression(p):
            return p[1]

        @self.pg.production("nestedconstantvalue : builtinvalue")
        def expression(p):
            return p[0]

        @self.pg.production("nestedconstantvalue : plutusvalue")
        def expression(p):
            return p[0]

        @self.pg.production("plutusvalue : B HEX")
        def expression(p):
            assert p[0].value == "B", f"Invalid plutus bytestring constant {p}"
            return PlutusByteString(bytes.fromhex(p[1].value[1:]))

        @self.pg.production("plutusvalue : I NUMBER")
        def expression(p):
            return PlutusInteger(int(p[1].value))

        @self.pg.production("plutusvalue : CONSTR NUMBER BRACK_OPEN plutusvaluelist")
        def expression(p):
            return PlutusConstr(int(p[1].value), p[3])

        @self.pg.production("plutusvalue : LIST BRACK_OPEN plutusvaluelist")
        @self.pg.production("plutusvalue : MAP BRACK_OPEN plutusvaluelist")
        def expression(p):
            if p[0].value == "List":
                return PlutusList(p[2])
            elif p[0].value == "Map":
                assert p[2] == [], f"Invalid plutus map constant"
                return PlutusMap(dict())
            raise ValueError(f"Invalid plutus constant {p[0]}")

        @self.pg.production("plutusvaluelist : plutusvalue COMMA plutusvaluelist ")
        def expression(p):
            return [p[0]] + p[2]

        @self.pg.production("plutusvaluelist : plutusvalue BRACK_CLOSE ")
        def expression(p):
            return [p[0]]

        @self.pg.production("plutusvaluelist : BRACK_CLOSE ")
        def expression(p):
            return []

        @self.pg.production(
            "plutusvaluepair : PAREN_OPEN plutusvalue COMMA plutusvalue PAREN_CLOSE"
        )
        def expression(p):
            return (p[1], p[3])

        @self.pg.production(
            "plutusvaluepairlist : plutusvaluepair COMMA plutusvaluepairlist "
        )
        def expression(p):
            return [p[0]] + p[2]

        @self.pg.production("plutusvaluepairlist : plutusvaluepair BRACK_CLOSE ")
        def expression(p):
            return [p[0]]

        @self.pg.production("plutusvalue : MAP BRACK_OPEN plutusvaluepairlist")
        def expression(p):
            assert p[0].value == "Map", f"Invalid plutus map {p[0]} constant"
            return PlutusMap(dict(p[2]))

    def get_parser(self):
        lrparser = self.pg.build()
        lrparser_imp = LRParserImproved(lrparser.lr_table, lrparser.error_handler)
        return lrparser_imp


class LRParserImproved(rply.parser.LRParser):
    def parse(self, tokenizer, state=None):
        from rply.token import Token

        lookahead = None
        lookaheadstack = []
        processing = None

        statestack = [0]
        symstack = [Token("$end", "$end")]

        current_state = 0
        try:
            while True:
                if self.lr_table.default_reductions[current_state]:
                    t = self.lr_table.default_reductions[current_state]
                    current_state = self._reduce_production(
                        t, symstack, statestack, state
                    )
                    continue

                if lookahead is None:
                    if lookaheadstack:
                        lookahead = lookaheadstack.pop()
                    else:
                        try:
                            lookahead = next(tokenizer)
                        except StopIteration:
                            lookahead = None

                    if lookahead is None:
                        lookahead = Token(
                            "$end",
                            "$end",
                            source_pos=rply.token.SourcePosition(
                                idx=None, lineno=1, colno=1
                            ),
                        )
                    processing = lookahead

                ltype = lookahead.gettokentype()
                if ltype in self.lr_table.lr_action[current_state]:
                    t = self.lr_table.lr_action[current_state][ltype]
                    if t > 0:
                        statestack.append(t)
                        current_state = t
                        symstack.append(lookahead)
                        processing = lookahead
                        lookahead = None
                        continue
                    elif t < 0:
                        current_state = self._reduce_production(
                            t, symstack, statestack, state
                        )
                        continue
                    else:
                        n = symstack[-1]
                        return n
                else:
                    # TODO: actual error handling here
                    if self.error_handler is not None:
                        if state is None:
                            self.error_handler(lookahead)
                        else:
                            self.error_handler(state, lookahead)
                        raise AssertionError("For now, error_handler must raise.")
                    else:
                        # erase trace
                        raise rply.errors.ParsingError(
                            None, lookahead.getsourcepos()
                        ) from None
        except Exception as e:
            if isinstance(e, rply.errors.ParsingError) or isinstance(
                e, rply.errors.LexingError
            ):
                raise e
            else:
                # annotate error with position in code where it occurred
                raise rply.errors.ParsingError(
                    str(e), processing.getsourcepos()
                ) from None


def wrap_builtin_type(typ: ast.Constant, val):
    """Hmmmmmmm wraps ...."""
    if isinstance(typ, ast.PlutusData):
        if isinstance(val, bytes):
            return ast.data_from_cbor(val)
        if isinstance(val, PlutusData):
            return val
        raise SyntaxError(f"Invalid plutus data constant {val}")
    if isinstance(typ, ast.BuiltinList):
        assert isinstance(val, list), f"Expected list but found {type(val)}"
        return ast.BuiltinList(
            [wrap_builtin_type(typ.sample_value, v) for v in val], typ.sample_value
        )
    if isinstance(typ, ast.BuiltinPair):
        assert isinstance(val, tuple) or isinstance(
            val, list
        ), f"Expected tuple but found {type(val)}"
        return ast.BuiltinPair(
            wrap_builtin_type(typ.l_value, val[0]),
            wrap_builtin_type(typ.r_value, val[1]),
        )
    if isinstance(typ, ast.BuiltinUnit):
        assert val is None, f"Expected () but found {type(val)}"
        return ast.BuiltinUnit()
    if isinstance(typ, ast.BuiltinByteString):
        assert isinstance(val, bytes), f"Expected bytes but found {type(val)}"
    if isinstance(typ, ast.BuiltinString):
        assert isinstance(val, str), f"Expected str but found {type(val)}"
    if isinstance(typ, ast.BuiltinInteger):
        assert isinstance(val, int), f"Expected int but found {type(val)}"
    if isinstance(typ, ast.BuiltinBool):
        assert isinstance(val, bool), f"Expected bool but found {type(val)}"
    return typ.__class__(val)
