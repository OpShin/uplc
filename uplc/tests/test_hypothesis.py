import datetime
import unittest
import hypothesis
from hypothesis import strategies as hst
import frozenlist as fl

from .. import *
from ..optimizer import pre_evaluation
from ..transformer import unique_variables
from ..ast import *
from .. import lexer


def frozenlist(l):
    l = fl.FrozenList(l)
    l.freeze()
    return l


pos_int = hst.integers(min_value=0)


uplc_data_integer = hst.builds(PlutusInteger, hst.integers())
uplc_data_bytestring = hst.builds(PlutusByteString, hst.binary())


def rec_data_strategies(uplc_data):
    uplc_data_list = hst.builds(
        lambda x: PlutusList(frozenlist(x)), hst.lists(uplc_data)
    )
    uplc_data_constr = hst.builds(
        lambda x, y: PlutusConstr(x, frozenlist(y)),
        hst.integers(),
        hst.lists(uplc_data),
    )
    uplc_data_map = hst.builds(
        PlutusMap,
        hst.dictionaries(
            hst.one_of(
                uplc_data_integer, uplc_data_bytestring
            ),  # TODO technically constr is legal too, but causes hashing error
            uplc_data,
            dict_class=frozendict.frozendict,
        ),
    )
    return hst.one_of(uplc_data_map, uplc_data_list, uplc_data_constr)


uplc_data = hst.recursive(
    hst.one_of(uplc_data_bytestring, uplc_data_integer),
    rec_data_strategies,
    max_leaves=4,
)
uplc_builtin_boolean = hst.builds(BuiltinBool, hst.booleans())
uplc_builtin_integer = hst.builds(BuiltinInteger, hst.integers())
uplc_builtin_bytestring = hst.builds(BuiltinByteString, hst.binary())
uplc_builtin_string = hst.builds(
    BuiltinString, hst.from_regex(r'([^\n\r"]|\\")*', fullmatch=True)
)
uplc_builtin_unit = hst.just(BuiltinUnit())


def rec_const_strategies(uplc_constant):
    uplc_builtin_pair = hst.builds(BuiltinPair, uplc_constant, uplc_constant)
    uplc_builtin_list = hst.builds(
        lambda x, y: BuiltinList(frozenlist([x] * y), x),
        uplc_constant,
        hst.integers(min_value=0, max_value=10),
    )
    return hst.one_of(uplc_builtin_list, uplc_builtin_pair)


uplc_constant = hst.recursive(
    hst.one_of(
        uplc_builtin_unit,
        uplc_builtin_string,
        uplc_builtin_bytestring,
        uplc_builtin_integer,
        uplc_builtin_boolean,
        uplc_data,
    ),
    rec_const_strategies,
    max_leaves=4,
)
uplc_error = hst.just(Error())
uplc_name = hst.from_regex(r"[a-z_~'][\w~!'#]*", fullmatch=True)
uplc_builtin_fun = hst.builds(BuiltIn, hst.sampled_from(BuiltInFun))
uplc_variable = hst.builds(Variable, uplc_name)


def rec_expr_strategies(uplc_expr):
    uplc_delay = hst.builds(Delay, uplc_expr)
    uplc_force = hst.builds(Force, uplc_expr)
    uplc_apply = hst.builds(Apply, uplc_expr, uplc_expr)
    uplc_lambda = hst.builds(Lambda, uplc_name, uplc_expr)
    return hst.one_of(uplc_lambda, uplc_delay, uplc_force, uplc_apply)


uplc_expr = hst.recursive(
    hst.one_of(uplc_error, uplc_constant, uplc_builtin_fun, uplc_variable),
    rec_expr_strategies,
    max_leaves=10,
)


uplc_version = hst.builds(lambda x, y, z: f"{x}.{y}.{z}", pos_int, pos_int, pos_int)
uplc_program = hst.builds(Program, uplc_version, uplc_expr)


uplc_token = hst.one_of(
    *(hst.from_regex(t, fullmatch=True) for t in lexer.TOKENS.values())
)
uplc_token_concat = hst.recursive(
    uplc_token,
    lambda uplc_token_concat: hst.builds(
        lambda x, y: x.join(y),
        hst.from_regex(r"[\n\r\s]+", fullmatch=True),
        hst.lists(uplc_token_concat, min_size=5),
    ),
)


class HypothesisTests(unittest.TestCase):
    @hypothesis.given(uplc_program, hst.sampled_from(UPLCDialect))
    @hypothesis.settings(max_examples=1000)
    @hypothesis.example(
        Program(version="0.0.0", term=BuiltinByteString(value=b"")), UPLCDialect.Aiken
    )
    @hypothesis.example(
        Program(version="0.0.0", term=BuiltIn(builtin=BuiltInFun.ConstrData)),
        UPLCDialect.Aiken,
    )
    @hypothesis.example(
        Program(version="0.0.0", term=BuiltinInteger(value=0)), UPLCDialect.Aiken
    )
    def test_dumps_parse_roundtrip(self, p, dialect):
        self.assertEqual(parse(dumps(p, dialect)), p)

    @hypothesis.given(uplc_program)
    @hypothesis.settings(max_examples=1000, deadline=datetime.timedelta(seconds=10))
    @hypothesis.example(parse("(program 0.0.0 (lam _ _))"))
    @hypothesis.example(parse("(program 0.0.0 [(lam x0 (lam _ x0)) (con integer 0)])"))
    @hypothesis.example(parse("(program 0.0.0 [(lam _ (delay _)) (con integer 0)])"))
    @hypothesis.example(parse("(program 0.0.0 (lam _ '))"))
    @hypothesis.example(parse("(program 0.0.0 (delay _))"))
    def test_rewrite_no_semantic_change(self, p):
        code = dumps(p)
        try:
            rewrite_p = unique_variables.UniqueVariableTransformer().visit(p)
        except unique_variables.FreeVariableError:
            return
        try:
            res = eval(p)
            res = unique_variables.UniqueVariableTransformer().visit(res)
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        except Exception as e:
            res = e.__class__
        try:
            rewrite_res = eval(rewrite_p)
            rewrite_res = unique_variables.UniqueVariableTransformer().visit(
                rewrite_res
            )
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        except Exception as e:
            rewrite_res = e.__class__
        self.assertEqual(
            res,
            rewrite_res,
            f"Two programs evaluate to different results even though only renamed in {code}",
        )

    @hypothesis.given(hst.one_of(hst.text(), uplc_token_concat))
    @hypothesis.settings(max_examples=1000)
    def test_raises_syntaxerror(self, p):
        try:
            parse(p)
        except SyntaxError:
            pass
        except Exception as e:
            self.fail(f"Failed with non-syntaxerror {e}")

    @hypothesis.given(uplc_program)
    @hypothesis.settings(max_examples=1000, deadline=datetime.timedelta(seconds=1))
    @hypothesis.example(
        Program(
            version="0.0.0",
            term=Lambda(
                var_name="_",
                term=Apply(
                    f=BuiltIn(
                        builtin=BuiltInFun.AddInteger,
                        applied_forces=0,
                        bound_arguments=[],
                    ),
                    x=Lambda(
                        var_name="_", term=Error(), state=frozendict.frozendict({})
                    ),
                ),
                state=frozendict.frozendict({}),
            ),
        )
    )
    def test_preeval_no_semantic_change(self, p):
        code = dumps(p)
        orig_p = parse(code).term
        rewrite_p = pre_evaluation.PreEvaluationOptimizer().visit(p).term
        params = []
        try:
            orig_res = orig_p
            for _ in range(100):
                if isinstance(orig_res, BoundStateLambda) or isinstance(
                    orig_res, ForcedBuiltIn
                ):
                    p = BuiltinUnit()
                    params.append(p)
                    orig_res = Apply(orig_res, p)
                if isinstance(orig_res, BoundStateDelay):
                    orig_res = Force(orig_res)
                orig_res = eval(orig_res)
            orig_res = unique_variables.UniqueVariableTransformer().visit(orig_res)
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        except Exception as e:
            orig_res = e.__class__
        try:
            rewrite_res = rewrite_p
            for _ in range(100):
                if isinstance(rewrite_res, BoundStateLambda) or isinstance(
                    rewrite_res, ForcedBuiltIn
                ):
                    p = params.pop(0)
                    rewrite_res = Apply(rewrite_res, p)
                if isinstance(rewrite_res, BoundStateDelay):
                    rewrite_res = Force(rewrite_res)
                rewrite_res = eval(rewrite_res)
            rewrite_res = unique_variables.UniqueVariableTransformer().visit(
                rewrite_res
            )
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        except Exception as e:
            rewrite_res = e.__class__
        self.assertEqual(
            orig_res,
            rewrite_res,
            f"Two programs evaluate to different results after optimization in {code}",
        )
