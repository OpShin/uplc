import datetime
import unittest

import hypothesis
from hypothesis import strategies as hst
import pyaiken
from parameterized import parameterized

from uplc import *
from uplc.flat_decoder import unzigzag
from uplc.flat_encoder import zigzag
from uplc.optimizer import pre_evaluation, pre_apply_args
from uplc.tools import unflatten
from uplc.transformer import unique_variables, debrujin_variables, undebrujin_variables
from uplc.ast import *
from uplc import lexer
from uplc.transformer.unique_variables import UniqueVariableTransformer
from uplc.util import UnboundVariableVisitor

pos_int = hst.integers(min_value=0, max_value=2**64 - 1)


uplc_data_integer = hst.builds(PlutusInteger, hst.integers())
uplc_data_bytestring = hst.builds(PlutusByteString, hst.binary())


def rec_data_strategies(uplc_data):
    uplc_data_list = hst.builds(
        lambda x: PlutusList(frozenlist(x)), hst.lists(uplc_data)
    )
    uplc_data_constr = hst.builds(
        lambda x, y: PlutusConstr(x, frozenlist(y)),
        pos_int,
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
# TODO reenable all text as soon as aiken issue for escaped strings in complex data is fixed
uplc_builtin_string = hst.builds(BuiltinString, hst.text())
# uplc_builtin_string = hst.builds(BuiltinString, hst.from_regex(r"\w*", fullmatch=True))
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


@hst.composite
def uplc_expr_all_bound(draw, uplc_expr):
    x = draw(uplc_expr)
    unbound_var_visitor = UnboundVariableVisitor()
    unbound_var_visitor.visit(x)
    unbound_vars = list(unbound_var_visitor.unbound)
    vars = draw(hst.permutations(unbound_vars))
    for v in vars:
        x = Lambda(v, x)
    return x


def rec_expr_strategies(uplc_expr):
    uplc_delay = hst.builds(Delay, uplc_expr)
    uplc_force = hst.builds(Force, uplc_expr)
    uplc_apply = hst.builds(Apply, uplc_expr, uplc_expr)
    uplc_lambda = hst.builds(Lambda, uplc_name, uplc_expr)
    uplc_lambda_bound = uplc_expr_all_bound(uplc_expr)
    return hst.one_of(
        uplc_lambda, uplc_delay, uplc_force, uplc_apply, uplc_lambda_bound
    )


uplc_expr = hst.recursive(
    hst.one_of(uplc_error, uplc_constant, uplc_builtin_fun, uplc_variable),
    rec_expr_strategies,
    max_leaves=10,
)


uplc_version = hst.builds(lambda x, y, z: (x, y, z), pos_int, pos_int, pos_int)
# This strategy also produces invalid programs (due to variables not being bound)
uplc_program_any = hst.builds(Program, uplc_version, uplc_expr)


uplc_expr_valid = uplc_expr_all_bound(uplc_expr)
# This strategy only produces valid programs (all variables are bound)
uplc_program_valid = hst.builds(Program, uplc_version, uplc_expr_valid)

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

uplc_program = hst.one_of(uplc_program_any, uplc_program_valid)


class HypothesisTests(unittest.TestCase):
    @hypothesis.given(uplc_program, hst.sampled_from(UPLCDialect))
    @hypothesis.settings(max_examples=1000, deadline=None)
    @hypothesis.example(
        Program(version=(0, 0, 0), term=BuiltinByteString(value=b"")),
        UPLCDialect.LegacyAiken,
    )
    @hypothesis.example(
        Program(version=(0, 0, 0), term=BuiltIn(builtin=BuiltInFun.ConstrData)),
        UPLCDialect.LegacyAiken,
    )
    @hypothesis.example(
        Program(version=(0, 0, 0), term=BuiltinInteger(value=0)),
        UPLCDialect.LegacyAiken,
    )
    @hypothesis.example(
        Program(version=(0, 0, 0), term=BuiltinString("\\")),
        UPLCDialect.LegacyAiken,
    )
    @hypothesis.example(
        Program(version=(0, 0, 0), term=BuiltinString('\\"')),
        UPLCDialect.LegacyAiken,
    )
    @hypothesis.example(
        Program(
            version=(0, 0, 0),
            term=BuiltinList([BuiltinString("\\"), BuiltinString("\\")]),
        ),
        UPLCDialect.LegacyAiken,
    )
    def test_dumps_parse_roundtrip(self, p, dialect):
        self.assertEqual(parse(dumps(p, dialect)), p)

    @hypothesis.given(uplc_program)
    @hypothesis.settings(max_examples=1000, deadline=datetime.timedelta(seconds=10))
    @hypothesis.example(parse('(program 0.0.0 (con -- string "--" --"))\n string ""))'))
    @hypothesis.example(parse('(program 0.0.0 (con string "--" --"))\n))'))
    @hypothesis.example(parse("(program 0.0.0 [(lam a (delay a)) (lam c c)])"))
    @hypothesis.example(
        parse("(program 0.0.0 [(lam a (lam b (error))) (lam _ (error))])")
    )
    @hypothesis.example(
        parse("(program 0.0.0 [(force (builtin mkCons)) (lam _ (error))])")
    )
    @hypothesis.example(
        parse("(program 0.0.0 (lam _ [(builtin mkPairData) (lam ' _)]))")
    )
    @hypothesis.example(parse("(program 0.0.0 (lam _ _))"))
    @hypothesis.example(parse("(program 0.0.0 [(lam x0 (lam _ x0)) (con integer 0)])"))
    @hypothesis.example(parse("(program 0.0.0 [(lam _ (delay _)) (con integer 0)])"))
    @hypothesis.example(parse("(program 0.0.0 (lam _ '))"))
    @hypothesis.example(parse("(program 0.0.0 (delay _))"))
    @hypothesis.example(parse('(program 0.0.0 (con string "---"))'))
    def test_rewrite_no_semantic_change(self, p):
        code = dumps(p)
        try:
            rewrite_p = unique_variables.UniqueVariableTransformer().visit(parse(code))
        except unique_variables.FreeVariableError:
            return
        try:
            res = eval(p)
            res = unique_variables.UniqueVariableTransformer().visit(res.result)
            res = res.dumps()
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        except Exception as e:
            res = e.__class__
        try:
            rewrite_res = eval(rewrite_p)
            rewrite_res = unique_variables.UniqueVariableTransformer().visit(
                rewrite_res.result
            )
            rewrite_res = rewrite_res.dumps()
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
    @hypothesis.settings(max_examples=1000, deadline=datetime.timedelta(seconds=1))
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
            version=(0, 0, 0),
            term=Apply(
                f=Apply(
                    f=BuiltIn(builtin=BuiltInFun.EqualsString),
                    x=BuiltIn(builtin=BuiltInFun.AddInteger),
                ),
                x=Delay(term=Error()),
            ),
        )
    )
    @hypothesis.example(
        Program(
            version=(0, 0, 0),
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
                if isinstance(orig_res, Exception):
                    break
                if isinstance(orig_res, BoundStateLambda) or isinstance(
                    orig_res, ForcedBuiltIn
                ):
                    p = BuiltinUnit()
                    params.append(p)
                    orig_res = Apply(orig_res, p)
                if isinstance(orig_res, BoundStateDelay):
                    orig_res = Force(orig_res)
                orig_res = eval(orig_res).result
            if not isinstance(orig_res, Exception):
                orig_res = unique_variables.UniqueVariableTransformer().visit(orig_res)
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        try:
            rewrite_res = rewrite_p
            for _ in range(100):
                if isinstance(rewrite_res, Exception):
                    break
                if isinstance(rewrite_res, BoundStateLambda) or isinstance(
                    rewrite_res, ForcedBuiltIn
                ):
                    p = params.pop(0)
                    rewrite_res = Apply(rewrite_res, p)
                if isinstance(rewrite_res, BoundStateDelay):
                    rewrite_res = Force(rewrite_res)
                rewrite_res = eval(rewrite_res).result
            if not isinstance(rewrite_res, Exception):
                rewrite_res = unique_variables.UniqueVariableTransformer().visit(
                    rewrite_res
                )
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        if not isinstance(rewrite_res, Exception):
            if isinstance(orig_res, Exception):
                self.assertIsInstance(
                    orig_res,
                    RuntimeError,
                    "Original code resulted in something different than a runtime error (exceeding budget) and rewritten result is ok",
                )
            self.assertEqual(
                orig_res,
                rewrite_res,
                f"Two programs evaluate to different results after optimization in {code}",
            )
        else:
            self.assertIsInstance(
                orig_res,
                Exception,
                "Rewrite result was exception but orig result is not an exception",
            )

    @hypothesis.given(uplc_program_valid, hst.floats(min_value=1, max_value=10))
    @hypothesis.settings(max_examples=1000, deadline=datetime.timedelta(seconds=1))
    @hypothesis.example(Program(version=(0, 0, 0), term=BuiltinString(value="𐀀")), 10.0)
    @hypothesis.example(
        Program(
            version=(0, 0, 0),
            term=Apply(
                Lambda(
                    var_name="x",
                    term=Apply(
                        f=Apply(
                            f=BuiltIn(builtin=BuiltInFun.EqualsString),
                            x=Variable(name="x"),
                        ),
                        x=Variable(name="x"),
                    ),
                ),
                x=BuiltinString(value="longstring" * 100),
            ),
        ),
        1.0,
    )
    @hypothesis.example(
        Program(
            version=(0, 0, 0),
            term=Apply(
                Lambda(
                    var_name="x",
                    term=Apply(
                        f=Apply(
                            f=BuiltIn(builtin=BuiltInFun.EqualsString),
                            x=Variable(name="x"),
                        ),
                        x=Variable(name="x"),
                    ),
                ),
                x=BuiltinString(value="short"),
            ),
        ),
        100.0,
    )
    def test_apply_lambda_no_semantic_change_and_size_increase(self, p, max_increase):
        code = dumps(p)
        orig_p = parse(code).term
        rewrite_p = (
            pre_apply_args.ApplyLambdaTransformer(max_increase)
            .visit(UniqueVariableTransformer().visit(p))
            .term
        )
        orig_p_size = len(flatten(orig_p))
        rewrite_p_size = len(flatten(rewrite_p))
        self.assertLessEqual(
            rewrite_p_size,
            orig_p_size * max_increase,
            f"Size increased too much from {orig_p_size} to {rewrite_p_size} in {code}",
        )
        params = []
        try:
            orig_res = orig_p
            for _ in range(100):
                if isinstance(orig_res, Exception):
                    break
                if isinstance(orig_res, BoundStateLambda) or isinstance(
                    orig_res, ForcedBuiltIn
                ):
                    p = BuiltinUnit()
                    params.append(p)
                    orig_res = Apply(orig_res, p)
                if isinstance(orig_res, BoundStateDelay):
                    orig_res = Force(orig_res)
                orig_res = eval(orig_res).result
            if not isinstance(orig_res, Exception):
                orig_res = unique_variables.UniqueVariableTransformer().visit(orig_res)
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        try:
            rewrite_res = rewrite_p
            for _ in range(100):
                if isinstance(rewrite_res, Exception):
                    break
                if isinstance(rewrite_res, BoundStateLambda) or isinstance(
                    rewrite_res, ForcedBuiltIn
                ):
                    p = params.pop(0)
                    rewrite_res = Apply(rewrite_res, p)
                if isinstance(rewrite_res, BoundStateDelay):
                    rewrite_res = Force(rewrite_res)
                rewrite_res = eval(rewrite_res).result
            if not isinstance(rewrite_res, Exception):
                rewrite_res = unique_variables.UniqueVariableTransformer().visit(
                    rewrite_res
                )
        except unique_variables.FreeVariableError:
            self.fail(f"Free variable error occurred after evaluation in {code}")
        if not isinstance(rewrite_res, Exception):
            if isinstance(orig_res, Exception):
                self.assertIsInstance(
                    orig_res,
                    RuntimeError,
                    "Original code resulted in something different than a runtime error (exceeding budget) and rewritten result is ok",
                )
            self.assertEqual(
                orig_res,
                rewrite_res,
                f"Two programs evaluate to different results after optimization in {code}",
            )
        else:
            self.assertIsInstance(
                orig_res,
                Exception,
                "Rewrite result was exception but orig result is not an exception",
            )

    @hypothesis.given(uplc_program_valid)
    @hypothesis.settings(max_examples=1000, deadline=datetime.timedelta(seconds=10))
    @hypothesis.example(Program(version=(0, 0, 0), term=PlutusConstr(0, [])))
    @hypothesis.example(
        Program(
            version=(0, 0, 0),
            term=PlutusByteString(
                b"asdjahsdhjddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
            ),
        )
    )
    @hypothesis.example(
        Program(
            version=(0, 0, 0),
            term=PlutusInteger(2**64 + 2),
        )
    )
    @hypothesis.example(
        Program(
            version=(0, 0, 0),
            term=BuiltinString("\x00"),
        )
    )
    @hypothesis.example(
        Program((0, 0, 0), BuiltinPair(BuiltinUnit(), BuiltinString("\x00")))
    )
    @hypothesis.example(Program(version=(0, 0, 0), term=BuiltinString(value="\\")))
    @hypothesis.example(Program((0, 0, 0), BuiltinList([BuiltinString("\x00")])))
    @hypothesis.example(
        Program(version=(0, 0, 0), term=PlutusInteger(value=18446744073709551618))
    )
    def test_flat_encode_pyaiken_hypothesis(self, p):
        self.flat_encode_pyaiken_base(p)

    @parameterized.expand((v.name, v) for v in BuiltInFun)
    def test_flat_encode_pyaiken_builtins(self, _, b: BuiltInFun):
        self.flat_encode_pyaiken_base(Program(version=(0, 0, 0), term=BuiltIn(b)))

    def flat_encode_pyaiken_base(self, p):
        flattened = flatten(p)
        unflattened_aiken_string = pyaiken.uplc.unflat(flattened.hex())
        unflattened_aiken = parse(unflattened_aiken_string)

        p_unique = unique_variables.UniqueVariableTransformer().visit(p)
        unflattened_aiken_unique = unique_variables.UniqueVariableTransformer().visit(
            unflattened_aiken
        )
        self.assertEqual(
            p_unique,
            unflattened_aiken_unique,
            "Aiken unable to unflatten encoded flat or decodes to wrong program",
        )

    @hypothesis.given(hst.integers(), hst.booleans())
    def test_zigzag(self, i, b):
        self.assertEqual(i, unzigzag(zigzag(i, b), b)), "Incorrect roundtrip"

    @hypothesis.given(uplc_program_valid)
    @hypothesis.settings(max_examples=1000, deadline=datetime.timedelta(seconds=10))
    def test_debrujin_undebrujin(self, p: Program):
        p_unique = unique_variables.UniqueVariableTransformer().visit(p)
        debrujin = debrujin_variables.DeBrujinVariableTransformer().visit(p_unique)
        undebrujin = undebrujin_variables.UnDeBrujinVariableTransformer().visit(
            debrujin
        )
        self.assertEqual(p_unique, undebrujin, "incorrect flatten roundtrip")

    @hypothesis.given(uplc_program_valid)
    @hypothesis.settings(max_examples=1000, deadline=datetime.timedelta(seconds=10))
    @hypothesis.example(
        Program(version=(0, 0, 0), term=PlutusMap(value=frozendict.frozendict({})))
    )
    @hypothesis.example(
        Program(version=(0, 0, 0), term=Lambda(var_name="_", term=Variable(name="_")))
    )
    @hypothesis.example(Program(version=(1, 0, 0), term=BuiltinUnit()))
    def test_flat_unflat_roundtrip(self, p: Program):
        p_unique = unique_variables.UniqueVariableTransformer().visit(p)
        self.assertEqual(p_unique, unflatten(flatten(p)), "incorrect flatten roundtrip")

    # TODO test invalid programs being detected with an free variable error

    @hypothesis.given(uplc_data)
    def test_cbor_plutus_data_roundtrip(self, p: PlutusData):
        encoded = plutus_cbor_dumps(p)
        decoded = data_from_cbor(encoded)
        self.assertEqual(p, decoded, "incorrect cbor roundtrip")

    @hypothesis.given(uplc_data)
    def test_json_plutus_data_roundtrip(self, p: PlutusData):
        encoded = p.to_json()
        decoded = data_from_json_dict(encoded)
        self.assertEqual(p, decoded, "incorrect json roundtrip")
