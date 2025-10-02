import cbor2
import rply.errors

from .compiler_config import DEFAULT_CONFIG
from .cost_model import (
    default_budget,
    Budget,
    CekMachineCostModel,
    default_cek_machine_cost_model_plutus_v2,
    BuiltinCostModel,
    default_builtin_cost_model_plutus_v2,
)
from .lexer import strip_comments, Lexer
from .optimizer.pre_apply_args import ApplyLambdaTransformer
from .optimizer.pre_evaluation import PreEvaluationOptimizer
from .optimizer.remove_force_delay import ForceDelayRemover
from .parser import Parser
from .machine import Machine
from .ast import AST, UPLCDialect, Program, plutus_cbor_dumps, PlutusByteString, Apply
from .flat_encoder import FlatEncodingVisitor
from .flat_decoder import UplcDeserializer
from .transformer.debrujin_variables import DeBrujinVariableTransformer
from .transformer.undebrujin_variables import UnDeBrujinVariableTransformer
from .transformer.unique_variables import UniqueVariableTransformer
from .util import NoOp


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
    """
    Parses the given UPLC program and returns the AST
    """
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


def apply(code: Program, *args: AST) -> Program:
    """
    Applies the given arguments to the given code and returns the program
    """
    version = code.version
    code = code.term
    for d in args:
        code = Apply(code, d)
    code = Program(version, code)
    return code


def eval(
    u: AST,
    *args: AST,
    budget: Budget = default_budget(),
    cek_machine_cost_model: CekMachineCostModel = default_cek_machine_cost_model_plutus_v2(),
    builtin_cost_model: BuiltinCostModel = default_builtin_cost_model_plutus_v2(),
):
    """
    Evaluates the given UPLC program and returns the result
    """
    m = Machine(budget, cek_machine_cost_model, builtin_cost_model)
    if not isinstance(u, Program):
        u = Program((1, 0, 0), u)
    u = apply(u, *args)
    return m.eval(u)


def dumps(u: AST, dialect=UPLCDialect.Plutus):
    """
    Print the AST as UPLC code
    """
    return u.dumps(dialect)


def compile(
    x: Program,
    config=DEFAULT_CONFIG,
) -> Program:
    """
    Returns compiled UPLC code in... UPLC
    This is useful for applying low-level optimizations to the program
    :param x: the program to compile
    """
    # pre-processing: ensure that the input has unique variable names for later steps
    for step in [
        UniqueVariableTransformer() if config.unique_variable_names else NoOp(),
    ]:
        x = step.visit(x)
    prev_dump = None
    new_dump = x.dumps(UPLCDialect.Plutus)
    while prev_dump != new_dump:
        for step in [
            (
                PreEvaluationOptimizer(
                    skip_traces=config.constant_folding_keep_traces is None
                    or config.constant_folding_keep_traces
                )
                if config.constant_folding
                else NoOp()
            ),
            ForceDelayRemover() if config.remove_force_delay else NoOp(),
            (
                ApplyLambdaTransformer(max_increase=config.fold_apply_lambda_increase)
                if config.unique_variable_names
                and config.fold_apply_lambda_increase is not None
                else NoOp()
            ),
        ]:
            x = step.visit(x)
        prev_dump = new_dump
        new_dump = x.dumps(UPLCDialect.Plutus)
    # post-processing: ensure that the output has unique variable names (and minimal)
    for step in [
        UniqueVariableTransformer() if config.unique_variable_names else NoOp(),
    ]:
        x = step.visit(x)
    return x
