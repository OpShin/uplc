import json
from pathlib import Path
import os

from parameterized import parameterized
import unittest

from .. import parse, dumps, UPLCDialect, eval
from ..cost_model import Budget
from ..util import NodeTransformer
from ..transformer import unique_variables
from ..optimizer import pre_evaluation, remove_traces, remove_force_delay

acceptance_test_path = Path("examples/acceptance_tests")


def acceptance_test_dirs():
    res = []
    for dirpath, dirs, files in sorted(os.walk(acceptance_test_path, topdown=True)):
        if dirs:
            # not a leaf directory
            continue
        res.append(dirpath)
    return res


rewriters = [
    # No transformation -> Checks conformance of implementation of the CEK machine
    NodeTransformer,
    # Rewriting variable names for unique names
    unique_variables.UniqueVariableTransformer,
    # Pre-evaluating subterms - here it will always evaluate the whole expression as there are no missing variables
    pre_evaluation.PreEvaluationOptimizer,
    # Trace removal
    remove_traces.TraceRemover,
    # Force Delay Removal
    remove_force_delay.ForceDelayRemover,
]


class AcceptanceTests(unittest.TestCase):
    # check that none of these transformers change semantics
    @parameterized.expand(
        (f"{path}_{rewriter.__name__}", path, rewriter)
        for path in acceptance_test_dirs()
        for rewriter in rewriters
    )
    def test_acceptance_tests(self, _, dirpath, rewriter):
        files = os.listdir(dirpath)
        input_file = next(f for f in files if f.endswith("uplc"))
        input_file_path = os.path.join(dirpath, input_file)
        with open(input_file_path, "r") as fp:
            input = fp.read()
        output_file = next(f for f in files if f.endswith("uplc.expected"))
        output_file_path = os.path.join(dirpath, output_file)
        with open(output_file_path, "r") as fp:
            output = fp.read().strip()
        try:
            input_parsed = parse(input, filename=input_file_path)
        except Exception:
            self.assertEqual(
                "parse error", output, "Parsing program failed unexpectedly"
            )
            return
        try:
            input_parsed = rewriter().visit(input_parsed)
        except unique_variables.FreeVariableError:
            # will raise an evaluation error anyways
            pass
        comp_res = eval(input_parsed)
        res = comp_res.result
        if isinstance(res, Exception):
            self.assertEqual(
                output, "evaluation failure", "Machine failed but should not fail."
            )
            return
        self.assertNotIn(
            output,
            ("parse error", "evaluation failure"),
            "Program parsed and evaluated but should have thrown error",
        )
        output_parsed = parse(output, filename=output_file_path).term
        res_parsed_unique = unique_variables.UniqueVariableTransformer().visit(res)
        output_parsed_unique = unique_variables.UniqueVariableTransformer().visit(
            output_parsed
        )
        res_dumps = dumps(res_parsed_unique, dialect=UPLCDialect.LegacyAiken)
        output_dumps = dumps(output_parsed_unique, dialect=UPLCDialect.LegacyAiken)
        self.assertEqual(output_dumps, res_dumps, "Program evaluated to wrong output")
        cost_file = next(f for f in files if f.endswith("cost"))
        with open(Path(dirpath).joinpath(cost_file)) as f:
            cost_content = f.read()
        if cost_content == "error":
            return
        cost = json.loads(cost_content)
        expected_spent_budget = Budget(cost["cpu"], cost["mem"])
        if rewriter in (
            pre_evaluation.PreEvaluationOptimizer,
            remove_force_delay.ForceDelayRemover,
        ):
            self.assertGreaterEqual(
                expected_spent_budget,
                comp_res.cost,
                "Program cost more after preeval/trace removal rewrite",
            )
        elif rewriter == remove_traces.TraceRemover:
            pass
        else:
            self.assertEqual(
                expected_spent_budget,
                comp_res.cost,
                "Program evaluated with wrong cost.",
            )
