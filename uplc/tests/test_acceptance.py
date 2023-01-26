from pathlib import Path
import os

import unittest

from .. import *
from ..transformer import unique_variables

acceptance_test_path = Path("examples/acceptance_tests")


class AcceptanceTests(unittest.TestCase):
    def test_acceptance_tests(self):
        for dirpath, dirs, files in os.walk(acceptance_test_path, topdown=True):
            if dirs:
                # not a leaf directory
                continue
            with self.subTest("Acceptance test", path=dirpath):
                input_file = next(f for f in files if f.endswith("uplc"))
                with open(os.path.join(dirpath, input_file), "r") as fp:
                    input = fp.read()
                output_file = next(f for f in files if f.endswith("uplc.expected"))
                with open(os.path.join(dirpath, output_file), "r") as fp:
                    output = fp.read().strip()
                try:
                    input_parsed = parse(input)
                except Exception:
                    self.assertEqual(
                        "parse error", output, "Parsing program failed unexpectedly"
                    )
                    continue
                try:
                    res = eval(input_parsed)
                except Exception:
                    self.assertEqual(
                        "evaluation failure",
                        output,
                        "Evaluating program failed unexpectedly",
                    )
                    continue
                if output in ("parse error", "evaluation failure"):
                    # self.fail(
                    #     "Program parsed and evaluated but should have thrown error"
                    # )
                    continue
                output_parsed = parse(output).term
                res_parsed_unique = unique_variables.UniqueVariableTransformer().visit(
                    res
                )
                output_parsed_unique = (
                    unique_variables.UniqueVariableTransformer().visit(output_parsed)
                )
                res_dumps = dumps(res_parsed_unique, dialect=UPLCDialect.Aiken)
                output_dumps = dumps(output_parsed_unique, dialect=UPLCDialect.Aiken)
                self.assertEqual(
                    output_dumps, res_dumps, "Program evaluated to wrong output"
                )
