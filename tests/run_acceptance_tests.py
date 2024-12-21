import json
import sys
from pathlib import Path
import os

from uplc import parse, dumps, UPLCDialect, eval
from uplc.ast import AST
from uplc.cost_model import Budget
from uplc.transformer import unique_variables


def acceptance_test_dirs(root):
    res = []
    for dirpath, dirs, files in sorted(os.walk(root, topdown=True)):
        if dirs:
            # not a leaf directory
            continue
        res.append(dirpath)
    return res


def run_acceptance_test(dirpath, log=False):
    files = os.listdir(dirpath)
    input_file = next(f for f in files if f.endswith("uplc"))
    input_file_path = os.path.join(dirpath, input_file)
    with open(input_file_path, "r") as fp:
        input = fp.read()
    if log:
        print("----- Input -------")
        print(input)
    output_file = next(f for f in files if f.endswith("uplc.expected"))
    output_file_path = os.path.join(dirpath, output_file)
    with open(output_file_path, "r") as fp:
        output = fp.read().strip()

    if log:
        print("----- Expected output -------")
        print(output)
    try:
        input_parsed = parse(input, filename=input_file_path)
    except Exception:
        assert "parse error" == output, "Parsing program failed unexpectedly"
        return
    comp_res = eval(input_parsed)
    res = comp_res.result
    if log:
        print("----- Actual output -------")
        if isinstance(res, AST):
            print(dumps(res))
        else:
            print(res)
    if isinstance(res, Exception):
        assert output == "evaluation failure", "Machine failed but should not fail."
        return
    assert output not in (
        "parse error",
        "evaluation failure",
    ), "Program parsed and evaluated but should have thrown error"
    output_parsed = parse(output, filename=output_file_path).term
    res_parsed_unique = unique_variables.UniqueVariableTransformer().visit(res)
    output_parsed_unique = unique_variables.UniqueVariableTransformer().visit(
        output_parsed
    )
    res_dumps = dumps(res_parsed_unique, dialect=UPLCDialect.LegacyAiken)
    output_dumps = dumps(output_parsed_unique, dialect=UPLCDialect.LegacyAiken)
    assert output_dumps == res_dumps, "Program evaluated to wrong output"
    try:
        cost_file = next(f for f in files if f.endswith("cost"))
        with open(Path(dirpath).joinpath(cost_file)) as f:
            cost_content = f.read()
        if cost_content == "error":
            return
        cost = json.loads(cost_content)
        expected_spent_budget = Budget(cost["cpu"], cost["mem"])
        assert (
            expected_spent_budget == comp_res.cost
        ), "Program evaluated with wrong cost."
    except StopIteration:
        pass


def main(test_root: str):
    for path in acceptance_test_dirs(test_root):
        failed = False
        try:
            run_acceptance_test(path)
        except AssertionError:
            failed = True
        if failed:
            print(path)
            run_acceptance_test(path, log=True)


if __name__ == "__main__":
    path = sys.argv[1]
    main(path)
