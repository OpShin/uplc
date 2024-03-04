from ..util import NodeTransformer, NodeVisitor
from ..ast import *

"""
Optimizes code by pre-evaluating each subterm
If it throws an error, assume it is not safe to pre-evaluate and don't replace, otherwise replace by result.
"""


class TraceFinder(NodeVisitor):
    found = False

    def visit_BuiltIn(self, node: BuiltIn):
        return self.visit_ForcedBuiltIn(node)

    def visit_ForcedBuiltIn(self, node: ForcedBuiltIn):
        if node.builtin == BuiltInFun.Trace:
            self.found = True
            return
        return

    def contains_trace(self, node: AST) -> bool:
        self.visit(node)
        return self.found


_CONSTANT_FINAL_RESULTS = (
    Constant,
    Variable,
    Error,
)


class PreEvaluationOptimizer(NodeTransformer):
    def __init__(self, skip_traces: bool = True):
        from ..tools import eval

        self.eval = eval
        self.skip_traces = skip_traces

    def visit_Program(self, node: Program) -> Program:
        return Program(version=node.version, term=self.visit(node.term))

    def generic_visit(self, node: AST) -> AST:
        if self.skip_traces and TraceFinder().contains_trace(node):
            return super().generic_visit(node)
        try:
            nc = self.eval(node).result
        except Exception as e:
            nc = node
        else:
            if isinstance(nc, Exception) or not any(
                isinstance(nc, cfr) for cfr in _CONSTANT_FINAL_RESULTS
            ):
                nc = node
        return super().generic_visit(nc)
