from ..util import NodeTransformer
from ..ast import Program, AST
from ..tools import eval

"""
Optimizes code by pre-evaluating each subterm
If it throws an error, assume it is not safe to pre-evaluate and don't replace, otherwise replace by result.
"""


class PreEvaluationOptimizer(NodeTransformer):
    def visit_Program(self, node: Program) -> Program:
        return Program(version=node.version, term=self.visit(node.term))

    def generic_visit(self, node: AST) -> AST:
        try:
            nc = eval(node).result
        except Exception as e:
            nc = node
        else:
            if isinstance(nc, Exception):
                nc = node
        return super().generic_visit(nc)
