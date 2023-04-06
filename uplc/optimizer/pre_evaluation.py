from ..util import NodeTransformer
from ..ast import Program, AST
from ..machine import Machine

"""
Optimizes code by pre-evaluating each subterm
If it throws an error, assume it is not safe to pre-evaluate and don't replace, otherwise replace by result.
"""


class PreEvaluationOptimizer(NodeTransformer):
    def visit_Program(self, node: Program) -> Program:
        return Program(version=node.version, term=self.visit(node.term))

    def generic_visit(self, node: AST) -> AST:
        try:
            nc = Machine(node).eval()
        except Exception as e:
            nc = node
        return super().generic_visit(nc)
