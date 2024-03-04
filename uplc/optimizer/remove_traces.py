from ..util import NodeTransformer
from ..ast import Apply, ForcedBuiltIn, BuiltInFun, Lambda, Delay, Variable

"""
Removes traces from the AST, as traces have only side-effects and no value
"""


class TraceRemover(NodeTransformer):
    def visit_BuiltIn(self, node: Apply):
        return self.visit_ForcedBuiltIn(node)

    def visit_ForcedBuiltIn(self, node: ForcedBuiltIn):
        if node.builtin == BuiltInFun.Trace:
            return Delay(term=Lambda("y", Lambda("x", Variable("x"))))
        return node
