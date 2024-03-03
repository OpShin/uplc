from ..util import NodeTransformer
from ..ast import Apply, ForcedBuiltIn, BuiltInFun

"""
Removes traces from the AST, as traces have only side-effects and no value
"""


class TraceRemover(NodeTransformer):
    def visit_Apply(self, node: Apply):
        if isinstance(node.f, ForcedBuiltIn) and node.f.builtin == BuiltInFun.Trace:
            return self.visit(node.x)
        return Apply(f=self.visit(node.f), x=self.visit(node.x))
