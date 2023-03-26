from copy import copy

from ..util import NodeTransformer
from ..ast import *


class OutOfBoundsIndex(ValueError):
    pass


class UnDeBrujinVariableTransformer(NodeTransformer):
    """Reverts the debrujin renaming and at the same time renames variables to a unique name"""

    def __init__(self):
        self.scope = []
        self.count = 0

    def get_name(self, index: str) -> str:
        try:
            return self.scope[-int(index)]
        except IndexError:
            raise OutOfBoundsIndex(f"Variable index {index} is too far back")

    def visit_Lambda(self, node: Lambda):
        nc = copy(node)
        nc.var_name = f"v{self.count}"
        self.count += 1
        self.scope.append(nc.var_name)
        nc.term = self.visit(node.term)
        self.scope.pop()
        return nc

    def visit_Variable(self, node: Variable):
        nc = copy(node)
        nc.name = self.get_name(node.name)
        return nc
