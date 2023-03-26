from copy import copy

from ..util import NodeTransformer
from ..ast import *


class FreeVariableError(ValueError):
    pass


class DeBrujinVariableTransformer(NodeTransformer):
    def __init__(self):
        self.scope = []

    def get_index(self, name: str):
        for index, var in enumerate(reversed(self.scope), start=1):
            if var == name:
                return index
        raise FreeVariableError(f"Variable {name} is never assigned")

    def visit_Lambda(self, node: Lambda):
        nc = copy(node)
        self.scope.append(node.var_name)
        nc.term = self.visit(node.term)
        self.scope.pop()
        return nc

    def visit_Variable(self, node: Variable):
        nc = copy(node)
        nc.name = self.get_index(node.name)
        return nc
