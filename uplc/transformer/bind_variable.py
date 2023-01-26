from copy import deepcopy

from .util import NodeTransformer


class BindVariableTransformer(NodeTransformer):
    def __init__(self, var_name: str, var_val):
        self.name = var_name
        self.value = var_val

    def visit_Lambda(self, node):
        if node.var_name == self.name:
            return node
        nc = deepcopy(node)
        nc.term = self.visit(node.term)
        return nc

    def visit_Variable(self, node):
        if node.name == self.name:
            return deepcopy(self.value)
        return node
