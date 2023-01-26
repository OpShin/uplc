from copy import copy

from ..util import NodeTransformer
from ..ast import *


class FreeVariableError(ValueError):
    pass


class UniqueVariableTransformer(NodeTransformer):
    def __init__(self):
        self.scopes = []
        self.count = 0

    def get_map(self, name: str):
        for orig, remap in reversed(self.scopes):
            if orig == name:
                return remap
        raise FreeVariableError(f"Variable {name} is never assigned")

    def push_map(self, name: str):
        new_name = f"v{self.count}"
        self.count += 1
        self.scopes.append((name, new_name))
        return new_name

    def pop_map(self):
        self.scopes.pop(-1)

    def visit_BoundStateLambda(self, node: BoundStateLambda):
        new_map = {}
        for k, v in node.state.items():
            nk = self.push_map(k)
            new_map[nk] = v
        n = self.push_map(node.var_name)
        nc = copy(node)
        nc.state = frozendict.frozendict(new_map)
        nc.var_name = n
        nc.term = self.visit(node.term)
        self.pop_map()
        for _ in node.state.keys():
            self.pop_map()
        return nc

    def visit_Lambda(self, node: Lambda):
        return self.visit_BoundStateLambda(node)

    def visit_BoundStateDelay(self, node: BoundStateDelay):
        for k in node.state.keys():
            self.push_map(k)
        nc = copy(node)
        nc.term = self.visit(node.term)
        for _ in node.state.keys():
            self.pop_map()
        return nc

    def visit_Variable(self, node: Variable):
        nc = copy(node)
        nc.name = self.get_map(node.name)
        return nc
