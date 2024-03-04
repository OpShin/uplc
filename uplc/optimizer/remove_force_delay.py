from ..util import NodeTransformer
from ..ast import *

"""
Removes occurrences of force(delay(x)) and replaces them with x
"""


class ForceDelayRemover(NodeTransformer):
    def visit_Force(self, node: Force):
        if isinstance(node.term, Delay):
            return self.visit(node.term.term)
        return super().generic_visit(node)
