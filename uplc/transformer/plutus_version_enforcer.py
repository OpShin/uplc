from copy import copy

from ..parser import PLUTUS_V3
from ..util import NodeTransformer, NodeVisitor
from ..ast import *


class UnsupportedTerm(ValueError):

    def __init__(self, message):
        self.message = message


class PlutusVersionEnforcer(NodeVisitor):
    def __init__(self):
        self.version = (1, 0, 0)

    def visit_Program(self, node: Program):
        self.version = node.version
        self.visit(node.term)

    def visit_Constr(self, node: Constr):
        if not self.version >= PLUTUS_V3:
            raise UnsupportedTerm(
                "Constr is only available after version 1.1.0 (PlutusV3)"
            )
        self.generic_visit(node)

    def visit_Case(self, node: Constr):
        if not self.version >= PLUTUS_V3:
            raise UnsupportedTerm(
                "Case is only available after version 1.1.0 (PlutusV3)"
            )
        self.generic_visit(node)
