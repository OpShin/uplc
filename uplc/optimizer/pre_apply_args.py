from ..util import NodeTransformer, bind_variables
from ..ast import *

"""
Pre-applies Apply(Lambda(x, body), arg) to body[x := arg], iff this leads to a smaller AST.
Note: this step expects unique variable names, so it should be run after UniqueVariableTransformer.
"""


class Substitute(NodeTransformer):
    def __init__(self, var_name: str, value: AST):
        self.var_name = var_name
        self.value = value

    def visit_Variable(self, node: Variable) -> AST:
        if node.name == self.var_name:
            return self.value
        return node


# Terms that are not further evaluated and can be folded into the apply without side effects
# If we ignore this, the optimizer can change the order of side effects (traces, errors, ...)
# or even their appearance (e.g. if we apply an error to a lambda in which the errors is not in the body)
_CONSTANT_FINAL_RESULTS = (
    Constant,
    Variable,
    ForcedBuiltIn,
    BuiltIn,
    Lambda,
    Delay,
)


class ApplyLambdaTransformer(NodeTransformer):
    def __init__(self, max_increase=1):
        """
        :param max_increase: the maximum allowed increase in AST size when applying the lambda in percentage (e.g. 1.1 = 10% increase allowed)
        """
        self.max_increase = max_increase

    def visit_Apply(self, node: Apply) -> AST:

        if isinstance(node.f, Lambda) and isinstance(node.x, _CONSTANT_FINAL_RESULTS):
            from ..tools import flatten

            # try to apply
            body = node.f.term
            var_name = node.f.var_name
            arg = node.x
            # substitute var_name in body with arg
            new_body = Substitute(var_name, arg).visit(body)
            # flatten to check size
            new_body_size = flatten(bind_variables(new_body))
            node_size = flatten(bind_variables(node))

            # check if smaller
            if len(new_body_size) <= len(node_size) * self.max_increase:
                return self.visit(new_body)
        return super().generic_visit(node)
