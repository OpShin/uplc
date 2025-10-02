from copy import deepcopy

from ..transformer.unique_variables import UniqueVariableTransformer
from ..util import (
    NodeTransformer,
    bind_variables,
    UnboundVariableVisitor,
    NodeVisitor,
    VariableVisitor,
)
from ..ast import *

"""
Extracts terms that occur multiple times and substitutes them with a variable.
It greedily removes the terms in order of largest impact first.
"""


class Substitute(NodeTransformer):
    def __init__(self, var_name: str, sub_terms: List[AST]):
        self.var_name = var_name
        self.sub_terms = sub_terms

    def generic_visit(self, node: AST):
        if node.dumps() in self.sub_terms:
            return Variable(self.var_name)
        return super().generic_visit(node)


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


class OccurrenceCounter(NodeVisitor):
    def __init__(self, max_increase=1):
        """
        :param max_increase: the maximum allowed increase in AST size when applying the lambda in percentage (e.g. 1.1 = 10% increase allowed)
        """
        # map from AST (flattened) to count
        self.occurence_counter = defaultdict(int)
        # map from flattened AST to original AST
        self.flattened_map = defaultdict(list)

    def generic_visit(self, node: AST):
        from ..tools import flatten

        super().generic_visit(node)
        # first check whether there are any unbound variables
        unbound = UnboundVariableVisitor()
        unbound.visit(node)
        if len(unbound.unbound) > 0:
            # cannot optimize terms with unbound variables
            return
        # flatten the node and insert into map
        flat = flatten(node)
        self.flattened_map[flat].append(node)
        self.occurence_counter[flat] += 1


class Deduplicate(NodeTransformer):

    def deduplicate(self, node: AST) -> AST:
        # first count occurrences of all subterms
        occ = OccurrenceCounter()
        occ.visit(node)
        # now find the best candidates to substitute
        # we do this by calculating the impact of substituting a term
        # impact = (count - 1) * size
        # where count is the number of occurrences of the term
        candidates = []
        for term, count in occ.occurence_counter.items():
            # find the original term
            size = len(term)
            # subtracting 2 to account for the fact that we need to add an apply and a lambda
            impact = (count - 1) * size - 2
            if impact <= 0:
                continue
            candidates.append((impact, term))
        candidates.sort(reverse=True, key=lambda x: x[0])
        # now substitute the first candidate
        if not candidates:
            return node
        # create a new variable name that does not clash with existing ones
        var_name = "_dedup_var"
        variable_vistor = VariableVisitor()
        variable_vistor.visit(node)
        existing_vars = variable_vistor.vars
        i = 0
        while var_name + str(i) in existing_vars:
            i += 1
        var_name = var_name + str(i)
        # substitute the term with the variable
        term_to_sub_flat = candidates[0][1]
        # find the original term
        sub_list = occ.flattened_map[term_to_sub_flat]
        sub_list_dumped = [x.dumps() for x in sub_list]
        new_node = Substitute(var_name, sub_list_dumped).visit(node)
        subbed_term = UniqueVariableTransformer().visit(sub_list[0])
        new_node = Apply(Lambda(var_name, new_node), subbed_term)

        return new_node

    def visit_Program(self, node: Program) -> AST:
        from ..tools import flatten, parse

        # make a copy of the node by dumping and parsing again
        copied_node = parse(node.dumps())
        # perform deduplication
        subbed_node = Program(
            version=node.version, term=self.deduplicate(copied_node.term)
        )
        # check size has decreased
        size_orig = len(flatten(node))
        size_new = len(flatten(subbed_node))
        if size_new >= size_orig:
            return node
        return subbed_node
