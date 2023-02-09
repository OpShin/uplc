from ast import iter_fields

import rply.errors

from .lexer import strip_comments, Lexer
from .parser import Parser
from .machine import Machine
from .ast import AST, UPLCDialect


def parse(s: str, filename=None):
    s = strip_comments(s)
    l = Lexer().get_lexer()
    p = Parser().get_parser()
    try:
        tks = l.lex(s)
        program = p.parse(tks)
    except rply.errors.LexingError as e:
        source = s.splitlines()[e.source_pos.lineno - 1]
        raise SyntaxError(
            f"Lexing failed, invalid token: {e.message}",
            (filename, e.source_pos.lineno, e.source_pos.colno, source),
        ) from None
    except rply.errors.ParsingError as e:
        source = s.splitlines()[e.source_pos.lineno - 1] if s else ""
        raise SyntaxError(
            f"Parsing failed, invalid production: {e.message}",
            (filename, e.source_pos.lineno, e.source_pos.colno, source),
        ) from None
    return program


def eval(u: AST):
    m = Machine(u)
    return m.eval()


def dumps(u: AST, dialect=UPLCDialect.Aiken):
    return u.dumps(dialect)


class NodeVisitor(object):
    """
    A node visitor base class that walks the abstract syntax tree and calls a
    visitor function for every node found.  This function may return a value
    which is forwarded by the `visit` method.

    This class is meant to be subclassed, with the subclass adding visitor
    methods.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `visit` method.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.

    Don't use the `NodeVisitor` if you want to apply changes to nodes during
    traversing.  For this a special visitor exists (`NodeTransformer`) that
    allows modifications.
    """

    def visit(self, node):
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            self.visit(value)


class NodeTransformer(NodeVisitor):
    """
    A :class:`NodeVisitor` subclass that walks the abstract syntax tree and
    allows modification of nodes.

    The `NodeTransformer` will walk the AST and use the return value of the
    visitor methods to replace or remove the old node.  If the return value of
    the visitor method is ``None``, the node will be removed from its location,
    otherwise it is replaced with the return value.  The return value may be the
    original node in which case no replacement takes place.

    Here is an example transformer that rewrites all occurrences of name lookups
    (``foo``) to ``data['foo']``::

       class RewriteName(NodeTransformer):

           def visit_Name(self, node):
               return Subscript(
                   value=Name(id='data', ctx=Load()),
                   slice=Constant(value=node.id),
                   ctx=node.ctx
               )

    Keep in mind that if the node you're operating on has child nodes you must
    either transform the child nodes yourself or call the :meth:`generic_visit`
    method for the node first.

    For nodes that were part of a collection of statements (that applies to all
    statement nodes), the visitor may also return a list of nodes rather than
    just a single node.

    Usually you use the transformer like this::

       node = YourTransformer().visit(node)
    """

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            new_node = self.visit(old_value)
            if new_node is None:
                delattr(node, field)
            else:
                setattr(node, field, new_node)
        return node
