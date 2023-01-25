"""
DeBrujin Machine to evaluate UPLC AST
"""

import frozendict
import logging
from dataclasses import dataclass

from .ast import *

_LOGGER = logging.getLogger(__name__)


class Machine:
    def __init__(self, program: AST, max_steps=1000000):
        self.program = program
        self.rem_steps = max_steps

    def eval(self):
        stack = [
            Compute(
                NoFrame(),
                frozendict.frozendict(),
                self.program,
            )
        ]

        while stack:
            self.rem_steps -= 1
            if self.rem_steps < 0:
                raise RuntimeError("Maximum steps exceeded")
            step = stack.pop()
            if isinstance(step, Compute):
                stack.append(step.term.eval(step.ctx, step.env))
            elif isinstance(step, Return):
                stack.append(self.return_compute(step.context, step.value))
            elif isinstance(step, Done):
                stack.append(step.term)
                break

        return stack.pop()

    def return_compute(self, context, value):
        if isinstance(context, FrameApplyFun):
            return self.apply_evaluate(context.ctx, context.val, value)
        if isinstance(context, FrameApplyArg):
            return Compute(
                FrameApplyFun(
                    value,
                    context.ctx,
                ),
                context.env,
                context.term,
            )
        if isinstance(context, FrameForce):
            return self.force_evaluate(context.ctx, value)
        if isinstance(context, NoFrame):
            term = value
            return Done(term)

    def apply_evaluate(self, context, function, argument):
        if isinstance(function, BoundStateLambda):
            return Compute(
                context,
                function.state | {function.var_name: argument},
                function.term,
            )
        if isinstance(function, ForcedBuiltIn):
            eval_fun = BuiltInFunEvalMap[function.builtin]
            needs_forces = BuiltInFunForceMap[function.builtin]
            if function.applied_forces == needs_forces:
                if eval_fun.__code__.co_argcount == len(function.bound_arguments) + 1:
                    res = eval_fun(*function.bound_arguments, argument)
                else:
                    res = ForcedBuiltIn(
                        function.builtin,
                        function.applied_forces,
                        function.bound_arguments + [argument],
                    )
                return Return(context, res)
        raise RuntimeError("Tried to apply arguments to unapplyiable object")

    def force_evaluate(self, context, value):
        if isinstance(value, BoundStateDelay):
            return Compute(context, value.state, value.term)
        if isinstance(value, ForcedBuiltIn):
            needs_forces = BuiltInFunForceMap[value.builtin]
            if value.applied_forces < needs_forces:
                res = ForcedBuiltIn(
                    value.builtin, value.applied_forces + 1, value.bound_arguments
                )
                # Theoretically we could check if the builtin requires 0 arguments, but this is not the case for any function
                return Return(context, res)
        raise RuntimeError("Forcing an unforceable object")
