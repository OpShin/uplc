"""
DeBrujin Machine to evaluate UPLC AST
"""

import copy

from .ast import *
from .transformer.unique_variables import UniqueVariableTransformer, FreeVariableError
from .cost_model import CekMachineCostModel, BuiltinCostModel, CekOp, Budget

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ComputationResult:
    result: Union[AST, Exception]
    logs: List[str]
    cost: Budget


def budget_cost_of_op_on_model(
    model: Union[BuiltinCostModel, CekMachineCostModel],
    op: Union[BuiltInFun, CekOp],
    *args: int,
):
    if op not in model.cpu or op not in model.memory:
        return Budget(0, 0)
    return Budget(
        cpu=model.cpu[op].cost(*args),
        memory=model.memory[op].cost(*args),
    )


AST_TO_CEK_OP_MAP = {
    Constant: CekOp.Const,
    Program: CekOp.Startup,
    Variable: CekOp.Var,
    BoundStateLambda: CekOp.Lam,
    ForcedBuiltIn: CekOp.Builtin,
    BoundStateDelay: CekOp.Delay,
    Force: CekOp.Force,
    Apply: CekOp.Apply,
}


class Machine:
    def __init__(
        self,
        budget: Budget,
        cek_machine_cost_model: CekMachineCostModel,
        builtin_cost_model: BuiltinCostModel,
        slippage: int = 5,
    ):
        self.budget = budget
        self.unbudgeted_steps = defaultdict(int)
        self.cek_machine_cost_model = cek_machine_cost_model
        self.builtin_cost_model = builtin_cost_model
        self.slippage = slippage

    # Cost methods

    def spend_budget(self, budget: Budget):
        self.remaining_budget -= budget
        if self.remaining_budget.exhausted():
            raise RuntimeError("Exhausted budget")

    def step_and_maybe_spend(self, term: AST):
        step = [op for ast, op in AST_TO_CEK_OP_MAP.items() if isinstance(term, ast)][0]
        self.unbudgeted_steps[step] += 1
        self.unbudgeted_steps[None] += 1
        if self.unbudgeted_steps[None] >= self.slippage:
            self.spend_unbudgeted_steps()

    def spend_unbudgeted_steps(self):
        for cek_op in CekOp:
            self.spend_budget(
                self.unbudgeted_steps[cek_op]
                * budget_cost_of_op_on_model(self.cek_machine_cost_model, cek_op, 0)
            )
        self.unbudgeted_steps = defaultdict(int)

    # Compute methods

    def eval(self, program: Program):
        try:
            program = UniqueVariableTransformer().visit(program)
        except FreeVariableError as e:
            return ComputationResult(
                e,
                [],
                Budget(0, 0),
            )
        self.remaining_budget = copy.copy(self.budget)
        self.logs = []
        stack = [
            Compute(
                NoFrame(),
                frozendict.frozendict(),
                program.term,
            )
        ]
        self.spend_budget(
            budget_cost_of_op_on_model(self.cek_machine_cost_model, CekOp.Startup, 0)
        )

        try:
            while stack:
                step = stack.pop()
                if isinstance(step, Compute):
                    stack.append(self.compute(step.term, step.ctx, step.env))
                elif isinstance(step, Return):
                    stack.append(self.return_compute(step.context, step.value))
                elif isinstance(step, Done):
                    stack.append(step.term)
                    break
            self.spend_unbudgeted_steps()
            res = stack.pop()
        except Exception as e:
            res = e

        return ComputationResult(
            res,
            self.logs,
            self.budget - self.remaining_budget,
        )

    def compute(self, term: AST, context: Context, state: frozendict.frozendict):
        if isinstance(term, Error):
            raise RuntimeError(f"Execution called Error")
        self.step_and_maybe_spend(term)
        if isinstance(term, Constant):
            return Return(context, term)
        elif isinstance(term, BoundStateLambda):
            return Return(
                context,
                BoundStateLambda(term.var_name, term.term, term.state | state),
            )
        elif isinstance(term, BoundStateDelay):
            return Return(context, BoundStateDelay(term.term, term.state | state))
        elif isinstance(term, Force):
            return Compute(
                FrameForce(
                    context,
                ),
                state,
                term.term,
            )
        elif isinstance(term, ForcedBuiltIn):
            return Return(context, term)
        elif isinstance(term, Apply):
            return Compute(
                FrameApplyArg(
                    state,
                    term.x,
                    context,
                ),
                state,
                term.f,
            )
        elif isinstance(term, Variable):
            try:
                return Return(context, state[term.name])
            except KeyError as e:
                _LOGGER.error(
                    f"Access to uninitialized variable {term.name} in {term.dumps()}"
                )
                raise e
        raise NotImplementedError(f"Invalid term to compute: {term}")

    def return_compute(self, context, value):
        if isinstance(context, FrameApplyFun):
            return self.apply_evaluate(context.ctx, context.val, value)
        elif isinstance(context, FrameApplyArg):
            return Compute(
                FrameApplyFun(
                    value,
                    context.ctx,
                ),
                context.env,
                context.term,
            )
        elif isinstance(context, FrameForce):
            return self.force_evaluate(context.ctx, value)
        elif isinstance(context, NoFrame):
            term = value
            return Done(term)
        raise NotImplementedError(f"Invalid context to return compute: {context}")

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
                    arguments = [*function.bound_arguments, argument]
                    cost = budget_cost_of_op_on_model(
                        self.builtin_cost_model,
                        function.builtin,
                        *(arg.ex_mem() for arg in arguments),
                    )
                    self.spend_budget(cost)
                    if function.builtin == BuiltInFun.Trace:
                        # Hack to add this side effect to the machine
                        self.logs.append(arguments[0].value)
                    res = eval_fun(*arguments)
                else:
                    res = ForcedBuiltIn(
                        function.builtin,
                        function.applied_forces,
                        function.bound_arguments + [argument],
                    )
                return Return(context, res)
        raise RuntimeError(f"Tried to apply arguments to improper object: {function}")

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
        raise RuntimeError(f"Forcing an unforceable object: {context}")
