import dataclasses
import math
from typing import Dict
from enum import Enum


from .ast import BuiltInFun


class BudgetMode(Enum):
    CPU = "cpu"
    Memory = "memory"


class CostingFun:
    def cost(self, *memories: int) -> int:
        raise NotImplementedError("Abstract cost not implemented")

    def from_cost_model(
        self, cost_model: Dict[str, int], fun: BuiltInFun, mode: BudgetMode
    ) -> None:
        """Initializes the parameters in this costing function based on the cost model"""
        fun_name = fun.name[0].lower() + fun.name[1:]
        budget_name = mode.value
        self._from_cost_model(cost_model, f"{fun_name}-{budget_name}")

    def _from_cost_model(self, cost_model: Dict[str, int], prefix: str) -> None:
        raise NotImplementedError()


@dataclasses.dataclass
class ConstantCost(CostingFun):
    constant: int = 0

    def cost(self, *memories: int) -> int:
        return self.constant

    def _from_cost_model(self, cost_model: Dict[str, int], prefix: str) -> None:
        self.constant = cost_model[f"{prefix}-arguments"]


@dataclasses.dataclass
class LinearSize(CostingFun):
    intercept: int = 0
    slope: int = 0

    def cost(self, *memories: int) -> int:
        return self.intercept + self.slope * memories[0]

    def _from_cost_model(self, cost_model: Dict[str, int], prefix: str) -> None:
        self.intercept = cost_model[f"{prefix}-arguments-intercept"]
        self.slope = cost_model[f"{prefix}-arguments-slope"]


@dataclasses.dataclass
class Derived(CostingFun):
    model: CostingFun

    def _from_cost_model(self, cost_model: Dict[str, int], prefix: str) -> None:
        self.model._from_cost_model(cost_model, prefix)


@dataclasses.dataclass
class SizeX(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(memories[0])


@dataclasses.dataclass
class SizeY(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(memories[1])


@dataclasses.dataclass
class SizeZ(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(memories[2])


@dataclasses.dataclass
class AddedSizes(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(sum(memories))


@dataclasses.dataclass
class SubtractedSizes(Derived):
    pass
    minimum: int = 0

    def cost(self, *memories: int) -> int:
        return self.model.cost(max(memories[0] - memories[1], self.minimum))


@dataclasses.dataclass
class MultipliedSizes(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(math.prod(memories))


@dataclasses.dataclass
class MinSize(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(min(memories))


@dataclasses.dataclass
class MaxSize(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(max(memories))


@dataclasses.dataclass
class LinearOnDiagonal(CostingFun):
    model_on_diagonal: LinearSize
    model_off_diagonal: ConstantCost = ConstantCost(0)

    def cost(self, x: int, y: int) -> int:
        if x == y:
            return self.model_on_diagonal.cost(x)
        return self.model_off_diagonal.cost(x, y)

    def _from_cost_model(self, cost_model: Dict[str, int], prefix: str) -> None:
        self.model_on_diagonal._from_cost_model(cost_model, prefix)
        self.model_off_diagonal._from_cost_model(cost_model, prefix)


@dataclasses.dataclass
class ConstantAboveDiagonal(CostingFun):
    model_below_equal_diagonal: CostingFun
    model_above_diagonal: ConstantCost = ConstantCost(0)

    def cost(self, x: int, y: int) -> int:
        if x > y:
            return self.model_above_diagonal.cost(x, y)
        return self.model_below_equal_diagonal.cost(x, y)

    def _from_cost_model(self, cost_model: Dict[str, int], prefix: str) -> None:
        self.model_above_diagonal._from_cost_model(cost_model, prefix)
        self.model_below_equal_diagonal._from_cost_model(cost_model, f"{prefix}-model")


@dataclasses.dataclass
class ConstantBelowDiagonal(CostingFun):
    model_above_equal_diagonal: CostingFun
    model_below_diagonal: ConstantCost = ConstantCost(0)

    def cost(self, x: int, y: int) -> int:
        if x >= y:
            return self.model_above_equal_diagonal.cost(x, y)
        return self.model_below_diagonal.cost(x, y)

    def _from_cost_model(self, cost_model: Dict[str, int], prefix: str) -> None:
        self.model_below_diagonal._from_cost_model(cost_model, prefix)
        self.model_above_equal_diagonal._from_cost_model(cost_model, f"{prefix}-model")


# TODO automatically parse from
# https://github.com/input-output-hk/plutus/blob/43ecfc3403cf908c55af57c8461e96e8b131b97c/plutus-core/cost-model/data/builtinCostModel.json
# or similar files

PlutusV2_mem_FunctionModel = {
    BuiltInFun.AddInteger: MaxSize(LinearSize()),
    BuiltInFun.SubtractInteger: MaxSize(LinearSize()),
    BuiltInFun.MultiplyInteger: AddedSizes(LinearSize()),
    BuiltInFun.DivideInteger: SubtractedSizes(LinearSize()),
    BuiltInFun.QuotientInteger: SubtractedSizes(LinearSize()),
    BuiltInFun.RemainderInteger: SubtractedSizes(LinearSize()),
    BuiltInFun.ModInteger: SubtractedSizes(LinearSize()),
    BuiltInFun.EqualsInteger: ConstantCost(),
    BuiltInFun.LessThanInteger: ConstantCost(),
    BuiltInFun.LessThanEqualsInteger: ConstantCost(),
    BuiltInFun.AppendByteString: AddedSizes(LinearSize()),
    BuiltInFun.ConsByteString: AddedSizes(LinearSize()),
    BuiltInFun.SliceByteString: AddedSizes(LinearSize()),
}

PlutusV2_cpu_FunctionModel = {
    BuiltInFun.AddInteger: MaxSize(LinearSize()),
    BuiltInFun.AppendByteString: AddedSizes(LinearSize()),
    BuiltInFun.AppendString: AddedSizes(LinearSize()),
    BuiltInFun.BData: ConstantCost(),
    # BuiltInFun.Blake2b_224: LinearSize(),
    BuiltInFun.Blake2b_256: LinearSize(),
    BuiltInFun.ChooseData: ConstantCost(),
    BuiltInFun.ChooseList: ConstantCost(),
}
