import dataclasses
import math

from .ast import BuiltInFun


class CostingFun:
    def cost(self, *memories: int) -> int:
        raise NotImplementedError("Abstract cost not implemented")


@dataclasses.dataclass
class ConstantCost(CostingFun):
    constant: int

    def cost(self, *memories: int) -> int:
        return self.constant


@dataclasses.dataclass
class LinearSize(CostingFun):
    intercept: int
    slope: int

    def cost(self, *memories: int) -> int:
        return self.intercept + self.slope * memories[0]


@dataclasses.dataclass
class Derived(CostingFun):
    model: CostingFun


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
    minimum: int

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
class ModelOnDiagonal(CostingFun):
    model_on_diagonal: CostingFun
    model_off_diagonal: CostingFun

    def cost(self, x: int, y: int) -> int:
        if x == y:
            return self.model_on_diagonal.cost(x)
        return self.model_off_diagonal.cost(x, y)


@dataclasses.dataclass
class ModelAboveDiagonal(CostingFun):
    model_above_diagonal: CostingFun
    model_below_equal_diagonal: CostingFun

    def cost(self, x: int, y: int) -> int:
        if x > y:
            return self.model_above_diagonal.cost(x, y)
        return self.model_below_equal_diagonal.cost(x, y)


@dataclasses.dataclass
class ModelBelowDiagonal(CostingFun):
    model_above_equal_diagonal: CostingFun
    model_below_diagonal: CostingFun

    def cost(self, x: int, y: int) -> int:
        if x >= y:
            return self.model_above_equal_diagonal.cost(x, y)
        return self.model_below_diagonal.cost(x, y)


BuiltinCostMap = {}
