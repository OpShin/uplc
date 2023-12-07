import dataclasses
import enum
import functools
import json
import math
from pathlib import Path
from typing import Dict, Any, Union
from enum import Enum


from .ast import BuiltInFun


class BudgetMode(Enum):
    CPU = "cpu"
    Memory = "memory"


@dataclasses.dataclass
class Budget:
    cpu: int
    memory: int

    def __add__(self, other: "Budget") -> "Budget":
        return Budget(self.cpu + other.cpu, self.memory + other.memory)

    def __sub__(self, other: "Budget") -> "Budget":
        return Budget(self.cpu - other.cpu, self.memory - other.memory)

    def __mul__(self, other: int) -> "Budget":
        return Budget(self.cpu * other, self.memory * other)

    def __isub__(self, other: "Budget") -> "Budget":
        self.cpu -= other.cpu
        self.memory -= other.memory
        return self

    def __iadd__(self, other: "Budget") -> "Budget":
        self.cpu += other.cpu
        self.memory += other.memory
        return self

    def __imul__(self, other: int) -> "Budget":
        self.cpu *= other
        self.memory *= other
        return self

    def __radd__(self, other: "Budget") -> "Budget":
        return Budget(self.cpu + other.cpu, self.memory + other.memory)

    def __rsub__(self, other: "Budget") -> "Budget":
        return Budget(self.cpu - other.cpu, self.memory - other.memory)

    def __rmul__(self, other: int) -> "Budget":
        return Budget(self.cpu * other, self.memory * other)

    def exhausted(self):
        return self.cpu < 0 or self.memory < 0


class CostingFun:
    def cost(self, *memories: int) -> int:
        raise NotImplementedError("Abstract cost not implemented")

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        """Parses the arguments of the Plutus Core cost model description"""
        if isinstance(arguments, int):
            return ConstantCost.from_arguments(arguments)
        if "intercept" in arguments and "slope" in arguments and len(arguments) == 2:
            return LinearCost(arguments["intercept"], arguments["slope"])
        raise NotImplementedError("Cost model unknown")


@dataclasses.dataclass
class ConstantCost(CostingFun):
    constant: int = 0

    def cost(self, *memories: int) -> int:
        return self.constant

    @classmethod
    def from_arguments(cls, arguments: int):
        return cls(arguments)


@dataclasses.dataclass
class LinearCost(CostingFun):
    intercept: int = 0
    slope: int = 0

    def cost(self, *memories: int) -> int:
        return self.intercept + self.slope * memories[0]

    @classmethod
    def from_arguments(cls, arguments: Dict[str, int]):
        return cls(**arguments)


@dataclasses.dataclass
class Derived(CostingFun):
    model: CostingFun

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        return cls(CostingFun.from_arguments(arguments))


@dataclasses.dataclass
class LinearInX(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(memories[0])


@dataclasses.dataclass
class LinearInY(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(memories[1])


@dataclasses.dataclass
class LinearInZ(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(memories[2])


@dataclasses.dataclass
class AddedSizes(Derived):
    def cost(self, *memories: int) -> int:
        return self.model.cost(sum(memories))


@dataclasses.dataclass
class SubtractedSizes(Derived):
    minimum: int = 0

    def cost(self, *memories: int) -> int:
        return self.model.cost(max(memories[0] - memories[1], self.minimum))

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        min = arguments["minimum"]
        del arguments["minimum"]
        return cls(
            CostingFun.from_arguments(arguments),
            min,
        )


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
    model_on_diagonal: LinearCost
    model_off_diagonal: ConstantCost = ConstantCost(0)

    def cost(self, x: int, y: int) -> int:
        if x == y:
            return self.model_on_diagonal.cost(x)
        return self.model_off_diagonal.cost(x, y)

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        const_model = ConstantCost(arguments["constant"])
        linear_model = LinearCost(arguments["intercept"], arguments["slope"])
        return cls(linear_model, const_model)


@dataclasses.dataclass
class ConstAboveDiagonal(CostingFun):
    model_below_equal_diagonal: CostingFun
    model_above_diagonal: ConstantCost = ConstantCost(0)

    def cost(self, x: int, y: int) -> int:
        if x > y:
            return self.model_above_diagonal.cost(x, y)
        return self.model_below_equal_diagonal.cost(x, y)

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        model = parse_costing_fun(arguments["model"])
        const = ConstantCost(arguments["constant"])
        return cls(model, const)


@dataclasses.dataclass
class ConstBelowDiagonal(CostingFun):
    model_above_equal_diagonal: CostingFun
    model_below_diagonal: ConstantCost = ConstantCost(0)

    def cost(self, x: int, y: int) -> int:
        if x >= y:
            return self.model_above_equal_diagonal.cost(x, y)
        return self.model_below_diagonal.cost(x, y)

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        model = parse_costing_fun(arguments["model"])
        const = ConstantCost(arguments["constant"])
        return cls(model, const)


# TODO automatically parse from
# https://github.com/input-output-hk/plutus/blob/43ecfc3403cf908c55af57c8461e96e8b131b97c/plutus-core/cost-model/data/builtinCostModel.json
# or similar files


@dataclasses.dataclass
class BuiltinCostModel:
    cpu: Dict[BuiltInFun, CostingFun]
    memory: Dict[BuiltInFun, CostingFun]


class CekOp(enum.Enum):
    Const = enum.auto()
    Var = enum.auto()
    Lam = enum.auto()
    Apply = enum.auto()
    Delay = enum.auto()
    Force = enum.auto()
    Builtin = enum.auto()
    Startup = enum.auto()
    Constr = enum.auto()
    Case = enum.auto()


@dataclasses.dataclass
class CekMachineCostModel:
    cpu: Dict[CekOp, CostingFun]
    memory: Dict[CekOp, CostingFun]


COSTING_FUN_DICT = {
    fun.__name__: fun
    for fun in (
        ConstBelowDiagonal,
        ConstantCost,
        ConstAboveDiagonal,
        LinearOnDiagonal,
        LinearCost,
        AddedSizes,
        MultipliedSizes,
        MinSize,
        MaxSize,
        LinearInX,
        LinearInY,
        LinearInZ,
        SubtractedSizes,
    )
}


def parse_costing_fun(model: dict):
    type, arguments = model["type"], model["arguments"]
    CamelCaseType = "".join(x.capitalize() for x in type.split("_"))
    costing_fun = COSTING_FUN_DICT[CamelCaseType]
    return costing_fun.from_arguments(arguments)


def parse_builtin_cost_model(model: dict):
    cost_model = BuiltinCostModel({}, {})
    for fun, d in model.items():
        builtin_fun_name = fun[:1].capitalize() + fun[1:]
        builtin_fun = BuiltInFun.__dict__[builtin_fun_name]
        cost_model.memory[builtin_fun] = parse_costing_fun(d["memory"])
        cost_model.cpu[builtin_fun] = parse_costing_fun(d["cpu"])
    return cost_model


@functools.lru_cache(maxsize=1)
def default_builtin_cost_model_plutus_v2():
    return default_builtin_cost_model("v2")


def default_builtin_cost_model(plutus_version: str):
    builtinCostModel = (
        Path(__file__)
        .parent.joinpath("cost_model_files")
        .joinpath(plutus_version)
        .joinpath("builtinCostModel.json")
    )
    with open(builtinCostModel) as f:
        d = json.load(f)
    return parse_builtin_cost_model(d)


def parse_cek_machine_cost_model(model: dict):
    cost_model = CekMachineCostModel({}, {})
    for op, d in model.items():
        enum_name = str(op).removeprefix("cek").removesuffix("Cost")
        cek_op = CekOp.__dict__[enum_name]
        cost_model.memory[cek_op] = ConstantCost(d["exBudgetMemory"])
        cost_model.cpu[cek_op] = ConstantCost(d["exBudgetCPU"])
    return cost_model


def default_cek_machine_cost_model(plutus_version: str):
    builtinCostModel = (
        Path(__file__)
        .parent.joinpath("cost_model_files")
        .joinpath(plutus_version)
        .joinpath("cekMachineCosts.json")
    )
    with open(builtinCostModel) as f:
        d = json.load(f)
    return parse_cek_machine_cost_model(d)


@functools.lru_cache(maxsize=1)
def default_cek_machine_cost_model_plutus_v2():
    return default_cek_machine_cost_model("v2")


def default_budget():
    return Budget(memory=14000000, cpu=10000000000)
