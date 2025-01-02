import copy
import dataclasses
import datetime
import enum
import functools
import json
import math
from pathlib import Path
from typing import Dict, Any, Union
from enum import Enum

import pycardano

from .ast import BuiltInFun

# 30000000 appears to be default (see https://github.com/aiken-lang/aiken/blob/e1d46fa8f063445da8c0372e3c031c8a11ad0b14/crates/uplc/src/machine/cost_model.rs#L3376)
DEFAULT_COST_COEFF = 30000000000

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

    def __ge__(self, other: "Budget") -> bool:
        return self.cpu >= other.cpu and self.memory >= other.memory

    def __gt__(self, other: "Budget") -> bool:
        return self.cpu > other.cpu and self.memory > other.memory

    def __lt__(self, other: "Budget") -> bool:
        return self.cpu < other.cpu and self.memory < other.memory

    def __le__(self, other: "Budget") -> bool:
        return self.cpu <= other.cpu and self.memory <= other.memory

    def __eq__(self, other: "Budget") -> bool:
        return self.cpu == other.cpu and self.memory == other.memory

    def exhausted(self):
        return self.cpu < 0 or self.memory < 0


class CostingFun:
    def cost(self, *memories: int, values=[]) -> int:
        raise NotImplementedError("Abstract cost not implemented")

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        """Parses the arguments of the Plutus Core cost model description"""
        if isinstance(arguments, int):
            return ConstantCost.from_arguments(arguments)
        if "intercept" in arguments and "slope" in arguments and len(arguments) == 2:
            return LinearCost(arguments["intercept"], arguments["slope"])
        raise NotImplementedError("Cost model unknown")

    def update_from_network_config(
        self, network_config: Dict[str, int], prefix: str = ""
    ):
        """Updates the cost model from the network configuration"""
        raise NotImplementedError("Base model can not update")


@dataclasses.dataclass
class ConstantCost(CostingFun):
    constant: int = 0

    def cost(self, *memories: int, values=[]) -> int:
        return self.constant

    @classmethod
    def from_arguments(cls, arguments: int):
        return cls(arguments)

    def update_from_network_config(
        self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.constant = network_config.get(f"{prefix}-arguments", DEFAULT_COST_COEFF)


@dataclasses.dataclass
class LinearCost(CostingFun):
    intercept: int = 0
    slope: int = 0

    def cost(self, *memories: int, values=[]) -> int:
        return self.intercept + self.slope * memories[0]

    @classmethod
    def from_arguments(cls, arguments: Dict[str, int]):
        return cls(**arguments)

    def update_from_network_config(
        self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.intercept = network_config.get(f"{prefix}-arguments-intercept", DEFAULT_COST_COEFF)
        self.slope = network_config.get(f"{prefix}-arguments-slope", DEFAULT_COST_COEFF)


@dataclasses.dataclass
class Derived(CostingFun):
    model: CostingFun

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        return cls(CostingFun.from_arguments(arguments))

    def update_from_network_config(
        self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.model.update_from_network_config(network_config, prefix)


@dataclasses.dataclass
class LinearInX(Derived):
    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(memories[0])


@dataclasses.dataclass
class LinearInY(Derived):
    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(memories[1])

@dataclasses.dataclass
class LinearInMaxYz(Derived):
    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(max(memories[1], memories[2]))


@dataclasses.dataclass
class LinearInZ(Derived):
    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(memories[2])

@dataclasses.dataclass
class AddedSizes(Derived):
    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(sum(memories))


@dataclasses.dataclass
class SubtractedSizes(Derived):
    minimum: int = 0

    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(max(memories[0] - memories[1], self.minimum))

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        min = arguments.get("minimum", DEFAULT_COST_COEFF)
        if "minimum" in arguments:
            del arguments["minimum"]
        return cls(
            CostingFun.from_arguments(arguments),
            min,
        )

    def update_from_network_config(
        self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.model.update_from_network_config(network_config, prefix)
        self.minimum = network_config.get(f"{prefix}-arguments-minimum", DEFAULT_COST_COEFF)


@dataclasses.dataclass
class MultipliedSizes(Derived):
    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(math.prod(memories))


@dataclasses.dataclass
class MinSize(Derived):
    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(min(memories))


@dataclasses.dataclass
class MaxSize(Derived):
    def cost(self, *memories: int, values=[]) -> int:
        return self.model.cost(max(memories))


@dataclasses.dataclass
class LinearOnDiagonal(CostingFun):
    model_on_diagonal: LinearCost
    model_off_diagonal: ConstantCost = dataclasses.field(
        default_factory=lambda: ConstantCost(0)
    )

    def cost(self, *memories: int, values=[]) -> int:
        x, y = memories[0], memories[1]
        if x == y:
            return self.model_on_diagonal.cost(x)
        return self.model_off_diagonal.cost(x, y)

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        const_model = ConstantCost(arguments["constant"])
        linear_model = LinearCost(arguments["intercept"], arguments["slope"])
        return cls(linear_model, const_model)

    def update_from_network_config(
        self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.model_off_diagonal.constant = network_config.get(
            f"{prefix}-arguments-constant", DEFAULT_COST_COEFF
        )
        self.model_on_diagonal.update_from_network_config(network_config, prefix)


@dataclasses.dataclass
class ConstAboveDiagonal(CostingFun):
    model_below_equal_diagonal: CostingFun
    model_above_diagonal: ConstantCost = dataclasses.field(
        default_factory=lambda: ConstantCost(0)
    )

    def cost(self, *memories: int, values=[]) -> int:
        x, y = memories[0], memories[1]
        if x > y:
            return self.model_above_diagonal.cost(x, y)
        return self.model_below_equal_diagonal.cost(x, y)

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        model = parse_costing_fun(arguments["model"])
        const = ConstantCost(arguments["constant"])
        return cls(model, const)

    def update_from_network_config(
        self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.model_above_diagonal.constant = network_config.get(
            f"{prefix}-arguments-constant", DEFAULT_COST_COEFF)
        self.model_below_equal_diagonal.update_from_network_config(
            network_config, f"{prefix}-arguments-model" if not isinstance(self.model_below_equal_diagonal, QuadraticInXAndY) else prefix
        )


@dataclasses.dataclass
class ConstBelowDiagonal(CostingFun):
    model_above_equal_diagonal: CostingFun
    model_below_diagonal: ConstantCost = dataclasses.field(
        default_factory=lambda: ConstantCost(0)
    )

    def cost(self, *memories: int, values=[]) -> int:
        x, y = memories[0], memories[1]
        if x >= y:
            return self.model_above_equal_diagonal.cost(x, y)
        return self.model_below_diagonal.cost(x, y)

    @classmethod
    def from_arguments(cls, arguments: Union[Dict[str, Any]]):
        model = parse_costing_fun(arguments["model"])
        const = ConstantCost(arguments["constant"])
        return cls(model, const)

    def update_from_network_config(
        self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.model_below_diagonal.constant = network_config.get(
            f"{prefix}-arguments-constant", DEFAULT_COST_COEFF
        )
        self.model_above_equal_diagonal.update_from_network_config(
            network_config, f"{prefix}-arguments-model" if not isinstance(self.model_above_equal_diagonal, QuadraticInXAndY) else prefix
        )

@dataclasses.dataclass
class QuadraticInY(CostingFun):
    c0: int = 0
    c1: int = 0
    c2: int = 0

    def cost(self, *memories: int, values=[]) -> int:
        return self.c0 + self.c1 * memories[1] + self.c2 * memories[1]**2

    @classmethod
    def from_arguments(cls, arguments: Dict[str, int]):
        return cls(**arguments)

    def update_from_network_config(
            self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.c0 = network_config.get(f"{prefix}-arguments-c0", DEFAULT_COST_COEFF)
        self.c1 = network_config.get(f"{prefix}-arguments-c1", DEFAULT_COST_COEFF)
        self.c2 = network_config.get(f"{prefix}-arguments-c2", DEFAULT_COST_COEFF)

@dataclasses.dataclass
class QuadraticInZ(CostingFun):
    c0: int = 0
    c1: int = 0
    c2: int = 0

    def cost(self, *memories: int, values=[]) -> int:
        return self.c0 + self.c1 * memories[2] + self.c2 * memories[2]**2

    @classmethod
    def from_arguments(cls, arguments: Dict[str, int]):
        return cls(**arguments)

    def update_from_network_config(
            self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.c0 = network_config.get(f"{prefix}-arguments-c0", DEFAULT_COST_COEFF)
        self.c1 = network_config.get(f"{prefix}-arguments-c1", DEFAULT_COST_COEFF)
        self.c2 = network_config.get(f"{prefix}-arguments-c2", DEFAULT_COST_COEFF)

@dataclasses.dataclass
class QuadraticInXAndY(CostingFun):
    c00: int = 0
    c01: int = 0
    c02: int = 0
    c10: int = 0
    c11: int = 0
    c20: int = 0
    minimum: int = 0

    def cost(self, *memories: int, values=[]) -> int:
        x, y = memories[0], memories[1]
        poly = self.c00 + self.c10 * x + self.c01 * y + + self.c20 * x * x  + self.c11 * x * y + self.c02 * y * y
        return max(poly, self.minimum)

    @classmethod
    def from_arguments(cls, arguments: Dict[str, int]):
        return cls(**arguments)

    def update_from_network_config(
            self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.c00 = network_config.get(f"{prefix}-arguments-c00", DEFAULT_COST_COEFF)
        self.c10 = network_config.get(f"{prefix}-arguments-c10", DEFAULT_COST_COEFF)
        self.c01 = network_config.get(f"{prefix}-arguments-c01", DEFAULT_COST_COEFF)
        self.c20 = network_config.get(f"{prefix}-arguments-c20", DEFAULT_COST_COEFF)
        self.c11 = network_config.get(f"{prefix}-arguments-c11", DEFAULT_COST_COEFF)
        self.c02 = network_config.get(f"{prefix}-arguments-c02", DEFAULT_COST_COEFF)
        self.minimum = network_config.get(f"{prefix}-arguments-minimum", DEFAULT_COST_COEFF)

@dataclasses.dataclass
class LiteralInYOrLinearInZ(CostingFun):
    intercept: int = 0
    slope: int = 0

    def cost(self, *memories: int, values=[]) -> int:
        # LOL good for you having implemented this extra step to pass actual values to the costing function
        # because the specs say you need it
        # (correct implementation based on official specifications below)
        y = values[1].value
        if y == 0:
          return self.intercept + self.slope * memories[2]
        return int(math.ceil((abs(y) - 1) / 8))
        # NO FCK YOU and instead use this completely broken implementation because possibly some IOHK engineer
        # was too lazy to implement the specs or they realized that the specs were a terrible idea
        # y = memories[1]
        # if y == 0:
        #   return self.intercept + self.slope * memories[2]
        # return y

    @classmethod
    def from_arguments(cls, arguments: Dict[str, int]):
        return cls(**arguments)

    def update_from_network_config(
            self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.c00 = network_config.get(f"{prefix}-arguments-c00", DEFAULT_COST_COEFF)
        self.c10 = network_config.get(f"{prefix}-arguments-c10", DEFAULT_COST_COEFF)
        self.c01 = network_config.get(f"{prefix}-arguments-c01", DEFAULT_COST_COEFF)
        self.c20 = network_config.get(f"{prefix}-arguments-c20", DEFAULT_COST_COEFF)
        self.c11 = network_config.get(f"{prefix}-arguments-c11", DEFAULT_COST_COEFF)
        self.c02 = network_config.get(f"{prefix}-arguments-c02", DEFAULT_COST_COEFF)
        self.minimum = network_config.get(f"{prefix}-arguments-minimum", DEFAULT_COST_COEFF)

@dataclasses.dataclass
class LinearInYAndZ(CostingFun):
    intercept: int = 0
    slope1: int = 0
    slope2: int = 0

    def cost(self, *memories: int, values=[]) -> int:
        return self.intercept + self.slope1 * memories[1] + self.slope2 * memories[2]

    @classmethod
    def from_arguments(cls, arguments: Dict[str, int]):
        return cls(**arguments)

    def update_from_network_config(
            self, network_config: Dict[str, int], prefix: str = ""
    ):
        self.intercept = network_config.get(f"{prefix}-arguments-intercept", DEFAULT_COST_COEFF)
        self.slope1 = network_config.get(f"{prefix}-arguments-slope1", DEFAULT_COST_COEFF)
        self.slope2 = network_config.get(f"{prefix}-arguments-slope2", DEFAULT_COST_COEFF)

# we automatically parse the model of each builtin from
# https://github.com/input-output-hk/plutus/blob/43ecfc3403cf908c55af57c8461e96e8b131b97c/plutus-core/cost-model/data/builtinCostModel.json


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
    cpu: Dict[CekOp, ConstantCost]
    memory: Dict[CekOp, ConstantCost]


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
        LinearInMaxYz,
        LinearInYAndZ,
        SubtractedSizes,
        QuadraticInY,
        QuadraticInZ,
        QuadraticInXAndY,
        LiteralInYOrLinearInZ,
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
        builtin_fun_name = "_".join(x[:1].capitalize() + x[1:] for x in fun.split("_"))
        builtin_fun = BuiltInFun.__dict__[builtin_fun_name]
        cost_model.memory[builtin_fun] = parse_costing_fun(d["memory"])
        cost_model.cpu[builtin_fun] = parse_costing_fun(d["cpu"])
    return cost_model


def updated_builtin_cost_model_from_network_config(
    builtin_cost_model: BuiltinCostModel, network_config: Dict[str, int]
):
    builtin_cost_model = copy.deepcopy(builtin_cost_model)
    for mode, cost_fun_dicts in (
        ("cpu", builtin_cost_model.cpu),
        ("memory", builtin_cost_model.memory),
    ):
        for builtin, cost_model in cost_fun_dicts.items():
            builtin_fun_name = builtin.name[:1].lower() + builtin.name[1:]
            prefix = f"{builtin_fun_name}-{mode}"
            cost_model.update_from_network_config(network_config, prefix)
    return builtin_cost_model


class PlutusVersion(enum.Enum):
    PlutusV1 = "PlutusV1"
    PlutusV2 = "PlutusV2"
    PlutusV3 = "PlutusV3"


@functools.lru_cache()
def default_builtin_cost_model_base():
    # TODO choose different base cost model based on Plutus Version
    builtinCostModel = (
        Path(__file__)
        .parent.joinpath("cost_model_files")
        .joinpath("base")
        .joinpath("builtinCostModel.json")
    )
    with open(builtinCostModel) as f:
        d = json.load(f)
    return parse_builtin_cost_model(d)


NETWORK_CONFIG_DIR = Path(__file__).parent.joinpath("cost_model_files")


def load_network_config(config_date: datetime.date):
    """
    Loads the network config from the network config directory that was released last before the given date
    """
    latest_date = None
    latest_dir_name = None
    for dir in NETWORK_CONFIG_DIR.iterdir():
        if not dir.is_dir():
            continue
        try:
            dir_date = datetime.date.fromisoformat(dir.name)
        except ValueError:
            continue
        if dir_date > config_date:
            continue
        if latest_date is None or dir_date > latest_date:
            latest_date = dir_date
            latest_dir_name = dir.name
    network_config_dir = NETWORK_CONFIG_DIR.joinpath(latest_dir_name)
    file = None
    for file in network_config_dir.iterdir():
        if file.suffix == "json":
            break
    if file is None:
        raise ValueError("Latest network config could not be loaded")
    with open(file) as f:
        d = json.load(f)
    return d


@functools.lru_cache(maxsize=1)
def latest_network_config():
    """
    Loads the latest network config from the network config directory that is most recent
    """
    return load_network_config(datetime.date.today())


def latest_network_config_plutus(plutus_version: PlutusVersion):
    return latest_network_config()[plutus_version.value]


def default_builtin_cost_model(plutus_version: PlutusVersion):
    return updated_builtin_cost_model_from_network_config(
        default_builtin_cost_model_base(), latest_network_config_plutus(plutus_version)
    )


@functools.lru_cache()
def default_builtin_cost_model_plutus_v1():
    return default_builtin_cost_model(PlutusVersion.PlutusV1)


@functools.lru_cache()
def default_builtin_cost_model_plutus_v2():
    return default_builtin_cost_model(PlutusVersion.PlutusV2)


@functools.lru_cache()
def default_builtin_cost_model_plutus_v3():
    return default_builtin_cost_model(PlutusVersion.PlutusV3)


def parse_cek_machine_cost_model(model: dict):
    cost_model = CekMachineCostModel({}, {})
    for op, d in model.items():
        enum_name = str(op)[len("cek") :][: -len("Cost")]
        cek_op = CekOp.__dict__[enum_name]
        cost_model.memory[cek_op] = ConstantCost(d["exBudgetMemory"])
        cost_model.cpu[cek_op] = ConstantCost(d["exBudgetCPU"])
    return cost_model


@functools.lru_cache()
def default_cek_machine_cost_model_base():
    builtinCostModel = (
        Path(__file__)
        .parent.joinpath("cost_model_files")
        .joinpath("base")
        .joinpath("cekMachineCosts.json")
    )
    with open(builtinCostModel) as f:
        d = json.load(f)
    return parse_cek_machine_cost_model(d)


def updated_cek_machine_cost_model_from_network_config(
    cek_machine_cost_model: CekMachineCostModel, network_config: Dict[str, int]
):
    cek_machine_cost_model = copy.deepcopy(cek_machine_cost_model)
    for mode, cost_fun_dicts in (
        ("CPU", cek_machine_cost_model.cpu),
        ("Memory", cek_machine_cost_model.memory),
    ):
        for cek_op, cost_model in cost_fun_dicts.items():
            cek_op_name = f"cek{cek_op.name}Cost"
            cost_model.constant = network_config.get(f"{cek_op_name}-exBudget{mode}", DEFAULT_COST_COEFF)
    return cek_machine_cost_model


def default_cek_machine_cost_model(plutus_version: PlutusVersion):
    return updated_cek_machine_cost_model_from_network_config(
        default_cek_machine_cost_model_base(),
        latest_network_config_plutus(plutus_version),
    )


@functools.lru_cache()
def default_cek_machine_cost_model_plutus_v1():
    return default_cek_machine_cost_model(PlutusVersion.PlutusV1)


@functools.lru_cache()
def default_cek_machine_cost_model_plutus_v2():
    return default_cek_machine_cost_model(PlutusVersion.PlutusV2)

@functools.lru_cache()
def default_cek_machine_cost_model_plutus_v3():
    return default_cek_machine_cost_model(PlutusVersion.PlutusV3)

def default_budget():
    return Budget(memory=14000000, cpu=10000000000)
