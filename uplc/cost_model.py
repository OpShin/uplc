from .ast import BuiltInFun


class CostingFun:
    def cost(self, *memories) -> int:
        raise NotImplementedError("Abstract cost not implemented")


BuiltinCostMap = {}
