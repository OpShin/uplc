import logging
from dataclasses import dataclass
from functools import partial
from enum import Enum, auto
import hashlib
from typing import List, Optional, Any, Tuple, Dict, Union

import cbor2

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlutusData:
    pass

    def to_cbor(self) -> Any:
        """Returns a CBOR encodable representation of this object"""
        raise NotImplementedError


@dataclass(frozen=True)
class PlutusAtomic(PlutusData):
    value: Any

    def to_cbor(self):
        return self.value


@dataclass(frozen=True)
class PlutusInteger(PlutusAtomic):
    value: int


@dataclass(frozen=True)
class PlutusByteString(PlutusAtomic):
    value: bytes


@dataclass(frozen=True)
class PlutusList(PlutusData):
    value: List[PlutusData]

    def to_cbor(self):
        return [d.to_cbor() for d in self.value]


@dataclass(frozen=True)
class PlutusMap(PlutusData):
    value: Dict[PlutusData, PlutusData]

    def to_cbor(self):
        return {k.to_cbor(): v.to_cbor() for k, v in self.value.items()}


@dataclass(frozen=True)
class PlutusConstr(PlutusData):
    constructor: int
    fields: List[PlutusData]

    def to_cbor(self):
        return cbor2.dumps(
            cbor2.CBORTag(self.constructor + 121, [f.to_cbor() for f in self.fields])
        )


def data_from_cbortag(cbor) -> PlutusData:
    if isinstance(cbor, cbor2.CBORTag):
        constructor = cbor.tag - 121
        fields = list(map(data_from_cbortag, cbor.value))
        return PlutusConstr(constructor, fields)
    if isinstance(cbor, int):
        return PlutusInteger(cbor)
    if isinstance(cbor, bytes):
        return PlutusByteString(cbor)
    if isinstance(cbor, list):
        return PlutusList(list(map(data_from_cbortag, cbor)))
    if isinstance(cbor, dict):
        return PlutusMap(
            {data_from_cbortag(k): data_from_cbortag(v) for k, v in cbor.items()}
        )


def data_from_cbor(cbor: bytes) -> PlutusData:
    raw_datum = cbor2.loads(cbor)
    return data_from_cbortag(raw_datum)


class ConstantType(Enum):
    integer = auto()
    bytestring = auto()
    string = auto()
    unit = auto()
    bool = auto()
    pair = auto()
    list = auto()
    data = auto()


ConstantEvalMap = {
    ConstantType.integer: int,
    ConstantType.bytestring: bytes,
    ConstantType.string: str,
    ConstantType.unit: lambda _: (),
    ConstantType.bool: bool,
    ConstantType.pair: lambda x: (ConstantEvalMap[x[0]], ConstantEvalMap[x[1]]),
    ConstantType.list: lambda xs: [ConstantEvalMap[x] for x in xs],
    ConstantType.data: data_from_cbor,
}

ConstantPrintMap = {
    ConstantType.integer: str,
    ConstantType.bytestring: lambda b: f"#{b.hex()}",
    ConstantType.string: str,
    ConstantType.unit: str,
    ConstantType.bool: bool,
    ConstantType.pair: lambda x: (ConstantPrintMap[x[0]], ConstantPrintMap[x[1]]),
    ConstantType.list: lambda xs: [ConstantPrintMap[x] for x in xs],
    ConstantType.data: data_from_cbor,
}


# As found in https://plutonomicon.github.io/plutonomicon/builtin-functions
class BuiltInFun(Enum):
    AddInteger = auto()
    SubtractInteger = auto()
    MultiplyInteger = auto()
    DivideInteger = auto()
    QuotientInteger = auto()
    RemainderInteger = auto()
    ModInteger = auto()
    EqualsInteger = auto()
    LessThanInteger = auto()
    LessThanEqualsInteger = auto()
    AppendByteString = auto()
    ConsByteString = auto()
    SliceByteString = auto()
    LengthOfByteString = auto()
    IndexByteString = auto()
    EqualsByteString = auto()
    LessThanByteString = auto()
    LessThanEqualsByteString = auto()
    Sha2_256 = auto()
    Sha3_256 = auto()
    Blake2b_256 = auto()
    VerifySignature = auto()
    AppendString = auto()
    EqualsString = auto()
    EncodeUtf8 = auto()
    DecodeUtf8 = auto()
    IfThenElse = auto()
    ChooseUnit = auto()
    Trace = auto()
    FstPair = auto()
    SndPair = auto()
    ChooseList = auto()
    MkCons = auto()
    HeadList = auto()
    TailList = auto()
    NullList = auto()
    ChooseData = auto()
    ConstrData = auto()
    MapData = auto()
    ListData = auto()
    IData = auto()
    BData = auto()
    UnConstrData = auto()
    UnMapData = auto()
    UnListData = auto()
    UnIData = auto()
    UnBData = auto()
    EqualsData = auto()
    MkPairData = auto()
    MkNilData = auto()
    MkNilPairData = auto()


def _ChooseList(_, d, v, w, x, y, z):
    if isinstance(d, PlutusConstr):
        return v
    if isinstance(d, PlutusMap):
        return w
    if isinstance(d, PlutusList):
        return x
    if isinstance(d, PlutusInteger):
        return y
    if isinstance(d, PlutusByteString):
        return z


BuiltInFunEvalMap = {
    BuiltInFun.AddInteger: lambda x, y: x + y,
    BuiltInFun.SubtractInteger: lambda x, y: x - y,
    BuiltInFun.MultiplyInteger: lambda x, y: x * y,
    # TODO difference with negative values?
    BuiltInFun.DivideInteger: lambda x, y: x // y,
    BuiltInFun.QuotientInteger: lambda x, y: x // y,
    # TODO difference with negative values?
    BuiltInFun.RemainderInteger: lambda x, y: x % y,
    BuiltInFun.ModInteger: lambda x, y: x % y,
    BuiltInFun.EqualsInteger: lambda x, y: x == y,
    BuiltInFun.LessThanInteger: lambda x, y: x < y,
    BuiltInFun.LessThanEqualsInteger: lambda x, y: x <= y,
    BuiltInFun.AppendByteString: lambda x, y: x + y,
    BuiltInFun.ConsByteString: lambda x, y: bytes([x]) + y,
    BuiltInFun.SliceByteString: lambda x, y, z: z[x : y + 1],
    BuiltInFun.LengthOfByteString: lambda x: len(x),
    BuiltInFun.IndexByteString: lambda x, y: x[y],
    BuiltInFun.EqualsByteString: lambda x, y: x == y,
    BuiltInFun.LessThanByteString: lambda x, y: x < y,
    BuiltInFun.LessThanEqualsByteString: lambda x, y: x <= y,
    BuiltInFun.Sha2_256: lambda x: hashlib.sha256(x).digest(),
    BuiltInFun.Sha3_256: lambda x: hashlib.sha3_256(x).digest(),
    BuiltInFun.Blake2b_256: lambda x: hashlib.blake2b(x).digest(),
    # TODO how to emulate this?
    BuiltInFun.VerifySignature: lambda pk, m, s: True,
    BuiltInFun.AppendString: lambda x, y: x + y,
    BuiltInFun.EqualsString: lambda x, y: x == y,
    BuiltInFun.EncodeUtf8: lambda x: x.encode("utf8"),
    BuiltInFun.DecodeUtf8: lambda x: x.decode("utf8"),
    BuiltInFun.IfThenElse: lambda _, x, y, z: y if x else z,
    BuiltInFun.ChooseUnit: lambda _, y: y,
    BuiltInFun.Trace: lambda _, x, y: print(x) or y,
    BuiltInFun.FstPair: lambda _, _2, x: x[0],
    BuiltInFun.SndPair: lambda _, _2, x: x[1],
    BuiltInFun.ChooseList: lambda _, _2, l, x, y: x if l == [] else y,
    BuiltInFun.MkCons: lambda _, e, l: [e] + l,
    BuiltInFun.HeadList: lambda _, l: l[0],
    BuiltInFun.TailList: lambda _, l: l[1:],
    BuiltInFun.NullList: lambda _, l: l == [],
    BuiltInFun.ChooseData: _ChooseList,
    BuiltInFun.ConstrData: lambda x, y: PlutusConstr(x, y),
    BuiltInFun.MapData: lambda x: PlutusMap({k: v for k, v in x}),
    BuiltInFun.ListData: lambda x: PlutusList(x),
    BuiltInFun.IData: lambda x: PlutusInteger(x),
    BuiltInFun.BData: lambda x: PlutusByteString(x),
    BuiltInFun.UnConstrData: lambda x: (x.constructor, x.fields),
    BuiltInFun.UnMapData: lambda x: [(k, v) for k, v in x.value.items()],
    BuiltInFun.UnListData: lambda x: x.value,
    BuiltInFun.UnIData: lambda x: x.value,
    BuiltInFun.UnBData: lambda x: x.value,
    BuiltInFun.EqualsData: lambda x, y: x == y,
    BuiltInFun.MkPairData: lambda x, y: (x, y),
    BuiltInFun.MkNilData: lambda _: [],
    BuiltInFun.MkNilPairData: lambda _: [],
}


class AST:
    def eval(self, state: dict):
        raise NotImplementedError()

    def dumps(self) -> str:
        raise NotImplementedError()


@dataclass
class Program(AST):
    version: str
    term: AST

    def eval(self, state):
        return self.term.eval(state)

    def dumps(self) -> str:
        return f"(program {self.version} {self.term.dumps()})"


@dataclass
class Variable(AST):
    name: str

    def eval(self, state):
        try:
            return state[self.name]
        except KeyError as e:
            _LOGGER.error(
                f"Access to uninitialized variable {self.name} in {self.dumps()}"
            )
            raise e

    def dumps(self) -> str:
        return self.name


@dataclass
class Constant(AST):
    type: ConstantType
    value: Union[Tuple, List, bytes, int, bool, str]
    type_params: Optional[List[ConstantType]] = None

    def eval(self, state):
        if self.type == ConstantType.pair:
            return (
                Constant(self.type_params[0], self.value[0]).eval(state),
                Constant(self.type_params[1], self.value[1]).eval(state),
            )
        if self.type == ConstantType.list:
            return [ConstantType(self.type_params[0], v) for v in self.value]
        return ConstantEvalMap[self.type](self.value)

    def dumps(self) -> str:
        type_params_str = (
            "<" + ",".join(x.name for x in self.type_params) + ">"
            if self.type_params is not None
            else ""
        )
        return f"(con {self.type.name}{type_params_str} {self.value})"


@dataclass
class Lambda(AST):
    var_name: str
    term: AST

    def eval(self, state):
        def f(x):
            return self.term.eval(state | {self.var_name: x})

        return partial(f)

    def dumps(self) -> str:
        return f"(lam {self.var_name} {self.term.dumps()})"


@dataclass
class Delay(AST):
    term: AST

    def eval(self, state):
        def f():
            return self.term.eval(state)

        return f

    def dumps(self) -> str:
        return f"(delay {self.term.dumps()})"


@dataclass
class Force(AST):
    term: AST

    def eval(self, state):
        try:
            return self.term.eval(state)()
        except TypeError as e:
            _LOGGER.error(
                f"Trying to force an uncallable object, probably not delayed? in {self.dumps()}"
            )
            raise e

    def dumps(self) -> str:
        return f"(force {self.term.dumps()})"


@dataclass
class BuiltIn(AST):
    builtin: BuiltInFun

    def eval(self, state):
        return partial(BuiltInFunEvalMap[self.builtin])

    def dumps(self) -> str:
        return f"(builtin {self.builtin.name[0].lower()}{self.builtin.name[1:]})"


@dataclass
class Error(AST):
    def eval(self, state):
        raise RuntimeError(f"Execution called Error")

    def dumps(self) -> str:
        return f"(error)"


@dataclass
class Apply(AST):
    f: AST
    x: AST

    def eval(self, state):
        f = self.f.eval(state)
        x = self.x.eval(state)
        try:
            res = partial(f, x)
            # If this function has as many arguments bound as it takes, reduce i.e. call
            if len(f.args) == f.func.__code__.co_argcount:
                res = f()
            return res
        except AttributeError as e:
            _LOGGER.warning(f"Tried to apply value to non-function in {self.dumps()}")
            raise e

    def dumps(self) -> str:
        return f"[{self.f.dumps()} {self.x.dumps()}]"
