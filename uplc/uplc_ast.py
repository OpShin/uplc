import dataclasses
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import hashlib
from typing import List, Any, Dict

import cbor2
import frozendict


class Context:
    pass


@dataclass
class FrameApplyFun(Context):
    val: Any
    ctx: Context


@dataclass
class FrameApplyArg(Context):
    env: frozendict.frozendict
    term: "AST"
    ctx: Context


@dataclass
class FrameForce(Context):
    ctx: Context


@dataclass
class NoFrame(Context):
    pass


class Step:
    pass


@dataclass
class Return:
    context: Context
    value: Any


@dataclass
class Compute:
    ctx: Context
    env: frozendict.frozendict
    term: "AST"


@dataclass
class Done:
    term: "AST"


_LOGGER = logging.getLogger(__name__)


class AST:
    def eval(self, context: Context, state: frozendict.frozendict):
        raise NotImplementedError()

    def dumps(self) -> str:
        raise NotImplementedError()


@dataclass(frozen=True)
class Constant(AST):
    def eval(self, context, state):
        return Return(context, self)

    def dumps(self) -> str:
        return f"(con {self.typestring()} {self.valuestring()})"

    def valuestring(self):
        raise NotImplementedError()

    def typestring(self):
        raise NotImplementedError()


@dataclass(frozen=True)
class BuiltinUnit(Constant):
    def typestring(self):
        return "unit"

    def valuestring(self):
        return "()"


@dataclass(frozen=True)
class BuiltinBool(Constant):
    value: bool

    def typestring(self):
        return "bool"

    def valuestring(self):
        return str(self.value)


@dataclass(frozen=True)
class BuiltinInteger(Constant):
    value: int

    def typestring(self):
        return "integer"

    def valuestring(self):
        return str(self.value)

    def __add__(self, other):
        assert isinstance(
            other, BuiltinInteger
        ), "Trying to add two non-builtin-integers"
        return BuiltinInteger(self.value + other.value)

    def __sub__(self, other):
        assert isinstance(
            other, BuiltinInteger
        ), "Trying to sub two non-builtin-integers"
        return BuiltinInteger(self.value - other.value)

    def __mul__(self, other):
        assert isinstance(
            other, BuiltinInteger
        ), "Trying to mul two non-builtin-integers"
        return BuiltinInteger(self.value * other.value)

    def __floordiv__(self, other):
        assert isinstance(
            other, BuiltinInteger
        ), "Trying to floordiv two non-builtin-integers"
        return BuiltinInteger(self.value // other.value)

    def __mod__(self, other):
        assert isinstance(
            other, BuiltinInteger
        ), "Trying to mod two non-builtin-integers"
        return BuiltinInteger(self.value % other.value)

    def __eq__(self, other):
        assert isinstance(
            other, BuiltinInteger
        ), "Trying to eq two non-builtin-integers"
        return BuiltinBool(self.value == other.value)

    def __le__(self, other):
        assert isinstance(
            other, BuiltinInteger
        ), "Trying to le two non-builtin-integers"
        return BuiltinBool(self.value <= other.value)

    def __lt__(self, other):
        assert isinstance(
            other, BuiltinInteger
        ), "Trying to lt two non-builtin-integers"
        return BuiltinBool(self.value < other.value)


@dataclass(frozen=True)
class BuiltinByteString(Constant):
    value: bytes

    def typestring(self):
        return "bytestring"

    def valuestring(self):
        return f"#{self.value.hex()}"

    def __add__(self, other):
        assert isinstance(
            other, BuiltinByteString
        ), "Trying to add two non-builtin-bytestrings"
        return BuiltinByteString(self.value + other.value)

    def __len__(self):
        return BuiltinInteger(len(self.value))

    def __getitem__(self, item):
        # To implement slicing of bytestrings
        if isinstance(item, slice):
            assert isinstance(
                slice.start, BuiltinInteger
            ), "Trying to access a slice (start) with a non-builtin-integer"
            assert isinstance(
                slice.stop, BuiltinInteger
            ), "Trying to access a slice (stop) with a non-builtin-integer"
            assert slice.step is None, "Trying to access a slice with non-none step"
            return BuiltinByteString(self.value[slice.start.value : slice.stop.value])
        elif isinstance(item, BuiltinInteger):
            return BuiltinInteger(self.value[item.value])
        raise ValueError(f"Invalid slice {item}")

    def __eq__(self, other):
        assert isinstance(
            other, BuiltinByteString
        ), "Trying to eq two non-builtin-bytestrings"
        return BuiltinBool(self.value == other.value)

    def __le__(self, other):
        assert isinstance(
            other, BuiltinByteString
        ), "Trying to le two non-builtin-bytestrings"
        return BuiltinBool(self.value <= other.value)

    def __lt__(self, other):
        assert isinstance(
            other, BuiltinByteString
        ), "Trying to lt two non-builtin-bytestrings"
        return BuiltinBool(self.value < other.value)

    def decode(self, *args):
        return BuiltinString(self.value.decode("utf8"))


@dataclass(frozen=True)
class BuiltinString(Constant):
    value: str

    def typestring(self):
        return "string"

    def valuestring(self):
        return f'"{self.value}"'

    def __add__(self, other):
        assert isinstance(other, BuiltinString)
        return BuiltinString(self.value + other.value)

    def __eq__(self, other):
        assert isinstance(other, BuiltinString)
        return BuiltinBool(self.value == other.value)

    def encode(self, *args):
        return BuiltinByteString(self.value.encode())


@dataclass(frozen=True)
class BuiltinPair(Constant):
    l_value: Constant
    r_value: Constant

    def typestring(self):
        return f"pair<{self.l_value.typestring()}, {self.r_value.typestring()}>"

    def valuestring(self):
        return f"[{self.l_value.valuestring()}, {self.r_value.valuestring()}]"

    def __getitem__(self, item):
        if isinstance(item, int):
            if item == 0:
                return self.l_value
            elif item == 1:
                return self.r_value
        raise ValueError()


@dataclass(frozen=True, init=False)
class BuiltinList(Constant):
    values: List[Constant]
    # dirty hack to handle the type of empty lists
    sample_value: Constant

    def __init__(self, values, sample_value=None):
        object.__setattr__(self, "values", values)
        if not values:
            assert (
                sample_value is not None
            ), "Need to provide a sample value for empty lists to infer the type"
            object.__setattr__(self, "sample_value", sample_value)
        else:
            object.__setattr__(self, "sample_value", values[0])

    def typestring(self):
        return f"list<{self.sample_value.typestring()}>"

    def valuestring(self):
        return f"[{', '.join(v.valuestring() for v in self.values)}]"

    def __add__(self, other):
        assert isinstance(other, BuiltinList)
        assert (
            other.typestring() == self.typestring()
        ), f"Expected {self.typestring()} but got {other.typestring()}"
        return BuiltinList(self.values + other.values)

    def __eq__(self, other):
        assert isinstance(other, BuiltinList)
        return self.values == other.values

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.values[item]
        elif isinstance(item, slice):
            return BuiltinList(self.values[item], self.sample_value)


@dataclass(frozen=True)
class PlutusData(Constant):
    pass

    def typestring(self):
        return "data"

    def valuestring(self):
        return f"#{cbor2.dumps(self.to_cbor()).hex()}"

    def to_cbor(self) -> bytes:
        """Returns a CBOR encodable representation of this object"""
        raise NotImplementedError


@dataclass(frozen=True)
class PlutusAtomic(PlutusData):
    value: Any

    def to_cbor(self):
        return self.value


@dataclass(frozen=True, eq=True)
class PlutusInteger(PlutusAtomic):
    value: int


@dataclass(frozen=True, eq=True)
class PlutusByteString(PlutusAtomic):
    value: bytes


@dataclass(frozen=True, eq=True)
class PlutusList(PlutusData):
    value: List[PlutusData]

    def to_cbor(self):
        return [d.to_cbor() for d in self.value]


@dataclass(frozen=True, eq=True)
class PlutusMap(PlutusData):
    value: Dict[PlutusData, PlutusData]

    def to_cbor(self):
        return {k.to_cbor(): v.to_cbor() for k, v in self.value.items()}


@dataclass(frozen=True, eq=True)
class PlutusConstr(PlutusData):
    constructor: int
    fields: List[PlutusData]

    def to_cbor(self):
        return cbor2.CBORTag(self.constructor + 121, [f.to_cbor() for f in self.fields])


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


def data_from_json(j: Dict) -> PlutusData:
    if "bytes" in j:
        return PlutusByteString(bytes.fromhex(j["bytes"]))
    if "int" in j:
        return PlutusInteger(int(j["int"]))
    if "list" in j:
        return PlutusList(list(map(data_from_json, j["list"])))
    if "map" in j:
        return PlutusMap({d["k"]: d["v"] for d in j["map"]})
    if "constructor" in j and "fields" in j:
        return PlutusConstr(j["constructor"], j["fields"])
    raise NotImplementedError(f"Unknown datum representation {j}")


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


def _IfThenElse(i, t, e):
    assert isinstance(
        i, BuiltinBool
    ), "Trying to compute ifthenelse with non-builtin-bool"
    return t if i.value else e


def _ChooseData(_, d, v, w, x, y, z):
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
    BuiltInFun.ConsByteString: lambda x, y: BuiltinByteString(bytes([x])) + y,
    BuiltInFun.SliceByteString: lambda x, y, z: z[x : y + 1],
    BuiltInFun.LengthOfByteString: lambda x: len(x),
    BuiltInFun.IndexByteString: lambda x, y: x[y],
    BuiltInFun.EqualsByteString: lambda x, y: x == y,
    BuiltInFun.LessThanByteString: lambda x, y: x < y,
    BuiltInFun.LessThanEqualsByteString: lambda x, y: x <= y,
    BuiltInFun.Sha2_256: lambda x: BuiltinByteString(hashlib.sha256(x).digest()),
    BuiltInFun.Sha3_256: lambda x: BuiltinByteString(hashlib.sha3_256(x).digest()),
    BuiltInFun.Blake2b_256: lambda x: BuiltinByteString(hashlib.blake2b(x).digest()),
    # TODO how to emulate this?
    BuiltInFun.VerifySignature: lambda pk, m, s: BuiltinBool(True),
    BuiltInFun.AppendString: lambda x, y: x + y,
    BuiltInFun.EqualsString: lambda x, y: x == y,
    BuiltInFun.EncodeUtf8: lambda x: x.encode("utf8"),
    BuiltInFun.DecodeUtf8: lambda x: x.decode("utf8"),
    BuiltInFun.IfThenElse: _IfThenElse,
    BuiltInFun.ChooseUnit: lambda x, y: y,
    BuiltInFun.Trace: lambda x, y: print(x.value) or y,
    BuiltInFun.FstPair: lambda x: x[0],
    BuiltInFun.SndPair: lambda x: x[1],
    BuiltInFun.ChooseList: lambda l, x, y: x
    if l == BuiltinList([], l.sample_value)
    else y,
    BuiltInFun.MkCons: lambda e, l: BuiltinList([e]) + l,
    BuiltInFun.HeadList: lambda l: l[0],
    BuiltInFun.TailList: lambda l: l[1:],
    BuiltInFun.NullList: lambda l: BuiltinBool(l == BuiltinList([], l.sample_value)),
    BuiltInFun.ChooseData: _ChooseData,
    BuiltInFun.ConstrData: lambda x, y: PlutusConstr(x.value, y.values),
    BuiltInFun.MapData: lambda x: PlutusMap({p.l_value: p.r_value for p in x.values}),
    BuiltInFun.ListData: lambda x: PlutusList(x.values),
    BuiltInFun.IData: lambda x: PlutusInteger(x.value),
    BuiltInFun.BData: lambda x: PlutusByteString(x.value),
    BuiltInFun.UnConstrData: lambda x: BuiltinPair(
        BuiltinInteger(x.constructor), BuiltinList(x.fields, PlutusData())
    ),
    BuiltInFun.UnMapData: lambda x: BuiltinList(
        [BuiltinPair(k, v) for k, v in x.value.items()],
        BuiltinPair(PlutusData(), PlutusData()),
    ),
    BuiltInFun.UnListData: lambda x: BuiltinList(x.value, PlutusData()),
    BuiltInFun.UnIData: lambda x: BuiltinInteger(x.value),
    BuiltInFun.UnBData: lambda x: BuiltinByteString(x.value),
    BuiltInFun.EqualsData: lambda x, y: BuiltinBool(x == y),
    BuiltInFun.MkPairData: lambda x, y: BuiltinPair(x, y),
    BuiltInFun.MkNilData: lambda _: BuiltinList([], PlutusData()),
    BuiltInFun.MkNilPairData: lambda _: BuiltinList(
        [], BuiltinPair(PlutusData(), PlutusData())
    ),
}

BuiltInFunForceMap = defaultdict(int)
BuiltInFunForceMap.update(
    {
        BuiltInFun.IfThenElse: 1,
        BuiltInFun.ChooseUnit: 1,
        BuiltInFun.Trace: 1,
        BuiltInFun.FstPair: 2,
        BuiltInFun.SndPair: 2,
        BuiltInFun.ChooseList: 2,
        BuiltInFun.MkCons: 1,
        BuiltInFun.HeadList: 1,
        BuiltInFun.TailList: 1,
        BuiltInFun.NullList: 1,
        BuiltInFun.ChooseData: 1,
    }
)


@dataclass
class Program(AST):
    version: str
    term: AST

    def eval(self, context, state):
        return self.term.eval(context, state)

    def dumps(self) -> str:
        return f"(program {self.version} {self.term.dumps()})"


@dataclass
class Variable(AST):
    name: str

    def eval(self, context, state):
        try:
            return Return(context, state[self.name])
        except KeyError as e:
            _LOGGER.error(
                f"Access to uninitialized variable {self.name} in {self.dumps()}"
            )
            raise e

    def dumps(self) -> str:
        return self.name


@dataclass
class BoundStateLambda(AST):
    var_name: str
    term: AST
    state: frozendict.frozendict

    def eval(self, context, state):
        return Return(
            context,
            BoundStateLambda(self.var_name, self.term, self.state | state),
        )

    def dumps(self) -> str:
        s = f"(lam {self.var_name} {self.term.dumps()})"
        for k, v in reversed(self.state.items()):
            s = f"[(lam {k} {s}) {v}]"
        return s


@dataclass
class Lambda(BoundStateLambda):
    var_name: str
    term: AST
    state: frozendict.frozendict = dataclasses.field(
        default_factory=frozendict.frozendict
    )


@dataclass
class BoundStateDelay(AST):
    term: AST
    state: frozendict.frozendict

    def eval(self, context, state):
        return Return(context, BoundStateDelay(self.term, self.state | state))

    def dumps(self) -> str:
        return f"(delay {self.term.dumps()})"


@dataclass
class Delay(BoundStateDelay):
    term: AST
    state: frozendict.frozendict = dataclasses.field(
        default_factory=frozendict.frozendict
    )


@dataclass
class Force(AST):
    term: AST

    def eval(self, context, state):
        return Compute(
            FrameForce(
                context,
            ),
            state,
            self.term,
        )

    def dumps(self) -> str:
        return f"(force {self.term.dumps()})"


@dataclass
class ForcedBuiltIn(AST):
    builtin: BuiltInFun
    applied_forces: int
    bound_arguments: List[AST]

    def eval(self, context, state):
        return Return(context, self)

    def dumps(self) -> str:
        if self.applied_forces > 0:
            return Force(
                ForcedBuiltIn(
                    self.builtin, self.applied_forces - 1, self.bound_arguments
                )
            ).dumps()
        if len(self.bound_arguments):
            return Apply(
                ForcedBuiltIn(
                    self.builtin, self.applied_forces, self.bound_arguments[:-1]
                ),
                self.bound_arguments[-1],
            ).dumps()
        return f"(builtin {self.builtin.name[0].lower()}{self.builtin.name[1:]})"


@dataclass
class BuiltIn(ForcedBuiltIn):
    builtin: BuiltInFun
    applied_forces: int = dataclasses.field(default=0)
    bound_arguments: list = dataclasses.field(default_factory=lambda: [])


@dataclass
class Error(AST):
    def eval(self, context, state):
        raise RuntimeError(f"Execution called Error")

    def dumps(self) -> str:
        return f"(error)"


@dataclass
class Apply(AST):
    f: AST
    x: AST

    def eval(self, context, state):
        return Compute(
            FrameApplyArg(
                state,
                self.x,
                context,
            ),
            state,
            self.f,
        )

    def dumps(self) -> str:
        return f"[{self.f.dumps()} {self.x.dumps()}]"
