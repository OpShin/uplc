import dataclasses
import enum
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import hashlib
from typing import List, Any, Dict, Union

import cbor2
import frozendict
import frozenlist
import nacl.exceptions
from pycardano.crypto.bip32 import BIP32ED25519PublicKey
from nacl import bindings


class UPLCDialect(enum.Enum):
    Aiken = "aiken"
    Plutus = "plutus"


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
    _fields = []

    def eval(self, context: Context, state: frozendict.frozendict):
        raise NotImplementedError()

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        raise NotImplementedError()


@dataclass(frozen=True)
class Constant(AST):
    def eval(self, context, state):
        return Return(context, self)

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        return f"(con {self.typestring(dialect=dialect)} {self.valuestring(dialect=dialect)})"

    def valuestring(self, dialect=UPLCDialect.Aiken):
        raise NotImplementedError()

    def typestring(self, dialect=UPLCDialect.Aiken):
        raise NotImplementedError()


@dataclass(frozen=True)
class BuiltinUnit(Constant):
    def typestring(self, dialect=UPLCDialect.Aiken):
        return "unit"

    def valuestring(self, dialect=UPLCDialect.Aiken):
        return "()"


@dataclass(frozen=True)
class BuiltinBool(Constant):
    value: bool

    def typestring(self, dialect=UPLCDialect.Aiken):
        return "bool"

    def valuestring(self, dialect=UPLCDialect.Aiken):
        return str(self.value)


@dataclass(frozen=True)
class BuiltinInteger(Constant):
    value: int

    def typestring(self, dialect=UPLCDialect.Aiken):
        return "integer"

    def valuestring(self, dialect=UPLCDialect.Aiken):
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

    def __neg__(self):
        return BuiltinInteger(-self.value)

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

    def typestring(self, dialect=UPLCDialect.Aiken):
        return "bytestring"

    def valuestring(self, dialect=UPLCDialect.Aiken):
        return f"#{self.value.hex()}"

    def __add__(self, other):
        assert isinstance(
            other, BuiltinByteString
        ), "Trying to add two non-builtin-bytestrings"
        return BuiltinByteString(self.value + other.value)

    def __len__(self):
        return BuiltinInteger(len(self.value))

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

    def __getitem__(self, item):
        if isinstance(item, BuiltinInteger):
            assert 0 <= item.value <= len(self.value), "Out of bounds"
            return BuiltinInteger(self.value[item.value])
        raise NotImplementedError()

    def decode(self, *args):
        return BuiltinString(self.value.decode("utf8"))


@dataclass(frozen=True)
class BuiltinString(Constant):
    value: str

    def typestring(self, dialect=UPLCDialect.Aiken):
        return "string"

    def valuestring(self, dialect=UPLCDialect.Aiken):
        return json.dumps(self.value)

    def __add__(self, other):
        assert isinstance(other, BuiltinString), "Can only add two bytestrings"
        return BuiltinString(self.value + other.value)

    def __eq__(self, other):
        assert isinstance(
            other, BuiltinString
        ), "Can only compare two bytestrings for equality"
        return BuiltinBool(self.value == other.value)

    def encode(self, *args):
        return BuiltinByteString(self.value.encode())


@dataclass(frozen=True)
class BuiltinPair(Constant):
    l_value: Constant
    r_value: Constant

    def typestring(self, dialect=UPLCDialect.Aiken):
        if dialect == UPLCDialect.Aiken:
            return f"pair<{self.l_value.typestring(dialect=dialect)}, {self.r_value.typestring(dialect=dialect)}>"
        elif dialect == UPLCDialect.Plutus:
            return f"(pair {self.l_value.typestring(dialect=dialect)} {self.r_value.typestring(dialect=dialect)})"

    def valuestring(self, dialect=UPLCDialect.Aiken):
        if dialect == UPLCDialect.Aiken:
            return f"[{self.l_value.valuestring(dialect=dialect)}, {self.r_value.valuestring(dialect=dialect)}]"
        elif dialect == UPLCDialect.Plutus:
            return f"({self.l_value.valuestring(dialect=dialect)}, {self.r_value.valuestring(dialect=dialect)})"

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

    def typestring(self, dialect=UPLCDialect.Aiken):
        if dialect == UPLCDialect.Aiken:
            return f"list<{self.sample_value.typestring(dialect=dialect)}>"
        elif dialect == UPLCDialect.Plutus:
            return f"(list {self.sample_value.typestring(dialect=dialect)})"

    def valuestring(self, dialect=UPLCDialect.Aiken):
        return f"[{', '.join(v.valuestring(dialect=dialect) for v in self.values)}]"

    def __add__(self, other):
        assert isinstance(other, BuiltinList), "Can only append two lists"
        assert (
            other.typestring() == self.typestring()
        ), f"Expected {self.typestring()} but got {other.typestring()}"
        return BuiltinList(self.values + other.values)

    def __eq__(self, other):
        assert isinstance(other, BuiltinList), "Can only compare two lists"
        return self.values == other.values

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.values[item]
        elif isinstance(item, slice):
            return BuiltinList(self.values[item], self.sample_value)


@dataclass(frozen=True)
class PlutusData(Constant):
    pass

    def typestring(self, dialect=UPLCDialect.Aiken):
        return "data"

    def valuestring(self, dialect=UPLCDialect.Aiken):
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
    value: Union[List[PlutusData], frozenlist.FrozenList]

    def to_cbor(self):
        return [d.to_cbor() for d in self.value]


@dataclass(frozen=True, eq=True)
class PlutusMap(PlutusData):
    value: Union[Dict[PlutusData, PlutusData], frozendict.frozendict]

    def to_cbor(self):
        return {k.to_cbor(): v.to_cbor() for k, v in self.value.items()}


@dataclass(frozen=True, eq=True)
class PlutusConstr(PlutusData):
    constructor: int
    fields: Union[List[PlutusData], frozenlist.FrozenList]

    def to_cbor(self):
        if 0 <= self.constructor < 7:
            return cbor2.CBORTag(
                self.constructor + 121, [f.to_cbor() for f in self.fields]
            )
        elif 7 <= self.constructor < 128:
            return cbor2.CBORTag(
                (self.constructor - 7) + 1280, [f.to_cbor() for f in self.fields]
            )
        else:
            return cbor2.CBORTag(
                102, [self.constructor, [f.to_cbor() for f in self.fields]]
            )


def data_from_cbortag(cbor) -> PlutusData:
    if isinstance(cbor, cbor2.CBORTag):
        if 121 <= cbor.tag <= 121 + 6:
            constructor = cbor.tag - 121
            fields = cbor.value
        elif 1280 <= cbor.tag <= 1280 + (127 - 7):
            constructor = cbor.tag - 1280 + 7
            fields = cbor.value
        elif cbor.tag == 102:
            constructor, fields = cbor.value
        else:
            raise ValueError(f"Invalid cbor with tag {cbor.tag}")
        fields = frozenlist.FrozenList(list(map(data_from_cbortag, fields)))
        fields.freeze()
        return PlutusConstr(constructor, fields)
    if isinstance(cbor, int):
        return PlutusInteger(cbor)
    if isinstance(cbor, bytes):
        return PlutusByteString(cbor)
    if isinstance(cbor, list):
        entries = frozenlist.FrozenList(list(map(data_from_cbortag, cbor)))
        entries.freeze()
        return PlutusList(entries)
    if isinstance(cbor, dict):
        return PlutusMap(
            frozendict.frozendict(
                {data_from_cbortag(k): data_from_cbortag(v) for k, v in cbor.items()}
            )
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
    VerifyEd25519Signature = auto()
    VerifyEcdsaSecp256k1Signature = auto()
    VerifySchnorrSecp256k1Signature = auto()
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


def _ChooseData(d, v, w, x, y, z):
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


def verify_ed25519(pk: BuiltinByteString, m: BuiltinByteString, s: BuiltinByteString):
    assert len(pk.value) == 32, "Ed25519S PublicKey should be 32 bytes"
    assert len(s.value) == 64, "Ed25519S Signature should be 64 bytes"
    try:
        BIP32ED25519PublicKey(pk.value[:32], pk.value[32:]).verify(s.value, m.value)
        return BuiltinBool(True)
    except nacl.exceptions.BadSignatureError:
        return BuiltinBool(False)


def verify_ecdsa_secp256k1(
    pk: BuiltinByteString, m: BuiltinByteString, s: BuiltinByteString
):
    raise NotImplementedError("EcdsaSecp256k1 not implemented yet")


def verify_schnorr_secp256k1(
    pk: BuiltinByteString, m: BuiltinByteString, s: BuiltinByteString
):
    raise NotImplementedError("SchnorrSecp256k1 not implemented yet")


def _quot(a, b):
    return a // b if (a * b > BuiltinInteger(0)).value else (a + (-a % b)) // b


def _TailList(xs: BuiltinList):
    if xs.values == []:
        raise RuntimeError("Can not tailList on an empty list")
    return xs[1:]


BuiltInFunEvalMap = {
    BuiltInFun.AddInteger: lambda x, y: x + y,
    BuiltInFun.SubtractInteger: lambda x, y: x - y,
    BuiltInFun.MultiplyInteger: lambda x, y: x * y,
    # round towards -inf
    BuiltInFun.DivideInteger: lambda x, y: x // y,
    # round towards 0
    BuiltInFun.QuotientInteger: _quot,
    # (x `quot` y)*y + (x `rem` y) == x
    BuiltInFun.RemainderInteger: lambda x, y: x - _quot(x, y) * y,
    # (x `div` y)*y + (x `mod` y) == x
    BuiltInFun.ModInteger: lambda x, y: x % y,
    BuiltInFun.EqualsInteger: lambda x, y: x == y,
    BuiltInFun.LessThanInteger: lambda x, y: x < y,
    BuiltInFun.LessThanEqualsInteger: lambda x, y: x <= y,
    BuiltInFun.AppendByteString: lambda x, y: x + y,
    BuiltInFun.ConsByteString: lambda x, y: BuiltinByteString(bytes([x.value])) + y,
    BuiltInFun.SliceByteString: lambda x, y, z: BuiltinByteString(
        z.value[max(x.value, 0) :][: max(y.value, 0)]
    ),
    BuiltInFun.LengthOfByteString: lambda x: BuiltinInteger(len(x.value)),
    BuiltInFun.IndexByteString: lambda x, y: x[y],
    BuiltInFun.EqualsByteString: lambda x, y: x == y,
    BuiltInFun.LessThanByteString: lambda x, y: x < y,
    BuiltInFun.LessThanEqualsByteString: lambda x, y: x <= y,
    BuiltInFun.Sha2_256: lambda x: BuiltinByteString(hashlib.sha256(x.value).digest()),
    BuiltInFun.Sha3_256: lambda x: BuiltinByteString(
        hashlib.sha3_256(x.value).digest()
    ),
    BuiltInFun.Blake2b_256: lambda x: BuiltinByteString(
        hashlib.blake2b(x.value, digest_size=32).digest()
    ),
    BuiltInFun.VerifySignature: verify_ed25519,
    BuiltInFun.VerifyEd25519Signature: verify_ed25519,
    # TODO how to emulate this?
    BuiltInFun.VerifyEcdsaSecp256k1Signature: lambda pk, m, s: BuiltinBool(True),
    BuiltInFun.VerifySchnorrSecp256k1Signature: lambda pk, m, s: BuiltinBool(True),
    BuiltInFun.AppendString: lambda x, y: x + y,
    BuiltInFun.EqualsString: lambda x, y: x == y,
    BuiltInFun.EncodeUtf8: lambda x: BuiltinByteString(x.value.encode("utf8")),
    BuiltInFun.DecodeUtf8: lambda x: BuiltinString(x.value.decode("utf8")),
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
    BuiltInFun.TailList: _TailList,
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
    _fields = ["term"]

    def eval(self, context, state):
        return self.term.eval(context, state)

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        return f"(program {self.version} {self.term.dumps(dialect=dialect)})"


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

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        return self.name


@dataclass
class BoundStateLambda(AST):
    var_name: str
    term: AST
    state: frozendict.frozendict
    _fields = ["term"]

    def eval(self, context, state):
        return Return(
            context,
            BoundStateLambda(self.var_name, self.term, self.state | state),
        )

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        s = f"(lam {self.var_name} {self.term.dumps(dialect=dialect)})"
        for k, v in reversed(self.state.items()):
            s = f"[(lam {k} {s}) {v.dumps(dialect=dialect)}]"
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
    _fields = ["term"]

    def eval(self, context, state):
        return Return(context, BoundStateDelay(self.term, self.state | state))

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        return f"(delay {self.term.dumps(dialect=dialect)})"


@dataclass
class Delay(BoundStateDelay):
    term: AST
    state: frozendict.frozendict = dataclasses.field(
        default_factory=frozendict.frozendict
    )


@dataclass
class Force(AST):
    term: AST
    _fields = ["term"]

    def eval(self, context, state):
        return Compute(
            FrameForce(
                context,
            ),
            state,
            self.term,
        )

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        return f"(force {self.term.dumps(dialect=dialect)})"


@dataclass
class ForcedBuiltIn(AST):
    builtin: BuiltInFun
    applied_forces: int
    bound_arguments: List[AST]

    def eval(self, context, state):
        return Return(context, self)

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        if len(self.bound_arguments):
            return Apply(
                ForcedBuiltIn(
                    self.builtin, self.applied_forces, self.bound_arguments[:-1]
                ),
                self.bound_arguments[-1],
            ).dumps(dialect=dialect)
        if self.applied_forces > 0:
            return Force(
                ForcedBuiltIn(
                    self.builtin, self.applied_forces - 1, self.bound_arguments
                )
            ).dumps(dialect=dialect)
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

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        return f"(error)"


@dataclass
class Apply(AST):
    f: AST
    x: AST
    _fields = ["f", "x"]

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

    def dumps(self, dialect=UPLCDialect.Aiken) -> str:
        return f"[{self.f.dumps(dialect=dialect)} {self.x.dumps(dialect=dialect)}]"
