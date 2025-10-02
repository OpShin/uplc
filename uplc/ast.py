import dataclasses
import enum
import json
import logging
import math
import typing
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import hashlib
from typing import List, Any, Dict, Union

import cbor2
import frozendict
from frozenlist2 import frozenlist
import nacl.exceptions
from _cbor2 import CBOREncoder
from pycardano.crypto.bip32 import BIP32ED25519PublicKey
from pycardano.serialization import IndefiniteFrozenList, IndefiniteList

try:
    import pysecp256k1
except ImportError:
    pysecp256k1 = None

try:
    import pysecp256k1.extrakeys
    import pysecp256k1.schnorrsig as schnorrsig
except (RuntimeError, ImportError):
    schnorrsig = None


class UPLCDialect(enum.Enum):
    LegacyAiken = "legacy-aiken"
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

    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        raise NotImplementedError()

    def ex_mem(self) -> int:
        """The memory consumption of this element"""
        raise NotImplementedError()


@dataclass(frozen=True)
class Constant(AST):
    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        return f"(con {self.typestring(dialect=dialect)} {self.valuestring(dialect=dialect)})"

    def valuestring(self, dialect=UPLCDialect.Plutus):
        raise NotImplementedError()

    def typestring(self, dialect=UPLCDialect.Plutus):
        raise NotImplementedError()


@dataclass(frozen=True)
class BuiltinUnit(Constant):
    def typestring(self, dialect=UPLCDialect.Plutus):
        return "unit"

    def valuestring(self, dialect=UPLCDialect.Plutus):
        return "()"

    def ex_mem(self) -> int:
        return 1


@dataclass(frozen=True)
class BuiltinBool(Constant):
    value: bool

    def typestring(self, dialect=UPLCDialect.Plutus):
        return "bool"

    def valuestring(self, dialect=UPLCDialect.Plutus):
        return str(self.value)

    def ex_mem(self) -> int:
        return 1


@dataclass(frozen=True)
class BuiltinInteger(Constant):
    value: int

    def typestring(self, dialect=UPLCDialect.Plutus):
        return "integer"

    def valuestring(self, dialect=UPLCDialect.Plutus):
        return str(self.value)

    def ex_mem(self) -> int:
        if self.value == 0:
            return 1
        return (math.ceil(math.log2(abs(self.value))) // 64) + 1

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

    def typestring(self, dialect=UPLCDialect.Plutus):
        return "bytestring"

    def valuestring(self, dialect=UPLCDialect.Plutus):
        return f"#{self.value.hex()}"

    def ex_mem(self) -> int:
        if not self.value:
            return 1
        return ((len(self.value) - 1) // 8) + 1

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

    def typestring(self, dialect=UPLCDialect.Plutus):
        return "string"

    def valuestring(self, dialect=UPLCDialect.Plutus):
        escaped = self.value.encode("unicode_escape").decode()
        if '"' in escaped:
            escaped = escaped.replace('"', '\\"')
        return f'"{escaped}"'

    def ex_mem(self) -> int:
        return len(self.value)

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

    def typestring(self, dialect=UPLCDialect.Plutus):
        if dialect == UPLCDialect.LegacyAiken:
            return f"pair<{self.l_value.typestring(dialect=dialect)}, {self.r_value.typestring(dialect=dialect)}>"
        return f"(pair {self.l_value.typestring(dialect=dialect)} {self.r_value.typestring(dialect=dialect)})"

    def valuestring(self, dialect=UPLCDialect.Plutus):
        if dialect == UPLCDialect.LegacyAiken:
            return f"[{self.l_value.valuestring(dialect=dialect)}, {self.r_value.valuestring(dialect=dialect)}]"
        return f"({self.l_value.valuestring(dialect=dialect)}, {self.r_value.valuestring(dialect=dialect)})"

    def ex_mem(self) -> int:
        return self.l_value.ex_mem() + self.r_value.ex_mem()

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
        object.__setattr__(self, "values", frozenlist(values))
        if not values:
            assert (
                sample_value is not None
            ), "Need to provide a sample value for empty lists to infer the type"
            object.__setattr__(self, "sample_value", sample_value)
        else:
            object.__setattr__(self, "sample_value", values[0])

    def typestring(self, dialect=UPLCDialect.Plutus):
        if dialect == UPLCDialect.LegacyAiken:
            return f"list<{self.sample_value.typestring(dialect=dialect)}>"
        return f"(list {self.sample_value.typestring(dialect=dialect)})"

    def valuestring(self, dialect=UPLCDialect.Plutus):
        return f"[{', '.join(v.valuestring(dialect=dialect) for v in self.values)}]"

    def ex_mem(self) -> int:
        return sum(v.ex_mem() for v in self.values)

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
    def dumps(self, dialect=UPLCDialect.Plutus):
        internal_string = self.valuestring(dialect=dialect)
        if dialect == UPLCDialect.Plutus:
            internal_string = f"({internal_string})"
        return f"(con data {internal_string})"

    def valuestring(self, dialect=UPLCDialect.Plutus) -> str:
        if dialect == UPLCDialect.LegacyAiken:
            return f"#{plutus_cbor_dumps(self).hex()}"
        return f"{self.plutus_valuestring()}"

    def plutus_valuestring(self):
        raise NotImplementedError

    def typestring(self, dialect=UPLCDialect.Plutus):
        return "data"

    def to_cbor(self) -> Any:
        """Returns a CBOR encodable representation of this object"""
        raise NotImplementedError

    def to_json(self) -> dict:
        """Returns a JSON encodable representation of this object"""
        raise NotImplementedError

    def ex_mem(self) -> int:
        return 4 + self.d_ex_mem()

    def d_ex_mem(self) -> int:
        """Ex-Mem without the constant 4 cost for deconstruction"""
        raise NotImplementedError()


@dataclass(frozen=True)
class PlutusAtomic(PlutusData):
    value: Any

    def to_cbor(self):
        return self


@dataclass(frozen=True, eq=True)
class PlutusInteger(PlutusAtomic):
    value: int

    def to_json(self):
        return {"int": self.value}

    def plutus_valuestring(self):
        return f"I {self.value}"

    def d_ex_mem(self) -> int:
        return BuiltinInteger(self.value).ex_mem()


@dataclass(frozen=True, eq=True)
class PlutusByteString(PlutusAtomic):
    value: bytes

    def to_json(self):
        return {"bytes": self.value.hex()}

    def plutus_valuestring(self):
        return f"B #{self.value.hex()}"

    def d_ex_mem(self) -> int:
        return BuiltinByteString(self.value).ex_mem()


@dataclass(frozen=True, eq=True)
class PlutusList(PlutusData):
    value: Union[List[PlutusData], frozenlist]

    def __post_init__(self):
        object.__setattr__(self, "value", frozenlist(self.value))

    def to_cbor(self):
        return [d.to_cbor() for d in self.value]

    def to_json(self):
        return {"list": [v.to_json() for v in self.value]}

    def plutus_valuestring(self):
        return f"List [{', '.join(x.plutus_valuestring() for x in self.value)}]"

    def d_ex_mem(self) -> int:
        return sum(v.ex_mem() for v in self.value)


@dataclass(frozen=True, eq=True)
class PlutusMap(PlutusData):
    value: Union[Dict[PlutusData, PlutusData], frozendict.frozendict]

    def __post_init__(self):
        frozen_value = frozendict.frozendict(self.value)
        object.__setattr__(self, "value", frozen_value)

    def to_cbor(self):
        return {k.to_cbor(): v.to_cbor() for k, v in self.value.items()}

    def to_json(self):
        return {
            "map": [{"k": k.to_json(), "v": v.to_json()} for k, v in self.value.items()]
        }

    def plutus_valuestring(self):
        recursive_val_strings = (
            f"({x.plutus_valuestring()}, {y.plutus_valuestring()})"
            for x, y in self.value.items()
        )
        return f"Map [{', '.join(recursive_val_strings)}]"

    def d_ex_mem(self) -> int:
        return sum(v.ex_mem() + k.ex_mem() for k, v in self.value.items())


@dataclass(frozen=True, eq=True)
class PlutusConstr(PlutusData):
    constructor: int
    fields: Union[List[PlutusData], frozenlist]

    def __post_init__(self):
        object.__setattr__(self, "fields", frozenlist(self.fields))

    def to_cbor(self):
        if self.fields:
            fields = IndefiniteFrozenList([f.to_cbor() for f in self.fields])
            fields.freeze()
        else:
            fields = []
        if 0 <= self.constructor < 7:
            return cbor2.CBORTag(self.constructor + 121, fields)
        elif 7 <= self.constructor < 128:
            return cbor2.CBORTag((self.constructor - 7) + 1280, fields)
        else:
            return cbor2.CBORTag(102, [self.constructor, fields])

    def to_json(self):
        return {
            "constructor": self.constructor,
            "fields": [v.to_json() for v in self.fields],
        }

    def plutus_valuestring(self):
        return f"Constr {self.constructor} [{', '.join(x.plutus_valuestring() for x in self.fields)}]"

    def d_ex_mem(self) -> int:
        return sum(v.ex_mem() for v in self.fields)


def _int_to_bytes(x: int):
    return x.to_bytes((x.bit_length() + 7) // 8, byteorder="big")


def default_encoder(encoder: CBOREncoder, value: Union[PlutusData, IndefiniteList]):
    """A fallback function that encodes PlutusData objects"""
    if isinstance(value, IndefiniteList):
        # Currently, cbor2 doesn't support indefinite list, therefore we need special
        # handling here to explicitly write header (b'\x9f'), each body item, and footer (b'\xff') to
        # the output bytestring.
        encoder.write(b"\x9f")
        for item in value:
            encoder.encode(item)
        encoder.write(b"\xff")
        return
    if not isinstance(value, PlutusData):
        raise NotImplementedError(f"Can not encode type {type(value)}")
    value = value.to_cbor()
    if isinstance(value, PlutusByteString):
        # the encoder can not handle indefinite length arrays, but the plutus standard
        # requires encoding bytes as indefinite byte sequence where each chunk is at most 64 bytes long
        byts = value.value
    elif isinstance(value, PlutusInteger):
        if -(2**64) < value.value < 2**64 - 1:
            encoder.encode(value.value)
            return
        if value.value >= 0:
            byts = _int_to_bytes(value.value)
            encoder.write(b"\xc2")
        else:
            byts = _int_to_bytes(-value.value - 1)
            encoder.write(b"\xc3")
    else:
        encoder.encode(value)
        return
    if len(byts) < 64:
        encoder.encode(byts)
        return
    encoder.write(b"\x5f")
    max_chunk_len = 64
    n = len(byts)
    pos = 0
    while pos < n:
        n_chunk = min(n - pos, max_chunk_len)
        chunk = byts[pos : pos + n_chunk]
        encoder.encode(chunk)
        pos += n_chunk
    encoder.write(b"\xff")


def plutus_cbor_dumps(x):
    return cbor2.dumps(x, default=default_encoder)


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
        fields = frozenlist(list(map(data_from_cbortag, fields)))
        return PlutusConstr(constructor, fields)
    if isinstance(cbor, int):
        return PlutusInteger(cbor)
    if isinstance(cbor, bytes):
        return PlutusByteString(cbor)
    if isinstance(cbor, list):
        entries = frozenlist(list(map(data_from_cbortag, cbor)))
        return PlutusList(entries)
    if isinstance(cbor, dict):
        return PlutusMap(
            frozendict.frozendict(
                {data_from_cbortag(k): data_from_cbortag(v) for k, v in cbor.items()}
            )
        )
    raise NotImplementedError(f"Unknown cbor type notation in {cbor}")


def data_from_cbor(cbor: bytes) -> PlutusData:
    raw_datum = cbor2.loads(cbor)
    return data_from_cbortag(raw_datum)


def data_from_json_dict(d: dict) -> PlutusData:
    if not isinstance(d, dict):
        raise ValueError("Expected a dictionary")
    if "constructor" in d:
        assert isinstance(
            d["constructor"], int
        ), "Expected integer in 'constructor' field"
        assert isinstance(d["fields"], list), "Expected a list in 'fields' field"
        fields = frozenlist([data_from_json_dict(f) for f in d["fields"]])
        return PlutusConstr(d["constructor"], fields)
    if "int" in d:
        assert (
            isinstance(d["int"], int) and round(d["int"]) == d["int"]
        ), "Expected integer in 'int' field"
        return PlutusInteger(d["int"])
    if "bytes" in d:
        assert isinstance(d["bytes"], str), "Expected bytes in 'bytes' field"
        try:
            return PlutusByteString(bytes.fromhex(d["bytes"]))
        except ValueError as e:
            raise ValueError("Invalid hex string in 'bytes' field") from e
    if "list" in d:
        assert isinstance(d["list"], list), "Expected a list in 'list' field"
        entries = frozenlist(list(map(data_from_json_dict, d["list"])))
        return PlutusList(entries)
    if "map" in d:
        assert isinstance(
            d["map"], list
        ), "Expected a list in 'map' field (entries are dicts with field 'k' and 'v')"
        return PlutusMap(
            frozendict.frozendict(
                {
                    data_from_json_dict(m["k"]): data_from_json_dict(m["v"])
                    for m in d["map"]
                }
            )
        )
    raise NotImplementedError(f"Unknown JSON notation in {d}")


def data_from_json(json_string: str) -> PlutusData:
    try:
        raw_datum = json.loads(json_string)
        return data_from_json_dict(raw_datum)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON") from e
    except KeyError as e:
        raise ValueError(f"Invalid PlutusData JSON, expected key {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid PlutusData JSON, {e}") from e


def plutus_json_dumps(x: PlutusData):
    return json.dumps(x.to_json())


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
# NOTE it is crucial that the values matches table C.3 in the plutus core spec
# https://ci.iog.io/build/1230997/download/1/plutus-core-specification.pdf
class BuiltInFun(Enum):
    # Integers
    AddInteger = 0
    SubtractInteger = 1
    MultiplyInteger = 2
    DivideInteger = 3
    QuotientInteger = 4
    RemainderInteger = 5
    ModInteger = 6
    EqualsInteger = 7
    LessThanInteger = 8
    LessThanEqualsInteger = 9
    # Bytestrings
    AppendByteString = 10
    ConsByteString = 11
    SliceByteString = 12
    LengthOfByteString = 13
    IndexByteString = 14
    EqualsByteString = 15
    LessThanByteString = 16
    LessThanEqualsByteString = 17
    # Cryptography and hashes
    Sha2_256 = 18
    Sha3_256 = 19
    Blake2b_256 = 20
    # Keccak_256 = 71
    # Blake2b_224 = 72
    VerifyEd25519Signature = 21  # formerly verifySignature
    VerifyEcdsaSecp256k1Signature = 52
    VerifySchnorrSecp256k1Signature = 53
    # Strings
    AppendString = 22
    EqualsString = 23
    EncodeUtf8 = 24
    DecodeUtf8 = 25
    # Bool
    IfThenElse = 26
    # Unit
    ChooseUnit = 27
    # Tracing
    Trace = 28
    # Pairs
    FstPair = 29
    SndPair = 30
    # Lists
    ChooseList = 31
    MkCons = 32
    HeadList = 33
    TailList = 34
    NullList = 35
    # Data
    ChooseData = 36
    ConstrData = 37
    MapData = 38
    ListData = 39
    IData = 40
    BData = 41
    UnConstrData = 42
    UnMapData = 43
    UnListData = 44
    UnIData = 45
    UnBData = 46
    EqualsData = 47
    SerialiseData = 51
    # Misc monomorphized constructors
    MkPairData = 48
    MkNilData = 49
    MkNilPairData = 50
    # BLS Builtins
    # Bls12_381_G1_Add = 54
    # Bls12_381_G1_Neg = 55
    # Bls12_381_G1_ScalarMul = 56
    # Bls12_381_G1_Equal = 57
    # Bls12_381_G1_Compress = 58
    # Bls12_381_G1_Uncompress = 59
    # Bls12_381_G1_HashToGroup = 60
    # Bls12_381_G2_Add = 61
    # Bls12_381_G2_Neg = 62
    # Bls12_381_G2_ScalarMul = 63
    # Bls12_381_G2_Equal = 64
    # Bls12_381_G2_Compress = 65
    # Bls12_381_G2_Uncompress = 66
    # Bls12_381_G2_HashToGroup = 67
    # Bls12_381_MillerLoop = 68
    # Bls12_381_MulMlResult = 69
    # Bls12_381_FinalVerify = 70


def typechecked(*typs):
    def typecheck_decorator(fun):
        if len(typs) == 1:

            def wrapped_fun(a1):
                for i, (arg, typ) in enumerate(zip([a1], typs)):
                    assert isinstance(
                        arg, typ
                    ), f"Argument {i} has invalid type, expected type {typ} got {type(arg)} ({arg})"
                return fun(a1)

        elif len(typs) == 2:

            def wrapped_fun(a1, a2):
                for i, (arg, typ) in enumerate(zip([a1, a2], typs)):
                    assert isinstance(
                        arg, typ
                    ), f"Argument {i} has invalid type, expected type {typ} got {type(arg)} ({arg})"
                return fun(a1, a2)

        elif len(typs) == 3:

            def wrapped_fun(a1, a2, a3):
                for i, (arg, typ) in enumerate(zip([a1, a2, a3], typs)):
                    assert isinstance(
                        arg, typ
                    ), f"Argument {i} has invalid type, expected type {typ} got {type(arg)} ({arg})"
                return fun(a1, a2, a3)

        elif len(typs) == 4:

            def wrapped_fun(a1, a2, a3, a4):
                for i, (arg, typ) in enumerate(zip([a1, a2, a3, a4], typs)):
                    assert isinstance(
                        arg, typ
                    ), f"Argument {i} has invalid type, expected type {typ} got {type(arg)} ({arg})"
                return fun(a1, a2, a3, a4)

        elif len(typs) == 5:

            def wrapped_fun(a1, a2, a3, a4, a5):
                for i, (arg, typ) in enumerate(zip([a1, a2, a3, a4, a5], typs)):
                    assert isinstance(
                        arg, typ
                    ), f"Argument {i} has invalid type, expected type {typ} got {type(arg)} ({arg})"
                return fun(a1, a2, a3, a4, a5)

        elif len(typs) == 6:

            def wrapped_fun(a1, a2, a3, a4, a5, a6):
                for i, (arg, typ) in enumerate(zip([a1, a2, a3, a4, a5, a6], typs)):
                    assert isinstance(
                        arg, typ
                    ), f"Argument {i} has invalid type, expected type {typ} got {type(arg)} ({arg})"
                return fun(a1, a2, a3, a4, a5, a6)

        else:
            raise NotImplementedError("Too many arguments")
        return wrapped_fun

    return typecheck_decorator


@typechecked(BuiltinBool, AST, AST)
def _IfThenElse(i, t, e):
    return t if i.value else e


@typechecked(PlutusData, AST, AST, AST, AST, AST)
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


@typechecked(BuiltinByteString, BuiltinByteString, BuiltinByteString)
def verify_ed25519(pk: BuiltinByteString, m: BuiltinByteString, s: BuiltinByteString):
    assert len(pk.value) == 32, "Ed25519S PublicKey should be 32 bytes"
    assert len(s.value) == 64, "Ed25519S Signature should be 64 bytes"
    try:
        BIP32ED25519PublicKey(pk.value[:32], pk.value[32:]).verify(s.value, m.value)
        return BuiltinBool(True)
    except nacl.exceptions.BadSignatureError:
        return BuiltinBool(False)


@typechecked(BuiltinByteString, BuiltinByteString, BuiltinByteString)
def verify_ecdsa_secp256k1(
    pk: BuiltinByteString, m: BuiltinByteString, s: BuiltinByteString
):
    # TODO length checks
    if pysecp256k1 is None:
        _LOGGER.error("libsecp256k1 is not installed. ECDSA verification will not work")
        raise RuntimeError("ECDSA not supported")
    pubkey = pysecp256k1.ec_pubkey_parse(pk.value)
    sig = pysecp256k1.ecdsa_signature_parse_compact(s.value)
    res = pysecp256k1.ecdsa_verify(sig, pubkey, m.value)
    return BuiltinBool(res)


@typechecked(BuiltinByteString, BuiltinByteString, BuiltinByteString)
def verify_schnorr_secp256k1(
    pk: BuiltinByteString, m: BuiltinByteString, s: BuiltinByteString
):
    # TODO length checks
    if pysecp256k1 is None:
        _LOGGER.error("libsecp256k1 is not installed. ECDSA verification will not work")
        raise RuntimeError("ECDSA not supported")
    if schnorrsig is None:
        _LOGGER.error(
            "libsecp256k1 is installed without schnorr support. Schnorr verification will not work"
        )
        raise RuntimeError("Schnorr not supported")
    pubkey = pysecp256k1.extrakeys.xonly_pubkey_parse(pk.value)
    res = schnorrsig.schnorrsig_verify(s.value, m.value, pubkey)
    return BuiltinBool(res)


def _quot(a, b):
    return a // b if (a * b > BuiltinInteger(0)).value else (a + (-a % b)) // b


@typechecked(BuiltinList)
def _TailList(xs: BuiltinList):
    if xs.values == []:
        raise RuntimeError("Can not tailList on an empty list")
    return xs[1:]


def _MkCons(x, xs):
    assert isinstance(xs, BuiltinList), "Can only cons onto a list"
    assert isinstance(x, xs.sample_value.__class__) or (
        isinstance(x, PlutusData) and (isinstance(xs.sample_value, PlutusData))
    ), "Can only cons elements of the same type"
    return BuiltinList([x]) + xs


def _MapData(x):
    assert isinstance(x, BuiltinList), "Can only map over a list"
    assert isinstance(x.sample_value, BuiltinPair), "Can only map over a list of pairs"
    return PlutusMap({p.l_value: p.r_value for p in x.values})


two_ints = typechecked(BuiltinInteger, BuiltinInteger)
two_bytestrings = typechecked(BuiltinByteString, BuiltinByteString)
two_strings = typechecked(BuiltinString, BuiltinString)
single_bytestring = typechecked(BuiltinByteString)
single_data = typechecked(PlutusData)
single_data_int = typechecked(PlutusInteger)
single_data_bytes = typechecked(PlutusByteString)
single_data_list = typechecked(PlutusList)
single_data_map = typechecked(PlutusMap)
single_data_constr = typechecked(PlutusConstr)

BuiltInFunEvalMap = {
    BuiltInFun.AddInteger: two_ints(lambda x, y: x + y),
    BuiltInFun.SubtractInteger: two_ints(lambda x, y: x - y),
    BuiltInFun.MultiplyInteger: two_ints(lambda x, y: x * y),
    # round towards -inf
    BuiltInFun.DivideInteger: two_ints(lambda x, y: x // y),
    # round towards 0
    BuiltInFun.QuotientInteger: two_ints(_quot),
    # (x `quot` y)*y + (x `rem` y) == x
    BuiltInFun.RemainderInteger: two_ints(lambda x, y: x - _quot(x, y) * y),
    # (x `div` y)*y + (x `mod` y) == x
    BuiltInFun.ModInteger: two_ints(lambda x, y: x % y),
    BuiltInFun.EqualsInteger: two_ints(lambda x, y: x == y),
    BuiltInFun.LessThanInteger: two_ints(lambda x, y: x < y),
    BuiltInFun.LessThanEqualsInteger: two_ints(lambda x, y: x <= y),
    BuiltInFun.AppendByteString: two_bytestrings(lambda x, y: x + y),
    BuiltInFun.ConsByteString: typechecked(BuiltinInteger, BuiltinByteString)(
        lambda x, y: BuiltinByteString(bytes([x.value])) + y
    ),
    BuiltInFun.SliceByteString: typechecked(
        BuiltinInteger, BuiltinInteger, BuiltinByteString
    )(lambda x, y, z: BuiltinByteString(z.value[max(x.value, 0) :][: max(y.value, 0)])),
    BuiltInFun.LengthOfByteString: single_bytestring(
        lambda x: BuiltinInteger(len(x.value))
    ),
    BuiltInFun.IndexByteString: typechecked(BuiltinByteString, BuiltinInteger)(
        lambda x, y: x[y]
    ),
    BuiltInFun.EqualsByteString: two_bytestrings(lambda x, y: x == y),
    BuiltInFun.LessThanByteString: two_bytestrings(lambda x, y: x < y),
    BuiltInFun.LessThanEqualsByteString: two_bytestrings(lambda x, y: x <= y),
    BuiltInFun.Sha2_256: single_bytestring(
        lambda x: BuiltinByteString(hashlib.sha256(x.value).digest())
    ),
    BuiltInFun.Sha3_256: single_bytestring(
        lambda x: BuiltinByteString(hashlib.sha3_256(x.value).digest())
    ),
    BuiltInFun.Blake2b_256: single_bytestring(
        lambda x: BuiltinByteString(hashlib.blake2b(x.value, digest_size=32).digest())
    ),
    # BuiltInFun.VerifySignature: verify_ed25519,
    BuiltInFun.VerifyEd25519Signature: verify_ed25519,
    BuiltInFun.VerifyEcdsaSecp256k1Signature: verify_ecdsa_secp256k1,
    BuiltInFun.VerifySchnorrSecp256k1Signature: verify_schnorr_secp256k1,
    BuiltInFun.AppendString: two_strings(lambda x, y: x + y),
    BuiltInFun.EqualsString: two_strings(lambda x, y: x == y),
    BuiltInFun.EncodeUtf8: typechecked(BuiltinString)(
        lambda x: BuiltinByteString(x.value.encode("utf8"))
    ),
    BuiltInFun.DecodeUtf8: single_bytestring(
        lambda x: BuiltinString(x.value.decode("utf8"))
    ),
    BuiltInFun.IfThenElse: _IfThenElse,
    BuiltInFun.ChooseUnit: typechecked(BuiltinUnit, AST)(lambda x, y: y),
    BuiltInFun.Trace: typechecked(BuiltinString, AST)(lambda x, y: y),
    BuiltInFun.FstPair: typechecked(BuiltinPair)(lambda x: x[0]),
    BuiltInFun.SndPair: typechecked(BuiltinPair)(lambda x: x[1]),
    BuiltInFun.ChooseList: typechecked(BuiltinList, AST, AST)(
        lambda l, x, y: x if BuiltinList([], l.sample_value) == l else y
    ),
    BuiltInFun.MkCons: _MkCons,
    BuiltInFun.HeadList: typechecked(BuiltinList)(lambda l: l[0]),
    BuiltInFun.TailList: _TailList,
    BuiltInFun.NullList: typechecked(BuiltinList)(
        lambda l: BuiltinBool(l.values == [])
    ),
    BuiltInFun.ChooseData: _ChooseData,
    BuiltInFun.ConstrData: typechecked(BuiltinInteger, BuiltinList)(
        lambda x, y: PlutusConstr(x.value, y.values)
    ),
    BuiltInFun.MapData: _MapData,
    BuiltInFun.ListData: typechecked(BuiltinList)(lambda x: PlutusList(x.values)),
    BuiltInFun.IData: typechecked(BuiltinInteger)(lambda x: PlutusInteger(x.value)),
    BuiltInFun.BData: single_bytestring(lambda x: PlutusByteString(x.value)),
    BuiltInFun.UnConstrData: single_data_constr(
        lambda x: BuiltinPair(
            BuiltinInteger(x.constructor), BuiltinList(x.fields, PlutusData())
        )
    ),
    BuiltInFun.UnMapData: single_data_map(
        lambda x: BuiltinList(
            [BuiltinPair(k, v) for k, v in x.value.items()],
            BuiltinPair(PlutusData(), PlutusData()),
        )
    ),
    BuiltInFun.UnListData: single_data_list(
        lambda x: BuiltinList(x.value, PlutusData())
    ),
    BuiltInFun.UnIData: single_data_int(lambda x: BuiltinInteger(x.value)),
    BuiltInFun.UnBData: single_data_bytes(lambda x: BuiltinByteString(x.value)),
    BuiltInFun.EqualsData: typechecked(PlutusData, PlutusData)(
        lambda x, y: BuiltinBool(x == y)
    ),
    BuiltInFun.MkPairData: typechecked(Constant, Constant)(
        lambda x, y: BuiltinPair(x, y)
    ),
    BuiltInFun.MkNilData: lambda _: BuiltinList([], PlutusData()),
    BuiltInFun.MkNilPairData: lambda _: BuiltinList(
        [], BuiltinPair(PlutusData(), PlutusData())
    ),
    BuiltInFun.SerialiseData: single_data(
        lambda x: BuiltinByteString(plutus_cbor_dumps(x))
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
    version: typing.Tuple[int, int, int]
    term: AST
    _fields = ["term"]

    def eval(self, context, state):
        return self.term.eval(context, state)

    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        return f"(program {'.'.join(str(x) for x in self.version)} {self.term.dumps(dialect=dialect)})"


@dataclass
class Variable(AST):
    name: str

    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        return self.name


@dataclass
class BoundStateLambda(AST):
    var_name: str
    term: AST
    state: frozendict.frozendict
    _fields = ["term"]

    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        s = f"(lam {self.var_name} {self.term.dumps(dialect=dialect)})"
        for k, v in reversed(self.state.items()):
            s = f"[(lam {k} {s}) {v.dumps(dialect=dialect)}]"
        return s

    def ex_mem(self) -> int:
        return 1


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

    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        s = f"(delay {self.term.dumps(dialect=dialect)})"
        for k, v in reversed(self.state.items()):
            s = f"[(lam {k} {s}) {v.dumps(dialect=dialect)}]"
        return s

    def ex_mem(self) -> int:
        return 1


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

    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        return f"(force {self.term.dumps(dialect=dialect)})"


@dataclass
class ForcedBuiltIn(AST):
    builtin: BuiltInFun
    applied_forces: int
    bound_arguments: List[AST]

    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
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

    def ex_mem(self) -> int:
        return 1


@dataclass
class BuiltIn(ForcedBuiltIn):
    builtin: BuiltInFun
    applied_forces: int = dataclasses.field(default=0)
    bound_arguments: list = dataclasses.field(default_factory=lambda: [])


@dataclass
class Error(AST):
    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        return f"(error)"


@dataclass
class Apply(AST):
    f: AST
    x: AST
    _fields = ["f", "x"]

    def dumps(self, dialect=UPLCDialect.Plutus) -> str:
        return f"[{self.f.dumps(dialect=dialect)} {self.x.dumps(dialect=dialect)}]"
