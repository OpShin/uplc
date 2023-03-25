from ast import NodeVisitor
from .ast import *

UPLC_TAG_WIDTHS = {
    "term": 4,
    "type": 3,
    "constType": 4,
    "builtin": 7,
    "constant": 4,
    "kind": 1,
}

from typing import Callable, List, Union


class UplcDeserializer(BitReader):
    def __init__(self, bytes: List[int]):
        super().__init__(bytes)

    def tag_width(self, category: str) -> int:
        assert category in UPLC_TAG_WIDTHS, f"unknown tag category {category}"

        return UPLC_TAG_WIDTHS[category]

    def builtin_name(self, id: int) -> Union[str, int]:
        all = UPLC_BUILTINS

        if 0 <= id < len(all):
            return all[id].name
        else:
            print(f"Warning: builtin id {id} out of range")

            return id

    def read_linked_list(self, elem_size: int) -> List[int]:
        nil_or_cons = self.read_bits(1)

        if nil_or_cons == 0:
            return []
        else:
            return [self.read_bits(elem_size)] + self.read_linked_list(elem_size)

    def read_term(self) -> UplcTerm:
        tag = self.read_bits(self.tag_width("term"))

        if tag == 0:
            return self.read_variable()
        elif tag == 1:
            return self.read_delay()
        elif tag == 2:
            return self.read_lambda()
        elif tag == 3:
            return self.read_call()
        elif tag == 4:
            return self.read_constant()
        elif tag == 5:
            return self.read_force()
        elif tag == 6:
            return UplcError(Site.dummy())
        elif tag == 7:
            return self.read_builtin()
        else:
            raise ValueError(f"term tag {tag} unhandled")

    def read_integer(self, signed: bool = False) -> UplcInt:
        bytes = []

        b = self.read_byte()
        bytes.append(b)

        while not UplcInt.raw_byte_is_last(b):
            b = self.read_byte()
            bytes.append(b)

        res = UplcInt(
            Site.dummy(),
            UplcInt.bytes_to_big_int([UplcInt.parse_raw_byte(b) for b in bytes]),
            False,
        )

        if signed:
            res = res.to_signed()

        return res

    def read_bytes(self) -> List[int]:
        self.move_to_byte_boundary(True)

        bytes = []

        n_chunk = self.read_byte()

        while n_chunk > 0:
            for _ in range(n_chunk):
                bytes.append(self.read_byte())

            n_chunk = self.read_byte()

        return bytes

    def read_byte_array(self) -> UplcByteArray:
        bytes = self.read_bytes()

        return UplcByteArray(Site.dummy(), bytes)

    def read_string(self) -> UplcString:
        bytes = self.read_bytes()

        s = bytes_to_text(bytes)

        return UplcString(Site.dummy(), s)

    def read_list(self, typed_reader: Callable[[], UplcValue]) -> List[UplcValue]:
        items = []

        while self.read_bits(1) == 1:
            items.append(typed_reader())

        return items

    def read_data(self) -> UplcData:
        bytes = self.read_bytes()

        return UplcData.from_cbor(bytes)

    def read_variable(self) -> UplcVariable:
        index = self.read_integer()

        return UplcVariable(Site.dummy(), index)

    def read_lambda(self) -> UplcLambda:
        rhs = self.read_term()

        return UplcLambda(Site.dummy(), rhs)

    def read_call(self) -> UplcCall:
        a = self.read_term()
        b = self.read_term()

        return UplcCall(Site.dummy(), a, b)

    def read_constant(self) -> UplcConst:
        type_list = self.read_linked_list(self.tag_width("constType"))

        res = UplcConst(self.read_typed_value(type_list))

        return res

    def read_typed_value(self, type_list: List[int]) -> UplcValue:
        typed_reader = self.construct_typed_reader(type_list)

        assert len(type_list) == 0, "Did not consume all type parameters"

        return typed_reader()

    def construct_typed_reader(self, type_list: List[int]) -> Callable[[], UplcValue]:
        type = type_list.pop(0)

        if type == 0:
            return lambda: self.read_integer(True)
        elif type == 1:
            return lambda: self.read_byte_array()
        elif type == 2:
            return lambda: self.read_string()
        elif type == 3:
            return lambda: UplcUnit(Site.dummy())
        elif type == 4:
            return lambda: UplcBool(Site.dummy(), self.read_bits(1) == 1)
        elif type in (5, 6):
            raise ValueError("unexpected type tag without type application")
        elif type == 7:
            container_type = type_list.pop(0)
            if container_type == 5:
                list_type = UplcType.from_numbers(type_list)
                type_reader = self.construct_typed_reader(type_list)

                return lambda: UplcList(
                    Site.dummy(), list_type, self.read_list(type_reader)
                )
            else:
                assert container_type == 7, "Unexpected type tag"
                container_type = type_list.pop(0)
                if container_type == 6:
                    left_reader = self.construct_typed_reader(type_list)
                    right_reader = self.construct_typed_reader(type_list)
                    return lambda: UplcPair(Site.dummy(), left_reader(), right_reader())
        elif type == 8:
            return lambda: UplcDataValue(Site.dummy(), self.read_data())
        else:
            raise ValueError(f"unhandled constant type {type}")

    def read_delay(self) -> UplcDelay:
        expr = self.read_term()

        return UplcDelay(Site.dummy(), expr)

    def read_force(self) -> UplcForce:
        expr = self.read_term()

        return UplcForce(Site.dummy(), expr)

    def read_builtin(self) -> UplcBuiltin:
        id = self.read_bits(self.tag_width("builtin"))

        name = self.builtin_name(id)

        return UplcBuiltin(Site.dummy(), name)

    def finalize(self):
        self.move_to_byte_boundary(True)


def deserialize_uplc_bytes(bytes: List[int]) -> UplcProgram:
    reader = UplcDeserializer(bytes)

    version = [
        reader.read_integer(),
        reader.read_integer(),
        reader.read_integer(),
    ]

    version_key = ".".join(str(v) for v in version)

    if version_key != UPLC_VERSION:
        print(
            f"Warning: Plutus-core script doesn't match version of Helios (expected {UPLC_VERSION}, got {version_key})"
        )

    expr = reader.read_term()

    reader.finalize()

    return UplcProgram(expr, None, version)
