from ast import NodeVisitor
from typing import Callable

from .ast import *

UPLC_TAG_WIDTHS = {
    "term": 4,
    "type": 3,
    "constType": 4,
    "builtin": 7,
    "constant": 4,
    "kind": 1,
}


def parse_raw_byte(b):
    """
    Parses a single byte in the Plutus-core byte-list representation of an int
    :param b: int
    :return: int
    """
    return b & 0b01111111


def raw_byte_is_last(b):
    """
    Returns true if 'b' is the last byte in the Plutus-core byte-list representation of an int.
    :param b: int
    :return: bool
    """
    return (b & 0b10000000) == 0


def bytes_to_int(bytes):
    """
    Combines a list of Plutus-core bytes into a int (leading bit of each byte is ignored).
    Differs from int.from_bytes because only 7 bits are used from each byte.
    :param bytes: list[int]
    :return: int
    """
    value = 0

    n = len(bytes)

    for i in range(n):
        b = bytes[i]

        # 7 (not 8), because leading bit isn't used here
        value += b * (2 ** (i * 7))

    return value


def unzigzag(value: int, signed: bool):
    """
    Unapplies zigzag encoding
    :return: int
    """
    if not signed:
        return value
    else:
        if value % 2 == 0:
            return value // 2
        else:
            return -((value + 1) // 2)


class UplcDeserializer:
    def __init__(self, bits: str):
        self._bits = bits
        self._pos = 0

    def tag_width(self, category: str) -> int:
        assert category in UPLC_TAG_WIDTHS, f"unknown tag category {category}"

        return UPLC_TAG_WIDTHS[category]

    def built_in_fun(self, id: int) -> BuiltInFun:
        return BuiltInFun(id)

    def read_linked_fixed_width_integer_list(self, elem_size: int) -> List[int]:
        nil_or_cons = self.read_bit()

        if nil_or_cons == 0:
            return []
        else:
            elem = self.read_fixed_width_integer(elem_size)
            return [elem] + self.read_linked_fixed_width_integer_list(elem_size)

    def read_term(self) -> AST:
        tag = self.read_tag("term")

        if tag == 0:
            return self.read_variable()
        elif tag == 1:
            return self.read_delay()
        elif tag == 2:
            return self.read_lambda()
        elif tag == 3:
            return self.read_apply()
        elif tag == 4:
            return self.read_constant()
        elif tag == 5:
            return self.read_force()
        elif tag == 6:
            return Error()
        elif tag == 7:
            return self.read_builtin()
        else:
            raise ValueError(f"term tag {tag} unhandled")

    def read_integer(self, signed: bool = False) -> int:
        byts = []

        b = self.read_byte()
        byts.append(b)

        while not raw_byte_is_last(b):
            b = self.read_byte()
            byts.append(b)

        res = bytes_to_int([parse_raw_byte(b) for b in byts])

        res = unzigzag(res, signed)

        return res

    def move_to_byte_boundary(self, force=False):
        """
        Moves position to the next byte boundary.

        Args:
            force (bool): If True, move to the next byte boundary even if already at one.

        Returns:
            None
        """
        if self._pos % 8 != 0:
            n = 8 - self._pos % 8
            self.read_bits(n)
        elif force:
            self.read_bits(8)

    def read_bytes(self) -> bytes:
        self.move_to_byte_boundary(True)

        byts = []

        n_chunk = self.read_byte()

        while n_chunk > 0:
            for _ in range(n_chunk):
                byts.append(self.read_byte())

            n_chunk = self.read_byte()

        return bytes(byts)

    def read_builtin_byte_string(self) -> BuiltinByteString:
        byts = self.read_bytes()

        return BuiltinByteString(byts)

    def read_builtin_string(self) -> BuiltinString:
        byts = self.read_bytes()

        s = byts.decode("utf8")

        return BuiltinString(s)

    def read_list(self, typed_reader: Callable[[], Constant]) -> List[Constant]:
        items = []

        while self.read_bit() == 1:
            items.append(typed_reader())

        return items

    def read_data(self) -> PlutusData:
        byts = self.read_bytes()

        return data_from_cbor(byts)

    def read_variable(self) -> Variable:
        index = self.read_integer(signed=False)

        return Variable(str(index))

    def read_lambda(self) -> Lambda:
        rhs = self.read_term()

        return Lambda("_", rhs)

    def read_apply(self) -> Apply:
        a = self.read_term()
        b = self.read_term()

        return Apply(a, b)

    def read_constant(self) -> Constant:
        type_list = self.read_linked_fixed_width_integer_list(
            self.tag_width("constType")
        )

        res = self.read_typed_value(type_list)

        return res

    def read_typed_value(self, type_list: List[int]) -> Constant:
        typed_reader = self.construct_typed_reader(type_list)

        assert len(type_list) == 0, "Did not consume all type parameters"

        return typed_reader()

    def sample_value(self, type_list: List[int]):
        typ = type_list.pop(0)
        if typ == 0:
            return BuiltinInteger(0)
        elif typ == 1:
            return BuiltinByteString(b"")
        elif typ == 2:
            return BuiltinString("")
        elif typ == 3:
            return BuiltinUnit()
        elif typ == 4:
            return BuiltinBool(False)
        elif typ in (5, 6):
            raise ValueError("unexpected type tag without type application")
        elif typ == 7:
            container_type = type_list.pop(0)
            if container_type == 5:
                list_type = self.sample_value(type_list)
                return BuiltinList([], list_type)
            else:
                assert container_type == 7, "Unexpected type tag"
                container_type = type_list.pop(0)
                if container_type == 6:
                    return BuiltinPair(
                        self.sample_value(type_list), self.sample_value(type_list)
                    )
        elif typ == 8:
            return PlutusInteger(0)
        else:
            raise ValueError(f"unhandled constant type {typ}")

    def construct_typed_reader(self, type_list: List[int]) -> Callable[[], Constant]:
        typ = type_list.pop(0)

        if typ == 0:
            return lambda: BuiltinInteger(self.read_integer(signed=True))
        elif typ == 1:
            return lambda: self.read_builtin_byte_string()
        elif typ == 2:
            return lambda: self.read_builtin_string()
        elif typ == 3:
            return lambda: BuiltinUnit()
        elif typ == 4:
            return lambda: BuiltinBool(self.read_bit() == 1)
        elif typ in (5, 6):
            raise ValueError("unexpected type tag without type application")
        elif typ == 7:
            container_type = type_list.pop(0)
            if container_type == 5:
                list_type = self.sample_value(type_list.copy())
                type_reader = self.construct_typed_reader(type_list)

                return lambda: BuiltinList(self.read_list(type_reader), list_type)
            else:
                assert container_type == 7, "Unexpected type tag"
                container_type = type_list.pop(0)
                if container_type == 6:
                    left_reader = self.construct_typed_reader(type_list)
                    right_reader = self.construct_typed_reader(type_list)
                    return lambda: BuiltinPair(left_reader(), right_reader())
                else:
                    raise ValueError(f"unhandled container type {container_type}")
        elif typ == 8:
            return lambda: self.read_data()
        else:
            raise ValueError(f"unhandled constant type {typ}")

    def read_delay(self) -> Delay:
        expr = self.read_term()

        return Delay(expr)

    def read_force(self) -> Force:
        expr = self.read_term()

        return Force(expr)

    def read_builtin(self) -> BuiltIn:
        id = self.read_tag("builtin")

        builtin = self.built_in_fun(id)

        return BuiltIn(builtin)

    def finalize(self):
        self.move_to_byte_boundary(True)

    def read_bits(self, num: int) -> str:
        bits = self._bits[self._pos : self._pos + num]
        self._pos += num
        return bits

    def read_fixed_width_integer(self, width: int) -> int:
        return int(self.read_bits(width), 2)

    def read_tag(self, name: str) -> int:
        return self.read_fixed_width_integer(self.tag_width(name))

    def read_bit(self) -> int:
        return self.read_fixed_width_integer(1)

    def read_byte(self) -> int:
        return self.read_fixed_width_integer(8)

    def read_program(self) -> Program:
        version = (
            self.read_integer(signed=False),
            self.read_integer(signed=False),
            self.read_integer(signed=False),
        )

        expr = self.read_term()

        self.finalize()

        return Program(version, expr)
