from ast import NodeVisitor
from .ast import *


def _int_to_bits(x: int):
    return bin(x)[2:]


class BitWriter:
    """
    BitWriter turns a string of '0's and '1's into a list of bytes.
    Finalization pads the bits using '0*1' if not yet aligned with the byte boundary.
    """

    def __init__(self):
        self._parts = []
        self._n = 0

    @property
    def length(self):
        """Number of bits written so far"""
        return self._n

    def write(self, bit_chars):
        """
        Write a string of '0's and '1's to the BitWriter.
        :param bit_chars: string
        """
        assert all(
            c in ("0", "1") for c in bit_chars
        ), "did pass invalid value to write in bitwriter"

        self._parts.append(bit_chars)
        self._n += len(bit_chars)

    def write_fixed_width_int(self, i: int, width: int):
        """
        Write an integer with fixed width (number of bits)
        :param i: int
        :param width: int
        """
        self.write(self.pad_zeroes(_int_to_bits(i), width))

    def write_nibble(self, nibble: int):
        """
        Write a nibble to the BitWriter.
        :param nibble: int
        """
        self.write_fixed_width_int(nibble, 4)

    def write_byte(self, byte: int):
        """
        Write a byte to the BitWriter.
        :param byte: int
        """
        self.write_fixed_width_int(byte, 8)

    def write_bytes(self, byts: bytes):
        """
        Write a bytestring to the BitWriter.
        :param bytes: bytearray
        """
        self.pad_to_byte_boundary(True)
        n = len(byts)
        pos = 0
        while pos < n:
            n_chunk = min(n - pos, 255)
            self.write_byte(n_chunk)
            for byte in byts[pos : pos + n_chunk]:
                self.write_byte(byte)
            pos += n_chunk
        self.write_byte(0)

    @staticmethod
    def pad_zeroes(s, n):
        return s.zfill(n)

    def pad_to_byte_boundary(self, force=False):
        """
        Add padding to the BitWriter in order to align with the byte boundary.
        If 'force == True' then 8 bits are added if the BitWriter is already aligned.
        :param force: bool
        """
        n_pad = 0
        if self._n % 8 != 0:
            n_pad = 8 - self._n % 8
        elif force:
            n_pad = 8

        if n_pad != 0:
            padding = ["0"] * n_pad
            padding[n_pad - 1] = "1"

            self._parts.append("".join(padding))
            self._n += n_pad

    def finalize(self, force=True):
        """
        Pads the BitWriter to align with the byte boundary and returns the resulting bytes.
        :param force: bool - force padding (will add one byte if already aligned)
        :return: list[int]
        """
        self.pad_to_byte_boundary(force)

        chars = "".join(self._parts)

        bytes_list = []

        for i in range(0, len(chars), 8):
            byte_chars = chars[i : i + 8]
            byte = int(byte_chars, 2)

            bytes_list.append(byte)

        return bytes(bytes_list)

    def write_int(self, i: int, signed: bool):
        i = int(i)
        assert signed or i >= 0, f"Tried to encode unsigned int {i} but is negative"
        zigzagged = zigzag(i, signed)
        bitstring = pad_zeroes(_int_to_bits(zigzagged), 7)

        # split every 7th
        parts = list(chunkstring(bitstring, 7))
        parts.reverse()

        # write all but the last
        for chunk in parts[:-1]:
            self.write("1" + chunk)
        # write the last
        self.write("0" + parts[-1])


def zigzag(i: int, signed: bool):
    """Zigzag-encode an integer"""
    if not signed:
        return i
    else:
        if i < 0:
            return 2 * (-i) - 1
        else:
            return 2 * i


def chunkstring(string, length):
    """Chunk a string into parts of fixed length"""
    return (string[0 + i : length + i] for i in range(0, len(string), length))


def pad_zeroes(bits, n):
    """Prepends zeroes to a bit-string so that 'len(result) == n'."""
    if len(bits) % n != 0:
        n_pad = n - (len(bits) % n)
        bits = n_pad * "0" + bits
    return bits


class FlatEncodingVisitor(NodeVisitor):
    def __init__(self, bw: typing.Optional[BitWriter] = None):
        self.bit_writer = BitWriter() if bw is None else bw

    def visit_Program(self, n: Program):
        for i in n.version:
            self.bit_writer.write_int(i, signed=False)
        self.visit(n.term)

    def visit_Variable(self, n: Variable):
        self.bit_writer.write("0000")
        # this requires the deBrujin encoding rather than the actual variable name
        self.bit_writer.write_int(int(n.name), signed=False)

    def visit_Delay(self, n: Delay):
        self.bit_writer.write("0001")
        self.visit(n.term)

    def visit_Lambda(self, n: Lambda):
        self.bit_writer.write("0010")
        self.visit(n.term)

    def visit_Apply(self, n: Apply):
        self.bit_writer.write("0011")
        self.visit(n.f)
        self.visit(n.x)

    def visit_Constant(self, n: Constant):
        self.bit_writer.write("0100")
        self.bit_writer.write("1")
        ConstantTypeFlatEncodingVisitor(self.bit_writer).visit(n)
        self.bit_writer.write("0")
        ConstantValueFlatEncodingVisitor(self.bit_writer).visit(n)

    def visit_Force(self, n: Force):
        self.bit_writer.write("0101")
        self.visit(n.term)

    def visit_Error(self, n: Error):
        self.bit_writer.write("0110")

    def visit_BuiltIn(self, n: BuiltIn):
        self.bit_writer.write("0111")
        # write index of uplc builtin
        index = n.builtin.value
        self.bit_writer.write_fixed_width_int(index, width=7)

    def visit_BuiltinUnit(self, n: BuiltinUnit):
        self.visit_Constant(n)

    def visit_BuiltinBool(self, n: BuiltinBool):
        self.visit_Constant(n)

    def visit_BuiltinInteger(self, n: BuiltinInteger):
        self.visit_Constant(n)

    def visit_BuiltinByteString(self, n: BuiltinByteString):
        self.visit_Constant(n)

    def visit_BuiltinString(self, n: BuiltinString):
        self.visit_Constant(n)

    def visit_BuiltinPair(self, n: BuiltinPair):
        self.visit_Constant(n)

    def visit_BuiltinList(self, n: BuiltinList):
        self.visit_Constant(n)

    def visit_PlutusData(self, n: PlutusData):
        self.visit_Constant(n)

    def visit_PlutusInteger(self, n: PlutusInteger):
        self.visit_Constant(n)

    def visit_PlutusByteString(self, n: PlutusByteString):
        self.visit_Constant(n)

    def visit_PlutusList(self, n: PlutusList):
        self.visit_Constant(n)

    def visit_PlutusMap(self, n: PlutusMap):
        self.visit_Constant(n)

    def visit_PlutusConstr(self, n: PlutusConstr):
        self.visit_Constant(n)


class ConstantValueFlatEncodingVisitor(NodeVisitor):
    def __init__(self, bw: typing.Optional[BitWriter] = None):
        self.bit_writer = BitWriter() if bw is None else bw

    def visit_BuiltinUnit(self, n: BuiltinUnit):
        pass

    def visit_BuiltinBool(self, n: BuiltinBool):
        self.bit_writer.write(str(int(n.value)))

    def visit_BuiltinInteger(self, n: BuiltinInteger):
        self.bit_writer.write_int(n.value, signed=True)

    def visit_BuiltinByteString(self, n: BuiltinByteString):
        self.bit_writer.write_bytes(n.value)

    def visit_BuiltinString(self, n: BuiltinString):
        self.bit_writer.write_bytes(n.value.encode("utf8"))

    def visit_BuiltinPair(self, n: BuiltinPair):
        self.visit(n.l_value)
        self.visit(n.r_value)

    def visit_BuiltinList(self, n: BuiltinList):
        for v in n.values:
            self.bit_writer.write("1")
            self.visit(v)
        self.bit_writer.write("0")

    def visit_PlutusData(self, n: PlutusData):
        self.bit_writer.write_bytes(plutus_cbor_dumps(n))

    def visit_PlutusInteger(self, n: PlutusInteger):
        self.visit_PlutusData(n)

    def visit_PlutusByteString(self, n: PlutusByteString):
        self.visit_PlutusData(n)

    def visit_PlutusList(self, n: PlutusList):
        self.visit_PlutusData(n)

    def visit_PlutusMap(self, n: PlutusMap):
        self.visit_PlutusData(n)

    def visit_PlutusConstr(self, n: PlutusConstr):
        self.visit_PlutusData(n)


class ConstantTypeFlatEncodingVisitor(NodeVisitor):
    def __init__(self, bw: typing.Optional[BitWriter] = None):
        self.bit_writer = BitWriter() if bw is None else bw

    def visit_BuiltinUnit(self, n: BuiltinUnit):
        self.bit_writer.write_nibble(3)

    def visit_BuiltinBool(self, n: BuiltinBool):
        self.bit_writer.write_nibble(4)

    def visit_BuiltinInteger(self, n: BuiltinInteger):
        self.bit_writer.write_nibble(0)

    def visit_BuiltinByteString(self, n: BuiltinByteString):
        self.bit_writer.write_nibble(1)

    def visit_BuiltinString(self, n: BuiltinString):
        self.bit_writer.write_nibble(2)

    def visit_BuiltinPair(self, n: BuiltinPair):
        self.bit_writer.write_nibble(7)
        self.bit_writer.write("1")
        self.bit_writer.write_nibble(7)
        self.bit_writer.write("1")
        self.bit_writer.write_nibble(6)
        self.bit_writer.write("1")
        self.visit(n.l_value)
        self.bit_writer.write("1")
        self.visit(n.r_value)

    def visit_BuiltinList(self, n: BuiltinList):
        self.bit_writer.write_nibble(7)
        self.bit_writer.write("1")
        self.bit_writer.write_nibble(5)
        self.bit_writer.write("1")
        self.visit(n.sample_value)

    def visit_PlutusData(self, n: PlutusData):
        self.bit_writer.write_nibble(8)

    def visit_PlutusInteger(self, n: PlutusInteger):
        self.visit_PlutusData(n)

    def visit_PlutusByteString(self, n: PlutusByteString):
        self.visit_PlutusData(n)

    def visit_PlutusList(self, n: PlutusList):
        self.visit_PlutusData(n)

    def visit_PlutusMap(self, n: PlutusMap):
        self.visit_PlutusData(n)

    def visit_PlutusConstr(self, n: PlutusConstr):
        self.visit_PlutusData(n)
