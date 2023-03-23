from ast import NodeVisitor
from .ast import *


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
        for c in bit_chars:
            if c != "0" and c != "1":
                raise ValueError("bad bit char")

        self._parts.append(bit_chars)
        self._n += len(bit_chars)

    def write_byte(self, byte):
        """
        Write a byte to the BitWriter.
        :param byte: int
        """
        self.write(self.pad_zeroes(bin(byte)[2:], 8))

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

        return bytes_list


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


def flatten_int(i: int, signed: bool, bw: BitWriter):
    assert signed or i > 0, f"Tried to encode unsigned int {i} but is negative"
    zigzagged = zigzag(i, signed)
    bitstring = pad_zeroes(bin(zigzagged)[2:], 7)

    # split every 7th
    parts = list(chunkstring(bitstring, 7))
    parts.reverse()

    # write all but the last
    for chunk in parts[:-1]:
        bw.write("0" + chunk)
    # write the last
    bw.write("1" + parts[-1])


class FlatEncodingVisitor(NodeVisitor):
    def __init__(self):
        self.bit_writer = BitWriter()

    def visit_Program(self, n: Program):
        for i in n.version:
            flatten_int(i, False, self.bit_writer)
        self.visit(n.term)

    def visit_Variable(self, n: Variable):
        self.bit_writer.write("0000")
        # TODO this requires the deBrujin encoding rather than the actual variable name
        flatten_int(n.index)
