from typing import Callable, List, TypeVar, Union
from .zigzag import to_usize, to_u128

T = TypeVar("T")


class Error(Exception):
    pass


class BufferNotByteAlignedError(Error):
    pass


class Encoder:
    def __init__(self):
        self.buffer = bytearray()
        self.used_bits = 0
        self.current_byte = 0

    def encode(self, x: T) -> "Encoder":
        x.encode(self)
        return self

    def u8(self, x: int) -> "Encoder":
        if self.used_bits == 0:
            self.current_byte = x
            self.next_word()
        else:
            self.byte_unaligned(x)

        return self

    def bool(self, x: bool) -> "Encoder":
        if x:
            self.one()
        else:
            self.zero()

        return self

    def bytes(self, x: bytes) -> "Encoder":
        self.filler()
        return self.byte_array(x)

    def byte_array(self, arr: bytes) -> "Encoder":
        if self.used_bits != 0:
            raise BufferNotByteAlignedError()

        self.write_blk(arr)
        return self

    def integer(self, i: int) -> "Encoder":
        i = to_usize(i)
        return self.word(i)

    def big_integer(self, i: int) -> "Encoder":
        i = to_u128(i)
        return self.big_word(i)

    def char(self, c: str) -> "Encoder":
        return self.word(ord(c))

    def string(self, s: str) -> "Encoder":
        for i in s:
            self.one()
            self.char(i)

        self.zero()
        return self

    def utf8(self, s: str) -> "Encoder":
        return self.bytes(s.encode("utf-8"))

    def word(self, c: int) -> "Encoder":
        d = c
        while True:
            w = d & 127
            d >>= 7

            if d != 0:
                w |= 128
            self.bits(8, w)

            if d == 0:
                break

        return self

    def big_word(self, c: int) -> "Encoder":
        d = c
        while True:
            w = d & 127
            d >>= 7

            if d != 0:
                w |= 128
            self.bits(8, w)

            if d == 0:
                break

        return self

    def encode_list_with(
        self, list_: List[T], encoder_func: Callable[[T, "Encoder"], None]
    ) -> "Encoder":
        for item in list_:
            self.one()
            encoder_func(item, self)

        self.zero()
        return self

    def bits(self, num_bits: int, val: int) -> "Encoder":
        if num_bits == 1:
            if val == 0:
                self.zero()
            elif val == 1:
                self.one()
        elif num_bits == 2:
            if val == 0:
                self.zero()
                self.zero()
            elif val == 1:
                self.zero()
                self.one()
            elif val == 2:
                self.one()
                self.zero()
            elif val == 3:
                self.one()
                self.one()
        else:
            self.used_bits += num_bits
            unused_bits = 8 - self.used_bits
            if unused_bits > 0:
                self.current_byte |= val << unused_bits
            elif unused_bits == 0:
                self.current_byte |= val
                self.next_word()
            else:
                used = -unused_bits
                self.current_byte |= val >> used
                self.next_word()
                self.current_byte = val << (8 - used)
                self.used_bits = used

        return self

    def filler(self):
        self.current_byte |= 1
        self.next_word()
        return self

    def zero(self):
        if self.used_bits == 7:
            self.next_word()
        else:
            self.used_bits += 1

    def one(self):
        if self.used_bits == 7:
            self.current_byte |= 1
            self.next_word()
        else:
            self.current_byte |= 128 >> self.used_bits
            self.used_bits += 1

    def byte_unaligned(self, x: int):
        x_shift = self.current_byte | (x >> self.used_bits)
        self.buffer.append(x_shift)
        self.current_byte = x << (8 - self.used_bits)

    def next_word(self):
        self.buffer.append(self.current_byte)
        self.current_byte = 0
        self.used_bits = 0

    def write_blk(self, arr: bytes):
        chunks = [arr[i : i + 255] for i in range(0, len(arr), 255)]
        for chunk in chunks:
            self.buffer.append(len(chunk))
            self.buffer.extend(chunk)
        self.buffer.append(0)
