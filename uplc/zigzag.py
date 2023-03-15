def to_usize(x: int) -> int:
    double_x = x << 1

    if x > 0 or x == 0:
        return double_x
    else:
        return -double_x - 1


def to_isize(u: int) -> int:
    return (u >> 1) ^ (-((u & 1)))


def to_u128(x: int) -> int:
    double_x = x << 1

    if x > 0 or x == 0:
        return double_x
    else:
        return -double_x - 1


def to_i128(u: int) -> int:
    return (u >> 1) ^ (-((u & 1)))
