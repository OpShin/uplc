from .classes import *

# Generator of G1
P1 = G1Point(1, 2)
# Generator of G2
P2 = G2Point(
    10857046999023057135944570762232829481370756359578518086990519993285655852781,
    11559732032986387107991004021392285783925812861821192530917403151452391805634,
    8495653923123431417604973247489272438418190587263600148770280649306958101930,
    4082367875863433681332203403145435568316851327593401208105741076214120093531,
)

# prime q in the base field F_q for G1
q = 21888242871839275222246405745257275088696311157297823662689037894645226208583


def bn256_addition(p1: G1Point, p2: G1Point) -> G1Point:
    """
    the sum of two points of G1
    """
    # equivalent to precompiled bn256add
    # https://docs.moonbeam.network/builders/pallets-precompiles/precompiles/eth-mainnet/
    # TODO black magic addition
    from .bn128.bn128_curve import add, FQ2, FQ

    mp1 = (FQ(p1.x), FQ(p1.y))
    mp2 = (FQ(p2.x), FQ(p2.y))
    mr = add(mp1, mp2)
    r = G1Point(mr[0].n, mr[1].n)
    return r


def bn256_scalar_mul(p: G1Point, s: int) -> G1Point:
    """
    returns the product of a point p on G1 and a scalar s, i.e.
    p == p.scalar_mul(1) and p.addition(p) == p.scalar_mul(2) for all points p.
    """
    # equivalent to precompiled bn256mul
    # https://docs.moonbeam.network/builders/pallets-precompiles/precompiles/eth-mainnet/
    # TODO black magic scalar mul
    from .bn128.bn128_curve import multiply, FQ

    mp = (FQ(p.x), FQ(p.y))
    mr = multiply(mp, s)
    r = G1Point(mr[0].n, mr[1].n)
    return r


def bn128_pairing(ps1: List[G1Point], ps2: List[G2Point]) -> bool:
    """
    return the result of computing the pairing check
    e(p1[0], p2[0]) *  .... * e(p1[n], p2[n]) == 1
    For example pairing([P1(), P1().negate()], [P2(), P2()]) should
    return true.
    """
    # equivalent to precompiled bn128pairing
    # https://docs.moonbeam.network/builders/pallets-precompiles/precompiles/eth-mainnet/
    # go implementation https://github.com/ethereum/go-ethereum/blob/c7c84ca16c724edbf9adad10893de502a5ee7e0a/core/vm/contracts.go#L511
    # TODO black magic pairing check
    from .bn128.bn128_pairing import optimal_ate_pairing_check, FQ2, FQ

    mp1 = [(FQ(p1.x), FQ(p1.y)) for p1 in ps1]
    mp2 = [(FQ2([FQ(p2.x1), FQ(p2.x2)]), FQ2([FQ(p2.y1), FQ(p2.y2)])) for p2 in ps2]
    return optimal_ate_pairing_check(list(zip(mp1, mp2)))
