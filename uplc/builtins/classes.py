from pycardano import PlutusData
from dataclasses import dataclass
from typing import List


@dataclass
class G1Point(PlutusData):
    x: int
    y: int


@dataclass
class G2Point(PlutusData):
    x1: int
    x2: int
    y1: int
    y2: int


@dataclass
class Proof(PlutusData):
    a: G1Point
    b: G2Point
    c: G1Point


@dataclass
class VerifyingKey(PlutusData):
    alpha: G1Point
    beta: G2Point
    gamma: G2Point
    delta: G2Point
    gamma_abc: List[G1Point]
