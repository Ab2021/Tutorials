"""Instructor solution: affine quantization and error analysis (standard library only)."""
from __future__ import annotations
import math
import random


def quantize_affine(x: list[float], bits: int = 8):
    qmin, qmax = 0, 2**bits - 1
    xmin, xmax = min(x), max(x)
    scale = (xmax - xmin) / (qmax - qmin) if xmax > xmin else 1.0
    zero_point = round(-xmin / scale)
    q = [min(qmax, max(qmin, round(v / scale) + zero_point)) for v in x]
    x_hat = [scale * (v - zero_point) for v in q]
    return q, x_hat, scale, zero_point


def quantize_symmetric(x: list[float], bits: int = 8):
    qmax = 2**(bits - 1) - 1
    peak = max(abs(v) for v in x)
    scale = peak / qmax if peak > 0 else 1.0
    q = [min(qmax, max(-qmax - 1, round(v / scale))) for v in x]
    return q, [scale * v for v in q], scale


def mse(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


if __name__ == "__main__":
    random.seed(7)
    x = [random.gauss(0, 1) for _ in range(1000)]
    _, xh, s, z = quantize_affine(x)
    print("affine scale", round(s, 4), "zero point", z, "MSE", round(mse(x, xh), 6))
    _, xh2, s2 = quantize_symmetric(x)
    print("symmetric scale", round(s2, 4), "MSE", round(mse(x, xh2), 6))
